import numpy as np
from sklearn.cluster import KMeans
import json, pickle, os, math, re
from time import time
from collections import Counter
try:
    import jieba
except ImportError:
    jieba = None


class BM25:
    """轻量 Okapi BM25 实现，无第三方依赖（仅 numpy/math）。
    用于混合检索中的关键词匹配分支，零 LLM token 消耗。"""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avgdl = 0.0
        self.doc_len = []
        self.tf = []        # 每个文档的 term -> freq
        self.idf = {}
        self.corpus_tokens = []

    def fit(self, corpus):
        self.doc_count = len(corpus)
        self.corpus_tokens = corpus
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = (sum(self.doc_len) / self.doc_count) if self.doc_count else 0.0
        self.tf = [Counter(d) for d in corpus]
        df = {}
        for c in self.tf:
            for t in c:
                df[t] = df.get(t, 0) + 1
        self.idf = {}
        for t, d in df.items():
            self.idf[t] = math.log((self.doc_count - d + 0.5) / (d + 0.5) + 1.0)

    def get_scores_subset(self, query_tokens, indices):
        """只给候选文档算 BM25 分，避免全量扫描（配合聚类加速）。"""
        res = {}
        qt = set(query_tokens)
        for i in indices:
            dl = self.doc_len[i]
            tf_i = self.tf[i]
            score = 0.0
            for q in qt:
                f = tf_i.get(q, 0)
                if f == 0:
                    continue
                denom = (f + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                         ) if self.avgdl > 0 else (f + self.k1)
                score += self.idf.get(q, 0.0) * f * (self.k1 + 1) / denom
            res[i] = score
        return res

    # ---------- 序列化：JSON 替代 pickle ----------
    # 为什么不直接用 pickle：pickle 反序列化会执行 GLOBAL/REDUCE 指令重建对象，
    # 加载不可信文件等于任意代码执行；且它绑定类定义，改个字段名旧索引就静默失效。
    # 这里只落纯数据（dict/list/数值），可读、可 diff、跨语言，加载也更快。
    def to_dict(self):
        """导出为纯 JSON 可序列化结构。corpus_tokens 不落盘——检索时不用，
        需要时可由 chunks 重新分词得到，落下只会让文件白白翻倍。"""
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avgdl": self.avgdl,
            "doc_len": list(self.doc_len),
            "tf": [dict(c) for c in self.tf],
            "idf": dict(self.idf),
        }

    @classmethod
    def from_dict(cls, d):
        """从 JSON 结构还原。tf 里的 Counter 需重建，其余直接赋值。"""
        obj = cls(k1=d.get("k1", 1.5), b=d.get("b", 0.75))
        obj.doc_count = int(d.get("doc_count", 0))
        obj.avgdl = float(d.get("avgdl", 0.0))
        obj.doc_len = list(d.get("doc_len", []))
        obj.tf = [Counter(x) for x in d.get("tf", [])]
        obj.idf = dict(d.get("idf", {}))
        obj.corpus_tokens = []
        return obj


# 索引格式版本。v1 = pickle（_cluster_indices.pkl / _bm25.pkl），v2 = JSON + npz。
# 加载时若只发现 v1 文件会自动迁移到 v2，旧文件改名为 .migrated 保留，可随时回滚。
INDEX_VERSION = 2
LEGACY_INDEX_VERSION = 1

# ===== 检索路由策略（用真实 932 条索引实测标定，见 MEMORY.md「检索性能实测」）=====
# 小库（N <= 阈值）直接全量参与，不路由。理由：路由是要漏召的——
# 实测 nprobe=3 召回 95.7%、nprobe=5 召回 97.7%，只有全量才是 100%。
# 代价是 search() 从 0.36 ms 涨到 2.23 ms（多出的是 BM25 全库扫描 1.58 ms），
# 而一次 DeepSeek 生成要 2000–8000 ms，2 ms 完全无感。用 2 ms 换 2.3~4.3% 召回，值得。
# 注意：别被「全库点积只要 0.039 ms」误导——那只算了 dense 矩阵乘，
# 真正的耗时大头是 BM25 的 Python 循环，dense 只占 0.045 ms。
BRUTE_FORCE_MAX_N = int(os.getenv("KNOWFLOW_BRUTE_FORCE_MAX_N", "5000"))
# 大库（N > 阈值）才走 IVF 路由，探测几个簇（即 nprobe）。
# 召回随 nprobe 递增但边际递减：3 → 95.7%，5 → 97.7%，全量 → 100%。
DEFAULT_ROUTING_TOP_K = int(os.getenv("KNOWFLOW_ROUTING_TOP_K", "5"))


class ClusteredRetriever:
    def __init__(self, embedder, n_clusters=20, routing_top_k=None,
                 alpha=0.4, use_hybrid=True):
        self.embedder = embedder
        self.n_clusters = n_clusters
        # 路由时选几个簇（nprobe）。None → 用 DEFAULT_ROUTING_TOP_K（实测标定值 5）
        self.routing_top_k = DEFAULT_ROUTING_TOP_K if routing_top_k is None else routing_top_k
        self.alpha = alpha                  # 融合权重：dense 占 alpha，bm25 占 1-alpha
        self.use_hybrid = use_hybrid        # 是否开启混合检索
        self._jieba = jieba                 # 中文分词（有则用，无则回退）
        self.chunks = []
        self.embeddings = None
        self.cluster_centers = None
        self.cluster_labels = None
        self.cluster_chunk_indices = {}  # cluster_id → [chunk_idx, ...]
        self.bm25 = None                  # BM25 索引（混合检索用）

    def index_documents(self, chunks_file, embeddings_file):
        """加载 chunks + embeddings，然后做 K-Means 聚类"""
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        # 防御：空知识库（chunks 为空或文件损坏）不聚类，直接置空状态，search 返回 []
        if not self.chunks:
            self.embeddings = None
            self.cluster_centers = None
            self.cluster_labels = None
            self.cluster_chunk_indices = {}
            self.bm25 = None
            print("⚠️ 知识库为空，索引置空（search 将返回空结果）")
            return
        self.embeddings = np.load(embeddings_file)

        # 防御：文档块数少于设定簇数时，KMeans 会直接报错；自动下调避免崩溃
        if len(self.chunks) < self.n_clusters:
            self.n_clusters = max(1, len(self.chunks))

        print(f"⏳ 开始聚类 {len(self.chunks)} 个文档块 → {self.n_clusters} 个簇...")
        t0 = time()

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        self.cluster_centers = kmeans.cluster_centers_

        # 建立 簇 → chunk索引 的映射
        self.cluster_chunk_indices = {
            i: [] for i in range(self.n_clusters)
        }
        for idx, label in enumerate(self.cluster_labels):
            self.cluster_chunk_indices[label].append(idx)

        elapsed = time() - t0
        sizes = [len(v) for v in self.cluster_chunk_indices.values()]
        print(f"✅ 聚类完成 ({elapsed:.1f}s) | 簇大小: min={min(sizes)} max={max(sizes)} avg={np.mean(sizes):.1f}")

        # ===== 构建 BM25（混合检索关键词分支）=====
        self._build_bm25()

    def save_index(self, path_prefix):
        """保存聚类结果，下次直接加载不用重新算。

        落盘格式：质心/标签用 npz（数值数组，本就不是 pickle），
        簇→chunk 映射和 BM25 用 JSON（纯数据，可读可 diff）。
        """
        np.savez(
            f"{path_prefix}_clusters.npz",
            centers=self.cluster_centers,
            labels=self.cluster_labels
        )

        # 簇 → chunk 下标映射。JSON 的 key 只能是字符串，读取时转回 int。
        with open(f"{path_prefix}_cluster_indices.json", 'w', encoding='utf-8') as f:
            json.dump({
                "version": INDEX_VERSION,
                "n_clusters": self.n_clusters,
                "cluster_chunk_indices": {
                    str(k): v for k, v in self.cluster_chunk_indices.items()
                },
            }, f, ensure_ascii=False)

        if self.bm25 is not None:
            with open(f"{path_prefix}_bm25.json", 'w', encoding='utf-8') as f:
                json.dump({"version": INDEX_VERSION, **self.bm25.to_dict()},
                          f, ensure_ascii=False)

        # 新的 JSON 已写成功，同名的旧 pickle 若还在就作废，避免新旧两份不一致
        self._drop_legacy_pkl(path_prefix)
        print(f"✅ 聚类索引已保存到 {path_prefix}_clusters.npz (+ JSON)")

    @staticmethod
    def _drop_legacy_pkl(path_prefix):
        """删除已被 JSON 取代的旧 pickle 索引（重命名而非 rm，出问题可回滚）。"""
        for suffix in ("_cluster_indices.pkl", "_bm25.pkl"):
            old = f"{path_prefix}{suffix}"
            if os.path.exists(old):
                os.replace(old, old + ".migrated")

    def _load_cluster_indices(self, path_prefix):
        """读取 簇→chunk 映射：优先 JSON，回退旧 pickle 并顺带迁移。"""
        json_path = f"{path_prefix}_cluster_indices.json"
        pkl_path = f"{path_prefix}_cluster_indices.pkl"

        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            return {int(k): v for k, v in payload.get("cluster_chunk_indices", {}).items()}

        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            print("🔁 检测到 v1 pickle 索引，自动迁移为 JSON")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "version": INDEX_VERSION,
                    "n_clusters": len(data),
                    "cluster_chunk_indices": {str(k): v for k, v in data.items()},
                }, f, ensure_ascii=False)
            os.replace(pkl_path, pkl_path + ".migrated")
            return {int(k): v for k, v in data.items()}

        print("⚠️ 未找到簇索引文件，将由 cluster_labels 现场重建")
        return {}

    def _load_bm25(self, path_prefix):
        """读取 BM25：优先 JSON，回退旧 pickle 并迁移，最后才从 chunks 重建。"""
        json_path = f"{path_prefix}_bm25.json"
        pkl_path = f"{path_prefix}_bm25.pkl"

        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            # 版本不符就当没有，走重建分支，避免旧结构喂出新行为
            if payload.get("version") == INDEX_VERSION:
                self.bm25 = BM25.from_dict(payload)
                print("✅ 加载 BM25 索引（JSON，混合检索启用）")
                return
            print("⚠️ BM25 索引版本不符，改为从 chunks 重建")

        elif os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                self.bm25 = pickle.load(f)
            print("🔁 检测到 v1 pickle BM25，自动迁移为 JSON")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({"version": INDEX_VERSION, **self.bm25.to_dict()},
                          f, ensure_ascii=False)
            os.replace(pkl_path, pkl_path + ".migrated")
            print("✅ 加载 BM25 索引（混合检索启用）")
            return

        if self.chunks:
            print("⚠️ 未找到 BM25 缓存，从已加载 chunks 重建（混合检索将生效）")
            self._build_bm25()
        else:
            self.bm25 = None
            print("⚠️ BM25 未构建，search 将回退为纯向量检索")

    def load_index(self, path_prefix, chunks_file=None, embeddings_file=None):
        """加载预计算的聚类结果。

        关键修复：之前只恢复了聚类中心 / 标签 / 簇索引，
        却没恢复 self.chunks 和 self.embeddings。
        这导致 search() 在 Stage 2 执行 self.embeddings[idx] 时，
        self.embeddings 为 None → 报 'NoneType' object is not subscriptable。
        现在一并把 chunks / embeddings 加载回来，search 才能正常跑。
        """
        data = np.load(f"{path_prefix}_clusters.npz")
        self.cluster_centers = data['centers']
        self.cluster_labels = data['labels']
        # 修复：让 n_clusters 与实际加载的簇数一致，
        # 否则热更新后删除到少于原簇数时，重启会带着错误的 n_clusters。
        self.n_clusters = len(self.cluster_centers)
        self.cluster_chunk_indices = self._load_cluster_indices(path_prefix)

        # 兜底：簇索引缺失/为空时，npz 里的 labels 足以现场重建映射
        if not self.cluster_chunk_indices and self.cluster_labels is not None:
            rebuilt = {i: [] for i in range(self.n_clusters)}
            for idx, label in enumerate(self.cluster_labels):
                rebuilt.setdefault(int(label), []).append(idx)
            self.cluster_chunk_indices = rebuilt
            print("✅ 已由 cluster_labels 重建簇→chunk 映射")

        # ===== 修复核心：恢复 chunks + embeddings =====
        if chunks_file and embeddings_file and os.path.exists(chunks_file) and os.path.exists(embeddings_file):
            with open(chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            self.embeddings = np.load(embeddings_file)
        else:
            print("⚠️ load_index: 未找到 chunks / embeddings 文件，"
                  "search 前请先调用 index_documents 或传入正确路径")

        # ===== 加载 / 迁移 / 重建 BM25（混合检索）=====
        # 必须放在 chunks 加载之后：重建分支要用 self.chunks 重新分词
        self._load_bm25(path_prefix)

        print(f"✅ 加载聚类索引: {self.n_clusters} 个簇")

    def _tokenize(self, text):
        """分词：优先 jieba（中文友好），无则回退到 英文词 + 中文字符。"""
        text = (text or "").lower()
        if self._jieba is not None:
            return [t for t in self._jieba.cut(text) if t.strip()]
        tokens = re.findall(r'[a-z0-9]+', text)
        tokens += re.findall(r'[\u4e00-\u9fff]', text)
        return tokens

    def _build_bm25(self):
        """用当前 chunks 的内容构建 BM25 索引。"""
        corpus = [self._tokenize(c.get("content", "")) for c in self.chunks]
        self.bm25 = BM25()
        self.bm25.fit(corpus)

    def search(self, query, top_k=5):
        """两阶段检索 + 混合融合：
        Stage 1 聚类路由（dense）→ Stage 2 候选集内 dense 与 BM25 融合。
        混合检索不调用 LLM，零额外 token。
        """
        # 防御：索引为空（知识库无文档 / 已删光）时，避免对 None 聚类中心迭代导致崩溃
        if not self.chunks or self.cluster_centers is None or self.embeddings is None:
            return []

        qv = self.embedder.embed_text(query)
        t0 = time()
        qv_norm = np.linalg.norm(qv)

        n_docs = len(self.chunks)
        # ===== 候选集选取：小库暴力 / 大库 IVF 路由 =====
        # 小库：全量参与，召回 100%，且省掉路由的 Python 开销（实测更快）。
        # 大库：KMeans 路由到 top-nprobe 个簇，只在这些簇内精算。
        use_brute = n_docs <= BRUTE_FORCE_MAX_N
        if use_brute:
            selected_clusters = []
            candidate_indices = list(range(n_docs))
        else:
            # Stage 1: 路由 —— 拿 query 和所有簇中心比（向量化）
            centers = np.asarray(self.cluster_centers, dtype=float)
            center_norms = np.linalg.norm(centers, axis=1)
            safe = (qv_norm > 0) & (center_norms > 0)
            center_sim = np.zeros(len(centers), dtype=float)
            if safe.any():
                center_sim[safe] = (centers[safe] @ qv) / (qv_norm * center_norms[safe])
            nprobe = max(1, min(self.routing_top_k, len(centers)))
            top_clusters = np.argpartition(-center_sim, nprobe)[:nprobe]
            selected_clusters = [(int(c), float(center_sim[c])) for c in top_clusters]
            selected_clusters.sort(key=lambda x: x[1], reverse=True)

            # Stage 2: 候选集
            candidate_indices = []
            for cluster_id, _ in selected_clusters:
                candidate_indices.extend(self.cluster_chunk_indices[cluster_id])

            # 兜底：路由异常导致候选为空时退化为全量，避免直接拒答
            if not candidate_indices:
                candidate_indices = list(range(n_docs))

        # ===== dense 相似度（向量化余弦）=====
        cand = np.asarray(candidate_indices, dtype=int)
        # 全量模式下候选就是全库，此时直接引用、不拷贝（省掉一次整库 1.4 MB 复制）。
        # 统一按 float32 计算：余弦相似度无需 float64，可再省一半内存带宽。
        src = np.asarray(self.embeddings)
        emb = src if len(cand) == len(src) else src[cand]
        emb = emb.astype(np.float32, copy=False)
        emb_norms = np.linalg.norm(emb, axis=1)
        dense_vals = np.zeros(len(cand), dtype=float)
        if qv_norm > 0:
            qv32 = np.asarray(qv, dtype=np.float32).reshape(-1)
            nz_mask = emb_norms > 0
            dense_vals[nz_mask] = (emb[nz_mask] @ qv32) / (qv_norm * emb_norms[nz_mask])
        raw = list(zip(cand.tolist(), dense_vals.tolist()))

        # BM25 关键词分（仅候选集，零 token）
        if self.use_hybrid and self.bm25 is not None:
            query_tokens = self._tokenize(query)
            bm25_map = self.bm25.get_scores_subset(query_tokens, candidate_indices)
        else:
            bm25_map = {idx: 0.0 for idx in candidate_indices}

        # 归一化 + 融合
        idxs = [i for i, _ in raw]
        dense_arr = np.array([s for _, s in raw], dtype=float)
        bm25_arr = np.array([bm25_map[i] for i in idxs], dtype=float)

        def _minmax(a):
            if a.size == 0:
                return a
            mn, mx = a.min(), a.max()
            return (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a)

        dense_n = _minmax(dense_arr)
        bm25_n = _minmax(bm25_arr)
        fused = self.alpha * dense_n + (1 - self.alpha) * bm25_n

        order = np.argsort(-fused)[:top_k]
        results = []
        for pos in order:
            i = idxs[pos]
            results.append({
                "chunk": self.chunks[i],
                "score": float(fused[pos]),
                "dense": float(dense_arr[pos]),
                "bm25": float(bm25_arr[pos]),
            })

        elapsed = (time() - t0) * 1000
        hybrid_tag = "混合" if (self.use_hybrid and self.bm25) else "纯向量"
        if use_brute:
            route_tag = f"全量暴力(N={n_docs}≤{BRUTE_FORCE_MAX_N})"
        else:
            route_tag = "路由→簇 " + str(
                [f"#{c}({s:.3f})" for c, s in selected_clusters])
        print(f"[{hybrid_tag}检索] {route_tag} | "
              f"候选 {len(candidate_indices)} 个 | 耗时 {elapsed:.1f}ms")

        return results