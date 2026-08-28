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


class ClusteredRetriever:
    def __init__(self, embedder, n_clusters=20, routing_top_k=3,
                 alpha=0.6, use_hybrid=True):
        self.embedder = embedder
        self.n_clusters = n_clusters
        self.routing_top_k = routing_top_k  # 路由时选几个簇
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
        """保存聚类结果，下次直接加载不用重新算"""
        np.savez(
            f"{path_prefix}_clusters.npz",
            centers=self.cluster_centers,
            labels=self.cluster_labels
        )
        with open(f"{path_prefix}_cluster_indices.pkl", 'wb') as f:
            pickle.dump(self.cluster_chunk_indices, f)
        if self.bm25 is not None:
            with open(f"{path_prefix}_bm25.pkl", 'wb') as f:
                pickle.dump(self.bm25, f)
        print(f"✅ 聚类索引已保存到 {path_prefix}_clusters.npz")

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
        with open(f"{path_prefix}_cluster_indices.pkl", 'rb') as f:
            self.cluster_chunk_indices = pickle.load(f)

        # ===== 修复核心：恢复 chunks + embeddings =====
        if chunks_file and embeddings_file and os.path.exists(chunks_file) and os.path.exists(embeddings_file):
            with open(chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            self.embeddings = np.load(embeddings_file)
        else:
            print("⚠️ load_index: 未找到 chunks / embeddings 文件，"
                  "search 前请先调用 index_documents 或传入正确路径")

        # ===== 加载 / 重建 BM25（混合检索）=====
        bm25_path = f"{path_prefix}_bm25.pkl"
        if os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                self.bm25 = pickle.load(f)
            print("✅ 加载 BM25 索引（混合检索启用）")
        elif self.chunks:
            print("⚠️ 未找到 BM25 缓存，从已加载 chunks 重建（混合检索将生效）")
            self._build_bm25()
        else:
            self.bm25 = None
            print("⚠️ BM25 未构建，search 将回退为纯向量检索")

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

        # ===== Stage 1: 路由 —— 拿 query 和所有簇中心比 =====
        t0 = time()
        center_scores = []
        qv_norm = np.linalg.norm(qv)
        for i, center in enumerate(self.cluster_centers):
            dot = np.dot(qv, center)
            center_norm = np.linalg.norm(center)
            sim = dot / (qv_norm * center_norm) if qv_norm and center_norm else 0
            center_scores.append((i, sim))

        center_scores.sort(key=lambda x: x[1], reverse=True)
        selected_clusters = center_scores[:self.routing_top_k]

        # ===== Stage 2: 候选集 =====
        candidate_indices = []
        for cluster_id, _ in selected_clusters:
            candidate_indices.extend(self.cluster_chunk_indices[cluster_id])

        # dense 相似度
        raw = []
        for idx in candidate_indices:
            c = self.embeddings[idx]
            c_norm = np.linalg.norm(c)
            sim = np.dot(qv, c) / (qv_norm * c_norm) if c_norm else 0
            raw.append((idx, sim))

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
        print(f"[{hybrid_tag}检索] 路由→簇 {[f'#{c}({s:.3f})' for c, s in selected_clusters]} | "
              f"候选 {len(candidate_indices)} 个 | 耗时 {elapsed:.1f}ms")

        return results