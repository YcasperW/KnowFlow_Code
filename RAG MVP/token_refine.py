"""KnowFlow Token 优化模块 —— 基于腾讯 Smart Context 方法论

三层优化策略：
  Layer 1: Smart Context — LLM 智能判断哪些上下文真正有用
  Layer 2: Handoff Compression — 结构化压缩为 key_facts + source_refs + artifact paths
  Layer 3: Artifact On-Demand Loading — 完整内容存外部，按需拉取

新增：
  QueryRouter — 零成本问题分类，简单问题走快车道
  TokenOptimizerCache — 磁盘缓存，Layer 1+2 命中后 Token 消耗归零
"""
import json
import os
import hashlib
import pickle
import re
from datetime import datetime
from pathlib import Path
import numpy as np


# ============================================================
# === NEW: 磁盘缓存层
# ============================================================
class TokenOptimizerCache:
    """三层管线的磁盘缓存 —— 高频问题第二次起 Layer 1+2 零 Token"""

    def __init__(self, cache_dir="output/cache", max_size=500,
                 vector_threshold=0.95):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.vector_threshold = vector_threshold

        self._index_file = self.cache_dir / "_vector_index.pkl"
        self._load_index()

    def _load_index(self):
        if self._index_file.exists():
            self._index = pickle.loads(self._index_file.read_bytes())
        else:
            self._index = {}

    def _save_index(self):
        self._index_file.write_bytes(pickle.dumps(self._index))

    @staticmethod
    def _cosine_similarity(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _evict_if_needed(self):
        files = list(self.cache_dir.glob("*"))
        files = [f for f in files if f.name != "_vector_index.pkl"]
        if len(files) > self.max_size:
            files.sort(key=lambda f: f.stat().st_mtime)
            for f in files[:len(files) - self.max_size]:
                f.unlink()
                self._index.pop(f.stem, None)
            self._save_index()

    def _make_key(self, *args):
        raw = "|".join(str(a) for a in args)
        return hashlib.md5(raw.encode()).hexdigest()

    def get_layer1_scores(self, query_embedding, chunk_ids):
        if query_embedding is None:
            return None

        exact_key = self._make_key("L1", *chunk_ids,
                                   *[f"{x:.4f}" for x in query_embedding[:16]])
        exact_path = self.cache_dir / f"{exact_key}.pkl"
        if exact_path.exists():
            return pickle.loads(exact_path.read_bytes())

        best_key = None
        best_sim = -1.0
        for key, cached_emb in self._index.items():
            sim = self._cosine_similarity(query_embedding, cached_emb)
            if sim > self.vector_threshold and sim > best_sim:
                best_sim = sim
                best_key = key

        if best_key:
            path = self.cache_dir / f"{best_key}.pkl"
            if path.exists():
                print(f"⚡ Layer 1 语义命中 (相似度 {best_sim:.3f})，跳过 LLM 打分")
                return pickle.loads(path.read_bytes())

        return None

    def set_layer1_scores(self, query_embedding, chunk_ids, scores):
        if query_embedding is None:
            return
        key = self._make_key("L1", *chunk_ids,
                             *[f"{x:.4f}" for x in query_embedding[:16]])
        path = self.cache_dir / f"{key}.pkl"
        path.write_bytes(pickle.dumps(scores))
        self._index[key] = query_embedding
        self._save_index()
        self._evict_if_needed()

    def get_layer2_result(self, chunk_contents_tuple):
        key = self._make_key("L2", *chunk_contents_tuple)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def set_layer2_result(self, chunk_contents_tuple, result):
        key = self._make_key("L2", *chunk_contents_tuple)
        path = self.cache_dir / f"{key}.json"
        storable = {k: v for k, v in result.items() if k != "artifact_loader"}
        path.write_text(json.dumps(storable, ensure_ascii=False, indent=2), encoding="utf-8")
        self._evict_if_needed()


# ============================================================
# === NEW: 问题路由器
# ============================================================
class QueryRouter:
    def __init__(self, simple_top_k=2, complex_top_k=8):
        self.simple_top_k = simple_top_k
        self.complex_top_k = complex_top_k

    def classify(self, query, retrieved_chunks_with_scores=None):
        if len(query) <= 15:
            return "simple", self.simple_top_k
        word_count = len(re.findall(r'[\w\u4e00-\u9fff]+', query))
        if word_count <= 5:
            return "simple", self.simple_top_k
        if retrieved_chunks_with_scores and len(retrieved_chunks_with_scores) >= 2:
            scores = [s for _, s in retrieved_chunks_with_scores]
            if scores[0] - scores[1] > 0.15:
                return "simple", self.simple_top_k
        return "complex", self.complex_top_k


# ============================================================
# Layer 1: Smart Context 智能筛选
# ============================================================
class SmartContextFilter:
    def __init__(self, client, model="deepseek-chat", cache=None):
        self.client = client
        self.model = model
        self.cache = cache

    def score_chunks(self, query, chunks, query_embedding=None):
        if not chunks:
            return []

        chunk_ids = [c.get("id", c["content"][:30]) for c in chunks]

        if self.cache and query_embedding is not None:
            cached_scores = self.cache.get_layer1_scores(query_embedding, chunk_ids)
            if cached_scores:
                print("⚡ Layer 1 语义缓存命中，跳过 LLM 打分")
                scored = list(zip(chunks, cached_scores))
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored

        chunk_list = "\n".join([
            f"[片段{i + 1}] {c['content'][:200]}..."
            for i, c in enumerate(chunks)
        ])

        prompt = f"""你是一个上下文质量评估器。用户提出了一个问题，
并从知识库中检索到了若干参考片段。请逐个评估每个片段对该问题的价值。

问题：{query}

参考片段：
{chunk_list}

请对每个片段给出一个 0-10 的相关度分数，标准：
10分 = 该片段直接、完整地回答了问题的核心
7-9分 = 该片段包含部分有用信息
4-6分 = 该片段与问题弱相关，可能有间接帮助
0-3分 = 该片段与问题基本无关

以 JSON 数组格式输出分数即可：
[分数1, 分数2, ..., 分数N]"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
                timeout=30
            )
            result_text = response.choices[0].message.content.strip()
            print(f"[Debug] Layer 1 LLM 原始返回: {result_text[:200]}")  # ← 新增 debug

            numbers = re.findall(r'[0-9]+(?:\.[0-9]+)?', result_text)
            scores = [float(n) for n in numbers[:len(chunks)]]

            while len(scores) < len(chunks):
                scores.append(5.0)

            if self.cache and query_embedding is not None:
                self.cache.set_layer1_scores(query_embedding, chunk_ids, scores)

            scored = list(zip(chunks, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            print(f"📊 Smart Context 评分: "
                  f"{[round(s, 1) for _, s in scored]}")

            return scored

        except Exception as e:
            print(f"⚠️ Smart Context 评分失败: {e}，使用全部片段")
            return [(c, 8.0) for c in chunks]

    def filter_by_threshold(self, scored_chunks, threshold=5.0, min_keep=1):
        kept = [(c, s) for c, s in scored_chunks if s >= threshold]
        if len(kept) < min_keep:
            kept = scored_chunks[:min_keep]
        removed_count = len(scored_chunks) - len(kept)
        if removed_count > 0:
            print(f"✂️ Smart Context 过滤: "
                  f"{len(scored_chunks)} → {len(kept)} 条 "
                  f"(丢弃 {removed_count} 条低分片段)")
        return [c for c, _ in kept]


# ============================================================
# Layer 2: Handoff 结构化压缩（修复 JSON 容错）
# ============================================================
class HandoffCompressor:
    def __init__(self, client, model="deepseek-chat",
                 artifact_dir="output/artifacts", cache=None):
        self.client = client
        self.model = model
        self.artifact_dir = artifact_dir
        self.cache = cache
        os.makedirs(artifact_dir, exist_ok=True)

    @staticmethod
    def _estimate_tokens(text):
        if not text:
            return 0
        return len(text) // 2

    def compress(self, query, chunks):
        if self.cache:
            contents_tuple = tuple(c["content"] for c in chunks)
            cached = self.cache.get_layer2_result(contents_tuple)
            if cached:
                print("⚡ Layer 2 命中缓存，跳过 LLM 压缩")
                from token_refine import ArtifactLoader
                cached["artifact_loader"] = {
                    "paths": cached.get("artifact_paths", []),
                    "loader": ArtifactLoader(),
                    "enabled": True
                }
                return cached

        combined_text = "\n\n".join([f"[{c.get('source', '?')}] {c['content']}"
                                     for c in chunks])

        prompt = f"""你是一个信息提取器。以下是从企业文档中检索到的参考资料，
以及用户的问题。请完成两件事：

1. 提取 3-5 个最关键的事实要点（每个不超过 20 字）
2. 标注每个事实的来源文档和大概位置

问题：{query}

参考资料：
{combined_text}

请以严格的 JSON 格式输出（不要输出其他内容）：
{{
  "key_facts": [
    {{"fact": "事实内容", "source_doc": "文档名", "location": "位置"}},
    ...
  ],
  "summary": "一段话概括这些材料的核心信息（50字以内）"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
                timeout=30
            )
            result_text = response.choices[0].message.content.strip()
            print(f"[Debug] Layer 2 LLM 原始返回: {result_text[:300]}")

            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                try:
                    compressed_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # 尝试修复截断的 JSON
                    fixed = json_match.group()
                    fixed = re.sub(r',\s*([\}\]])', r'\1', fixed)
                    fixed = fixed.replace("'", '"')
                    start = fixed.find('{')
                    if start != -1 and '}' not in fixed[start:]:
                        # JSON 被截断，补上结尾
                        fixed = fixed.rstrip() + ']}'
                    if start != -1:
                        try:
                            compressed_data = json.loads(fixed[start:])
                        except json.JSONDecodeError:
                            print(f"⚠️ Handoff JSON 修复失败（截断），使用轻量级压缩")
                            return self._fallback_compress(query, chunks)
                    else:
                        print(f"⚠️ Handoff JSON 结构不完整，使用轻量级压缩")
                        return self._fallback_compress(query, chunks)
            else:
                print("⚠️ 未找到 JSON 结构，使用轻量级压缩")
                return self._fallback_compress(query, chunks)

            # ===== 以下原逻辑不变 =====
            artifact_paths = []
            for i, c in enumerate(chunks):
                file_hash = hashlib.md5(c["content"].encode()).hexdigest()[:12]
                artifact_name = f"chunk_{file_hash}.txt"
                artifact_path = os.path.join(self.artifact_dir, artifact_name)
                with open(artifact_path, 'w', encoding='utf-8') as f:
                    f.write(f"Source: {c.get('source', 'unknown')}\n")
                    f.write(f"Saved: {datetime.now().isoformat()}\n")
                    f.write("-" * 40 + "\n")
                    f.write(c["content"])
                artifact_paths.append(artifact_path)

            compressed_prompt = self._build_compressed_prompt(compressed_data, query)
            original_tokens = self._estimate_tokens(combined_text)
            compressed_tokens = self._estimate_tokens(compressed_prompt)
            # 修复：原文为空文本时 original_tokens=0，这里会抛 ZeroDivisionError，
            # 异常被外层 except 吞掉后白白浪费一次 LLM 调用并降级。
            # 同时把节省率下限截到 0：提示语本身有固定开销，
            # 片段极短时算出负值会让前端显示「节省 -30%」。
            savings = max(0.0, (1 - compressed_tokens / original_tokens) * 100) if original_tokens > 0 else 0.0

            print(f"🗜️ Handoff 压缩: "
                  f"{original_tokens} → {compressed_tokens} tokens "
                  f"(节省 {savings:.0f}%)")

            result = {
                "key_facts": compressed_data.get("key_facts", []),
                "summary": compressed_data.get("summary", ""),
                "source_refs": [
                    (f.get("source_doc", "?"), f.get("location", "?"))
                    for f in compressed_data.get("key_facts", [])
                ],
                "artifact_paths": artifact_paths,
                "compressed_prompt": compressed_prompt,
                "stats": {
                    "original_tokens": original_tokens,
                    "compressed_tokens": compressed_tokens,
                    "savings_percent": round(savings, 1),
                    "fact_count": len(compressed_data.get("key_facts", []))
                }
            }

            if self.cache:
                contents_tuple = tuple(c["content"] for c in chunks)
                self.cache.set_layer2_result(contents_tuple, result)

            return result

        except Exception as e:
            print(f"⚠️ Handoff 压缩失败: {e}")
            return self._fallback_compress(query, chunks)

    def _build_compressed_prompt(self, data, query):
        lines = [
            f"【问题】{query}",
            f"",
            f"【参考资料摘要】{data.get('summary', '')}",
            f"",
            f"【关键事实】"
        ]
        for i, fact in enumerate(data.get("key_facts", [])):
            src = fact.get("source_doc", "")
            loc = fact.get("location", "")
            lines.append(f"  {i + 1}. {fact['fact']} （来源：{src} {loc}）")
        lines.append("")
        lines.append("请基于以上关键事实回答问题。如需更详细信息可引用具体来源。")
        return "\n".join(lines)

    def _fallback_compress(self, query, chunks):
        facts = []
        refs = []
        artifact_paths = []
        for c in chunks:
            first_sentence = c["content"].split("。")[0] + "。"
            facts.append({"fact": first_sentence[:80],
                          "source_doc": c.get("source", "?"),
                          "location": ""})
            refs.append((c.get("source", "?"), ""))
            file_hash = hashlib.md5(c["content"].encode()).hexdigest()[:12]
            ap = os.path.join(self.artifact_dir, f"chunk_{file_hash}.txt")
            with open(ap, 'w', encoding='utf-8') as f:
                f.write(c["content"])
            artifact_paths.append(ap)

        cp = self._build_compressed_prompt({"key_facts": facts, "summary": ""}, query)
        ot = self._estimate_tokens("\n".join([c["content"] for c in chunks]))
        ct = self._estimate_tokens(cp)

        result = {
            "key_facts": facts, "summary": "",
            "source_refs": refs, "artifact_paths": artifact_paths,
            "compressed_prompt": cp,
            "stats": {"original_tokens": ot, "compressed_tokens": ct,
                      # 同上：防除零 + 不为负
                      "savings_percent": round(max(0.0, (1 - ct / ot) * 100), 1) if ot > 0 else 0,
                      "fact_count": len(facts)}
        }
        if self.cache:
            contents_tuple = tuple(c["content"] for c in chunks)
            self.cache.set_layer2_result(contents_tuple, result)
        return result


# ============================================================
# Layer 3: Artifact 按需加载
# ============================================================
class ArtifactLoader:
    def __init__(self, artifact_dir="output/artifacts"):
        self.artifact_dir = artifact_dir

    def load(self, artifact_path):
        full_path = artifact_path
        if not os.path.isabs(full_path):
            full_path = os.path.join(self.artifact_dir,
                                     os.path.basename(artifact_path))
        if not os.path.exists(full_path):
            return None
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"📎 Artifact 加载: {os.path.basename(full_path)} "
              f"({len(content)} 字符)")
        return content

    def batch_load(self, artifact_paths, max_load=3):
        loaded = []
        for path in artifact_paths[:max_load]:
            content = self.load(path)
            if content:
                loaded.append(content)
        return loaded


# ============================================================
# 统一入口
# ============================================================
class KnowFlowTokenOptimizer:
    def __init__(self, client, model="deepseek-chat",
                 smart_context_threshold=5.0,
                 enable_artifact=True,
                 enable_cache=True,
                 cache_dir="output/cache",
                 artifact_dir="output/artifacts",
                 simple_top_k=2,
                 complex_top_k=8):

        self.client = client
        self.model = model
        # 修复：这两个目录原是相对路径，工作目录一变（例如从别处启动服务）
        # 缓存和产物就会散落到当前目录，导致缓存永不命中、artifact 找不到。
        # 统一按项目根目录解析成绝对路径。
        _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isabs(cache_dir):
            cache_dir = os.path.join(_BASE, cache_dir)
        if not os.path.isabs(artifact_dir):
            artifact_dir = os.path.join(_BASE, artifact_dir)
        self.cache = TokenOptimizerCache(cache_dir) if enable_cache else None

        self.layer1_filter = SmartContextFilter(client, model, cache=self.cache)
        # 修复：ArtifactLoader 与 HandoffCompressor 原先各自用默认目录，
        # 一旦传入自定义 artifact_dir，压缩写进去的路径加载器就找不到，
        # 按需加载会静默失效。这里统一用同一个目录。
        self.layer2_compressor = HandoffCompressor(
            client, model, artifact_dir=artifact_dir, cache=self.cache
        )
        self.layer3_loader = ArtifactLoader(artifact_dir)
        self.router = QueryRouter(
            simple_top_k=simple_top_k,
            complex_top_k=complex_top_k
        )
        self.smart_threshold = smart_context_threshold
        self.enable_artifact = enable_artifact
        self.stats = {}

    def optimize(self, question, chunks, query_embedding=None, retrieved_scores=None):
        route, top_k = self.router.classify(question, retrieved_scores)

        if route == "simple":
            return self._fast_path(question, chunks, top_k)

        print("\n" + "=" * 50)
        print("🚀 KnowFlow 三层管线启动（复杂问题）")
        print("=" * 50 + "\n")

        original_text = "\n".join([c["content"] for c in chunks])
        original_tokens = self._estimate_tokens(original_text)

        print("--- Layer 1: Smart Context 智能筛选 ---")
        scored_chunks = self.layer1_filter.score_chunks(
            question, chunks,
            query_embedding=query_embedding
        )
        filtered_chunks = self.layer1_filter.filter_by_threshold(
            scored_chunks, threshold=self.smart_threshold
        )

        print("\n--- Layer 2: Handoff 结构化压缩 ---")
        compression_result = self.layer2_compressor.compress(
            question, filtered_chunks
        )

        print("\n--- Layer 3: Artifact 模式注册 ---")
        artifact_info = {
            "paths": compression_result["artifact_paths"],
            "loader": self.layer3_loader if self.enable_artifact else None,
            "enabled": self.enable_artifact
        }
        print(f"   已注册 {len(artifact_info['paths'])} 个 artifact 文件")

        final_tokens = self._estimate_tokens(
            compression_result["compressed_prompt"]
        )
        # 下限截到 0：把「问题」「请基于以上资料回答」等提示开销也算进 final 后，
        # 短片段场景可能算出负值，前端会显示成「节省 -X%」。
        total_savings = max(0.0, (1 - final_tokens / original_tokens) * 100) if original_tokens > 0 else 0

        report = {
            "original_tokens": original_tokens,
            "after_smart_context": self._estimate_tokens(
                "\n".join([c["content"] for c in filtered_chunks])
            ),
            "after_compression": final_tokens,
            "total_savings_percent": round(total_savings, 1),
            "layer1_filtered": len(chunks) - len(filtered_chunks),
            "layer2_compression_rate": compression_result["stats"]["savings_percent"],
            "key_fact_count": compression_result["stats"]["fact_count"],
            "artifact_count": len(compression_result["artifact_paths"]),
            "route": "complex",
            "timestamp": datetime.now().isoformat()
        }

        self._print_report(report)

        return {
            "final_prompt": compression_result["compressed_prompt"],
            "source_refs": compression_result["source_refs"],
            "key_facts": compression_result["key_facts"],
            "artifact_loader": artifact_info,
            "report": report,
            "route": "complex",
            "layer1_result": {"scored": scored_chunks, "kept": filtered_chunks},
            "layer2_result": compression_result
        }

    def _fast_path(self, question, chunks, top_k=2):
        print("\n⚡ 快车道模式（简单问题，跳过三层管线）\n")
        kept = chunks[:top_k]
        context = "\n\n".join([f"[{c.get('source', '?')}] {c['content']}"
                                for c in kept])
        final_prompt = f"【问题】{question}\n\n【参考资料】\n{context}\n\n请基于以上资料回答问题。"
        original_tokens = self._estimate_tokens(
            "\n".join([c["content"] for c in chunks])
        )
        final_tokens = self._estimate_tokens(final_prompt)
        report = {
            "original_tokens": original_tokens,
            "after_smart_context": final_tokens,
            "after_compression": final_tokens,
            "total_savings_percent": round(
                max(0.0, (1 - final_tokens / original_tokens) * 100), 1
            ) if original_tokens > 0 else 0,
            "layer1_filtered": len(chunks) - len(kept),
            "layer2_compression_rate": 0,
            "key_fact_count": 0,
            "artifact_count": 0,
            "route": "simple",
            "timestamp": datetime.now().isoformat()
        }
        print(f"⚡ 快车道: {len(chunks)} → {len(kept)} 条, "
              f"~{final_tokens} tokens")
        return {
            "final_prompt": final_prompt,
            "source_refs": [(c.get("source", "?"), "") for c in kept],
            "key_facts": [],
            "artifact_loader": {"paths": [], "loader": None, "enabled": False},
            "report": report,
            "route": "simple",
            "layer1_result": None,
            "layer2_result": None
        }

    def expand_on_demand(self, artifact_paths, indices=None):
        if not self.enable_artifact or not artifact_paths:
            return []
        target_paths = ([artifact_paths[i] for i in indices]
                        if indices else artifact_paths)
        return self.layer3_loader.batch_load(target_paths)

    @staticmethod
    def _estimate_tokens(text):
        if not text:
            return 0
        return len(text) // 2

    def _print_report(self, report):
        print("\n" + "=" * 50)
        print("💰 KnowFlow Token 优化报告")
        print("=" * 50)
        print(f"  路由:             {report.get('route', 'complex')}")
        print(f"  原始检索文本:     {report['original_tokens']:>6} tokens")
        print(f"  Smart Context 后:  {report['after_smart_context']:>6} tokens")
        print(f"  压缩后:           {report['after_compression']:>6} tokens")
        print(f"  总节省:           {report['total_savings_percent']}%\n")