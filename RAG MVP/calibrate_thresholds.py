# -*- coding: utf-8 -*-
"""
标定 KnowFlow 低置信闸门阈值（KNOWFLOW_MIN_SCORE / KNOWFLOW_BM25_FLOOR）。

方法：
  · 在库查询：用每个 chunk 的内容作为查询，跑真实 search()，
    排除自身匹配(同一 chunk 对象)后取 top-5 内的 max dense / max bm25
    —— 模拟“用户问了语料里有的话题”。
  · 离库查询：一批明显与操作手册无关的问题，同样取 max dense / max bm25
    —— 模拟“知识库未覆盖的问题”。
  · 闸门逻辑 = AND拒答 / OR放行：best_dense<CONF 且 best_bm25<FLOOR 才拒。
    目标：在库查询几乎全放行、离库查询几乎全拒答。
"""
import os, json, contextlib, io
import numpy as np
from sentence_transformers import SentenceTransformer
from clustered_retriever import ClusteredRetriever

# 修复：原为硬编码绝对路径，换台机器/换个目录就 FileNotFoundError。
# 改为相对本文件定位项目根（与 Web Frame.py 的路径规则一致）。
BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
CHUNKS = os.path.join(BASE, "chunks.json")
EMB = os.path.join(BASE, "embeddings.npy")
IDX = os.path.join(BASE, "cluster_index")
TOP_K = 5

# —— embedder 包装（与 RAG_pipeline.TextEmbedder.embed_text 行为一致）——
class MiniEmbedder:
    def __init__(self, name):
        self.model = SentenceTransformer(name)
    def embed_text(self, text):
        return self.model.encode(text)

embedder = MiniEmbedder("paraphrase-multilingual-MiniLM-L12-v2")
retriever = ClusteredRetriever(embedder=embedder, n_clusters=20, routing_top_k=3, alpha=0.4)
retriever.load_index(IDX, chunks_file=CHUNKS, embeddings_file=EMB)

# 注意：必须用 retriever 内部加载的同一批 chunk 对象做身份比对，
# 否则“排除自身匹配”会因对象不同而不生效（会把自身 1.0 相似度算进去）。
chunks = retriever.chunks
print(f"chunks 总数: {len(chunks)}")

def best_scores(query):
    with contextlib.redirect_stdout(io.StringIO()):
        res = retriever.search(query, top_k=TOP_K)
    if not res:
        return 0.0, 0.0
    return max(r["dense"] for r in res), max(r["bm25"] for r in res)

def best_scores_excl_self(query, self_chunk):
    with contextlib.redirect_stdout(io.StringIO()):
        res = retriever.search(query, top_k=TOP_K)
    others = [r for r in res if r["chunk"] is not self_chunk]
    if not others:
        return 0.0, 0.0
    return max(r["dense"] for r in others), max(r["bm25"] for r in others)

# 1) 在库查询
on_d, on_b = [], []
for c in chunks:
    q = c.get("content", "")
    if not q.strip():
        continue
    d, b = best_scores_excl_self(q, c)
    on_d.append(d); on_b.append(b)
print(f"在库查询样本: {len(on_d)}")

# 2) 离库查询（明显无关）
off_queries = [
    "今天北京天气怎么样", "如何做麻婆豆腐", "2022世界杯冠军是谁", "1加1等于几",
    "周杰伦最新专辑叫什么", "新手怎么学吉他", "python 快速排序怎么写", "新冠病毒有什么症状",
    "相对论讲的是什么", "篮球比赛基本规则", "今天股市涨了吗", "怎么养猫",
    "红楼梦的作者是谁", "地球到月球有多远", "怎么考取驾照", "推荐一部好看的科幻电影",
    "失眠了有什么办法", "信用卡怎么还款", "用英语做个自我介绍", "怎么煮手冲咖啡",
    "如何训练马拉松", "量子计算机原理", "西红柿炒鸡蛋的做法", "怎么给手机贴膜",
]
off_d, off_b = [], []
for q in off_queries:
    d, b = best_scores(q)
    off_d.append(d); off_b.append(b)

def pct(vals, p):
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]

def show(title, vals):
    print(f"\n=== {title} (n={len(vals)}) ===")
    for p in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
        print(f"  p{p:>3}: {pct(vals, p):.3f}")

show("在库查询 best_dense", on_d)
show("在库查询 best_bm25", on_b)
show("离库查询 best_dense", off_d)
show("离库查询 best_bm25", off_b)

print(f"\n离库 best_dense 最大值: {max(off_d):.3f}")
print(f"离库 best_bm25 最大值: {max(off_b):.3f}")

# 多组阈值对比：在库(应放行)放行率越高越好，离库(应拒答)放行率越低越好
def pass_rate(d_list, b_list, c, f):
    p = sum(1 for d, b in zip(d_list, b_list) if not (d < c and b < f))
    return p, len(d_list)

candidates = [
    (0.22, 1.0,   "当前默认"),
    (0.52, 20.0,  "推荐A"),
    (0.55, 20.0,  "推荐B"),
    (0.55, 25.0,  "推荐C"),
    (0.608, 42.5, "脚本原推荐(偏激进)"),
]
print("\n=== 阈值对比（AND拒答/OR放行）===")
print(f"{'方案':>22} | {'CONF':>5} {'FLOOR':>6} | {'在库放行':>10} | {'离库放行':>10}")
for c, f, name in candidates:
    on_p, on_n = pass_rate(on_d, on_b, c, f)
    off_p, off_n = pass_rate(off_d, off_b, c, f)
    print(f"{name:>22} | {c:>5} {f:>6} | {on_p}/{on_n} ({on_p/on_n*100:4.1f}%) | "
          f"{off_p}/{off_n} ({off_p/off_n*100:4.1f}%)")

# 推荐结论
print("\n推荐: CONF=0.55, FLOOR=20")
print("  · 离库 max dense=0.516 < 0.55 → 离库全部卡在 dense 轴；")
print("  · 离库 max bm25=16.2  < 20   → 即便关键词有重叠也卡在 bm25 轴；")
print("  · 在库 min dense=0.50 (仅1条) < 0.55，但其在库 bm25=33.5 ≥ 20 → OR 救回；")
print("  · 在库 min bm25=33.5  > 20    → 在库全部靠 bm25 或 dense 至少一项越过下限。")
