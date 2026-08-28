"""RAG 系统自动评估模块
基于 RAGAS 思路实现（不依赖 ragas 库）。

当前唯一保留的评估器：
  EmbeddingRAGEvaluator —— 纯嵌入向量余弦相似度，**零 token 消耗、确定可复现**，
                          适合频繁跑回归、做「质量量化」证据。

（注：早期曾提供调用 LLM 打分的 RAGEvaluator 版本，因每次评估都烧 DeepSeek token、
且与检索阶段共用同一套嵌入空间，评分信号并无本质增益，已归档移除，仅保留下方余弦版。）
"""
import json  # noqa: F401  （保留 import 以兼容其他脚本可能的引用，当前未直接使用）
import os    # noqa: F401
import re
import numpy as np


def _cosine(a, b):
    """余弦相似度，自动对二维矩阵做 mean-pooling。返回 [0,1]（裁剪负值）。

    - a、b 可以是「单个向量」（一维）或「多个向量拼成的矩阵」（二维）。
    - 若传入二维（如一个文本块被切成多段嵌入），先沿行取平均压成一维，
      保证后面比较时维数一致。
    - 分母用 L2 范数归一化，结果落在 [-1,1]；这里用 np.clip 把负值裁成 0，
      只保留「相似」语义（0=不相关，1=完全相同方向）。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 2:
        a = a.mean(axis=0)
    if b.ndim == 2:
        b = b.mean(axis=0)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


class EmbeddingRAGEvaluator:
    """非 LLM 版 RAGAS 评估器。

    思路：用嵌入向量余弦相似度近似三项指标，**不调任何 LLM、零 token 消耗**，
    结果确定可复现（同一输入永远同分），适合 CI / 频繁回归。

      - faithfulness（忠实度）：把答案拆成句子，每句与任一参考片段做余弦；
                                被支撑（≥阈值）的句子占比即为忠实度。
      - relevancy（相关性）：问题向量 与 回答向量 的余弦。
      - context_recall（上下文召回）：有标准答案时算 GT 与参考片段的余弦；
                                      无 GT 时退化为 问题与参考片段的余弦。

    入参 embedder 需提供 embed_text(text) -> vector（与检索器共用同一 embedder 即可，
    保证「检索空间」和「评分空间」是同一套向量语义，可比、可解释）。
    """

    def __init__(self, embedder, faith_threshold=0.55, recall_threshold=0.55):
        # embedder：嵌入器（与 RAG 检索共用同一个），负责把文字转成向量。
        self.embedder = embedder
        # faith_threshold：判定「某句答案被上下文支撑」所需的最低余弦分。
        self.faith_threshold = faith_threshold
        # recall_threshold：residual 预留阈值（当前 recall 计算未用，保留扩展位）。
        self.recall_threshold = recall_threshold

    def _emb(self, text):
        """把一段文字交给 embedder 转成 numpy 向量，统一成 float 方便后面算余弦。"""
        e = self.embedder.embed_text(text)
        return np.asarray(e, dtype=float)

    def evaluate_faithfulness(self, answer, contexts):
        """忠实度：答案里有多少句话，能在参考片段里找到支撑。

        做法：
          1. 没有参考片段 → 直接 0 分（无据可查）。
          2. 用正则按中英文句号/问号/换行把答案切成句子。
          3. 每句向量与所有参考片段向量逐一算余弦，取最高分。
          4. 最高分 ≥ faith_threshold 算「被支撑」，统计支撑句占比。
        """
        if not contexts:
            return {"score": 0.0, "details": "no contexts", "supported": 0, "total": 0}
        ctx_embs = [self._emb(c) for c in contexts]
        sents = [s for s in re.split(r'[。！？!?\n]', answer) if s.strip()]
        if not sents:
            return {"score": 0.0, "details": "empty answer", "supported": 0, "total": 0}
        supported = 0
        for s in sents:
            s_emb = self._emb(s)
            sims = [_cosine(s_emb, ce) for ce in ctx_embs]
            if max(sims) >= self.faith_threshold:
                supported += 1
        score = supported / len(sents)
        print(f"✅ 忠实度(嵌入)评分: {score:.2f}  ({supported}/{len(sents)} 句被支撑)")
        return {"score": score, "supported": supported, "total": len(sents)}

    def evaluate_relevancy(self, question, answer):
        """相关性：问题与回答的语义贴合度。

        直接算「问题向量」和「回答向量」的余弦。分高 = 答非所问的概率低。
        """
        sim = _cosine(self._emb(question), self._emb(answer))
        print(f"✅ 相关性(嵌入)评分: {sim:.2f}")
        return {"score": sim}

    def evaluate_context_recall(self, question, contexts, ground_truth=None):
        """上下文召回：检索到的片段是否「足够覆盖」能回答问题所需的信息。

        - 有标准答案(gt)：算 gt 向量 与「所有参考片段均值向量」的余弦，
          越接近 1 说明检索到的内容越能支撑标准答案。
        - 无 gt：退化为「问题向量」与「片段均值向量」的余弦，
          衡量检索质量（能召回到相关信息吗）。
        """
        if not contexts:
            return {"score": 0.0, "details": "no contexts"}
        ctx_emb = np.mean([self._emb(c) for c in contexts], axis=0)
        if ground_truth:
            sim = _cosine(self._emb(ground_truth), ctx_emb)
        else:
            sim = _cosine(self._emb(question), ctx_emb)
        print(f"✅ 上下文召回(嵌入)评分: {sim:.2f}")
        return {"score": sim}

    def run_full_evaluation(self, test_cases):
        """跑一整批用例，打印每个用例分数 + 汇总均值。

        test_cases 格式：[{'question':..., 'answer':..., 'contexts':[...],
                          'ground_truth':可选}, ...]
        """
        print("\n" + "=" * 50)
        print("📊 RAG 系统全面评估（嵌入相似度版 · 零 token）")
        print("=" * 50 + "\n")

        all_f, all_r, all_c = [], [], []
        for i, case in enumerate(test_cases):
            print(f"--- 用例 {i + 1}/{len(test_cases)}: {case.get('question', '')[:30]}... ---")
            f = self.evaluate_faithfulness(case["answer"], case["contexts"])
            r = self.evaluate_relevancy(case["question"], case["answer"])
            c = self.evaluate_context_recall(
                case["question"], case["contexts"], case.get("ground_truth"))
            all_f.append(f.get("score", 0))
            all_r.append(r.get("score", 0))
            all_c.append(c.get("score", 0))
            print()

        avg_f, avg_r, avg_c = (sum(all_f)/len(all_f), sum(all_r)/len(all_r),
                               sum(all_c)/len(all_c))
        print("=" * 50)
        print("📋 评估汇总报告（嵌入相似度）")
        print("=" * 50)
        print(f"  忠实度 Faithfulness:  {avg_f:.2f}  {'✅ 优秀' if avg_f > 0.8 else '⚠️ 需改进' if avg_f > 0.5 else '❌ 较差'}")
        print(f"  相关性 Relevancy:     {avg_r:.2f}  {'✅ 优秀' if avg_r > 0.8 else '⚠️ 需改进' if avg_r > 0.5 else '❌ 较差'}")
        print(f"  上下文召回 Recall:    {avg_c:.2f}  {'✅ 优秀' if avg_c > 0.7 else '⚠️ 需改进' if avg_c > 0.4 else '❌ 较差'}")
        print(f"  综合得分:             {(avg_f + avg_r + avg_c) / 3:.2f}")
        print("=" * 50)
        return {
            "faithfulness": avg_f, "relevancy": avg_r,
            "context_recall": avg_c, "overall": (avg_f + avg_r + avg_c) / 3
        }
