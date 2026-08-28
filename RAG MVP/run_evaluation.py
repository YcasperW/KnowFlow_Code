"""KnowFlow RAGAS 离线评估脚本
================================
把原本是「孤儿文件」的 evaluator.py 接到真实的问答流程上，
跑出可量化的质量报告（忠实度 / 相关性 / 上下文召回）。

用法：
    python run_evaluation.py                  # 用内置示例问题（请替换成你手册真实问题）
    python run_evaluation.py test_cases.json  # 用自定义问题集
        test_cases.json 格式：
        [
          {"question": "如何申请年假？", "ground_truth": "..."},
          {"question": "报销流程是怎样的？"}
        ]

输出：
    - 控制台打印每个用例分数 + 汇总均值
    - output/evaluation_report.json 落盘，可直接贴进简历「质量量化」证据
"""
import os
import json
import sys

from dotenv import load_dotenv

from RAG_pipeline import TextEmbedder, RAGPipeline
from clustered_retriever import ClusteredRetriever
from evaluator import EmbeddingRAGEvaluator

# ===== 路径（与 Web Frame.py 保持一致）=====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

api_key = os.getenv("DEEPSEEK_API_KEY")
chunks_path = os.path.join(BASE_DIR, "output", "chunks.json")
embeddings_path = os.path.join(BASE_DIR, "output", "embeddings.npy")
cluster_index_path = os.path.join(BASE_DIR, "output", "cluster_index")


def build_retriever():
    """复用与线上完全一致的检索配置（alpha=0.4 等）。"""
    embedder = TextEmbedder("paraphrase-multilingual-MiniLM-L12-v2")
    retriever = ClusteredRetriever(
        embedder=embedder, n_clusters=20, routing_top_k=3, alpha=0.4
    )
    if os.path.exists(f"{cluster_index_path}_clusters.npz"):
        retriever.load_index(cluster_index_path,
                             chunks_file=chunks_path,
                             embeddings_file=embeddings_path)
    elif os.path.exists(chunks_path) and os.path.exists(embeddings_path):
        retriever.index_documents(chunks_file=chunks_path,
                                  embeddings_file=embeddings_path)
    else:
        raise FileNotFoundError("❌ 找不到索引/语料，请先上传文档并触发热更新。")
    return retriever


def load_test_cases(path=None):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # 内置示例（⚠️ 请替换为你操作手册里的真实问题，分数才有业务意义）
    return [
        {"question": "如何申请年假？"},
        {"question": "报销流程是怎样的？"},
        {"question": "试用期有多长？"},
    ]


def main():
    cases_file = sys.argv[1] if len(sys.argv) > 1 else None
    test_cases = load_test_cases(cases_file)

    print("🔧 构建检索器与管线...")
    retriever = build_retriever()
    rag = RAGPipeline(api_key=api_key, embedder=retriever.embedder, retriever=retriever)

    # 仅使用嵌入余弦相似度版评估器（零 token 消耗，确定可复现）。
    # 早期 LLM 打分版（RAGEvaluator）已归档移除。
    print("✅ 使用嵌入相似度版（零 token 消耗，确定可复现）")
    evaluator = EmbeddingRAGEvaluator(retriever.embedder)

    results = []
    for i, case in enumerate(test_cases):
        q = case["question"]
        print(f"\n--- 用例 {i + 1}/{len(test_cases)}: {q} ---")
        res = rag.ask(q, session_id="eval")
        answer = res.get("answer", "")
        contexts = res.get("contexts", [])
        if not contexts:
            print("⚠️ 未检索到上下文，跳过该用例")
            continue
        f = evaluator.evaluate_faithfulness(answer, contexts)
        r = evaluator.evaluate_relevancy(q, answer)
        c = evaluator.evaluate_context_recall(q, contexts, case.get("ground_truth"))
        results.append({
            "question": q,
            "faithfulness": f.get("score", 0),
            "relevancy": r.get("score", 0),
            "context_recall": c.get("score", 0),
        })

    if not results:
        print("无有效评估结果。")
        return

    n = len(results)
    avg_f = sum(x["faithfulness"] for x in results) / n
    avg_r = sum(x["relevancy"] for x in results) / n
    avg_c = sum(x["context_recall"] for x in results) / n

    print("\n" + "=" * 50)
    print("📋 KnowFlow 质量量化评估报告（RAGAS 思路）")
    print("=" * 50)
    print(f"  忠实度 Faithfulness:  {avg_f:.2f}  "
          f"{'✅ 优秀' if avg_f > 0.8 else '⚠️ 需改进' if avg_f > 0.5 else '❌ 较差'}")
    print(f"  相关性 Relevancy:     {avg_r:.2f}  "
          f"{'✅ 优秀' if avg_r > 0.8 else '⚠️ 需改进' if avg_r > 0.5 else '❌ 较差'}")
    print(f"  上下文召回 Recall:    {avg_c:.2f}  "
          f"{'✅ 优秀' if avg_c > 0.7 else '⚠️ 需改进' if avg_c > 0.4 else '❌ 较差'}")
    print(f"  综合得分:             {(avg_f + avg_r + avg_c) / 3:.2f}")
    print("=" * 50)

    out = os.path.join(BASE_DIR, "output", "evaluation_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "faithfulness": round(avg_f, 3),
                "relevancy": round(avg_r, 3),
                "context_recall": round(avg_c, 3),
                "overall": round((avg_f + avg_r + avg_c) / 3, 3),
            },
            "cases": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ 报告已保存: {out}")


if __name__ == "__main__":
    main()
