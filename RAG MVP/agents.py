"""KnowFlow 多 Agent 编排模块
================================
把原本「单管线 RAG」重构为 4 个职责单一的 Agent，Agent 之间用「消息字典」传递数据，
呼应立项初心「多 Agent 协作省 token」。每个 Agent 只持有完成自己职责必需的资源引用，互不继承。

四个 Agent：
  - RouterAgent     : 路由决策——判断「该不该检索」。问候/客套等不需要查资料的问题直接生成，
                      省下「检索 + 生成」两道 token，是本项目最贴卖点的省 token 开关。
  - RetrieverAgent  : 混合检索 + 低置信拒答闸门（复用现有 ClusteredRetriever）。
  - CompressorAgent : 三层 Token 压缩（复用 token_refine.KnowFlowTokenOptimizer）。
  - GeneratorAgent  : 调 LLM 生成最终答案。

编排逻辑放在 RAGPipeline.multi_agent_ask 里，agents.py 只负责「各 Agent 自己那点事」。
"""

import os
import re


class RouterAgent:
    """路由：决定走「检索生成」还是「免检索直接生成」。

    决策顺序（保守优先，宁多检不漏答，避免把真问题误判为闲聊导致幻觉）：
      1) 零 token 启发式：短且明显是问候/客套 → chat_direct（免检索）。
      2) 有历史上下文的追问 → 一定需要检索来对齐实体 → retrieve。
      3) 其余默认 retrieve。
    注：文档草图允许用一次 max_tokens=5 的 LLM 分类做更激进的省 token，
        但默认保守检索，以免误杀真问题。需要时可在此扩展。
    """

    # 零 token 启发式：这类句子基本不需要查知识库
    _GREETING_RE = re.compile(
        r"^(你好|您好|hi|hello|hey|嗨|在吗|在不在|你是谁|你叫什么|你是什么|谢谢|感谢|多谢|"
        r"好的|好吧|ok|okay|拜拜|再见|辛苦了|赞|牛|哈哈|嘿)\b", re.I)

    def __init__(self, client=None, model=None):
        self.client = client
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    def decide(self, question, history_context=""):
        q = (question or "").strip()
        # 1) 零 token：短且像问候/客套 → 免检索
        if len(q) <= 12 and self._GREETING_RE.search(q):
            return "chat_direct"
        # 2) 有历史上下文的追问，需要检索来对齐实体 → 检索
        if history_context:
            return "retrieve"
        # 3) 保守默认：宁多检，不漏答 / 不幻觉
        return "retrieve"


class RetrieverAgent:
    """混合检索 + 低置信拒答闸门。

    输入 retrieval_query（已含历史上下文拼接），输出消息字典：
      - 命中拒答条件 → {"reject": {...拒答结构...}}
      - 正常        → {"ok": True, "chunks": [...], "raw_results": [...]}
    与原 _retrieve_and_optimize 的检索/拒答/重排/字段统一逻辑一一对应，行为不变。
    """

    def __init__(self, retriever):
        self.retriever = retriever

    def retrieve(self, retrieval_query, top_k=5):
        if not self.retriever:
            raise RuntimeError("❌ retriever 未初始化")

        raw_results = self.retriever.search(retrieval_query, top_k=top_k)

        # 空检索兜底：知识库为空或路由无候选 → 直接拒答，杜绝幻觉
        if not raw_results:
            return {"reject": self._reject(["知识库检索为空"])}

        # 低置信度拒答（反幻觉兜底，省 token）：语义极低「且」关键词也未真正命中 → 拒答
        CONF_THRESHOLD = float(os.getenv("KNOWFLOW_MIN_SCORE", "0.55"))
        BM25_FLOOR = float(os.getenv("KNOWFLOW_BM25_FLOOR", "20.0"))
        best_dense = max(float(r.get("dense", 0.0)) for r in raw_results)
        best_bm25 = max(float(r.get("bm25", 0.0)) for r in raw_results)

        if best_dense < CONF_THRESHOLD and best_bm25 < BM25_FLOOR:
            reason = []
            if best_dense < CONF_THRESHOLD:
                reason.append("语义匹配度不足")
            if best_bm25 < BM25_FLOOR:
                reason.append("关键词命中不足")
            return {"reject": self._reject(reason)}

        # 主上下文重排：让「放行信号」对应的 chunk 居首，避免被融合冲淡
        bm25_driven = (best_bm25 >= BM25_FLOOR and best_dense < CONF_THRESHOLD)
        dense_driven = (best_dense >= CONF_THRESHOLD and best_bm25 < BM25_FLOOR)
        if bm25_driven:
            raw_results = sorted(raw_results, key=lambda r: float(r.get("bm25", 0.0)), reverse=True)
        elif dense_driven:
            raw_results = sorted(raw_results, key=lambda r: float(r.get("dense", 0.0)), reverse=True)

        # 统一字段名（chunk 可能带 text / content 两种键）
        chunks = []
        for r in raw_results:
            c = r["chunk"].copy() if isinstance(r["chunk"], dict) else {"content": str(r["chunk"])}
            if "text" in c and "content" not in c:
                c["content"] = c.pop("text")
            if "content" not in c:
                c["content"] = str(c)
            chunks.append(c)

        return {"ok": True, "chunks": chunks, "raw_results": raw_results}

    @staticmethod
    def _reject(reason):
        """构造与原管线一致的拒答返回结构。"""
        return {
            "answer": (
                "抱歉，当前知识库未覆盖该问题，暂时无法给出准确答复。"
                "建议：① 换一种表述重试；② 联系文档管理员补充对应操作手册。"
            ),
            "sources": [], "key_facts": [], "token_report": None,
            "low_confidence": True, "reject_reason": reason,
        }


class CompressorAgent:
    """三层 Token 压缩：复用 token_refine.KnowFlowTokenOptimizer。

    输入 (retrieval_query, chunks, query_embedding, raw_results)，
    输出 optimization 字典（含 final_prompt / source_refs / key_facts / report 等）。
    optimize 异常时退化为「直接拼接前 top_k 块」的兜底 prompt，保证不中断。
    """

    def __init__(self, token_optimizer):
        self.token_optimizer = token_optimizer

    def compress(self, retrieval_query, chunks, query_embedding=None, raw_results=None, top_k=5):
        optimization = None
        try:
            optimization = self.token_optimizer.optimize(
                retrieval_query, chunks, query_embedding=query_embedding)
        except Exception as e:
            print(f"[Error] optimize 异常: {e}")

        if (not optimization
                or not isinstance(optimization, dict)
                or "final_prompt" not in optimization):
            base_prompt = "\n\n".join([
                f"[{c.get('source', '?')}] {c.get('content', '')[:500]}"
                for c in chunks[:top_k]])
            source_refs = []
            if raw_results:
                source_refs = [
                    {"source": c.get("source", "?"), "relevance_score": r.get("score", 0)}
                    for r, c in zip(raw_results[:top_k], chunks[:top_k])]
            optimization = {
                "final_prompt": base_prompt,
                "source_refs": source_refs,
                "key_facts": [],
                "artifact_loader": {"enabled": False, "paths": []},
                "report": None,
            }
        return optimization


class GeneratorAgent:
    """调 LLM 生成最终答案（一次性返回字符串）。"""

    def __init__(self, client, model=None):
        self.client = client
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    def generate(self, final_prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.3,
            timeout=20,
        )
        return response.choices[0].message.content
