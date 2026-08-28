import numpy as np
from sentence_transformers import SentenceTransformer
from token_refine import KnowFlowTokenOptimizer
from memory_store import RAGMemoryStore
from openai import OpenAI
from clustered_retriever import ClusteredRetriever
from agents import RouterAgent, RetrieverAgent, CompressorAgent, GeneratorAgent
import json, pickle
import os
import time


# ========== 向量化 ==========
class TextEmbedder:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        print("⏳ 加载向量模型...")
        self.model = SentenceTransformer(model_name)
        print("✅ 向量模型就绪")

    def embed_text(self, text):
        return self.model.encode(text)

    def encode(self, text):
        """兼容：让 embedder 同时支持 .encode() 调用"""
        return self.model.encode(text)

    def embed_batch(self, texts):
        return self.model.encode(texts)


# ========== RAG 管线 ==========
class RAGPipeline:
    def __init__(self, api_key=None, api_base=None, model=None,
                 embedder=None, retriever=None,
                 smart_context_threshold=5.0, enable_artifact=True):
        self.embedder = embedder
        self.retriever = retriever
        self.sessions = {}
        self.memory_stores = {}
        self.max_memory_items = 20

        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = api_base or os.getenv("DEEPSEEK_BASE_URL")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        if not self.api_key:
            raise ValueError("❌ 未找到 DEEPSEEK_API_KEY。")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=20.0
        )

        self.token_optimizer = KnowFlowTokenOptimizer(
            client=self.client,
            model=self.model,
            smart_context_threshold=smart_context_threshold,
            enable_artifact=enable_artifact,
            enable_cache=True
        )

        # ===== 多 Agent 编排：实例化 4 个职责单一的 Agent =====
        # 编排逻辑见 multi_agent_ask()，各 Agent 只持有完成自身职责必需的资源引用。
        self.router = RouterAgent(client=self.client, model=self.model)
        self.retriever_agent = RetrieverAgent(retriever=self.retriever)
        self.compressor = CompressorAgent(token_optimizer=self.token_optimizer)
        self.generator = GeneratorAgent(client=self.client, model=self.model)

    # ===== 多 Agent 重构说明 =====
    # 原 _retrieve_and_optimize 的「记忆/检索/拒答/重排/压缩」逻辑已拆分到
    # agents.py（RetrieverAgent / CompressorAgent / GeneratorAgent），
    # 由 multi_agent_ask() 统一编排。本方法已废弃，请勿调用。

    @staticmethod
    def _build_final_prompt(question, state):
        optimization = state["optimization"]
        base_prompt = optimization["final_prompt"]
        is_ref = state["is_ref"]
        history_context = state["history_context"]
        if is_ref and history_context:
            return f'''【历史对话上下文】
        {history_context}

        【当前任务】
        基于上面的【历史对话上下文】，请回答用户的最新问题。
        注意：用户的最新问题中可能包含代词（如"其他人"、"他"、"这个"），请务必结合历史对话中的具体人名或实体进行解答，不要脱离上下文自行发散。

        【当前检索到的参考资料】
        {base_prompt}

        【最新问题】
        {question}
        '''
        return f"{base_prompt}\n\n【最新问题】\n{question}"

    # ===== 多 Agent 重构说明 =====
    # 原 _generate_answer 已迁移到 agents.GeneratorAgent.generate；
    # 原 _retrieve_and_optimize 已拆分为 agents.py 的 RetrieverAgent / CompressorAgent，
    # 统一由 multi_agent_ask() 编排。以下旧方法已废弃，请勿调用。

    def multi_agent_ask(self, question, session_id="default", top_k=5):
        """多 Agent 编排入口（对应 Phase4 文档 10.2 草图）。

        流程：记忆/历史 → ① Router 路由 → ② Retriever 检索+拒答闸门 →
        ③ Compressor 三层压缩 → ④ Generator 生成。
        问候/客套类问题走 chat_direct 分支，免检索、免压缩，直接生成，省 token。
        """
        # ===== RAG 记忆（原 _retrieve_and_optimize 前半，保留多轮对话能力）=====
        if session_id not in self.memory_stores:
            persist_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "output",
                f"memory_{session_id}.json")
            self.memory_stores[session_id] = RAGMemoryStore(
                embedder=self.embedder,
                max_items=self.max_memory_items,
                persist_path=persist_path)
        memory = self.memory_stores[session_id]
        history_context = memory.build_context(question, top_k=2)

        # ===== 分层检索：判断是否纯指代追问（复用既有逻辑）=====
        is_ref = False
        if history_context:
            try:
                is_ref = self._is_pure_reference(question, history_context)
            except Exception:
                is_ref = False
        retrieval_query = f"{history_context}\n{question}" if (is_ref and history_context) else question

        # ===== ① Router：要不要检索 =====
        route = self.router.decide(question, history_context)
        if route == "chat_direct":
            final_prompt = (
                "你是 KnowFlow 企业知识库智能助手。用户这句是问候或闲聊，"
                "不需要检索资料，请用一句友好、简洁的话回应。\n\n用户：" + question)
            try:
                answer_text = self.generator.generate(final_prompt)
            except Exception as e:
                safe_msg = self._sanitize_error(str(e))
                return {
                    "answer": f"抱歉，生成回答时出现了问题（{safe_msg}）。请稍后重试。",
                    "sources": [], "key_facts": [], "token_report": None,
                    "error": safe_msg, "session_id": session_id, "route": "chat_direct"}
            memory.add(question, answer_text)
            return {
                "answer": answer_text, "sources": [], "key_facts": [],
                "token_report": None, "contexts": [],
                "session_id": session_id, "route": "chat_direct"}

        # ===== ② Retriever（+ 低置信拒答闸门）=====
        ret = self.retriever_agent.retrieve(retrieval_query, top_k=top_k)
        if "reject" in ret:
            rej = ret["reject"]
            rej["session_id"] = session_id
            rej["route"] = "retrieve"
            return rej

        chunks = ret["chunks"]
        raw_results = ret["raw_results"]

        # ===== ③ Compressor：三层 Token 压缩（复用 token_refine）=====
        query_embedding = None
        if self.embedder:
            try:
                query_embedding = self.embedder.encode(retrieval_query).tolist()
            except Exception as e:
                print(f"[Debug] embedding 失败: {e}")
        optimization = self.compressor.compress(
            retrieval_query, chunks, query_embedding, raw_results, top_k)

        state = {
            "optimization": optimization, "chunks": chunks,
            "is_ref": is_ref, "history_context": history_context,
            "top_k": top_k, "session_id": session_id, "memory": memory,
        }

        # ===== ④ 组 prompt + Generator 生成 =====
        final_prompt = self._build_final_prompt(question, state)
        try:
            answer_text = self.generator.generate(final_prompt)
        except Exception as e:
            safe_msg = self._sanitize_error(str(e))
            return {
                "answer": f"抱歉，生成回答时出现了问题（{safe_msg}）。请稍后重试。",
                "sources": [], "key_facts": [], "token_report": None,
                "error": safe_msg, "session_id": session_id, "route": "retrieve"}

        memory.add(question, answer_text)

        # ===== Token 实测：补全库朴素基线（端到端节省比）=====
        _report = optimization.get("report")
        if _report and self.retriever and getattr(self.retriever, "chunks", None):
            _all_text = "\n".join(
                str(c.get("content", "")) for c in self.retriever.chunks if c.get("content"))
            _naive_tokens = len(_all_text) // 2
            if _naive_tokens > 0:
                _report["naive_baseline_tokens"] = _naive_tokens
                _final_tokens = _report.get("after_compression", 0)
                _report["end_to_end_saving_percent"] = round(
                    (1 - _final_tokens / _naive_tokens) * 100, 1)
            optimization["report"] = _report

        return {
            "answer": answer_text,
            "sources": optimization.get("source_refs", []),
            "key_facts": optimization.get("key_facts", []),
            "token_report": optimization.get("report"),
            "contexts": [c.get("content", "") for c in chunks[:top_k]],
            "session_id": session_id,
            "route": "retrieve",
        }

    def ask(self, question, session_id="default", top_k=5):
        """对外统一入口：委托 multi_agent_ask（多 Agent 编排）。"""
        return self.multi_agent_ask(question, session_id=session_id, top_k=top_k)

    def _is_pure_reference(self, question, history_context=""):
        if not history_context:
            return False
        prompt = f"""判断当前问题是否能在不依赖历史对话的情况下独立理解。

规则：
- 如果问题里包含了新的实体词（如"病假""加班""报销"），说明可以独立搜索 → 只输出 NO
- 如果问题完全依赖历史才能理解（如"这个呢""那我自己呢""那其他人呢""它也适用吗"），说明是纯指代 → 只输出 YES

特别注意："其他人呢"、"那其他人呢"、"其他人" 必须输出 YES。

历史对话摘要：
{history_context}

当前问题：{question}

只输出 YES 或 NO，不要输出任何其他内容："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
                timeout=10
            )
            answer = response.choices[0].message.content
            if not answer or not answer.strip():
                return False
            return "YES" in answer.strip().upper()
        except Exception:
            return False

    def _sanitize_error(self, msg):
        import re
        msg = re.sub(r'sk-[a-zA-Z0-9]{10,}', '***', msg)
        msg = re.sub(r'Bearer [a-zA-Z0-9\.\-_]+', '***', msg)
        return msg