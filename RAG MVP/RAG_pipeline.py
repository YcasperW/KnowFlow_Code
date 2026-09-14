import numpy as np
from sentence_transformers import SentenceTransformer
from token_refine import KnowFlowTokenOptimizer
from memory_store import RAGMemoryStore
from openai import OpenAI
from clustered_retriever import ClusteredRetriever
from agents import RouterAgent, RetrieverAgent, CompressorAgent, GeneratorAgent
from structured_cite import generate_cited_answer, get_standard_doc, get_task_prompt
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

    def multi_agent_ask(self, question, session_id="default", top_k=5, task=None):
        """多 Agent 编排入口（对应 Phase4 文档 10.2 草图）。

        流程：记忆/历史 → ① Router 路由 → ② Retriever 检索+拒答闸门 →
        ③ Compressor 三层压缩 → ④ Generator 生成。
        问候/客套类问题走 chat_direct 分支，免检索、免压缩，直接生成，省 token。
        task：任务键（如 revise_proposal），用于注入固定任务型系统指令，与流式路径保持一致。
        """
        # 解析任务型系统指令（与流式路径一致，用函数取值以支持管理员在线改提示词后热生效）
        sys_instr = get_task_prompt(task)

        # ===== 任务型快路径（与 multi_agent_ask_stream 保持一致）=====
        # 非流式接口此前完全没有 task 支持，导致 /ask 与 /ask_stream 行为分裂；
        # 这里补齐，使两个入口对同一请求给出一致的结果。
        if task and sys_instr:
            std_doc = get_standard_doc(task)
            combined = sys_instr + ("\n\n【参考标准文档】\n" + std_doc if std_doc else "")
            try:
                answer_text = self.generator.generate(question, system_instruction=combined)
            except Exception as e:
                safe_msg = self._sanitize_error(str(e))
                return {
                    "answer": f"抱歉，生成回答时出现了问题（{safe_msg}）。请稍后重试。",
                    "sources": [], "key_facts": [], "token_report": None,
                    "confidence": None, "low_confidence": True,
                    "error": safe_msg, "session_id": session_id, "route": "task",
                    "is_followup": False, "reject_reason": None}
            memory = self._get_memory(session_id)
            memory.add(question, answer_text)
            return {
                "answer": answer_text, "sources": [], "key_facts": [],
                "token_report": None, "confidence": None, "low_confidence": False,
                "contexts": [], "session_id": session_id, "route": "task",
                "is_followup": False, "reject_reason": None}

        # ===== RAG 记忆（原 _retrieve_and_optimize 前半，保留多轮对话能力）=====
        memory = self._get_memory(session_id)
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
                answer_text = self.generator.generate(final_prompt, system_instruction=sys_instr)
            except Exception as e:
                safe_msg = self._sanitize_error(str(e))
                return {
                    "answer": f"抱歉，生成回答时出现了问题（{safe_msg}）。请稍后重试。",
                    "sources": [], "key_facts": [], "token_report": None,
                    "confidence": None, "low_confidence": True,
                    "error": safe_msg, "session_id": session_id, "route": "chat_direct",
                    "is_followup": False, "reject_reason": None}
            memory.add(question, answer_text)
            return {
                "answer": answer_text, "sources": [], "key_facts": [],
                "token_report": None, "contexts": [],
                "confidence": None, "low_confidence": False,
                "session_id": session_id, "route": "chat_direct",
                "is_followup": False, "reject_reason": None}

        # ===== ② Retriever（+ 低置信拒答闸门）=====
        ret = self.retriever_agent.retrieve(retrieval_query, top_k=top_k)
        if "reject" in ret:
            rej = ret["reject"]
            rej["session_id"] = session_id
            rej["route"] = "retrieve"
            rej["is_followup"] = is_ref
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

        # ===== ④ 组 prompt + Generator 结构化溯源生成 =====
        final_prompt = self._build_final_prompt(question, state)
        try:
            cited = self.generator.generate_structured(
                question, chunks, max_chars=800, system_instruction=sys_instr)
            answer_text = cited.answer
            structured_sources = [s.model_dump() for s in cited.sources]
            confidence = cited.confidence
        except Exception as e:
            safe_msg = self._sanitize_error(str(e))
            return {
                "answer": f"抱歉，生成回答时出现了问题（{safe_msg}）。请稍后重试。",
                "sources": [], "key_facts": [], "token_report": None,
                "confidence": None, "low_confidence": True,
                "error": safe_msg, "session_id": session_id, "route": "retrieve",
                "is_followup": is_ref, "reject_reason": ["生成异常"]}

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
            "sources": structured_sources,
            "key_facts": optimization.get("key_facts", []),
            "token_report": optimization.get("report"),
            "confidence": confidence,
            # 与流式路径口径一致（<0.4 判低置信），此前非流式缺该字段，
            # 导致 /ask 的结果没有低置信提示、前端只能取默认值 False。
            "low_confidence": (confidence is not None and confidence < 0.4),
            "contexts": [c.get("content", "") for c in chunks[:top_k]],
            "session_id": session_id,
            "route": "retrieve",
            "is_followup": is_ref,
            "reject_reason": None,
        }

    def _get_memory(self, session_id):
        """按会话取（或惰性创建）记忆实例，避免多入口重复拼路径。"""
        if session_id not in self.memory_stores:
            persist_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "output",
                f"memory_{session_id}.json")
            self.memory_stores[session_id] = RAGMemoryStore(
                embedder=self.embedder,
                max_items=self.max_memory_items,
                persist_path=persist_path)
        return self.memory_stores[session_id]

    def ask(self, question, session_id="default", top_k=5, task=None):
        """对外统一入口：委托 multi_agent_ask（多 Agent 编排）。"""
        return self.multi_agent_ask(question, session_id=session_id, top_k=top_k, task=task)

    def clear_session(self, session_id="default"):
        """清空指定会话的记忆（Web 层 /clear 端点调用）。

        ⚠️ 修复说明：Web Frame.py 的 /clear 路由一直调用本方法，但 RAGPipeline
        从未定义它 → 每次点「新对话」都抛 AttributeError，前端拿到 500。
        这里补齐实现：清内存里的记忆实例 + 删掉对应的持久化文件，使重启后也不会复活。
        """
        # 1) 内存中的记忆实例：存在则清空条目
        mem = self.memory_stores.get(session_id)
        if mem is not None:
            try:
                mem.clear()
            except Exception as e:
                print(f"[clear_session] 清空内存记忆失败: {e}")
        # 2) 即便该会话从未在本次进程内实例化（例如服务刚重启、用户就点了新对话），
        #    也要把磁盘上的持久化文件删掉，否则刷新页面后旧对话会被重新加载。
        try:
            persist_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "output",
                f"memory_{session_id}.json")
            if os.path.exists(persist_path):
                os.remove(persist_path)
        except Exception as e:
            print(f"[clear_session] 删除记忆文件失败: {e}")
        # 3) 从索引里摘掉，避免 memory_stores 随会话数无限增长
        self.memory_stores.pop(session_id, None)
        self.sessions.pop(session_id, None)
        return {"status": "ok", "session_id": session_id}

    def multi_agent_ask_stream(self, question, session_id="default", top_k=5, task=None):
        """流式版编排入口：yield 事件元组给 Web 层转成 SSE。

        事件种类：
          ("token", text)   —— 答案文本增量（打字机效果）
          ("meta",  dict)   —— 结尾元数据（sources / confidence / key_facts / token_report / reject 等）
          ("error", msg)    —— 致命错误
        结构与原 multi_agent_ask 一致，仅把「一次性返回」改为「边生成边 yield」。
        task：前端「猜你想问」芯片携带的任务键，用于注入固定的任务型系统指令（prefix cache 友好）。
        """
        # 解析任务型系统指令（固定前缀，重复请求可命中 DeepSeek 前缀缓存）
        # 用 get_task_prompt() 而非直接引用 TASK_SYSTEM_PROMPTS 字典：
        # 管理员在 /admin/prompts 保存后模块变量会被重新绑定，函数取值才能读到最新提示词。
        sys_instr = get_task_prompt(task)
        # ===== 任务型快路径（前缀缓存友好）=====
        # 任务提示词 + 标准文档 合并为固定 system 前缀，跳过 RAG 检索，
        # 直接对用户提交的文档做审阅/梳理；重复请求可命中前缀缓存（省标准文档重算）。
        if task and sys_instr:
            std_doc = get_standard_doc(task)
            combined = sys_instr + ("\n\n【参考标准文档】\n" + std_doc if std_doc else "")
            try:
                for delta in self.generator.stream_generate(question, system_instruction=combined):
                    yield ("token", delta)
            except Exception as e:
                yield ("error", self._sanitize_error(str(e)))
                return
            yield ("meta", {
                "answer": "", "sources": [], "confidence": None, "low_confidence": False,
                "key_facts": [], "token_report": None,
                "is_followup": False, "reject_reason": None, "task": task,
            })
            return
        # ===== RAG 记忆 =====
        memory = self._get_memory(session_id)
        history_context = memory.build_context(question, top_k=2)

        # ===== 分层检索：判断是否纯指代追问 =====
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
                for delta in self.generator.stream_generate(final_prompt, system_instruction=sys_instr):
                    yield ("token", delta)
            except Exception as e:
                yield ("error", self._sanitize_error(str(e)))
                return
            yield ("meta", {
                "answer": "",
                "sources": [], "confidence": None, "low_confidence": False,
                "key_facts": [], "token_report": None,
                "is_followup": False, "reject_reason": None,
            })
            return

        # ===== ② Retriever（+ 低置信拒答闸门）=====
        ret = self.retriever_agent.retrieve(retrieval_query, top_k=top_k)
        if "reject" in ret:
            rej = ret["reject"]
            rej["is_followup"] = is_ref
            yield ("meta", rej)
            return

        chunks = ret["chunks"]
        raw_results = ret["raw_results"]

        # ===== ③ Compressor：三层 Token 压缩 =====
        query_embedding = None
        if self.embedder:
            try:
                query_embedding = self.embedder.encode(retrieval_query).tolist()
            except Exception:
                print("[Debug] embedding 失败")
        optimization = self.compressor.compress(
            retrieval_query, chunks, query_embedding, raw_results, top_k)

        state = {
            "optimization": optimization, "chunks": chunks,
            "is_ref": is_ref, "history_context": history_context,
            "top_k": top_k, "session_id": session_id, "memory": memory,
        }
        final_prompt = self._build_final_prompt(question, state)

        # ===== ④ 流式结构化生成 =====
        streamed_answer = []
        final_cited = None
        try:
            for delta, final in self.generator.stream_structured(question, chunks, max_chars=800, system_instruction=sys_instr):
                if delta:
                    streamed_answer.append(delta)
                    yield ("token", delta)
                if final is not None:
                    final_cited = final
        except Exception as e:
            # 流式中途崩溃 → 降级为非流式拿一份可靠结果
            try:
                final_cited = generate_cited_answer(self.client, self.model, question, chunks, max_chars=800, system_instruction=sys_instr)
            except Exception:
                final_cited = None

        # 若流式没吐出任何字（接口不支持 / 解析失败），用非流式兜底补出答案
        if final_cited is None:
            try:
                final_cited = generate_cited_answer(self.client, self.model, question, chunks, max_chars=800, system_instruction=sys_instr)
            except Exception:
                final_cited = None

        if final_cited is not None:
            answer_text = final_cited.answer
            structured_sources = [s.model_dump() for s in final_cited.sources]
            confidence = final_cited.confidence
            if not streamed_answer:
                yield ("token", answer_text)
            meta = {
                "answer": answer_text,
                "sources": structured_sources,
                "confidence": confidence,
                "low_confidence": (confidence is not None and confidence < 0.4),
                "key_facts": optimization.get("key_facts", []),
                "token_report": optimization.get("report"),
                "is_followup": is_ref,
                "reject_reason": None,
            }
            memory.add(question, answer_text)
        else:
            meta = {
                "answer": "抱歉，生成回答时出现问题，请稍后重试。",
                "sources": [], "confidence": 0.2, "low_confidence": True,
                "key_facts": [], "token_report": optimization.get("report"),
                "is_followup": is_ref, "reject_reason": ["生成异常"],
            }
        yield ("meta", meta)

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