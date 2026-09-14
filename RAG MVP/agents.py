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
import json

from structured_cite import (
    generate_cited_answer,
    build_citation_prompt,
    CITED_ANSWER_SCHEMA,
    CitedAnswer,
    PREFIX_CACHE_P,
)


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
            "confidence": 0.0,
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
        # 修复：原先没把检索分传给 optimize，导致 QueryRouter 的
        # 「首位与次位分差 > 0.15 就走快车道」判断永远拿不到数据，
        # 快慢车道退化成只看问题长度 —— 省 token 的一道闸门形同虚设。
        retrieved_scores = None
        if raw_results:
            retrieved_scores = [(r.get("chunk"), float(r.get("score", 0.0)))
                                for r in raw_results]
        try:
            optimization = self.token_optimizer.optimize(
                retrieval_query, chunks, query_embedding=query_embedding,
                retrieved_scores=retrieved_scores)
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


class _StreamingAnswerExtractor:
    """从流式 JSON 增量文本中抽取 answer 字符串字段，逐段吐出已确定的文本。

    原因：structured outputs + stream=True 时，LLM 返回的不再是现成答案，而是
    JSON 片段流（delta）。要做到「答案逐字出现」的打字机效果，就得自己把 JSON 里
    answer 字段的内容一点点抠出来。本抽取器只认第一个名为 answer 的「键」
    （其后紧跟冒号），逐字符解码（含转义），把已稳定的文本增量交给前端。
    """
    def __init__(self):
        self._seen_key = False
        self._in_str = False
        self._escaping = False
        self._buf_pos = 0
        self._chars = []
        self._emitted = 0
        self._done = False

    def feed(self, full_text):
        # 还没定位到 answer 键：在全文里找第一个形如 "answer": 的键
        if not self._seen_key:
            p = full_text.find('"answer"')
            if p == -1:
                return ""
            # 校验这真的是个「键」：闭合引号后紧跟冒号
            # （用来区分正文里恰好出现 "answer" 字样的情况）
            after = full_text[p + len('"answer"'):]
            if not after.lstrip().startswith(":"):
                return ""
            i = p + len('"answer"')
            # 跳过冒号及前后空白，定位到值开始的引号
            while i < len(full_text) and full_text[i] in ' \t\r\n:':
                i += 1
            if i >= len(full_text) or full_text[i] != '"':
                return ""  # 值的起始引号还没到，等下一次 feed
            self._seen_key = True
            self._in_str = True
            self._buf_pos = i + 1  # 进入字符串内容
        # 扫描字符串内容（可能跨多次 feed）。逐字符处理 JSON 转义：
        # \" \\ \/ 直接取该字符；\n \t \r \b \f 解码为对应控制符；\uXXXX 解码为 Unicode。
        n = len(full_text)
        while self._buf_pos < n and not self._done:
            c = full_text[self._buf_pos]
            if self._escaping:
                self._escaping = False
                self._buf_pos += 1
                if c == 'n':
                    self._chars.append('\n')
                elif c == 't':
                    self._chars.append('\t')
                elif c == 'r':
                    self._chars.append('\r')
                elif c == 'b':
                    self._chars.append('\b')
                elif c == 'f':
                    self._chars.append('\f')
                elif c == 'u':
                    hex4 = full_text[self._buf_pos:self._buf_pos + 4]
                    if len(hex4) == 4:
                        try:
                            self._chars.append(chr(int(hex4, 16)))
                            self._buf_pos += 4
                        except Exception:
                            self._chars.append('?')
                            self._buf_pos += 4
                    else:
                        self._chars.append('u')  # \u 收不齐，退化为字面（极罕见）
                else:
                    self._chars.append(c)  # \" \\ \/ 等
                continue
            if c == '\\':
                self._escaping = True
                self._buf_pos += 1
                continue
            if c == '"':
                self._in_str = False
                self._done = True
                self._buf_pos += 1
                continue
            self._chars.append(c)
            self._buf_pos += 1
        # 只返回「相比上次新出现」的已解码字符
        new = "".join(self._chars[self._emitted:])
        self._emitted = len(self._chars)
        return new


class GeneratorAgent:
    """调 LLM 生成最终答案。

    - generate()           : 传统自由文本生成（用于免检索的 chat_direct 分支）。
    - generate_structured(): 可靠溯源生成——强制 LLM 按结构输出
                              (答案 + 编号来源列表 + 置信度)，并用 pydantic 校验。
    """

    def __init__(self, client, model=None):
        self.client = client
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    def _cache_kwargs(self):
        """仅在 DeepSeek 模型下启用前缀缓存（cache_p），避免非 DeepSeek 模型报 unknown 参数。"""
        if PREFIX_CACHE_P is not None and "deepseek" in (self.model or "").lower():
            return {"extra_body": {"cache_p": PREFIX_CACHE_P}}
        return {}

    def generate(self, final_prompt, system_instruction=None):
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": final_prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            timeout=20,
            **self._cache_kwargs(),
        )
        return response.choices[0].message.content

    def generate_structured(self, question, contexts, max_chars=800, system_instruction=None):
        """返回 pydantic 校验过的 CitedAnswer（答案 + 来源列表 + 置信度）。

        任何异常都在 structured_cite 内部降级为「普通回答 + 空来源 + 低置信」，
        调用方无需额外 try/except 也能拿到完整结构。
        """
        return generate_cited_answer(
            self.client, self.model, question, contexts,
            max_chars=max_chars, system_instruction=system_instruction)

    def stream_generate(self, final_prompt, system_instruction=None):
        """流式纯文本生成（chat_direct 免检索分支用），逐块 yield 文本增量。"""
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": final_prompt})
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            timeout=20,
            stream=True,
            **self._cache_kwargs(),
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    def stream_structured(self, question, contexts, max_chars=800, system_instruction=None):
        """流式结构化溯源生成。yield (delta_text, final_cited)。

        - delta_text：增量答案文本（最终那次 yield 为空串）；
        - final_cited：结束时为 pydantic 校验过的 CitedAnswer；
                      若流式接口不可用或最终 JSON 解析失败，则为 None（交由上层降级）。
        说明：开启 stream=True 后，LLM 返回的是 JSON 片段流，我们用 _StreamingAnswerExtractor
        把 answer 字段内容逐步抽出来向前端吐字；流结束后整体解析出 sources / confidence。
        """
        prompt = build_citation_prompt(question, contexts, max_chars=max_chars)
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        schema_fmt = {
            "type": "json_schema",
            "json_schema": {"name": "CitedAnswer", "strict": True, "schema": CITED_ANSWER_SCHEMA},
        }
        extractor = _StreamingAnswerExtractor()
        buf = []
        final_cited = None
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                response_format=schema_fmt,
                stream=True,
                stream_options={"include_usage": False},
                timeout=40,
                **self._cache_kwargs(),
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                buf.append(delta)
                new_text = extractor.feed("".join(buf))
                if new_text:
                    yield (new_text, None)
            # 流结束 → 解析完整 JSON，校验成 CitedAnswer
            full = "".join(buf)
            try:
                data = json.loads(full)
                final_cited = CitedAnswer.model_validate(data)
            except Exception:
                final_cited = None
        except Exception:
            # 流式接口不可用（如当前模型/网关不支持）→ 标记降级
            final_cited = None
        yield ("", final_cited)
