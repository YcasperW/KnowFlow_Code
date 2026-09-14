# memory_store.py
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
from datetime import datetime


class RAGMemoryStore:
    """
    RAG 式对话记忆：把历史对话存成向量，按需检索最相关的几条。
    每个 session 独立一个实例（跟之前的 ConversationMemory 一样）。
    """

    def __init__(self, embedder, max_items=20, persist_path=None):
        """
        embedder: SentenceTransformer 实例
        max_items: 最多存多少条历史（超过就删最旧的）
        persist_path: 持久化文件路径（None = 不存盘）
        """
        self.embedder = embedder
        self.max_items = max_items
        self.persist_path = persist_path

        self.entries = []      # [{"text": "...", "timestamp": "...", "embedding": [...]}, ...]
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 如果有持久化文件，加载
        if persist_path and os.path.exists(persist_path):
            self.load(persist_path)

    def add(self, question, answer):
        """
        把一轮对话存进去（自动算向量）
        """
        text = f"Q: {question}\nA: {answer}"
        # embedder 缺失时不再抛 AttributeError 把整轮问答带崩：
        # 退化成「无向量」条目，后续 retrieve 会跳过它，对话流程不受影响。
        if self.embedder is None:
            embedding = None
        else:
            try:
                embedding = self.embedder.encode(text).tolist()
            except Exception as e:
                print(f"[memory] 向量化失败，按无向量条目存储: {e}")
                embedding = None

        self.entries.append({
            "text": text,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "embedding": embedding
        })

        # 超过上限删最旧的
        if len(self.entries) > self.max_items:
            self.entries.pop(0)

        # 自动持久化
        if self.persist_path:
            self.save(self.persist_path)

    def retrieve(self, query, top_k=2):
        """
        根据当前问题，从记忆里检索最相关的 top_k 条历史。
        返回格式：[{"text": "...", "score": 0.85}, ...]
        """
        if not self.entries:
            return []

        if self.embedder is None:
            return []

        query_emb = self.embedder.encode(query)

        # 余弦相似度
        results = []
        for entry in self.entries:
            # 兼容性修复：早期版本落盘的条目可能没有 embedding 字段
            # （或 embedder 缺失时存成了 None），直接取键会抛 KeyError，
            # 一旦有一条脏数据，整个会话的历史检索就全废了 —— 这里跳过即可。
            emb = entry.get("embedding")
            if emb is None:
                continue
            entry_emb = np.array(emb)
            score = self._cosine(query_emb, entry_emb)
            results.append({
                "text": entry["text"],
                "score": float(score)
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def build_context(self, query, top_k=2):
        """
        检索 + 格式化成 prompt 文本。
        如果没找到相关历史，返回空字符串。
        """
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return ""

        lines = ["\n【相关历史对话】"]
        for i, h in enumerate(hits):
            lines.append(f"[历史{i+1}] {h['text']}")

        return "\n".join(lines)

    def clear(self):
        self.entries = []
        if self.persist_path and os.path.exists(self.persist_path):
            os.remove(self.persist_path)
        print("💬 RAG 记忆已清空")

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "session_id": self.session_id,
                "entries": self.entries
            }, f, ensure_ascii=False, indent=2)

    def load(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.session_id = data.get("session_id", self.session_id)
            self.entries = data.get("entries", [])

    @staticmethod
    def _cosine(a, b):
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)