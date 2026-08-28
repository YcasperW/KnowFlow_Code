# -*- coding: utf-8 -*-
"""Web 端文档管理模块：上传、列表、删除

修复说明（对照手册原版）：
  1. 原代码 `from embedder import TextEmbedder` 是错的——
     你的 TextEmbedder 写在 RAG_pipeline.py 里，改成 `from RAG_pipeline import TextEmbedder`。
  2. 原代码 `from document_processor import DocumentProcessor` 需要配套文件，
     已新建 document_processor.py（见同目录）。
  3. 原代码漏了 `import datetime`，第 36 行会崩，已补 `from datetime import datetime`。
  4. 原代码调 `embedder.embed_batch(...)`，但你的 TextEmbedder 没有这个方法，
     改成统一的 `_embed()` 辅助函数，自动兼容 embed_batch / embed_text / encode 三种写法。
"""

import os
import json
import numpy as np
import threading
from datetime import datetime
from flask import request, jsonify, send_from_directory

from document_processor import DocumentProcessor
from RAG_pipeline import TextEmbedder


class DocumentManager:
    """文档管理器：统一管理知识库文档的生命周期（增 / 查 / 删 + 自动重建索引）"""

    def __init__(self, data_dir="data", output_dir="output", embedder=None,
                 retriever=None, cluster_index_path=None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)
        self.embedder = embedder  # ← 直接用传进来的，不再自己懒加载
        # ===== 热更新索引所需 =====
        # retriever：内存里的检索器（upload/delete 后需刷新它，免重启）
        # cluster_index_path：与 retriever.save_index 同前缀，热更新后落盘
        self.retriever = retriever
        self.cluster_index_path = cluster_index_path
        self._reload_lock = threading.Lock()  # 防止并发上传/删除打穿索引

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    # ---------- 统一的向量化入口（兼容多种 embedder 类型）----------
    def _embed(self, texts):
        """把一批文本变成向量。自动适配你传进来的 embedder 类型。"""
        e = self.embedder
        if e is None:
            raise RuntimeError("❌ embedder 未初始化，请在创建 DocumentManager 时传入 embedder")
        if hasattr(e, "embed_batch"):
            return e.embed_batch(texts)
        if hasattr(e, "embed_text"):
            return [e.embed_text(t) for t in texts]
        if hasattr(e, "encode"):      # 原生 SentenceTransformer
            return e.encode(texts)
        raise RuntimeError("❌ embedder 不支持向量化（需要 embed_batch / embed_text / encode 之一）")

    # ---------- 热更新：刷新内存检索索引（免重启）----------
    def _hot_reload_retriever(self):
        """磁盘索引（chunks.json / embeddings.npy）已由 _update_index / _rebuild_index
        更新完毕，这里把【内存里】的 retriever 也重建一遍，使 search() 立即生效，
        无需重启 Flask。

        做法：直接调 retriever.index_documents 从最新磁盘文件重建聚类 + BM25，
        再 save_index 落盘，保证下次重启也一致。
        """
        if self.retriever is None or self.cluster_index_path is None:
            print("⚠️ 未注入 retriever / cluster_index_path，跳过热更新（需重启生效）")
            return

        chunks_file = os.path.join(self.output_dir, "chunks.json")
        embeddings_file = os.path.join(self.output_dir, "embeddings.npy")

        # 语料已清空（embeddings 被删除）→ 把 retriever 置为空索引
        if not os.path.exists(embeddings_file):
            self.retriever.chunks = []
            self.retriever.embeddings = None
            self.retriever.cluster_centers = None
            self.retriever.cluster_labels = None
            self.retriever.cluster_chunk_indices = {}
            self.retriever.bm25 = None
            print("🔥 热更新完成：索引已清空（无文档）")
            return

        if not os.path.exists(chunks_file):
            print("⚠️ 热更新中止：chunks.json 不存在")
            return

        with self._reload_lock:
            # 从磁盘读取“最新” chunks + embeddings，重建聚类与 BM25
            self.retriever.index_documents(chunks_file, embeddings_file)
            # 新聚类落盘，保证下次重启也一致
            self.retriever.save_index(self.cluster_index_path)
        print("🔥 热更新完成：内存检索索引已刷新（无需重启）")

    # ---------- 查：列出所有文档 ----------
    def list_documents(self):
        documents = []
        allowed = ('.pdf', '.txt', '.md')
        for fname in os.listdir(self.data_dir):
            fpath = os.path.join(self.data_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(allowed):
                stat = os.stat(fpath)
                documents.append({
                    "filename": fname,
                    "size_kb": round(stat.st_size / 1024, 1),
                    "upload_time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "type": "PDF" if fname.endswith('.pdf') else "TXT"
                })
        documents.sort(key=lambda x: x["upload_time"], reverse=True)
        return documents

    # ---------- 增：上传并自动切块 + 向量化 + 更新索引 ----------
    def upload_document(self, file_obj):
        filename = file_obj.filename

        allowed = ('.pdf', '.txt', '.md')
        if not filename.lower().endswith(allowed):
            return {"error": "仅支持 PDF / TXT / MD 格式"}, 400

        save_path = os.path.join(self.data_dir, filename)
        file_obj.save(save_path)

        print(f"📄 处理文档: {filename}")
        chunks = self.processor.process_file(save_path)

        embeddings = self._embed([c["content"] for c in chunks])

        self._update_index(chunks, embeddings, filename)
        self._hot_reload_retriever()   # ← 热更新：刷新内存索引，免重启

        return {
            "filename": filename,
            "chunk_count": len(chunks),
            "status": "success",
            "message": f"文档已添加（{len(chunks)} 个文本块）"
        }

    # ---------- 删：删除并重建索引 ----------
    def delete_document(self, filename):
        fpath = os.path.join(self.data_dir, filename)
        if not os.path.exists(fpath):
            return {"error": "文件不存在", "code": 404}

        os.remove(fpath)
        self._rebuild_index(removed_filename=filename)   # ← 传文件名进去
        self._hot_reload_retriever()   # ← 热更新：刷新内存索引，免重启

        return {"status": "success", "message": f"已删除 {filename}", "code": 200}

    # ---------- 内部：追加式更新索引 ----------
    def _update_index(self, new_chunks, new_embeddings, filename):
        chunks_file = os.path.join(self.output_dir, "chunks.json")
        embeddings_file = os.path.join(self.output_dir, "embeddings.npy")

        existing_chunks = []
        existing_embeddings = None
        if os.path.exists(chunks_file):
            with open(chunks_file, 'r', encoding='utf-8') as f:
                existing_chunks = json.load(f)
        if os.path.exists(embeddings_file):
            existing_embeddings = np.load(embeddings_file)

        all_chunks = existing_chunks + new_chunks
        if existing_embeddings is not None and len(existing_embeddings) > 0:
            all_embeddings = np.vstack([existing_embeddings, new_embeddings])
        else:
            all_embeddings = new_embeddings

        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        np.save(embeddings_file, all_embeddings)

        print(f"✅ 索引已更新: 共 {len(all_chunks)} 个文档块")

    # ---------- 内部：从零重建索引（删除后调用）----------
    def _rebuild_index(self, removed_filename=None):
        """
        从已有的 chunks.json 里过滤掉被删文件的 chunks，然后重建索引。
        不再依赖 data/ 目录里有没有文件。
        """
        chunks_file = os.path.join(self.output_dir, "chunks.json")
        embeddings_file = os.path.join(self.output_dir, "embeddings.npy")

        # 读取现有 chunks
        if not os.path.exists(chunks_file):
            print("⚠️ chunks.json 不存在，无需重建")
            return

        with open(chunks_file, 'r', encoding='utf-8') as f:
            all_chunks = json.load(f)

        if removed_filename:
            # 找到要删除的 chunk 索引（source 字段里通常含文件名）
            removed_indices = set()
            for i, c in enumerate(all_chunks):
                source = c.get("source", "")
                if source == removed_filename:
                    removed_indices.add(i)

            if not removed_indices:
                # 没找到匹配的 chunk，不用重建
                print(f"⚠️ 未在索引中找到 {removed_filename} 对应的 chunks，跳过重建")
                return

            # 过滤 chunks
            remaining_chunks = [c for i, c in enumerate(all_chunks) if i not in removed_indices]
        else:
            # 没指定文件名，重建全部（从 data/ 目录重新处理）
            remaining_chunks = []
            for fname in os.listdir(self.data_dir):
                fpath = os.path.join(self.data_dir, fname)
                if os.path.isfile(fpath) and (fname.endswith('.pdf') or fname.endswith('.txt')):
                    chunks = self.processor.process_file(fpath)
                    remaining_chunks.extend(chunks)

        # 重新计算 embeddings
        if not remaining_chunks:
            # 全部删光了，清空索引
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            if os.path.exists(embeddings_file):
                os.remove(embeddings_file)
            print("✅ 索引已清空")
            return

        # 算向量
        all_embeddings = self._embed([c["content"] for c in remaining_chunks])
        np.save(embeddings_file, all_embeddings)

        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(remaining_chunks, f, ensure_ascii=False, indent=2)

        print(f"✅ 索引已重建: 共 {len(remaining_chunks)} 个文档块")
