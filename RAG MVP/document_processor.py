# -*- coding: utf-8 -*-
"""文档处理器：读取 PDF / TXT / MD 并切分为文本块（chunk）

这个文件是 document_manager.py 要 import 的 DocumentProcessor。
之前只在 .ipynb 里跑过切块逻辑，从没落成独立 .py 文件，所以补上。
现仅支持 PDF / TXT / MD（多格式解析 docx/xlsx/pptx/html 留作后续迭代，未启用）。

用法：
    from document_processor import DocumentProcessor
    proc = DocumentProcessor(chunk_size=300, chunk_overlap=50)
    chunks = proc.process_file("员工手册.pdf")
    # chunks = [{"content": "...", "source": "员工手册.pdf", "chunk_index": 0}, ...]
"""

import os


class DocumentProcessor:
    """把一份文档切成多段固定长度的小块。

    为什么要切块？
      向量模型一次只能理解有限长度的文本。
      把整本手册塞进去会超出长度限制，且检索时不精准。
      切成 300 字左右的小块后，用户提问时只检索最相关的几块，
      既省 Token 又准。

    支持格式：.pdf .txt .md
    """

    def __init__(self, chunk_size=300, chunk_overlap=50):
        """
        chunk_size: 每块多少字符（中文约 300 字 = 一小段）
        chunk_overlap: 相邻块重叠多少字符（防止一句话被切断在边界）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_file(self, file_path):
        """读取单个文件，返回文本块列表。支持 PDF / TXT / MD。"""
        ext = os.path.splitext(file_path)[1].lower()
        readers = {
            ".pdf": self._read_pdf,
            ".txt": self._read_txt,
            ".md": self._read_txt,
        }
        if ext not in readers:
            raise ValueError(f"不支持的文件类型: {ext}（支持 .pdf/.txt/.md）")
        text = readers[ext](file_path)

        source = os.path.basename(file_path)
        return self._chunk_text(text, source)

    # ---------- 内部方法：读文件 ----------

    def _read_pdf(self, path):
        """用 pypdf 读取 PDF 的全部文字（需要 pip install pypdf）"""
        from pypdf import PdfReader
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        return "\n".join(parts)

    def _read_txt(self, path):
        """读取 TXT / MD 文件（兼容各种编码，读不出的字符忽略）"""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # ---------- 内部方法：切块 ----------

    def _chunk_text(self, text, source):
        """滑动窗口切分：每次取 chunk_size 个字符，向后挪 (size-overlap)"""
        text = text.strip()
        if not text:
            return []

        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            piece = text[start:start + self.chunk_size]
            if piece.strip():   # 跳过纯空白块
                chunks.append({
                    "content": piece.strip(),
                    "source": source,
                    "chunk_index": idx
                })
                idx += 1
            start += step
        return chunks


# ===== 自测 =====
if __name__ == "__main__":
    import tempfile, time

    # 造一个临时 TXT 测试
    tmp = os.path.join(tempfile.gettempdir(), f"test_{int(time.time())}.txt")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("年假天数为：工作满1-5年享5天。" * 50)

    proc = DocumentProcessor(chunk_size=300, chunk_overlap=50)
    chunks = proc.process_file(tmp)
    print(f"✅ 测试文档切成 {len(chunks)} 块，每块约 {len(chunks[0]['content'])} 字")
    print(f"   第一块来源标记: {chunks[0]['source']}")
    os.remove(tmp)
