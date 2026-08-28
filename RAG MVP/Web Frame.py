# -*- coding: utf-8 -*-
import os
import time as _time
import hashlib
import secrets
import base64
from flask import Flask, request, jsonify, Response, session
from dotenv import load_dotenv
from RAG_pipeline import TextEmbedder
from document_manager import DocumentManager

import logging

from clustered_retriever import ClusteredRetriever
from RAG_pipeline import RAGPipeline

# ===== 路径配置 =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===== 加载 .env =====
env_path = os.path.join(BASE_DIR, ".env")
print(f"🔍 查找 .env: {env_path}")
print(f"📁 .env exists: {os.path.exists(env_path)}")
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("DEEPSEEK_API_KEY")
print(f"🔑 API Key loaded: {'Yes' if api_key else 'No'}")

# ===== Flask 配置 =====
logging.getLogger('werkzeug').setLevel(logging.ERROR)
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
# 会话签名密钥（来自 .env，部署时务必替换为随机强字符串）
app.secret_key = os.getenv("FLASK_SECRET_KEY", "knowflow-dev-secret-please-change")

# ===== 账号分层（RBAC）：管理员口令集 =====
# 从 .env 的 KNOWFLOW_ADMIN_PASSWORD 读取，支持逗号分隔多个管理员。
# 游客（未登录）只能调用 /ask 问答；管理员登录后才能 /upload、/delete。
_ADMIN_PASSWORDS = set(
    p.strip() for p in os.getenv("KNOWFLOW_ADMIN_PASSWORD", "").split(",") if p.strip()
)
if not _ADMIN_PASSWORDS:
    print("⚠️ 未设置 KNOWFLOW_ADMIN_PASSWORD：所有上传/删除将永久被拒（403）。请在 .env 配置。")

# ===== 口令哈希（PBKDF2-HMAC-SHA256，标准库实现，零依赖） =====
# .env 的 KNOWFLOW_ADMIN_PASSWORD 推荐存哈希串，格式：pbkdf2$<iter>$<salt_b64>$<hash_b64>
# 兼容过渡：若 .env 仍是明文（非 pbkdf2$ 前缀），verify_password 会按明文比对，但会告警。
def hash_password(plain: str, iterations: int = 200_000) -> str:
    """把明文口令变成可安全存储的哈希串（含随机盐，不可逆）。"""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, iterations)
    return f"pbkdf2${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"

def verify_password(plain: str, stored: str) -> bool:
    """验证明文是否匹配已存储的口令（哈希或过渡期明文）。"""
    if stored.startswith("pbkdf2$"):
        try:
            _, it_s, salt_b64, hash_b64 = stored.split("$")
            salt = base64.b64decode(salt_b64)
            dk = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, int(it_s))
            return base64.b64encode(dk).decode() == hash_b64
        except Exception:
            return False
    # 过渡期兼容：.env 仍是明文时的兜底（请尽快迁移为哈希）
    return plain == stored

# 启动告警：检测是否仍在用明文口令
if _ADMIN_PASSWORDS and not any(p.startswith("pbkdf2$") for p in _ADMIN_PASSWORDS):
    print("⚠️ 管理员口令仍以明文存储在 .env！请用 gen_password_hash.py 生成哈希串替换，避免泄露即失守。")

# ===== 会话 TTL（登录态过期） =====
# 管理员登录态有效期（秒）。到期后在任意受保护接口自动失效，需重新登录。
# 可通过 .env 的 KNOWFLOW_ADMIN_TTL 覆盖（默认 3600 秒 = 1 小时）。
ADMIN_SESSION_TTL = int(os.getenv("KNOWFLOW_ADMIN_TTL", "3600"))

# ===== 防暴破：登录失败限流（进程内存态，重启清零，非持久化） =====
# 同一 IP 连续失败达到上限后锁定一段时间；锁定期间拒绝登录尝试。
# 可通过 .env 覆盖：KNOWFLOW_MAX_LOGIN_FAILS / KNOWFLOW_LOGIN_LOCK_SECONDS
MAX_LOGIN_FAILS = int(os.getenv("KNOWFLOW_MAX_LOGIN_FAILS", "5"))        # 允许连续失败次数
LOGIN_LOCK_SECONDS = int(os.getenv("KNOWFLOW_LOGIN_LOCK_SECONDS", "900"))  # 锁定时长（秒），默认 15 分钟
LOGIN_FAILS = {}  # {ip: {"count": int, "lock_until": float}}

def _login_locked_until(ip):
    """返回该 IP 的锁定到期时间戳；未锁定返回 None。"""
    rec = LOGIN_FAILS.get(ip)
    if rec and rec["lock_until"] > _time.time():
        return rec["lock_until"]
    return None

def _record_login_fail(ip):
    """记录一次失败；达到上限则进入锁定。"""
    rec = LOGIN_FAILS.get(ip, {"count": 0, "lock_until": 0})
    rec["count"] += 1
    if rec["count"] >= MAX_LOGIN_FAILS:
        rec["lock_until"] = _time.time() + LOGIN_LOCK_SECONDS
        rec["count"] = 0
    LOGIN_FAILS[ip] = rec

def _reset_login_fails(ip):
    """登录成功后清空该 IP 的失败记录。"""
    LOGIN_FAILS.pop(ip, None)

def _is_admin():
    """当前请求会话是否为管理员（自动检查登录态是否过期）。"""
    if not session.get("is_admin"):
        return False
    if session.get("admin_expire", 0) <= _time.time():
        # 登录态已过期：清理并返回未登录
        session.pop("is_admin", None)
        session.pop("admin_expire", None)
        return False
    return True

# ===== 用户反馈闭环（点赞 / 点踩）=====
import json as _json
import threading as _thr
_FEEDBACK_LOCK = _thr.Lock()
FEEDBACK_STORE = []  # 内存暂存，同时落盘 output/feedback.jsonl

def _save_feedback(entry):
    FEEDBACK_STORE.append(entry)
    try:
        fb_path = os.path.join(BASE_DIR, "output", "feedback.jsonl")
        os.makedirs(os.path.dirname(fb_path), exist_ok=True)
        with _FEEDBACK_LOCK, open(fb_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[feedback] 落盘失败: {e}")

# ===== 数据文件路径 =====
chunks_path = os.path.join(BASE_DIR, "output", "chunks.json")
embeddings_path = os.path.join(BASE_DIR, "output", "embeddings.npy")
cluster_index_path = os.path.join(BASE_DIR, "output", "cluster_index")
print(f"📁 chunks exists: {os.path.exists(chunks_path)}")
print(f"📁 embeddings exists: {os.path.exists(embeddings_path)}")

# ===== 初始化 embedder =====
embedder = TextEmbedder("paraphrase-multilingual-MiniLM-L12-v2")

# ===== 初始化 retriever =====
# alpha=0.4：混合检索中 BM25 占 40%、dense 占 60%。
# 对"人名/工号/编号"等专有名词场景，BM25 权重需足够大才能纠正纯向量的误召回。
retriever = ClusteredRetriever(embedder=embedder, n_clusters=20, routing_top_k=3, alpha=0.4)

if os.path.exists(f"{cluster_index_path}_clusters.npz"):
    retriever.load_index(cluster_index_path,
                         chunks_file=chunks_path,
                         embeddings_file=embeddings_path)
elif os.path.exists(chunks_path) and os.path.exists(embeddings_path):
    retriever.index_documents(chunks_file=chunks_path, embeddings_file=embeddings_path)
    retriever.save_index(cluster_index_path)
else:
    # 全新部署（无语料 / 无索引）→ 空知识库模式，上传文档后自动建索引，避免启动即崩
    print("⚠️ 未找到已有索引与语料，启动空知识库模式（上传文档后将自动建索引）")

# ===== 初始化 RAG Pipeline =====
_rag = None

def get_rag():
    global _rag
    if _rag is None:
        print("loading rag...")
        _rag = RAGPipeline(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            embedder=embedder,
            retriever=retriever
        )
        print("rag ready")
    return _rag

# ===== 初始化文档管理器 =====
doc_manager = DocumentManager(
    data_dir=os.path.join(BASE_DIR, "data"),
    output_dir=os.path.join(BASE_DIR, "output"),
    embedder=embedder,
    retriever=retriever,                    # ← 热更新：上传/删除后刷新内存索引
    cluster_index_path=cluster_index_path   # ← 与 save_index 同前缀
)

# ===== 前端页面（你原来的完整页面）=====
_PAGE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KnowFlow 智流 · 知识库问答</title>
<style>
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: #f0f2f5;
    display: flex;
    justify-content: center;
  }
  .app {
    width: 100%;
    max-width: 820px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    background: #fff;
    box-shadow: 0 0 24px rgba(0,0,0,0.06);
  }
  .header {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    background: linear-gradient(135deg, #4f7cff, #6a5cff);
    color: #fff;
  }
  .header .title { font-size: 16px; font-weight: 600; }
  .header .title small { font-weight: 400; opacity: 0.85; margin-left: 6px; font-size: 12px; }
  .header-actions { display: flex; gap: 8px; }
  .header button {
    border: none;
    background: rgba(255,255,255,0.2);
    color: #fff;
    padding: 6px 14px;
    border-radius: 18px;
    cursor: pointer;
    font-size: 13px;
    transition: background 0.2s;
  }
  .header button:hover { background: rgba(255,255,255,0.35); }
  .file-panel {
    display: none;
    padding: 12px 16px;
    background: #f0f2ff;
    border-bottom: 1px solid #d9d6ff;
  }
  .file-panel-row {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
    align-items: center;
    flex-wrap: wrap;
  }
  .file-panel-row input[type="file"] {
    flex: 1;
    max-width: 240px;
    font-size: 13px;
  }
  .file-panel-row button {
    background: #4f7cff;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 6px 14px;
    cursor: pointer;
    font-size: 13px;
  }
  .file-panel-row button:hover { background: #3a66e0; }
  #fileList { font-size: 13px; color: #333; }
  #fileList .file-item {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid #e0dfff;
  }
  #fileList .file-item button {
    color: #e54d42;
    border: none;
    background: none;
    cursor: pointer;
    font-size: 12px;
  }
  .messages {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: 20px;
    background: #f7f8fa;
  }
  .msg { display: flex; margin-bottom: 16px; }
  .msg.user { justify-content: flex-end; }
  .msg.bot { justify-content: flex-start; }
  .bubble {
    max-width: 76%;
    padding: 11px 14px;
    border-radius: 14px;
    line-height: 1.65;
    font-size: 14px;
    word-wrap: break-word;
    white-space: pre-wrap;
  }
  .msg.user .bubble {
    background: #4f7cff;
    color: #fff;
    border-bottom-right-radius: 4px;
  }
  .msg.bot .bubble {
    background: #fff;
    color: #1f2329;
    border: 1px solid #e8eaed;
    border-bottom-left-radius: 4px;
  }
  .bubble .sources {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px dashed #e2e5ea;
    font-size: 12px;
    color: #8a9099;
    line-height: 1.5;
  }
  .bubble .token {
    margin-top: 6px;
    font-size: 12px;
    color: #4f7cff;
  }
  .bubble .facts {
    margin-top: 6px;
    font-size: 12px;
    color: #6a5cff;
  }
  .bubble .err { color: #e54d42; }
  .followup {
    display: inline-block;
    font-size: 11px;
    color: #4f7cff;
    border: 1px solid #c9d6ff;
    border-radius: 8px;
    padding: 1px 6px;
    margin-right: 6px;
  }
  .loading { display: flex; gap: 4px; align-items: center; }
  .loading span {
    width: 7px; height: 7px; border-radius: 50%;
    background: #c0c4cc; display: inline-block;
    animation: blink 1.2s infinite both;
  }
  .loading span:nth-child(2) { animation-delay: 0.2s; }
  .loading span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes blink { 0%, 80%, 100% { opacity: 0.3; } 40% { opacity: 1; } }
  .inputbar {
    flex: 0 0 auto;
    display: flex;
    gap: 10px;
    padding: 14px 16px;
    border-top: 1px solid #eceef1;
    background: #fff;
  }
  .inputbar textarea {
    flex: 1 1 auto;
    resize: none;
    border: 1px solid #d9dce1;
    border-radius: 12px;
    padding: 10px 12px;
    font-size: 14px;
    font-family: inherit;
    line-height: 1.5;
    max-height: 120px;
    outline: none;
  }
  .inputbar textarea:focus { border-color: #4f7cff; }
  .inputbar button {
    flex: 0 0 auto;
    align-self: flex-end;
    border: none;
    background: #4f7cff;
    color: #fff;
    padding: 10px 20px;
    border-radius: 12px;
    cursor: pointer;
    font-size: 14px;
  }
  .inputbar button:disabled { background: #b9c6ff; cursor: not-allowed; }
  .stream-toggle { color:#7c89c4; font-size:13px; display:flex; align-items:center; gap:4px; user-select:none; }
  .feedback { margin-top:8px; }
  .feedback button { background:#eef1ff; border:1px solid #c7d0ff; border-radius:8px; cursor:pointer; padding:2px 10px; font-size:13px; }
  .feedback button:hover { background:#dfe5ff; }
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <div class="title">KnowFlow 智流<small>基于你的文档智能答疑</small></div>
    <div class="header-actions">
      <button id="toggleFiles" style="display:none;">📁 文件管理</button>
      <button id="adminLoginBtn">🔐 管理员登录</button>
      <button id="adminLogoutBtn" style="display:none;">🔓 退出管理</button>
      <button id="newchat">＋ 新对话</button>
    </div>
  </div>

  <div id="filePanel" class="file-panel">
    <div class="file-panel-row">
      <input type="file" id="fileInput" accept=".pdf,.txt">
      <button id="uploadBtn">📤 上传</button>
      <span id="uploadStatus"></span>
    </div>
    <div id="fileList"></div>
  </div>

  <div class="messages" id="messages"></div>
  <div class="inputbar">
    <textarea id="input" rows="1" placeholder="输入你的问题，Enter 发送，Shift+Enter 换行"></textarea>
    <button id="send">发送</button>
  </div>
</div>
<script>
  var sessionId = 'sess-' + Math.random().toString(36).slice(2);
  var messagesEl = document.getElementById('messages');
  var inputEl = document.getElementById('input');
  var sendBtn = document.getElementById('send');

  function scrollToBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function(c) {
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
    });
  }

  function addMessage(role, html) {
    var wrap = document.createElement('div');
    wrap.className = 'msg ' + (role === 'user' ? 'user' : 'bot');
    var bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = html;
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    scrollToBottom();
  }

  function formatSources(sources) {
    if (!sources || !sources.length) return '';
    var items = sources.map(function(s) {
      if (typeof s === 'string') return escapeHtml(s);
      if (Array.isArray(s)) return escapeHtml(s.join(' — '));
      if (typeof s === 'object') {
        var name = s.source || s.doc || s.name || '';
        var score = (s.relevance_score !== undefined) ? '（相关度 ' + s.relevance_score + '）' : '';
        return escapeHtml(String(name) + score);
      }
      return escapeHtml(String(s));
    });
    return '<div class="sources">📚 <b>来源：</b>' + items.join('；') + '</div>';
  }

  function formatFacts(facts) {
    if (!facts || !facts.length) return '';
    var items = facts.map(function(f) {
      if (typeof f === 'string') return escapeHtml(f);
      if (typeof f === 'object') return escapeHtml(JSON.stringify(f));
      return escapeHtml(String(f));
    });
    return '<div class="facts">💡 <b>关键事实：</b>' + items.join('；') + '</div>';
  }

  function formatToken(report) {
    if (!report) return '';
    var saves = report.total_savings_percent !== undefined ? report.total_savings_percent
              : (report.savings_percent !== undefined ? report.savings_percent : '');
    var html = '';
    if (saves !== '' && saves !== undefined) {
      html += '<div class="token">💰 检索文本压缩：节省 ' + escapeHtml(String(saves)) + '%</div>';
    }
    if (report.end_to_end_saving_percent !== undefined) {
      html += '<div class="token">📊 对比全库直塞（全文扫描）：端到端节省 '
            + escapeHtml(String(report.end_to_end_saving_percent)) + '%</div>';
    }
    return html;
  }

  var lastUserQuestion = '';

  function send() {
    var q = inputEl.value.trim();
    if (!q) return;
    lastUserQuestion = q;
    inputEl.value = '';
    addMessage('user', escapeHtml(q));

    var loading = document.createElement('div');
    loading.className = 'msg bot';
    loading.innerHTML = '<div class="bubble"><div class="loading"><span></span><span></span><span></span></div></div>';
    messagesEl.appendChild(loading);
    scrollToBottom();
    sendBtn.disabled = true;

    // 把评分按钮挂到“真正显示出来的那条回答气泡”上（而非已移除的加载气泡）
    function attachFeedback(answerText) {
      var bubbles = messagesEl.querySelectorAll('.msg.bot .bubble');
      var bubble = bubbles[bubbles.length - 1];
      if (!bubble || bubble.querySelector('.feedback')) return;
      var fb = document.createElement('div');
      fb.className = 'feedback';
      fb.innerHTML = '<button class="fb-up">👍</button><button class="fb-down">👎</button>';
      fb.querySelector('.fb-up').onclick = function() { submitFeedback(answerText, 'up'); fb.remove(); };
      fb.querySelector('.fb-down').onclick = function() { submitFeedback(answerText, 'down'); fb.remove(); };
      bubble.appendChild(fb);
    }

    fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, session_id: sessionId })
    })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      loading.remove();
      sendBtn.disabled = false;
      if (d.error) {
        addMessage('bot', '<span class="err">⚠️ ' + escapeHtml(d.error) + '</span>');
        return;
      }
      var tag = d.is_followup ? '<span class="followup">↺ 多轮追问</span>' : '';
      var html = tag + escapeHtml(d.answer).replace(/\n/g, '<br>');
      html += formatSources(d.sources);
      html += formatFacts(d.key_facts);
      html += formatToken(d.token_report);
      addMessage('bot', html);
      attachFeedback(d.answer);
    })
    .catch(function(e) {
      loading.remove();
      sendBtn.disabled = false;
      addMessage('bot', '<span class="err">⚠️ 网络错误：' + escapeHtml(String(e)) + '</span>');
    });
  }

  function submitFeedback(answerText, rating) {
    fetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: lastUserQuestion, answer: answerText, rating: rating, session_id: sessionId })
    }).then(function() {}).catch(function() {});
  }

  sendBtn.onclick = send;
  inputEl.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });

  document.getElementById('newchat').onclick = function() {
    fetch('/clear', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId })
    }).finally(function() {
      messagesEl.innerHTML = '';
      sessionId = 'sess-' + Math.random().toString(36).slice(2);
      addMessage('bot', '👋 已开启新对话，基于你的文档，有什么想问的？');
    });
  };

  var filePanel = document.getElementById('filePanel');
  var fileListEl = document.getElementById('fileList');
  var uploadStatus = document.getElementById('uploadStatus');

  document.getElementById('toggleFiles').onclick = function() {
    if (filePanel.style.display === 'none' || !filePanel.style.display) {
      filePanel.style.display = 'block';
      loadFileList();
    } else {
      filePanel.style.display = 'none';
    }
  };

  function loadFileList() {
    fetch('/list_docs')
      .then(function(r) { return r.json(); })
      .then(function(docs) {
        if (docs.error) {
          fileListEl.innerHTML = '<div style="color:#e54d42;">加载失败：' + docs.error + '</div>';
          return;
        }
        if (docs.length === 0) {
          fileListEl.innerHTML = '<div style="color:#999;">暂无文档，上传一个 PDF 或 TXT 试试吧</div>';
          return;
        }
        fileListEl.innerHTML = docs.map(function(d) {
          return '<div class="file-item"><span>📄 ' + escapeHtml(d.filename) + ' <small style="color:#999;">(' + d.size_kb + ' KB, ' + d.upload_time + ')</small></span><button onclick="deleteFile(\'' + d.filename + '\')">🗑 删除</button></div>';
        }).join('');
      })
      .catch(function(e) {
        fileListEl.innerHTML = '<div style="color:#e54d42;">加载失败</div>';
      });
  }

  document.getElementById('uploadBtn').onclick = function() {
    var input = document.getElementById('fileInput');
    if (!input.files.length) {
      uploadStatus.textContent = '请先选择文件';
      uploadStatus.style.color = '#e54d42';
      return;
    }
    var formData = new FormData();
    formData.append('file', input.files[0]);
    uploadStatus.textContent = '上传中...';
    uploadStatus.style.color = '#6a5cff';
    fetch('/upload', {
      method: 'POST',
      body: formData
    })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.error) {
        uploadStatus.textContent = '❌ ' + d.error;
        uploadStatus.style.color = '#e54d42';
      } else {
        uploadStatus.textContent = '✅ ' + d.message;
        uploadStatus.style.color = '#4f7cff';
        input.value = '';
        loadFileList();
      }
    })
    .catch(function(e) {
      uploadStatus.textContent = '❌ 上传失败';
      uploadStatus.style.color = '#e54d42';
    });
  };

  window.deleteFile = function(filename) {
    if (!confirm('确定删除 ' + filename + '？')) return;
    fetch('/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: filename })
    })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.error) {
        alert('删除失败：' + d.error);
      } else {
        loadFileList();
      }
    });
  };

  // ===== 账号分层：管理员登录态 =====
  var isAdmin = false;

  function refreshAuth() {
    fetch('/auth_status')
      .then(function(r) { return r.json(); })
      .then(function(d) {
        isAdmin = !!d.is_admin;
        document.getElementById('toggleFiles').style.display = isAdmin ? '' : 'none';
        document.getElementById('adminLoginBtn').style.display = isAdmin ? 'none' : '';
        document.getElementById('adminLogoutBtn').style.display = isAdmin ? '' : 'none';
        if (!isAdmin) filePanel.style.display = 'none';
      })
      .catch(function() { /* 默认游客，忽略 */ });
  }

  document.getElementById('adminLoginBtn').onclick = function() {
    var pwd = prompt('请输入管理员口令：');
    if (pwd === null) return;
    fetch('/admin_login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password: pwd })
    })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.is_admin) {
        isAdmin = true;
        document.getElementById('toggleFiles').style.display = '';
        document.getElementById('adminLoginBtn').style.display = 'none';
        document.getElementById('adminLogoutBtn').style.display = '';
        alert('✅ 管理员已登录');
      } else {
        alert('❌ ' + (d.error || '登录失败'));
      }
    })
    .catch(function() { alert('❌ 登录请求失败'); });
  };

  document.getElementById('adminLogoutBtn').onclick = function() {
    fetch('/admin_logout', { method: 'POST' })
      .then(function() {
        isAdmin = false;
        document.getElementById('toggleFiles').style.display = 'none';
        document.getElementById('adminLoginBtn').style.display = '';
        document.getElementById('adminLogoutBtn').style.display = 'none';
        filePanel.style.display = 'none';
      });
  };

  refreshAuth();  // 页面加载即确认登录态
  addMessage('bot', '👋 你好，我是 KnowFlow 智流，基于你的文档为你答疑。试试问我点什么吧～');
</script>
</body>
</html>'''

# ===== 路由 =====
@app.route('/')
def index():
    return Response(_PAGE, mimetype='text/html')

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True)
        question = data.get("question", "").strip()
        session_id = data.get("session_id", "default")

        if not question:
            return jsonify({"error": "请输入问题"}), 400

        t0 = _time.time()
        result = get_rag().ask(question, session_id=session_id)
        elapsed = _time.time() - t0
        print(f"[Web] 问题: {question[:30]} | 总耗时: {elapsed:.1f}s")

        if "error" in result and result["error"]:
            return jsonify({"error": result["error"]}), 500

        return jsonify({
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "key_facts": result.get("key_facts", []),
            "token_report": result.get("token_report"),
            "is_followup": result.get("is_followup")
        })

    except Exception as e:
        safe_msg = str(e)
        print(f"[Web] /ask 异常: {safe_msg}")
        return jsonify({"error": f"服务暂时不可用，请稍后重试。（{safe_msg}）"}), 500

@app.route("/delete", methods=["POST"])
def delete():
    # ===== RBAC：仅管理员可删除 =====
    if not _is_admin():
        return jsonify({"error": "需要管理员权限才能删除文档"}), 403
    try:
        data = request.get_json(force=True)
        filename = data.get("filename", "").strip()
        if not filename:
            return jsonify({"error": "请指定文件名"}), 400
        result = doc_manager.delete_document(filename)
        if "error" in result:
            return jsonify(result), result.get("code", 400)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    # ===== RBAC：仅管理员可上传 =====
    if not _is_admin():
        return jsonify({"error": "需要管理员权限才能上传文档"}), 403
    try:
        if 'file' not in request.files:
            return jsonify({"error": "没有文件"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "未选择文件"}), 400
        result = doc_manager.upload_document(file)
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/list_docs", methods=["GET"])
def list_docs():
    # ===== RBAC：仅管理员可查看文档清单（避免游客枚举知识库）=====
    if not _is_admin():
        return jsonify({"error": "需要管理员权限才能查看文档列表"}), 403
    try:
        docs = doc_manager.list_documents()
        return jsonify(docs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/feedback", methods=["POST"])
def feedback():
    """用户反馈闭环：点赞 / 点踩，落库供后续反哺阈值与扩充评估集。游客即可提交。"""
    try:
        data = request.get_json(force=True)
        rating = data.get("rating")  # "up" / "down"
        if rating not in ("up", "down"):
            return jsonify({"error": "rating 必须为 up 或 down"}), 400
        entry = {
            "ts": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": data.get("session_id", "default"),
            "question": (data.get("question") or "").strip()[:500],
            "answer": (data.get("answer") or "").strip()[:2000],
            "rating": rating,
            "comment": (data.get("comment") or "").strip()[:500],
        }
        _save_feedback(entry)
        return jsonify({"status": "ok", "stored": len(FEEDBACK_STORE)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== 账号分层：管理员登录 / 登出 / 状态 =====
@app.route("/admin_login", methods=["POST"])
def admin_login():
    if not _ADMIN_PASSWORDS:
        return jsonify({"error": "服务端未配置管理员口令"}), 500
    ip = request.remote_addr
    locked_until = _login_locked_until(ip)
    if locked_until:
        remain = int(locked_until - _time.time())
        return jsonify({"error": f"登录尝试过于频繁，请 {remain} 秒后重试"}), 429
    data = request.get_json(force=True, silent=True) or {}
    pwd = (data.get("password") or "").strip()
    if any(verify_password(pwd, stored) for stored in _ADMIN_PASSWORDS):
        _reset_login_fails(ip)
        session["is_admin"] = True
        session["admin_expire"] = _time.time() + ADMIN_SESSION_TTL
        return jsonify({"status": "ok", "is_admin": True, "expires_in": ADMIN_SESSION_TTL})
    _record_login_fail(ip)
    return jsonify({"error": "口令错误"}), 401

@app.route("/admin_logout", methods=["POST"])
def admin_logout():
    session.pop("is_admin", None)
    session.pop("admin_expire", None)
    return jsonify({"status": "ok", "is_admin": False})

@app.route("/auth_status", methods=["GET"])
def auth_status():
    if _is_admin():
        remain = int(session.get("admin_expire", 0) - _time.time())
        return jsonify({"is_admin": True, "expires_in": remain})
    return jsonify({"is_admin": False})

@app.route("/clear", methods=["POST"])
def clear():
    try:
        data = request.get_json(force=True, silent=True) or {}
        session_id = data.get("session_id", "default")
        get_rag().clear_session(session_id)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== 启动 =====
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🚀 KnowFlow 智流 Web 服务启动中...")
    print("=" * 50)
    print("📌 浏览器访问: http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000, use_reloader=False)