# -*- coding: utf-8 -*-
import os
import html
import time as _time
import hashlib
import secrets
import base64
from flask import Flask, request, jsonify, Response, session
from dotenv import load_dotenv
from RAG_pipeline import TextEmbedder
from document_manager import DocumentManager
from document_processor import DocumentProcessor

import logging

from clustered_retriever import ClusteredRetriever
from RAG_pipeline import RAGPipeline
# 注意：提示词 / 标准文档会被管理员在线修改（save_prompts 会重新绑定模块级变量），
# 因此这里导入的是「取值函数」而不是字典对象本身，避免持有过期的旧引用。
from structured_cite import (get_task_prompt, get_task_prompts, get_standard_doc,
                            get_standard_docs, save_prompts, save_standard_docs)

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

# ===== 会话级任务记忆：让「猜你想问」选中的任务型系统指令在整轮对话中持续生效 =====
SESSION_TASKS = {}       # {session_id: task_key}
_SESSION_TASKS_MAX = 500  # 上限：会话 ID 由前端随机生成，不设限会随访问量无限增长

def _remember_task(session_id, task):
    """记录/读取会话的任务键，并做容量裁剪（简易 FIFO）。

    前端每次点「新对话」都会生成一个新 session_id，原实现只增不减，
    长时间运行会持续吃内存；超过上限时丢弃最早写入的一批。
    """
    if task:
        SESSION_TASKS[session_id] = task
        if len(SESSION_TASKS) > _SESSION_TASKS_MAX:
            for k in list(SESSION_TASKS.keys())[:_SESSION_TASKS_MAX // 10]:
                SESSION_TASKS.pop(k, None)
        return task
    return SESSION_TASKS.get(session_id)

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
# alpha=0.4：融合公式为 alpha*dense + (1-alpha)*bm25，即 dense 占 40%、BM25 占 60%。
# （原注释把两者写反了，与 clustered_retriever.py 的 fused 计算不符）
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
  .std-panel {
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px dashed #d0d6e8;
  }
  .std-title { font-size: 14px; font-weight: 600; color: #334; margin-bottom: 8px; }
  .std-title small { font-weight: 400; color: #8a9099; font-size: 12px; }
  .std-item {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    padding: 6px 0;
    border-bottom: 1px solid #eef0f5;
  }
  .std-item .std-label { width: 120px; font-size: 13px; color: #445; }
  .std-item input[type="file"] { flex: 1 1 140px; font-size: 12px; }
  .std-item .std-btn {
    border: none; background: #4f7cff; color: #fff; padding: 4px 12px;
    border-radius: 6px; cursor: pointer; font-size: 12px;
  }
  .std-item .std-del {
    border: none; background: none; color: #e54d42; cursor: pointer; font-size: 12px;
  }
  .std-item .std-status { font-size: 12px; color: #4f7cff; }
  .std-item .std-meta { font-size: 12px; color: #999; width: 100%; }
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
  .bubble .confidence {
    margin-top: 8px;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
  }
  .bubble .confidence.high { background: #e6f7ed; color: #1a8a4f; }
  .bubble .confidence.mid  { background: #fff4e0; color: #b9770b; }
  .bubble .confidence.low  { background: #fdecec; color: #c0392b; }
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
  .suggest {
    flex: 0 0 auto;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: #f7f8fa;
    border-top: 1px solid #eceef1;
  }
  .suggest-label { font-size: 13px; color: #8a9099; }
  .suggest .chip {
    border: 1px solid #c9d6ff;
    background: #eef1ff;
    color: #3a4fb5;
    padding: 5px 12px;
    border-radius: 16px;
    cursor: pointer;
    font-size: 13px;
    transition: background 0.15s, border-color 0.15s;
  }
  .suggest .chip:hover { background: #dfe5ff; border-color: #4f7cff; }
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

    <div class="std-panel">
      <div class="std-title">📋 标准文件管理 <small>（上传后并入请求前缀，命中 DeepSeek 前缀缓存）</small></div>
      <div id="stdList"></div>
    </div>
  </div>

  <div class="messages" id="messages"></div>
  <div class="suggest" id="suggest">
    <span class="suggest-label">hi，猜你想问：</span>
    <button class="chip" data-task="revise_proposal" data-q="帮我修改策划案">帮我修改策划案</button>
    <button class="chip" data-task="revise_art" data-q="帮我修改美术需求">帮我修改美术需求</button>
    <button class="chip" data-task="organize_ui" data-q="帮我整理UI需求">帮我整理UI需求</button>
  </div>
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
  var currentTask = null;  // 当前选中的「猜你想问」任务键，随本次发送带上，服务端按会话记忆

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
        var ref = (s.ref_id !== undefined) ? '[' + s.ref_id + '] ' : '';
        var name = s.source || s.doc || s.name || '';
        var ex = s.excerpt ? '：' + s.excerpt : '';
        return escapeHtml(ref + name + ex);
      }
      return escapeHtml(String(s));
    });
    return '<div class="sources">📚 <b>来源：</b>' + items.join('；') + '</div>';
  }

  function formatConfidence(confidence, low) {
    if (low) return '<div class="confidence low">⚠️ 置信度低：检索资料不足，回答可能不可靠</div>';
    if (confidence === undefined || confidence === null) return '';
    var pct = Math.round(confidence * 100);
    var cls = confidence >= 0.7 ? 'high' : (confidence >= 0.4 ? 'mid' : 'low');
    var label = cls === 'high' ? '高' : (cls === 'mid' ? '中' : '低');
    return '<div class="confidence ' + cls + '">🎯 置信度：' + pct + '%（' + label + '）</div>';
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
    // 修复：先取任务键再判空。原写法在输入为空时直接 return，
    // currentTask 却已被芯片设置好且没清掉，会「残留」到用户下一次手动提问上，
    // 导致普通问题莫名走成任务模式。
    var task = currentTask;      // 捕获本次任务键（芯片点击设置的）
    currentTask = null;          // 用后即清，避免污染后续手动输入
    if (!q) return;
    lastUserQuestion = q;
    inputEl.value = '';
    addMessage('user', escapeHtml(q));
    sendBtn.disabled = true;

    // 加载气泡（在拿到首个 token 前显示）
    var loading = document.createElement('div');
    loading.className = 'msg bot';
    loading.innerHTML = '<div class="bubble"><div class="loading"><span></span><span></span><span></span></div></div>';
    messagesEl.appendChild(loading);
    scrollToBottom();

    var bubble = null;
    var tokenBuf = '';

    // 首个 token 到达时把加载气泡换成真正的回答气泡
    function ensureBubble() {
      if (bubble) return bubble;
      loading.remove();
      var wrap = document.createElement('div');
      wrap.className = 'msg bot';
      bubble = document.createElement('div');
      bubble.className = 'bubble';
      wrap.appendChild(bubble);
      messagesEl.appendChild(wrap);
      scrollToBottom();
      return bubble;
    }

    function attachFeedback(answerText) {
      if (!bubble || bubble.querySelector('.feedback')) return;
      var fb = document.createElement('div');
      fb.className = 'feedback';
      fb.innerHTML = '<button class="fb-up">👍</button><button class="fb-down">👎</button>';
      fb.querySelector('.fb-up').onclick = function() { submitFeedback(answerText, 'up'); fb.remove(); };
      fb.querySelector('.fb-down').onclick = function() { submitFeedback(answerText, 'down'); fb.remove(); };
      bubble.appendChild(fb);
    }

    fetch('/ask_stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, session_id: sessionId, task: task })
    })
    .then(function(res) {
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var reader = res.body.getReader();
      var decoder = new TextDecoder('utf-8');
      var buf = '';

      function pump() {
        return reader.read().then(function(result) {
          if (result.done) { sendBtn.disabled = false; return; }
          buf += decoder.decode(result.value, { stream: true });
          // SSE 以空行（\n\n）分隔事件，把已完整的事件切出来处理，残留片段留到下次
          var parts = buf.split('\n\n');
          buf = parts.pop();
          for (var i = 0; i < parts.length; i++) {
            var line = parts[i].trim();
            if (line.indexOf('data: ') !== 0) continue;
            var msg;
            try { msg = JSON.parse(line.slice(6)); } catch (e) { continue; }
            if (msg.type === 'token') {
              var b = ensureBubble();
              tokenBuf += msg.text;
              b.innerHTML = escapeHtml(tokenBuf).replace(/\n/g, '<br>');
              scrollToBottom();
            } else if (msg.type === 'meta') {
              var b2 = ensureBubble();
              var d = msg.data;
              if (!tokenBuf) {
                // 没有流式正文（如拒答场景）：用 meta.answer 兜底显示
                b2.innerHTML = escapeHtml(d.answer || '').replace(/\n/g, '<br>');
              }
              if (d.is_followup) {
                b2.innerHTML = '<span class="followup">↺ 多轮追问</span>' + b2.innerHTML;
              }
              b2.innerHTML += formatConfidence(d.confidence, d.low_confidence);
              b2.innerHTML += formatSources(d.sources);
              b2.innerHTML += formatFacts(d.key_facts);
              b2.innerHTML += formatToken(d.token_report);
              attachFeedback(d.answer || tokenBuf);
            } else if (msg.type === 'error') {
              var b3 = ensureBubble();
              b3.innerHTML += '<span class="err">⚠️ ' + escapeHtml(msg.message) + '</span>';
              sendBtn.disabled = false;
            } else if (msg.type === 'done') {
              sendBtn.disabled = false;
            }
          }
          return pump();
        });
      }
      return pump();
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

  // ===== 猜你想问：快捷指令芯片，点击即按对应任务发送 =====
  document.querySelectorAll('#suggest .chip').forEach(function(chip) {
    chip.addEventListener('click', function() {
      currentTask = chip.getAttribute('data-task');   // 任务键：服务端据此注入固定系统指令
      inputEl.value = chip.getAttribute('data-q') || '';
      send();
    });
  });

  document.getElementById('newchat').onclick = function() {
    // 修复：/clear 原先返回 500 时 fetch 不会 reject，.finally 照样执行，
    // 前端表现成「新对话成功」而服务端记忆从未清除 —— 静默失败。
    // 这里显式检查响应状态并告警，界面重置则无论成败都照常进行。
    fetch('/clear', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId })
    })
    .then(function(r) { if (!r.ok) throw new Error('HTTP ' + r.status); })
    .catch(function(e) { console.warn('[clear] 服务端清空会话失败：', e); })
    .then(function() {
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
      loadStdList();
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
        // 安全修复：原先把文件名直接拼进 onclick="deleteFile('...')"，
        // 文件名里含单引号/引号时会截断 JS 字符串，既可造成按钮失效，也可被注入脚本。
        // 改为写进 data-* 属性（经 escapeHtml 转义）再用事件绑定读取。
        fileListEl.innerHTML = docs.map(function(d) {
          return '<div class="file-item">' +
            '<span>📄 ' + escapeHtml(d.filename) +
            ' <small style="color:#999;">(' + escapeHtml(String(d.size_kb)) + ' KB, ' +
            escapeHtml(String(d.upload_time)) + ')</small></span>' +
            '<button class="del-doc" data-file="' + escapeHtml(d.filename) + '">🗑 删除</button>' +
            '</div>';
        }).join('');
        fileListEl.querySelectorAll('.del-doc').forEach(function(btn) {
          btn.onclick = function() { deleteFile(btn.getAttribute('data-file')); };
        });
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

  // ===== 标准文件管理：在文件管理面板内上传/删除各任务的标准文档 =====
  var STD_TASKS = [
    { key: "revise_proposal", label: "标准策划案" },
    { key: "revise_art", label: "标准美术需求" },
    { key: "organize_ui", label: "标准UI需求" }
  ];
  var stdListEl = document.getElementById('stdList');

  function loadStdList() {
    fetch('/admin/standard_list')
      .then(function(r) { return r.json(); })
      .then(function(data) {
        if (data.error) { stdListEl.innerHTML = '<div style="color:#e54d42;">加载失败：' + data.error + '</div>'; return; }
        stdListEl.innerHTML = STD_TASKS.map(function(t) {
          var info = data[t.key] || {};
          var meta = info.has_doc
            ? '<div class="std-meta">已载入：' + info.length + ' 字' + (info.preview ? ' · 预览：' + info.preview + '…' : '') + '</div>'
            : '<div class="std-meta">尚未上传标准文档</div>';
          return '<div class="std-item">' +
            '<span class="std-label">' + t.label + '</span>' +
            '<input type="file" id="stdFile_' + t.key + '" accept=".pdf,.txt,.md">' +
            '<button class="std-btn" data-key="' + t.key + '">上传</button>' +
            (info.has_doc ? '<button class="std-del" data-key="' + t.key + '">清空</button>' : '') +
            '<span class="std-status" id="stdStatus_' + t.key + '"></span>' +
            meta +
            '</div>';
        }).join('');

        stdListEl.querySelectorAll('.std-btn').forEach(function(btn) {
          btn.onclick = function() {
            var key = btn.getAttribute('data-key');
            var input = document.getElementById('stdFile_' + key);
            var status = document.getElementById('stdStatus_' + key);
            if (!input.files.length) { status.textContent = '请先选文件'; status.style.color = '#e54d42'; return; }
            var fd = new FormData();
            fd.append('file', input.files[0]);
            fd.append('task', key);
            status.textContent = '解析中…'; status.style.color = '#6a5cff';
            fetch('/admin/standard_upload', { method: 'POST', body: fd })
              .then(function(r) { return r.json(); })
              .then(function(d) {
                if (d.error) { status.textContent = '❌ ' + d.error; status.style.color = '#e54d42'; }
                else { status.textContent = '✅ 已载入 ' + d.length + ' 字'; status.style.color = '#4f7cff'; loadStdList(); }
              })
              .catch(function() { status.textContent = '❌ 上传失败'; status.style.color = '#e54d42'; });
          };
        });
        stdListEl.querySelectorAll('.std-del').forEach(function(btn) {
          btn.onclick = function() {
            var key = btn.getAttribute('data-key');
            if (!confirm('清空该标准文档？')) return;
            fetch('/admin/standard_delete', {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ task: key })
            }).then(function() { loadStdList(); });
          };
        });
      })
      .catch(function() { stdListEl.innerHTML = '<div style="color:#e54d42;">加载失败</div>'; });
  }

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
        # 修复：非流式 /ask 此前完全忽略 task，与 /ask_stream 行为分裂
        # （流式带任务走快路径，非流式却仍去检索）。这里对齐两者语义。
        task = _remember_task(session_id, data.get("task"))

        if not question:
            return jsonify({"error": "请输入问题"}), 400

        t0 = _time.time()
        result = get_rag().ask(question, session_id=session_id, task=task)
        elapsed = _time.time() - t0
        print(f"[Web] 问题: {question[:30]} | 总耗时: {elapsed:.1f}s")

        if "error" in result and result["error"]:
            return jsonify({"error": result["error"]}), 500

        return jsonify({
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "key_facts": result.get("key_facts", []),
            "token_report": result.get("token_report"),
            "confidence": result.get("confidence"),
            "low_confidence": result.get("low_confidence", False),
            "reject_reason": result.get("reject_reason"),
            "is_followup": result.get("is_followup")
        })

    except Exception as e:
        safe_msg = str(e)
        print(f"[Web] /ask 异常: {safe_msg}")
        return jsonify({"error": f"服务暂时不可用，请稍后重试。（{safe_msg}）"}), 500

@app.route("/ask_stream", methods=["POST"])
def ask_stream():
    """流式问答端点：返回 text/event-stream（SSE），把答案逐字推给前端。"""
    try:
        data = request.get_json(force=True)
        question = data.get("question", "").strip()
        session_id = data.get("session_id", "default")
        # 任务键：优先取本次请求携带的；否则沿用本会话之前选过的（让任务指令贯穿整轮对话）
        task = _remember_task(session_id, data.get("task"))
        if not question:
            return jsonify({"error": "请输入问题"}), 400

        def event_stream():
            try:
                for kind, payload in get_rag().multi_agent_ask_stream(question, session_id=session_id, task=task):
                    if kind == "token":
                        yield "data: " + _json.dumps({"type": "token", "text": payload}, ensure_ascii=False) + "\n\n"
                    elif kind == "meta":
                        yield "data: " + _json.dumps({"type": "meta", "data": payload}, ensure_ascii=False) + "\n\n"
                    elif kind == "error":
                        yield "data: " + _json.dumps({"type": "error", "message": payload}, ensure_ascii=False) + "\n\n"
                yield "data: " + _json.dumps({"type": "done"}, ensure_ascii=False) + "\n\n"
            except Exception as e:
                yield "data: " + _json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n\n"

        # SSE 必须关闭缓冲，否则浏览器要等攒够才显示
        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # 关掉反向代理缓冲（如 cpolar / nginx）
                "Connection": "keep-alive",
            },
        )
    except Exception as e:
        return jsonify({"error": f"服务暂时不可用，请稍后重试。（{e}）"}), 500

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
        # 服务端未配置口令属于「配置缺失」，登录动作本身无权通过 → 403 而非 500
        return jsonify({"error": "服务端未配置管理员口令，请联系部署者在 .env 设置 KNOWFLOW_ADMIN_PASSWORD"}), 403
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
        SESSION_TASKS.pop(session_id, None)  # 清对话同时清掉任务记忆
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== 管理员：任务提示词 / 标准文档 在线编辑（复用 RBAC：_is_admin）=====
_ADMIN_TASKS = [
    ("revise_proposal", "修改策划案"),
    ("revise_art", "修改美术需求"),
    ("organize_ui", "整理 UI 需求"),
]

@app.route("/admin/prompts", methods=["GET"])
def admin_prompts_page():
    """提示词与标准文档编辑页（/admin/prompts）：编辑后并入请求前缀并命中 DeepSeek 缓存。"""
    if not _is_admin():
        return ("请先以管理员身份登录：POST /admin_login（口令来自 .env 的 "
                "KNOWFLOW_ADMIN_PASSWORD），再访问本页。"), 403
    sections = []
    for key, label in _ADMIN_TASKS:
        p = get_task_prompt(key) or ""
        d = get_standard_doc(key)
        sections.append(
            '<div class="task-card">'
            f'<h3>{label} <code>{key}</code></h3>'
            '<label>任务提示词（固定前缀，会被缓存）</label>'
            f'<textarea id="prompt_{key}" rows="6">{html.escape(p)}</textarea>'
            '<label>标准文档（并入前缀，会被缓存）</label>'
            f'<textarea id="doc_{key}" rows="10">{html.escape(d)}</textarea>'
            '</div>'
        )
    sections_html = "\n".join(sections)
    tasks_json = _json.dumps([k for k, _ in _ADMIN_TASKS])
    page = f'''<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>KnowFlow 管理页</title>
<style>
  body{{font-family:-apple-system,Segoe UI,Microsoft YaHei,sans-serif;margin:0;background:#f5f6fa;color:#222}}
  .wrap{{max-width:880px;margin:0 auto;padding:24px}}
  h1{{font-size:20px}} .task-card{{background:#fff;border:1px solid #e3e6ee;border-radius:10px;padding:16px;margin-bottom:16px}}
  label{{display:block;font-size:13px;color:#555;margin:10px 0 4px}}
  textarea{{width:100%;box-sizing:border-box;padding:8px;border:1px solid #ccd2e0;border-radius:6px;font-size:13px;line-height:1.5}}
  button{{margin-top:8px;background:#4f7cff;color:#fff;border:0;padding:10px 18px;border-radius:8px;cursor:pointer;font-size:14px}}
  .status{{margin-left:12px;font-size:13px;color:#2a8a4a}}
</style></head><body><div class="wrap">
  <h1>KnowFlow 管理页 · 任务提示词与标准文档</h1>
  <p style="font-size:13px;color:#666">改完点保存即可生效；这些内容会以「固定前缀」形式进入每次请求并命中 DeepSeek 前缀缓存。</p>
  {sections_html}
  <button onclick="saveAll()">保存全部</button><span class="status" id="status"></span>
  <script>
  async function saveAll(){{
    const tasks={tasks_json};
    const payload={{tasks:{{}}}};
    tasks.forEach(t=>{{
      payload.tasks[t]={{prompt:document.getElementById('prompt_'+t).value, standard_doc:document.getElementById('doc_'+t).value}};
    }});
    const r=await fetch('/admin/prompts/update',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}});
    const j=await r.json();
    document.getElementById('status').textContent = j.status==='ok'?'已保存 ✓':'保存失败：'+(j.error||'');
  }}
  </script>
</div></body></html>'''
    return page


@app.route("/admin/prompts/update", methods=["POST"])
def admin_prompts_update():
    """保存管理员编辑的任务提示词 / 标准文档，写回 config/*.json 并热更新。"""
    if not _is_admin():
        return jsonify({"error": "unauthorized"}), 403
    data = request.get_json(force=True, silent=True) or {}
    tasks = data.get("tasks", {})
    if not isinstance(tasks, dict):
        return jsonify({"error": "bad payload"}), 400
    new_prompts, new_docs = {}, {}
    for k, v in tasks.items():
        if not isinstance(v, dict):
            continue
        if "prompt" in v:
            new_prompts[k] = v["prompt"]
        if "standard_doc" in v:
            new_docs[k] = v["standard_doc"]
    # 用取值函数取当前值再合并，避免把上一次保存的内容覆盖掉
    if new_prompts:
        save_prompts({**get_task_prompts(), **new_prompts})
    if new_docs:
        save_standard_docs({**get_standard_docs(), **new_docs})
    return jsonify({"status": "ok"})


# ===== 标准文件管理：在文件管理面板内上传/删除各任务的标准文档 =====
# 复用同一套 DocumentProcessor 做文本抽取（PDF/TXT/MD），抽出全文存入标准文档配置。
_std_proc = DocumentProcessor()

def _extract_full_text(file_obj):
    """把上传文件落临时盘 → 用 DocumentProcessor 抽全文 → 删除临时盘，返回纯文本。"""
    import tempfile
    suffix = os.path.splitext(file_obj.filename)[1].lower()
    tmp = os.path.join(tempfile.gettempdir(), f"_std_{os.getpid()}_{_time.time()}{suffix}")
    file_obj.save(tmp)
    try:
        chunks = _std_proc.process_file(tmp)
        return "\n".join(c.get("content", "") for c in chunks)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

@app.route("/admin/standard_list", methods=["GET"])
def admin_standard_list():
    """返回各任务当前已载入的标准文档概况（供文件管理面板渲染）。"""
    if not _is_admin():
        return jsonify({"error": "unauthorized"}), 403
    out = {}
    for key, _ in _ADMIN_TASKS:
        d = get_standard_doc(key)
        out[key] = {"has_doc": bool(d), "length": len(d), "preview": d[:120]}
    return jsonify(out)

@app.route("/admin/standard_upload", methods=["POST"])
def admin_standard_upload():
    """上传某任务的标准文档文件（PDF/TXT/MD），抽取全文并入前缀缓存配置。"""
    if not _is_admin():
        return jsonify({"error": "unauthorized"}), 403
    task = (request.form.get("task") or "").strip()
    if task not in dict(_ADMIN_TASKS):
        return jsonify({"error": "bad task"}), 400
    if "file" not in request.files:
        return jsonify({"error": "没有文件"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400
    allowed = (".pdf", ".txt", ".md")
    if not file.filename.lower().endswith(allowed):
        return jsonify({"error": "仅支持 PDF / TXT / MD"}), 400
    try:
        text = _extract_full_text(file)
        if not text.strip():
            return jsonify({"error": "文件无可用文本（可能是扫描件/空文件）"}), 400
        save_standard_docs({**get_standard_docs(), task: text})
        return jsonify({"status": "ok", "length": len(text), "filename": file.filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/standard_delete", methods=["POST"])
def admin_standard_delete():
    """清空某任务的标准文档。"""
    if not _is_admin():
        return jsonify({"error": "unauthorized"}), 403
    data = request.get_json(force=True, silent=True) or {}
    task = (data.get("task") or "").strip()
    if task not in dict(_ADMIN_TASKS):
        return jsonify({"error": "bad task"}), 400
    cur = get_standard_docs()
    cur.pop(task, None)
    save_standard_docs(cur)
    return jsonify({"status": "ok"})



# ===== 启动 =====
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🚀 KnowFlow 智流 Web 服务启动中...")
    print("=" * 50)
    print("📌 浏览器访问: http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000, use_reloader=False)