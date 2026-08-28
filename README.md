# KnowFlow（智流）

基于 RAG 的企业知识库问答系统。具备多 Agent 编排、回答评分、Token 优化、RAGAS 评估，
以及策划案 / 需求案「对标精修」质量体检能力。

## 功能特性
- 两阶段混合检索（向量 + BM25）+ 低置信拒答闸门
- 多 Agent 编排：Router / Retriever / Compressor / Generator
- 回答评分（余弦相似度版）与 Token 节省实测
- RAGAS 评估（embedding 余弦版，零额外 token 消耗）
- 管理员权限加固：口令哈希（PBKDF2）+ 会话 TTL + 登录防暴破
- 策划案 / 需求案「对标精修」：上传标准案作参照，逐节指出"写得不够细"并给修改意见

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`、`KNOWFLOW_ADMIN_PASSWORD`、`FLASK_SECRET_KEY`
3. 启动：`python "RAG MVP/Web Frame.py"`
4. 浏览器访问 http://127.0.0.1:5000

## 目录结构
- `RAG MVP/`：核心源码（RAG 管线、检索、压缩、Agent、评估、Web 入口）
- `data/`：上传文档与生成的分块 / 向量（不入库，运行时生成）
- `output/`：运行产物（不入库）

> 注意：`data/`、`output/`、`.env` 均不纳入版本控制，请本地自行准备。
