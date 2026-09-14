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

## 联网访问（内网穿透，临时演示用）

本服务默认只监听 `127.0.0.1:5000`（仅本机浏览器可访问）。要让他人（如面试官）从任意网络打开，
用内网穿透工具将本地 5000 端口映射为一个公网 HTTPS 链接即可，**无需购买服务器、无需改代码**。

### 前置安全（必须）
1. **口令哈希迁移**：公网暴露前务必把 `.env` 里的明文管理员口令替换为哈希串，否则密码明文传输。
   运行 `python "RAG MVP/gen_password_hash.py"`，按提示输入口令，把输出的
   `KNOWFLOW_ADMIN_PASSWORD=pbkdf2$...` 整串替换进 `.env`。
2. **只用 HTTPS 链接**：穿透工具默认提供 HTTPS，把 HTTPS 链接发给对方，不要发 HTTP。

### 方式一：cpolar（国产，中文界面，推荐）
1. 访问 https://www.cpolar.com 下载 Windows 客户端并安装。
2. 先本地启动服务：`python "RAG MVP/Web Frame.py"`（保持运行）。
3. 打开 cpolar 客户端（或命令行 `cpolar http 5000`），它会生成一个形如
   `https://xxxx.cpolar.cn` 的公网地址。
4. 把该 HTTPS 链接发给对方，对方即可在浏览器登录使用。

### 方式二：ngrok
1. 访问 https://ngrok.com 注册并下载，配置 authtoken。
2. 本地启动服务后，命令行执行 `ngrok http 5000`，获得公网 HTTPS 链接。

### 注意
- 免费版链接会不定期变化 / 限时；演示前重新生成一次即可。
- 关闭本地服务或关机后，公网链接立即失效。
- 长期在线请走云服务器部署（waitress/gunicorn + nginx + HTTPS 证书）。

