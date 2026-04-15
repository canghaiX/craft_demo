<<<<<<< HEAD
# Agentic RAG with LangGraph + LightRAG

这是一个从零搭好的 Agentic RAG 项目骨架，目标是满足这条链路：

1. 上传 PDF
2. 解析文本
3. 写入 LightRAG
4. 由 LightRAG 抽取知识图谱并建立检索索引
5. 基于 LangGraph 让 Agent 自动路由：
   - 直接走模型自由问答
   - 或走 LightRAG 做基于文档知识图谱的回答

## 项目结构

```text
.
├─ pyproject.toml
├─ .env.example
├─ src/
│  └─ agentic_rag/
│     ├─ app.py
│     ├─ config.py
│     ├─ main.py
│     ├─ schemas.py
│     ├─ graph/
│     │  └─ workflow.py
│     └─ services/
│        ├─ lightrag_service.py
│        ├─ llm.py
│        └─ pdf_parser.py
└─ tests/
   └─ test_health.py
```

## 技术选型

- `LangGraph`
  - 负责任务编排和 Agent 路由
- `LightRAG`
  - 负责文档索引、实体关系抽取、知识图谱检索
- `FastAPI`
  - 提供上传 PDF 与问答 API
- `PyPDF`
  - 负责 PDF 文本解析
- `OpenAI / OpenAI-compatible API`
  - 负责路由判定、直接问答，以及供 LightRAG 调用 LLM/Embedding

## 先决条件

- Python `3.11+`
- 可用的 OpenAI API Key，或兼容 OpenAI 接口的模型服务

## 安装

### 1. 创建虚拟环境

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. 安装依赖

```powershell
pip install -e .[dev]
```

### 3. 配置环境变量

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`：

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=
AGENT_MODEL=gpt-4o-mini
ROUTER_MODEL=gpt-4o-mini
LIGHTRAG_LLM_MODEL=gpt-4o-mini
LIGHTRAG_EMBED_MODEL=text-embedding-3-large
LIGHTRAG_EMBED_DIM=3072
LIGHTRAG_WORKING_DIR=./data/lightrag
LIGHTRAG_QUERY_MODE=hybrid
```

如果你用的是兼容 OpenAI 的国内模型网关，也可以只改：

```env
OPENAI_BASE_URL=https://your-compatible-endpoint/v1
```

## 启动服务

```powershell
python -m agentic_rag.main
```

服务默认启动在 `http://127.0.0.1:8000`

Swagger 文档：

- `http://127.0.0.1:8000/docs`

## API 用法

### 1. 上传 PDF 并写入 LightRAG

```bash
curl -X POST "http://127.0.0.1:8000/ingest/pdf" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@sample.pdf"
```

返回示例：

```json
{
  "file_name": "tmpabc.pdf",
  "pages": 12,
  "chunks_indexed": 18,
  "characters": 21654,
  "storage_dir": "data/lightrag"
}
```

### 2. 发起问答

```bash
curl -X POST "http://127.0.0.1:8000/chat" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"这份PDF里提到的核心系统模块有哪些？\",\"force_route\":\"auto\"}"
```

返回示例：

```json
{
  "answer": "...",
  "route": "lightrag",
  "reason": "Question explicitly refers to document corpus."
}
```

## 路由逻辑

当前 LangGraph 工作流很清晰：

- `router`
  - 用 LLM 判断问题该走 `direct` 还是 `lightrag`
  - 如果请求里显式指定 `force_route`，则优先按用户指定执行
- `direct_answer`
  - 直接走聊天模型回答
- `lightrag_answer`
  - 走 LightRAG `aquery(...)`

适合后续继续扩展成更完整的 Agent：

- 增加 Web Search 节点
- 增加 SQL / Tool 调用节点
- 增加 Citation 后处理节点
- 增加 会话记忆 与 多 Agent 协作

## LightRAG 说明

项目里已经对 LightRAG 做了两点兼容处理：

1. 初始化时执行 `initialize_storages()`
2. 如果当前版本存在 `initialize_pipeline_status()`，则一并调用

这是因为 LightRAG 的部分版本在文档入库前如果没有初始化 pipeline 状态，可能触发 `history_messages` 相关错误。

## 建议的下一步增强

- 接入 `Neo4jStorage`
  - 便于图谱可视化与生产级图数据库管理
- 加入文档切分策略
  - 例如按标题、章节、页码做更细粒度 chunking
- 增加引用与出处返回
  - 让回答带来源页码/文件名
- 支持批量目录导入 PDF
- 增加前端上传与问答页面

## 参考

- LangGraph 官方文档: https://langchain-ai.github.io/langgraph/
- LightRAG 官方仓库: https://github.com/HKUDS/LightRAG

## 已知说明

当前工作区里没有可直接执行的 Python 运行时，因此这次我完成的是：

- 项目结构搭建
- 核心代码实现
- 环境变量与说明文档补齐

你在本机装好 Python 后，就可以按上面的步骤直接安装运行。
=======
# craft_demo
基于LangGraph搭建的MultiAgent+Agentic RAG项目
>>>>>>> f2d9fbbb28c630a02c4463182233461ba24fe199
