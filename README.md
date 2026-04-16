# Agentic RAG with LangGraph + LightRAG

这是一个基于 LangGraph 搭建的 Agentic RAG 项目，当前已经按正式 LightRAG Core 的方式接入，支持两种能力：

- 上传单个 PDF，解析后写入 LightRAG
- 批量扫描 `data` 目录下的 PDF，导入 LightRAG 并抽取知识图谱

## 核心流程

1. 读取 PDF
2. 用 `pypdf` 提取逐页文本
3. 把文本送入官方 `LightRAG` Core
4. LightRAG 在入库过程中完成实体关系抽取、图谱构建和混合索引
5. LangGraph 根据问题自动路由到：
   - 直接模型问答
   - LightRAG 图谱检索问答

## 当前架构

- `FastAPI + LangGraph` 负责 API、路由和工作流编排
- `LightRAG Core` 负责文档入库、图谱抽取、混合检索和问答
- `LocalInferenceService` 负责为 LightRAG 注入 LLM 与 embedding 回调
- `PdfParser` 负责把 PDF 转成可入库文本

## 项目结构

```text
.
├─ .env
├─ .env.example
├─ data/
├─ src/
│  └─ agentic_rag/
│     ├─ app.py
│     ├─ config.py
│     ├─ main.py
│     ├─ schemas.py
│     ├─ graph/
│     │  └─ workflow.py
│     └─ services/
│        ├─ data_ingest.py
│        ├─ lightrag_service.py
│        ├─ llm.py
│        └─ pdf_parser.py
└─ tests/
```

## 环境变量

项目默认按 vLLM 的 OpenAI 兼容接口调用你已经启动好的服务端模型。

```env
MODEL_BACKEND=vllm
OPENAI_API_KEY=EMPTY
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
AGENT_MODEL=qwen2.5-7B
ROUTER_MODEL=qwen2.5-7B
AGENT_MODEL_PATH=/data/models/qwen2.5-7B
ROUTER_MODEL_PATH=/data/models/qwen2.5-7B
LIGHTRAG_LLM_MODEL=qwen2.5-7B
LIGHTRAG_LLM_MODEL_PATH=/data/models/qwen2.5-7B
LIGHTRAG_EMBED_MODEL=bge-m3
LIGHTRAG_EMBED_MODEL_PATH=/data/models/bge-m3
LIGHTRAG_EMBED_DIM=1024
LIGHTRAG_WORKING_DIR=./data/lightrag
LIGHTRAG_QUERY_MODE=mix
LIGHTRAG_RESPONSE_TYPE=Multiple Paragraphs
LIGHTRAG_TOP_K=40
LIGHTRAG_CHUNK_TOP_K=20
LIGHTRAG_INCLUDE_REFERENCES=true
LIGHTRAG_ENABLE_LLM_CACHE=true
LIGHTRAG_CHUNK_TOKEN_SIZE=1200
LIGHTRAG_CHUNK_OVERLAP_TOKEN_SIZE=100
LIGHTRAG_EMBEDDING_MAX_TOKENS=8192
LIGHTRAG_LLM_MAX_ASYNC=8
LIGHTRAG_EMBEDDING_MAX_ASYNC=8
LIGHTRAG_TIKTOKEN_MODEL_NAME=gpt-4o-mini
PDF_SOURCE_DIR=./data
```

注意：

- `PDF_SOURCE_DIR` 是批量导入 PDF 的目录
- `LIGHTRAG_WORKING_DIR` 是 LightRAG 存图谱和索引数据的位置
- `LIGHTRAG_QUERY_MODE` 推荐使用 `mix` 或 `hybrid`
- `LIGHTRAG_INCLUDE_REFERENCES=true` 时，回答会附带参考片段
- `OPENAI_BASE_URL` 指向 Qwen 的 vLLM OpenAI 兼容接口
- `EMBEDDING_BASE_URL` 指向 bge-m3 的 vLLM embedding 接口
- `*_MODEL_PATH` 只有在 `MODEL_BACKEND=local` 时才会被本地推理代码使用

## 安装

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

如果你的环境还没有 `torch`，请先按你的 CUDA/CPU 环境安装对应版本，再执行上面的命令。

## 运行 API

```powershell
python -m agentic_rag.main serve
```

默认地址：

- `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`

## 批量加载 `data` 目录 PDF 并抽取知识图谱

先把 PDF 放进项目根目录下的 `data` 文件夹，例如：

```text
data/
├─ report-a.pdf
├─ report-b.pdf
└─ contracts/
   └─ contract-01.pdf
```

然后执行：

```powershell
python -m agentic_rag.main ingest-data-pdfs
```

这条命令会：

- 扫描 `PDF_SOURCE_DIR` 下所有 `.pdf`
- 逐个解析文本
- 调用官方 LightRAG `ainsert(...)`
- 让 LightRAG 建立图谱、实体关系与混合检索索引

执行完成后会打印 JSON 结果，包含：

- 发现了多少个 PDF
- 成功索引了多少个 PDF
- 每个 PDF 的页数、字符数、估算 chunk 数

LightRAG 会把图谱、向量索引、KV 状态等数据写入 `LIGHTRAG_WORKING_DIR`。同一路径文档会使用稳定 `doc_id` 更新导入，而不是无限重复累积。

## API 方式导入

### 上传单个 PDF

```powershell
curl -X POST "http://127.0.0.1:8000/ingest/pdf" `
  -H "accept: application/json" `
  -H "Content-Type: multipart/form-data" `
  -F "file=@sample.pdf"
```

### 批量导入 `data` 目录

```powershell
curl -X POST "http://127.0.0.1:8000/ingest/data-pdfs"
```

## 问答

```powershell
curl -X POST "http://127.0.0.1:8000/chat" `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"这些PDF中有哪些核心实体及其关系？\",\"force_route\":\"lightrag\"}"
```

建议你在验证图谱效果时先把 `force_route` 设成 `lightrag`，这样能确保命中 LightRAG，而不是直接模型问答。

## 当前实现重点

- [src/agentic_rag/services/lightrag_service.py](src/agentic_rag/services/lightrag_service.py) 现在是正式的 LightRAG Core 适配层
- 首次使用时会执行 `initialize_storages()`，关闭服务时会执行 `finalize_storages()`
- 文档导入会按来源路径生成稳定 `doc_id`，重复导入时先删除旧文档再重新写入
- 查询走官方 `QueryParam`，支持 `mode/top_k/chunk_top_k/response_type/include_references`

## PDF 解析位置

PDF 解析逻辑在：

- [src/agentic_rag/services/pdf_parser.py](src/agentic_rag/services/pdf_parser.py)

批量扫描 `data` 目录逻辑在：

- [src/agentic_rag/services/data_ingest.py](src/agentic_rag/services/data_ingest.py)

API 入口在：

- [src/agentic_rag/app.py](src/agentic_rag/app.py)

命令行入口在：

- [src/agentic_rag/main.py](src/agentic_rag/main.py)
