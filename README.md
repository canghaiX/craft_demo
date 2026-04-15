# Agentic RAG with LangGraph + LightRAG

这是一个基于 LangGraph 搭建的 Agentic RAG 项目，支持两种能力：

- 上传单个 PDF，解析后写入 LightRAG
- 批量扫描 `data` 目录下的 PDF，导入 LightRAG 并抽取知识图谱

## 核心流程

1. 读取 PDF
2. 用 `pypdf` 提取逐页文本
3. 把文本送入 LightRAG
4. LightRAG 在入库过程中完成实体关系抽取和图谱索引构建
5. LangGraph 根据问题自动路由到：
   - 直接模型问答
   - LightRAG 图谱检索问答

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

项目当前按本地模型推理运行，不再依赖 OpenAI API。

```env
MODEL_BACKEND=local
OPENAI_API_KEY=
OPENAI_BASE_URL=
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
LIGHTRAG_QUERY_MODE=hybrid
PDF_SOURCE_DIR=./data
```

注意：

- `PDF_SOURCE_DIR` 是批量导入 PDF 的目录
- `LIGHTRAG_WORKING_DIR` 是 LightRAG 存图谱和索引数据的位置
- `*_MODEL_PATH` 会被本地推理代码实际使用
- 请确保本机环境已安装与硬件匹配的 `torch`

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
- 调用 LightRAG `ainsert(...)`
- 让 LightRAG 建立图谱与索引

执行完成后会打印 JSON 结果，包含：

- 发现了多少个 PDF
- 成功索引了多少个 PDF
- 每个 PDF 的页数、字符数、估算 chunk 数

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

## PDF 解析位置

PDF 解析逻辑在：

- [src/agentic_rag/services/pdf_parser.py](src/agentic_rag/services/pdf_parser.py)

批量扫描 `data` 目录逻辑在：

- [src/agentic_rag/services/data_ingest.py](src/agentic_rag/services/data_ingest.py)

API 入口在：

- [src/agentic_rag/app.py](src/agentic_rag/app.py)

命令行入口在：

- [src/agentic_rag/main.py](src/agentic_rag/main.py)
