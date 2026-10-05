# AI 求职助手

一个用于**面试展示**的 AI 应用项目：简历结构化解析 + 面经知识库问答（RAG）+ 调用成本监控。

技术栈：Python · FastAPI · Streamlit · Pydantic · BM25/向量混合检索 · SSE 流式输出 · Docker

> 设计目标不是"功能最多"，而是**每个模块都对应一个面试考点，并且都能报出数字**。
> 详见 [INTERVIEW.md](INTERVIEW.md)。

---

## 项目位置

```
项目代码   C:\Users\17839\Desktop\ai-job-assistant
Python     E:\dev\Python312        （已加入 PATH，直接敲 python 即可）
Git        D:\Git                  （已加入 PATH）
```

---

## 它解决什么问题

求职时你会发现两件事：简历投出去石沉大海不知道差在哪；面试前翻自己的笔记效率极低。
这个项目把两件事一起做掉：

1. **简历解析** —— 把 PDF/Word 简历变成结构化数据（技能、项目、可量化亮点）
2. **面经问答** —— 把自己的面经笔记做成知识库，提问即得答案，**每个答案都带引用来源**
3. **成本与效果监控** —— 记录每次调用的 token、成本、延迟分位数，并用评测集量化检索效果

---

## 架构

```
┌─────────────────┐   HTTP    ┌──────────────────────────────────────┐
│  Streamlit UI   │ ────────► │            FastAPI (8000)            │
│   (8501)        │  SSE 流式  │                                      │
└─────────────────┘           │  api/routes.py   仅做校验与转发        │
                              └───────────────┬──────────────────────┘
                                              │
                              ┌───────────────▼──────────────────────┐
                              │           services/ 业务层            │
                              │  resume.py   简历 → 结构化字段         │
                              │  rag.py      检索 → 提示词 → 生成      │
                              │  retriever.py BM25 + 向量 + 混合融合   │
                              │  documents.py 加载 / 切块             │
                              └───────────────┬──────────────────────┘
                                              │
                              ┌───────────────▼──────────────────────┐
                              │  llm.py      唯一直接调用模型的地方    │
                              │  （重试 / 超时 / 埋点 / 错误归一化）    │
                              │  config.py   .env 集中配置 + 价格表    │
                              │  metrics.py  成本与延迟统计            │
                              └──────────────────────────────────────┘
```

**三条值得在面试里讲的架构决定：**

1. **前端不 import 后端代码**，只通过 HTTP 通信 → 同一套 API 可以同时给网页、脚本、别人的程序用
2. **模型调用只在一个文件里** → 换供应商只改 `.env`；测试时可以只 mock 这一层
3. **成本、延迟在调用发生的同一处记录** → 不存在"忘了埋点"的可能

---

## 快速开始

完整说明见 **[SETUP.md](SETUP.md)**。

```powershell
cd C:\Users\17839\Desktop\ai-job-assistant

# 1) 离线测试（不需要 Key、不联网、不花钱）
python -m pytest -v

# 2) 配 API Key
copy .env.example .env
notepad .env          # 填 LLM_API_KEY

# 3) 启动（开两个终端）
python -m app.main                 # 终端 A：后端 → http://127.0.0.1:8000/docs
python -m streamlit run ui/app.py  # 终端 B：前端 → http://localhost:8501
```

首次使用：前端左侧点「重建知识库索引」，然后到问答页提问。

---

## 目录结构

```
ai-job-assistant/
├── app/
│   ├── config.py            # .env 集中配置 + DeepSeek 价格表 + 峰谷计价
│   ├── llm.py               # 模型网关：重试、流式、向量化、埋点
│   ├── metrics.py           # 线程安全的调用记录与分位数统计
│   ├── schemas.py           # Pydantic 数据结构（自动生成接口文档）
│   ├── main.py              # FastAPI 应用入口
│   ├── api/routes.py        # 路由（含 SSE 流式接口、路径穿越防护）
│   └── services/
│       ├── documents.py     # 文档加载（md/txt/pdf/docx）+ 三层切块
│       ├── retriever.py     # BM25 + 向量 + 混合融合 + 持久化
│       ├── rag.py           # 检索增强问答（带引用）
│       └── resume.py        # 简历结构化（校验失败自动修复重试）
├── ui/app.py                # Streamlit 前端（三个标签页）
├── scripts/
│   ├── eval_rag.py          # 检索评测：产出 Hit@k / MRR（简历数字的来源）
│   ├── setup.ps1            # 一键初始化（本机未使用 venv，见 SETUP.md）
│   ├── run_api.ps1
│   └── run_ui.ps1
├── data/
│   ├── knowledge/           # 知识库语料（放你自己的面经笔记）
│   ├── resumes/             # 示例简历
│   └── eval/                # 评测集（问题 + 正确来源 + 必含关键词）
├── tests/test_core.py       # 15 个离线单元测试
├── requirements.txt
├── Dockerfile
└── SETUP.md / INTERVIEW.md
```

---

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查（Key 是否配置、索引块数） |
| GET | `/debug/connection` | 当前模型连接信息（**Key 已打码**） |
| GET | `/metrics` | 调用次数、成本、P50/P95 延迟、缓存命中率 |
| POST | `/metrics/reset` | 清空埋点 |
| POST | `/kb/index` | 建立/追加索引（限制在 data 目录内） |
| GET | `/kb/status` | 索引状态、来源文件、降级告警 |
| POST | `/kb/query` | 知识库问答（非流式） |
| POST | `/kb/stream` | 知识库问答（**SSE 流式**，先推引用再推正文） |
| POST | `/resume/parse` | 解析简历文本 |
| POST | `/resume/upload` | 上传简历文件（md/txt/pdf/docx）并解析 |

> 注：新版 FastAPI（0.142）把 `include_router` 的子路由包成了 `_IncludedRouter`，
> 不再拍平进 `app.routes`。**这不影响接口可用性**，但如果你写代码去数 `app.routes`
> 会数不到 —— 用 `TestClient` 实际调用才是可靠的验证方式。

---

## 现在的完成度（诚实说明）

**已实现并通过验证：**

- 文档加载与三层切块（标题面包屑 + 段落累积 + 硬切重叠）
- BM25 检索（纯 Python 实现 + jieba 中文分词，无 jieba 时降级字符二元组）
- 可插拔向量检索与混合融合（min-max 归一化后加权）
- 带引用的 RAG 问答 + SSE 流式输出
- 简历结构化抽取，含 **Pydantic 校验失败后的修复重试**
- 调用埋点：成本（区分缓存命中/未命中、区分峰谷时段）、延迟分位数
- 检索评测脚本，可扫描不同混合权重并输出对比报告
- 路径穿越防护、单文件解析失败隔离、向量失败的可见降级
- **15 个离线单元测试全部通过**；API 冒烟测试（TestClient）全通过

**尚未实现（后续阶段）：**

- 简历与 JD 的匹配打分（embedding 粗排 + LLM 精排）
- 模拟面试 Agent（function calling + 多轮状态机）
- rerank 精排模块
- 向量数据库接入（当前是进程内索引 + JSON 持久化）

---

## 为什么默认不开向量检索

**因为 DeepSeek 官方 API 不提供 embeddings 接口。** 硬要用就得额外接一家服务商或下载本地模型。
所以默认走 BM25：零额外依赖、立刻能跑通、也更容易看清楚检索的本质。

要开向量，改 `.env`：

```ini
EMBEDDING_BACKEND=api
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_API_KEY=你的key
EMBEDDING_MODEL=BAAI/bge-m3
HYBRID_ALPHA=0.5
```

然后跑 `python -m scripts.eval_rag --alpha 0 0.3 0.5 0.7 1` 对比效果——
**这个对比实验本身就是面试素材**。

---

## 下一步

1. 配 `.env` 的 API Key（否则问答和简历解析返回 502）
2. 把 `data/knowledge/` 换成你自己的面经笔记
3. **重做一份有区分度的评测集** —— 现在的题目太简单（见 SETUP.md 第四节）
4. 看 [INTERVIEW.md](INTERVIEW.md)：简历怎么写、项目怎么讲、追问怎么答
