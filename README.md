# RAGTrainingProgram

学习 Agent 的第一步：RAG。用 **Go + [Eino](https://github.com/cloudwego/eino)** 搭一个最小 RAG，逐步做检索对比实验与评测。

## 阶段计划

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 0. 骨架 | Go 工程 + Eino 依赖、`.env` 配置、17 篇中文语料 | 完成 |
| 1. 最小 RAG | 加载 → 切块 → 向量化 → 检索 → 生成（自写内存检索器，实现 Eino `Retriever` 接口） | 完成 |
| 2. 检索对比 | 切块策略（定长/按标题/重叠）× 检索（纯向量 / 手写 BM25 / RRF 混合） | 进行中 |
| 3. 评测 | 测试集 + recall@k / MRR / LLM 打分 | 待开始 |
| 4. 进阶 | Eino 的 MultiQuery / Router、Parent 策略；把 RAG 接入 ReAct Agent | 待开始 |

## 快速开始

**1. 开通火山方舟 Ark**（约 5 分钟）

1. 注册[火山引擎](https://www.volcengine.com/)账号并完成实名认证
2. 进入[方舟控制台](https://console.volcengine.com/ark)，开通要用的豆包 chat 模型与 embedding 模型
3. 在「API Key 管理」中创建 API Key

> 控制台菜单名称以实际为准；模型 ID 或推理接入点 ID（ep-…）直接复制控制台里的填法。

**2. 配置**

```bat
copy .env.example .env
:: 编辑 .env，填入 ARK_API_KEY、ARK_CHAT_MODEL、ARK_EMBEDDING_MODEL
```

**3. 运行自检**

```bat
go run ./cmd/smoke
```

**4. 建索引**（一次性：切块 → 向量化，约 1 分钟）

```bat
go run ./cmd/index
```

**5. 提问**

```bat
go run ./cmd/ask -q "什么是 RAG？"
```

**6. 检索对比**（纯向量 / BM25 / RRF 三路并排看排名与耗时；除向量路一次 API 调用外全本地）

```bat
go run ./cmd/compare -q "什么是 RAG？" [-k 5] [-n 20]
```

## 目录结构

```
cmd/ask/          在线问答：检索 top-k + 基于资料生成（-q 问题、-k 条数）
cmd/compare/      检索对比：纯向量 / BM25 / RRF 混合三路并排看排名与耗时
cmd/index/        离线建索引：语料 → 切块 → 向量化 → .data/vectors.gob
cmd/chunk/        切块观察：块数、长度分布、重叠效果
cmd/sim/          相似度自检：三句话验证 Cosine 方向是否正确
cmd/smoke/        API 冒烟测试：embedding + chat 连通性
corpus/           语料：17 篇中文公开文档，来源与许可见 corpus/SOURCES.md
internal/config/  环境配置加载（.env / 环境变量）
internal/llm/     Ark chat / embedding 适配器封装
internal/rag/     RAG 核心：加载、切块、索引、内存检索器、提示词组装、问答链
.data/            本地索引缓存（vectors.gob，已被 gitignore）
```
