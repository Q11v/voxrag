# voxrag

面向 Apple Silicon 的音频 RAG pipeline。使用 Whisper 转写音频，对 segment 做 chunk 与 embedding，存入 Qdrant，并通过 OpenRouter 以流式 LLM 生成回答。

## 架构

```
audio → transcribe (mlx-whisper) → chunk → embed (BGE-M3) → Qdrant
                                                                ↓
answer (stream) ← generate (LLM) ← rerank (BGE-reranker) ← retrieve
```

## 环境要求

- Apple Silicon Mac（mlx-whisper 依赖）
- 本地运行的 [Qdrant](https://qdrant.tech/)
- OpenRouter API key

## 安装

```bash
git clone https://github.com/yourname/voxrag
cd voxrag
uv sync
source .venv/bin/activate
```

## 配置

复制 `.env.example` 为 `.env`，填入所需的值：

```env
# 必填
OPENROUTER_API_KEY=sk-...

# 可选 —— 以下为默认值
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
QDRANT_HOST=localhost
QDRANT_PORT=6333

ASR_MODEL=mlx-community/whisper-turbo
EMBED_MODEL=BAAI/bge-m3
RERANK_MODEL=BAAI/bge-reranker-v2-m3
LLM_MODEL=z-ai/glm-4.5-air:free

COLLECTION_NAME=voxrag
VECTOR_SIZE=1024

CHUNK_MAX_SIZE=400
CHUNK_OVERLAP_SEGS=2
RETRIEVE_TOP_K=20
RERANK_TOP_K=3

# point id 的 UUID 命名空间，用于把 (source, chunk 序号) 映射为稳定的 id
ID_NAMESPACE=00000000-0000-0000-0000-000000000001

# Hugging Face
HF_HUB_OFFLINE=0        # 模型下载完成后设为 1 可跳过联网检查
HF_TOKEN=hf_xxx         # 仅访问受限模型时需要

# CLI 默认值（可选）
VOXRAG_VERBOSE=0        # 设为 1 则默认开启 debug 日志
VOXRAG_LOG_FILE=        # 设置路径则默认将日志写入文件
```

首次运行会下载模型权重到 `~/.cache/huggingface/`，BGE-M3 与 BGE-reranker-v2-m3 各约 2.3 GB。下载完成后可在 `.env` 中设置 `HF_HUB_OFFLINE=1` 跳过联网检查。

`VECTOR_SIZE` 必须与 `EMBED_MODEL` 的输出维度一致（BGE-M3 为 1024）。换 embedding 模型时需同步修改，并重建 collection。

## 使用

**导入音频文件：**

```bash
voxrag ingest './assets/lzy.wav'
voxrag ingest './assets/lzy.wav' --source meeting-1
```

`--source` 是检索时的过滤键，默认取文件名。重复导入同一个 source 会替换已有数据。

**提问：**

```bash
voxrag ask "我昨晚干啥了"
voxrag ask "我昨晚干啥了" --source meeting-2026-05
```

`--source` 可将检索范围限定在单个音频文件；省略则在全部已导入内容中检索。

**其他命令：**

```bash
voxrag info                              # 查看 collection 统计信息
voxrag --verbose ask "..."               # 开启 debug 日志（prompt、usage、score）
voxrag --log-file ./log/info.log ask "..." # 将日志写入文件
```

`--verbose` 和 `--log-file` 是全局 flag，必须放在子命令之前。两者都可通过 `.env` 中的 `VOXRAG_VERBOSE` 和 `VOXRAG_LOG_FILE` 长期生效。

## Pipeline 模块

| 模块         | 职责                               |
| ------------ | ---------------------------------- |
| `transcribe` | 通过 mlx-whisper 进行 Whisper 转写 |
| `chunker`    | 将 segment 合并为带重叠的文本 chunk |
| `embedder`   | 基于 BGE-M3 的稠密向量（懒加载）   |
| `store`      | Qdrant 按 source 的 upsert / delete |
| `retriever`  | ANN 检索 + CrossEncoder 重排       |
| `generator`  | 通过 OpenRouter 流式生成回答       |
| `pipeline`   | `ingest()` 与 `ask()` 的编排       |

## 详细文档

https://www.yuque.com/bangxw/it/aezhnphs2gzgdrrr?singleDoc# 《RAG 工程笔记：从语音到智能问答》
