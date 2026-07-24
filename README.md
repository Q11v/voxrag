# voxrag

面向 Apple Silicon 的音频 RAG pipeline：把一段音频变成可以提问的知识库。

转写、embedding、重排全部在本机跑（MLX / MPS 加速），只有最后的答案生成走 OpenRouter。模型权重下载完成后，除生成外的环节都可离线运行。

## 架构

```
audio → transcribe (mlx-whisper) → chunk → embed (BGE-M3) → Qdrant
                                                                ↓
answer (stream) ← generate (LLM) ← rerank (BGE-reranker) ← retrieve
```

## 环境要求

| 依赖             | 说明                                                       |
| ---------------- | ---------------------------------------------------------- |
| Apple Silicon Mac | mlx-whisper 只支持 Apple 芯片                              |
| Python ≥ 3.11    | 开发使用 3.13                                              |
| ffmpeg           | mlx-whisper 通过 ffmpeg CLI 解码音频，未安装会在转写时报错 |
| Qdrant           | 本地运行即可                                               |
| OpenRouter API key | 仅生成环节需要                                           |

```bash
brew install ffmpeg
docker run -d -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

## 安装

```bash
git clone https://github.com/Q11v/voxrag
cd voxrag
uv sync
```

之后既可以 `source .venv/bin/activate` 再用 `voxrag`，也可以不激活直接 `uv run voxrag ...`。

## 配置

复制 `.env.example` 为 `.env`，只有一个必填项：

```env
OPENROUTER_API_KEY=sk-or-...
```

其余配置项都有默认值，完整列表见 [.env.example](.env.example)。其中几项需要额外说明：

| 变量                 | 默认值                                 | 说明                                                                 |
| -------------------- | -------------------------------------- | -------------------------------------------------------------------- |
| `VECTOR_SIZE`        | `1024`                                 | 必须与 `EMBED_MODEL` 的输出维度一致（BGE-M3 为 1024）。换模型要同步改，并重建 collection |
| `ID_NAMESPACE`       | `00000000-...-0001`                    | UUID 命名空间，把 `(source, chunk 序号)` 映射为稳定的 point id，使重复导入是幂等替换而非追加 |
| `CHUNK_MAX_SIZE`     | `400`                                  | chunk 的字符数上限。1 token ≈ 1.5~2 个中文字符                       |
| `CHUNK_OVERLAP_SEGS` | `2`                                    | 相邻 chunk 重叠的 segment 数，以 segment 而非字符为单位              |
| `RETRIEVE_TOP_K`     | `20`                                   | 向量检索的召回数量，作为重排的候选集                                 |
| `RERANK_TOP_K`       | `3`                                    | 重排后送入 LLM 的片段数                                              |
| `HF_HUB_OFFLINE`     | `0`                                    | 模型下载完成后设为 `1`，可跳过每次启动的联网检查                     |
| `VOXRAG_VERBOSE`     | `0`                                    | 设为 `1` 则默认开启 debug 日志，等价于常驻 `--verbose`               |
| `VOXRAG_LOG_FILE`    | `./log/main.log`                       | 日志文件路径，留空则只输出到终端                                     |

首次运行会把模型权重下载到 `~/.cache/huggingface/`，BGE-M3 与 BGE-reranker-v2-m3 各约 2.3 GB。

## 使用

### 导入音频

```bash
voxrag ingest './assets/lzy.wav'
voxrag ingest './assets/lzy.wav' --source meeting-2026-05
```

`--source` 是检索时的过滤键，默认取文件名。重复导入同一个 source 会**替换**已有数据，不会累积重复 chunk。

```
$ voxrag ingest './assets/lzy.wav'
ingest  source=lzy.wav  segments=412  chunks=18
[INFO]: transcribe     52.4s
[INFO]: chunk           0.0s
[INFO]: embed           6.1s
[INFO]: store           0.3s
已写入 18 个 chunk
向量数:18 状态:green
```

### 提问

```bash
voxrag ask "我昨晚干啥了"
voxrag ask "我昨晚干啥了" --source meeting-2026-05
```

`--source` 可将检索范围限定在单个音频；省略则在全部已导入内容中检索。

回答前会先打印重排后的召回片段（分数、时间区间、正文摘要），然后流式输出答案：

```
$ voxrag ask "我昨晚干啥了"

============================================================
retrieve - rerank 3 个片段：
  [1] score=0.9971  (0.0-158.0s)  昨天晚上我喝大了 现在还有点晕 这个喝酒啊 跟任何事情一样 分三种情况 一种呢 能力强 胆子...
  [2] score=0.0021  (723.0-885.0s)  他就到王婆的茶餐厅去找 被王婆裹了一掌 接着呢 里也滚了一地 他就把事情的真相 告诉了武大郎 导致武大郎被害 武松回来的...
  [3] score=0.0004  (880.0-1019.0s)  我说 徐华说得好 作贼要脏 桌奸要霜 只要没捉住 咱们就宁信其无 不信其有 我说爱国 你仔细想一想 如果你把这么漂亮的...
============================================================

你昨晚喝大了，和十几个朋友一起吃饭喝酒，现在还有点晕（片段 1, 00:00）。
```

### 回答行为

生成环节的 prompt 对模型做了硬约束：

- 只允许基于召回片段作答，不得引入片段外的信息；
- 片段中信息不足时，固定回复「根据现有会议记录，无法回答这个问题。」而不是猜测；
- 引用具体内容时需标注片段编号和时间戳，形如「（片段 2, 02:15）」；
- 片段之间不连贯或互相冲突时，要显式指出。

### 其他命令

```bash
voxrag info                                 # 查看 collection 统计信息
voxrag --verbose ask "..."                  # 开启 debug 日志（完整 prompt、token usage、召回分数）
voxrag --log-file ./log/info.log ask "..."  # 同时把日志写入文件
```

`--verbose` 和 `--log-file` 是全局 flag，**必须放在子命令之前**。两者都可以通过 `.env` 里的 `VOXRAG_VERBOSE` / `VOXRAG_LOG_FILE` 长期生效。

### 重建 collection

改了 `EMBED_MODEL` 或 `VECTOR_SIZE` 后，旧向量的维度与新模型不匹配，需要删掉 collection 再重新导入：

```bash
curl -X DELETE http://localhost:6333/collections/voxrag
voxrag ingest './assets/lzy.wav'
```

## Pipeline 模块

| 模块         | 职责                                             |
| ------------ | ------------------------------------------------ |
| `config`     | 从 `.env` 加载配置，冻结为不可变的 `Settings`    |
| `transcribe` | 通过 mlx-whisper 转写，产出带时间戳的 segment    |
| `chunker`    | 按字符预算合并 segment，在句子边界切分并保留重叠 |
| `embedder`   | 基于 BGE-M3 的归一化稠密向量（模型懒加载）       |
| `store`      | Qdrant collection 初始化、按 source 的 upsert / delete |
| `retriever`  | 向量召回 + source 过滤，再交给 reranker 精排     |
| `reranker`   | BGE-reranker-v2-m3 CrossEncoder 重排（模型懒加载） |
| `generator`  | 拼 prompt 并通过 OpenRouter 流式生成回答         |
| `pipeline`   | `ingest()` 与 `ask()` 的编排，附带各阶段耗时统计 |
| `cli`        | argparse 子命令、日志初始化与终端输出            |

## 详细文档

[《RAG 工程笔记：从语音到智能问答》](https://www.yuque.com/bangxw/it/aezhnphs2gzgdrrr?singleDoc#)
