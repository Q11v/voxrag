# voxrag

Audio RAG pipeline for Apple Silicon. Transcribes audio with Whisper, chunks and embeds the segments, stores them in Qdrant, and answers questions with streaming LLM generation via OpenRouter.

## Architecture

```
audio → transcribe (mlx-whisper) → chunk → embed (BGE-M3) → Qdrant
                                                                ↓
answer (stream) ← generate (LLM) ← rerank (BGE-reranker) ← retrieve
```

