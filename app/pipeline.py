import logging
import os
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager

from .chunker import chunk_segments
from .embedder import embed
from .generator import answer_stream
from .retriever import Hit, retrieve
from .store import init_collection, store
from .transcribe import transcribe

_logger = logging.getLogger(__name__)


# @contextmanager 把一个生成器函数变成上下文管理器（可以用 with 的对象），省去手写类 + __enter__ / __exit__
@contextmanager
def _timed(label: str) -> Generator[None, None, None]:
    t = time.perf_counter()
    yield
    # %s: 字符串，%-14s 表示左对齐，占 14 个字符宽度
    # %.1f: 浮点数，保留 1 位小数
    _logger.info("%-14s %.1fs", label, time.perf_counter() - t)


def ingest(audio_path: str, source: str | None = None) -> int:
    source = source or os.path.basename(audio_path)

    _logger.info("ingest  source=%s  pid=%d", source, os.getpid())

    with _timed("transcribe"):
        segments = transcribe(audio_path)

    with _timed("chunk"):
        chunks = chunk_segments(segments)

    if not chunks:
        _logger.warning("no chunks generated for %s", source)
        return 0
    print(f"ingest  source={source}  segments={len(segments)}  chunks={len(chunks)}")

    with _timed("embed"):
        vectors = embed([c.text for c in chunks])

    with _timed("store"):
        init_collection()
        store(chunks, vectors, source=source, replace=True)

    return len(chunks)


def ask(question: str, source: str | None = None) -> tuple[list[Hit], Iterator[str]]:
    hits = retrieve(question, source=source)
    return hits, answer_stream(question, hits)
