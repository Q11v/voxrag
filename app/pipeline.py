import logging
import os
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager

from .generator import answer_stream
from .retriever import Hit, retrieve
from .transcribe import transcribe

_logger = logging.getLogger(__name__)


# @contextmanager 把一个生成器函数变成上下文管理器（可以用 with 的对象），省去手写类 + __enter__ / __exit__
@contextmanager
def _timed(label: str) -> Generator[None, None, None]:
    t = time.perf_counter()
    yield
    dt = time.perf_counter() - t
    # %-14s 左对齐占 14 字符
    _logger.info(
        "%-14s %5.1fs",
        label,
        dt,
    )


def ingest(audio_path: str, source: str | None = None) -> int:
    source = source or os.path.basename(audio_path)

    _logger.info("ingest  source=%s  pid=%d", source, os.getpid())

    with _timed("transcribe"):
        segments = transcribe(audio_path)

    print([s["text"] for s in segments])

    # with _timed("chunk"):
    #     chunks = chunk_segments(segments)

    # if not chunks:
    #     _logger.warning("no chunks generated for %s", source)
    #     return 0
    # print(f"ingest  source={source}  segments={len(segments)}  chunks={len(chunks)}")
    print(f"ingest  source={source}  segments={len(segments)}")

    # with _timed("embed"):
    #     vectors = embed([c.text for c in chunks])

    # with _timed("store"):
    #     init_collection()
    #     store(chunks, vectors, source=source, replace=True)

    # return len(chunks)

    return 1


def ask(question: str, source: str | None = None) -> tuple[list[Hit], Iterator[str]]:
    hits = retrieve(question, source=source)
    return hits, answer_stream(question, hits)
