import logging
import os
import sys
from pathlib import Path

from ..pipeline import ask as pipeline_ask
from ..pipeline import ingest as pipeline_ingest
from ..store import collection_info, init_collection


def env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def setup_logging(verbose: bool, log_file: str | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        # log/ 被 gitignore，新克隆的仓库里不存在，而 FileHandler 不会自动建目录
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    logging.getLogger(__name__).info("$ %s", " ".join(sys.argv))


def print_info() -> None:
    init_collection()
    info = collection_info()
    print(f"向量数:{info.points_count} 状态:{info.status}")


def run_ingest(audio: str, source: str | None) -> None:
    n = pipeline_ingest(audio, source=source)
    print(f"已写入 {n} 个 chunk")
    print_info()


def run_ask(question: str, source: str | None) -> None:
    hits, stream = pipeline_ask(question, source=source)

    print(f"\n{'=' * 60}")
    print(f"retrieve - rerank {len(hits)} 个片段：")
    for i, h in enumerate(hits):
        print(
            f"  [{i + 1}] score={h.score:.4f}  ({h.start:.1f}-{h.end:.1f}s)  {h.text[:60]}..."
        )
    print(f"{'=' * 60}\n")

    for chunk in stream:
        print(chunk, end="", flush=True)
    print()
