import logging
import sys

from ..pipeline import ask as pipeline_ask
from ..pipeline import ingest as pipeline_ingest
from ..store import collection_info, init_collection


def setup_logging(
    verbose: bool,
    log_file: str | None,
) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    # 配置日志文件路径，默认为 main.log
    # handlers.append(logging.FileHandler(os.getenv("VOXRAG_LOG_FILE", "./log/main.log")))
    if log_file:
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
