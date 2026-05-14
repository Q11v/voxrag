import logging
import os
import sys


def setup_logging(
    verbose: bool,
) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    # 配置日志文件路径，默认为 main.log
    handlers.append(logging.FileHandler(os.getenv("VOXRAG_LOG_FILE", "./log/main.log")))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    logging.getLogger(__name__).info("$ %s", " ".join(sys.argv))
