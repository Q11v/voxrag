import sys

import typer

from .shared import print_info, run_ask, run_ingest, setup_logging

app = typer.Typer(name="voxrag", help="voxrag 工具")


@app.callback()
def callback(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", envvar="VOXRAG_VERBOSE", help="输出调试日志"
    ),
    log_file: str | None = typer.Option(
        None, "--log-file", envvar="VOXRAG_LOG_FILE", help="日志写入文件路径"
    ),
) -> None:
    setup_logging(verbose, log_file)


@app.command()
def ingest(
    audio: str = typer.Argument(help="音频/视频文件路径"),
    source: str | None = typer.Option(None, help="指定 source 名称（默认文件名）"),
) -> None:
    """转写音频并入库"""
    run_ingest(audio, source)


@app.command()
def ask(
    question: str = typer.Argument(help="问题"),
    source: str | None = typer.Option(None, help="只在指定 source 内检索"),
) -> None:
    """提问"""
    run_ask(question, source)


@app.command()
def info() -> None:
    """查看集合信息"""
    print_info()


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        print("\n 已中断", file=sys.stderr)
        sys.exit(130)
