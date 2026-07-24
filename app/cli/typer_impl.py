import os

import typer

from .shared import env_flag, print_info, run_ask, run_ingest, setup_logging

app = typer.Typer(name="voxrag", help="voxrag 工具")

# -- 开头 → 可选参数（named argument），有名字，顺序任意，可以省略
# 没有 -- → 位置参数（positional argument），按顺序匹配，通常必填


# callback 上的参数是全局 flag，必须写在子命令之前：voxrag --verbose ask "..."
# 不用 typer 的 envvar=：它走 click 的 BOOL 解析，规则与 argparse 侧不一致
@app.callback()
def callback(
    verbose: bool = typer.Option(
        env_flag("VOXRAG_VERBOSE"),
        "--verbose",
        "-v",
        help="输出调试日志（也可用 VOXRAG_VERBOSE=1）",
    ),
    log_file: str | None = typer.Option(
        os.getenv("VOXRAG_LOG_FILE") or None,
        "--log-file",
        help="日志写入文件路径（也可用 VOXRAG_LOG_FILE）",
    ),
) -> None:
    setup_logging(verbose, log_file)


@app.command()
def ingest(
    audio: str = typer.Argument(help="音频/视频文件路径"),
    source: str | None = typer.Option(None, help="指定 source 名称（默认用文件名）"),
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
    app()


if __name__ == "__main__":
    main()
