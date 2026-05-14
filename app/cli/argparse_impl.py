import argparse
import os
import sys

from .shared import setup_logging


def cmd_ingest(args: argparse.Namespace) -> None:
    pass


def cmd_ask(args: argparse.Namespace) -> None:
    pass


def cmd_info(_: argparse.Namespace) -> None:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="voxrag 工具")

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",  # action="store_true" 表示这是一个开关型参数（flag），不需要跟值
        default=os.getenv("VOXRAG_VERBOSE") == "1",
        help="输出调试日志",
    )

    # add_subparsers 创建子命令系统
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 注册第一个子命令：ingest
    p_ingest = sub.add_parser("ingest", help="转写音频并入库")
    p_ingest.set_defaults(func=cmd_ingest)

    # 注册第二个子命令：ask
    p_ask = sub.add_parser("ask", help="提问")
    p_ask.set_defaults(func=cmd_ask)

    # 注册第三个子命令：info
    p_info = sub.add_parser("info", help="查看集合信息")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        sys.exit(130)
