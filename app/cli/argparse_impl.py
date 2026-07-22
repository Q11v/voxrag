import argparse
import os
import sys

from .shared import print_info, run_ask, run_ingest, setup_logging


def cmd_ingest(args: argparse.Namespace) -> None:
    run_ingest(args.audio, args.source)


def cmd_ask(args: argparse.Namespace) -> None:
    run_ask(args.question, args.source)


def cmd_info(_: argparse.Namespace) -> None:
    print_info()


def main() -> None:
    parser = argparse.ArgumentParser(description="voxrag 工具")

    # -- 开头 → 可选参数（named argument），有名字，顺序任意，可以省略
    # 没有 -- → 位置参数（positional argument），按顺序匹配，通常必填

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",  # action="store_true" 表示这是一个开关型参数（flag），不需要跟值
        default=os.getenv("VOXRAG_VERBOSE") == "1",
        help="输出调试日志",
    )
    parser.add_argument(
        "--log-file", default=os.getenv("VOXRAG_LOG_FILE"), help="日志写入文件路径"
    )

    # add_subparsers 创建子命令系统
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 注册第一个子命令：ingest
    # source 是存储在向量数据库中每条记录的元数据标签，用来标识这段内容来自哪个文件。ask 时, 会用 source 作为过滤条件，只在指定来源的 chunks 里做语义检索，而不是全库搜索。
    p_ingest = sub.add_parser("ingest", help="转写音频并入库")
    p_ingest.add_argument("audio", help="音频/视频文件路径")
    p_ingest.add_argument("--source", help="指定 source 名称（默认用文件名）")
    p_ingest.set_defaults(func=cmd_ingest)

    # 注册第二个子命令：ask
    p_ask = sub.add_parser("ask", help="提问")
    p_ask.add_argument("question", help="问题")
    p_ask.add_argument("--source", help="只在指定 source 内检索")
    p_ask.set_defaults(func=cmd_ask)

    # 注册第三个子命令：info
    p_info = sub.add_parser("info", help="查看集合信息")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    setup_logging(args.verbose, args.log_file)
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        sys.exit(130)
