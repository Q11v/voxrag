from dataclasses import dataclass

from .config import settings
from .transcribe import Segment

SENTENCE_END = "。！？!?"


@dataclass
class Chunk:
    text: str
    start: float
    end: float
    seg_ids: list[int]


def _is_sentence_boundary(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in SENTENCE_END


# [
#     {
#         "id": 0,
#         "seek": 0,
#         "start": 0.0,
#         "end": 1.36,
#         "text": "昨天晚上我喝大了",
#         "tokens": [50365, 47404, 6135, 50157, 1654, 26784, 3582, 2289, 50433],
#         "temperature": 0.0,
#         "avg_logprob": -0.24068851005740283,
#         "compression_ratio": 1.1958041958041958,
#         "no_speech_prob": 1.4620414254262748e-11,
#     },
#     {
#         "id": 1,
#         "seek": 0,
#         "start": 1.36,
#         "end": 4.16,
#         "text": "现在还有点晕",
#         "tokens": [50433, 25040, 35091, 12579, 5094, 243, 50573],
#         "temperature": 0.0,
#         "avg_logprob": -0.24068851005740283,
#         "compression_ratio": 1.1958041958041958,
#         "no_speech_prob": 1.4620414254262748e-11,
#     },
#     ...
# ]


def chunk_segments(
    segments: list[Segment],
    max_size: int | None = None,
    overlap_segs: int | None = None,
) -> list[Chunk]:
    # 用 is None 而非 or：显式传 0 是合法取值，不该被当成「没传」而回退到配置
    max_size = settings.chunk_max_size if max_size is None else max_size
    overlap_segs = (
        settings.chunk_overlap_segs if overlap_segs is None else overlap_segs
    )

    chunks: list[Chunk] = []
    buf: list[Segment] = []
    buf_len = 0

    def flush(buf: list[Segment]) -> Chunk:
        return Chunk(
            text=" ".join(s["text"] for s in buf).strip(),
            start=buf[0]["start"],
            end=buf[-1]["end"],
            seg_ids=[s["id"] for s in buf],
        )

    for seg in segments:
        seg_len = len(seg["text"])
        over_budget = buf_len + seg_len > max_size
        at_boundary = buf and _is_sentence_boundary(buf[-1]["text"])

        # 1. 加入后即将超过 max_size（字符数）
        # 2. 已达 70% 容量 且 当前末尾是句子结束符（。！？!?）
        if buf and (over_budget or (buf_len > max_size * 0.7 and at_boundary)):
            chunks.append(flush(buf))
            # 保留最后几个 segment 作为 overlap；
            # 不能直接写 buf[-overlap_segs:]，overlap_segs 为 0 时等价于 buf[0:]（整个 buf）
            buf = buf[-overlap_segs:] if overlap_segs else []
            buf_len = sum(len(seg["text"]) for seg in buf)

        buf.append(seg)
        buf_len += seg_len

    # 处理最后一批未满 max_size 的 segments。
    if buf:
        chunks.append(flush(buf))

    return chunks
