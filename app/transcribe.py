from typing import TypedDict

import mlx_whisper

from .config import settings


# dict 是自由的——任何键、任何值都能放。方便，但写错了要等运行时才发现
# TypedDict = 给字典写一份键名和类型的说明书，让编辑器帮你抓拼写和类型错误。 数据本身还是普通字典，一点没变
class Segment(TypedDict):
    id: int
    start: float
    end: float
    text: str


def transcribe(audio_path: str) -> list[Segment]:
    result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=settings.asr_model)
    segments = result["segments"]
    assert isinstance(segments, list), "segments should be a list"
    return segments
