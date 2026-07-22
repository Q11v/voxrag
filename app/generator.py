import logging
from collections.abc import Iterator

from openai import OpenAI

from .config import settings
from .retriever import Hit

_logger = logging.getLogger(__name__)
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )
    return _client


def _format_time(seconds: float) -> str:
    # divmod(a, b) = 一次同时算出商和余数，返回一个元组 (a // b, a % b)
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(
        f"[片段 {i + 1}] ({_format_time(h.start)}-{_format_time(h.end)}), 来源：{h.source}\n{h.text}"
        for i, h in enumerate(hits)
    )
    return f"""你是一个会议助手。请严格基于下面提供的会议片段回答用户问题。

【回答规则】
1. 答案必须基于会议片段，不得编造未在片段中出现的信息。
2. 如果片段中没有足够信息回答问题，直接回复："根据现有会议记录，无法回答这个问题。"
3. 引用具体内容时，标注片段编号和时间戳，例如 "（片段 2, 02:15）"。
4. 答案保持简洁清晰，避免重复。
5. 如果片段内容不连贯或有冲突，指出这一点。

【会议片段】
{context}

【用户问题】
{question}

【回答】"""


def answer_stream(question: str, hits: list[Hit]) -> Iterator[str]:
    prompt = _build_prompt(question, hits)
    _logger.debug("prompt:\n%s", prompt)
    stream = _get_client().chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        if chunk.usage:
            u = chunk.usage
            _logger.info(
                "usage prompt=%d completion=%d total=%d",
                u.prompt_tokens,
                u.completion_tokens,
                u.total_tokens,
            )
