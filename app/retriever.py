import logging
from dataclasses import dataclass

from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from .config import settings
from .embedder import embed
from .reranker import rerank
from .store import get_client

_logger = logging.getLogger(__name__)


@dataclass
class Hit:
    text: str
    score: float
    start: float
    end: float
    source: str


def retrieve(
    question: str,
    top_k: int | None = None,
    rerank_k: int | None = None,
    source: str | None = None,
) -> list[Hit]:
    top_k = top_k or settings.retrieve_top_k
    rerank_k = rerank_k or settings.rerank_top_k

    q_vec = embed([question])[0]

    query_filter = (
        Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
        if source
        else None
    )

    points = (
        get_client()
        .query_points(
            collection_name=settings.collection_name,
            query=q_vec,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        .points
    )

    candidates: list[Hit] = []
    for p in points:
        _logger.debug("point id=%s score=%.4f", p.id, p.score)
        if p.payload is None:
            raise ValueError(f"point {p.id} has no payload")
        candidates.append(
            Hit(
                text=p.payload["text"],
                score=p.score,
                start=p.payload["start"],
                end=p.payload["end"],
                source=p.payload["source"],
            )
        )

    if not candidates:
        _logger.warning(
            "no candidates found for question=%s source=%s", question, source
        )
        return []

    ranked = rerank(question, [h.text for h in candidates], rerank_k)
    return [
        Hit(
            text=candidates[i].text,
            source=candidates[i].source,
            start=candidates[i].start,
            end=candidates[i].end,
            score=score,
        )
        for i, score in ranked
    ]
