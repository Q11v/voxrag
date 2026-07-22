from sentence_transformers import CrossEncoder

from .config import settings

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(settings.rerank_model)
    return _model


def rerank(query: str, candidates: list[str], top_k: int) -> list[tuple[int, float]]:
    if not candidates:
        return []
    pairs = [(query, c) for c in candidates]
    scores = _get_model().predict(pairs).tolist()

    indexed = sorted(enumerate(scores), key=lambda x: -x[1])
    return indexed[:top_k]
