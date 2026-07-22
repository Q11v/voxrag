import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    CollectionInfo,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .chunker import Chunk
from .config import settings

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
    return _client


def _make_id(source: str, idx: int) -> str:
    return str(uuid.uuid5(settings.id_namespace, f"{source}#{idx}"))


def init_collection() -> None:
    client = get_client()
    if not client.collection_exists(settings.collection_name):
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(
                size=settings.vector_size, distance=Distance.COSINE
            ),
        )


def collection_info() -> CollectionInfo:
    return get_client().get_collection(settings.collection_name)


def delete_by_source(source: str) -> None:
    get_client().delete(
        collection_name=settings.collection_name,
        points_selector=Filter(
            # MatchValue(value=source)                        # ① 值要等于 source
            # FieldCondition(key="source", match=...)         # ② source 这个字段，要满足①
            # Filter(must=[...])                              # ③ 必须满足②里所有条件
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        ),
        # delete 默认异步返回，若不等待落盘，后续 upsert 的新数据可能被这次删除波及
        wait=True,
    )


def store(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    source: str,
    replace: bool = True,
) -> None:
    if replace:
        delete_by_source(source)

    points = [
        PointStruct(
            id=_make_id(source, i),
            vector=embedding,
            payload={
                "text": chunk.text,
                "start": chunk.start,
                "end": chunk.end,
                "seg_ids": chunk.seg_ids,
                "source": source,
            },
        )
        # zip(chunks, embeddings)：两个列表配对
        # strict=True：长度不等就报错（Python 3.10+）
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
    ]
    get_client().upsert(collection_name=settings.collection_name, points=points)
