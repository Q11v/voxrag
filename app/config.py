import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# @dataclass：自动生成样板代码，__init__ / __repr__ / __eq__ 全自动生成：
# 创建后不能改
@dataclass(frozen=True)
class Settings:
    asr_model: str
    embed_model: str
    rerank_model: str
    llm_model: str

    chunk_max_size: int
    chunk_overlap_segs: int

    qdrant_host: str
    qdrant_port: int

    collection_name: str
    vector_size: int

    id_namespace: uuid.UUID

    retrieve_top_k: int
    rerank_top_k: int

    openrouter_api_key: str
    openrouter_base_url: str


# 必填、缺了不能跑（密钥、连接串）	os.environ["KEY"]
# 可选、有合理默认	os.getenv("KEY", "默认")
def load_settings() -> Settings:
    return Settings(
        asr_model=os.getenv("ASR_MODEL", "mlx-community/whisper-turbo"),
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
        rerank_model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        llm_model=os.getenv("LLM_MODEL", "z-ai/glm-4.5-air:free"),
        # CHUNK_MAX_SIZE => 400: 经验性的长度预算，1 token ≈ 1.5 ~ 2 中文字符
        # CHUNK_OVERLAP_SEGS => 2: 以 segment 为单位（简单可靠）
        chunk_max_size=int(os.getenv("CHUNK_MAX_SIZE", "400")),
        chunk_overlap_segs=int(os.getenv("CHUNK_OVERLAP_SEGS", "2")),
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        collection_name=os.getenv("COLLECTION_NAME", "voxrag"),
        vector_size=int(os.getenv("VECTOR_SIZE", "1024")),
        id_namespace=uuid.UUID(
            os.getenv("ID_NAMESPACE", "00000000-0000-0000-0000-000000000001")
        ),
        retrieve_top_k=int(os.getenv("RETRIEVE_TOP_K", "20")),
        rerank_top_k=int(os.getenv("RERANK_TOP_K", "3")),
        openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
    )


settings = load_settings()
