from __future__ import annotations

from dataclasses import dataclass

from secondbrain.vectorstore.chroma_store import ChromaVectorStore
from secondbrain.vectorstore.store import VectorStoreProtocol


@dataclass(slots=True)
class VectorStoreDeps:
    dimension: int
    collection_name: str = "secondbrain_notes"


async def build_vector_store(persistence_path: str, deps: VectorStoreDeps) -> VectorStoreProtocol:
    return ChromaVectorStore(
        persistence_path,
        deps.collection_name,
        expected_dimension=deps.dimension,
    )
