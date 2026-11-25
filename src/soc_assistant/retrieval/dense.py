"""Semantic search with an embedding model and a Chroma vector store.

An embedding model turns a text into a vector (384 numbers here) so that texts with
similar *meaning* get vectors pointing in similar directions, even with different words
("Word launched PowerShell" ~ "Office application spawning a command interpreter").
Similarity is the cosine of the angle between two vectors (1 = same direction).

The index is built once (`make index`) and stored on disk in Chroma. One Chroma
collection per embedding model, so two models can be compared side by side.
"""

from __future__ import annotations

import re
from functools import lru_cache

import chromadb
import numpy as np
from chromadb.config import Settings

from soc_assistant.config import PROJECT_ROOT, load_config
from soc_assistant.ingest.documents import Chunk


def collection_name(model_name: str) -> str:
    return "chunks_" + re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()


@lru_cache(maxsize=4)
def load_embedder(model_name: str, device: str = "cpu"):
    from sentence_transformers import SentenceTransformer  # slow import, so done lazily

    # float32: some models default to bfloat16, which this CPU runs slowly and less precisely.
    kwargs = {"device": device, "model_kwargs": {"dtype": "float32"}}
    try:
        # Cached model: no request to the HuggingFace Hub, so nothing leaves the machine.
        return SentenceTransformer(model_name, local_files_only=True, **kwargs)
    except OSError:
        return SentenceTransformer(model_name, **kwargs)   # first run only: download it once


class Embedder:
    def __init__(self, model_name: str | None = None):
        cfg = load_config()["embeddings"]
        self.model_name = model_name or cfg["model"]
        self.query_prefix = cfg.get("query_prefix", {}).get(self.model_name, "")
        self.batch_size = cfg.get("batch_size", 32)
        self.model = load_embedder(self.model_name, cfg.get("device", "cpu"))

    def embed_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=show_progress, convert_to_numpy=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True,
                                 convert_to_numpy=True)[0]


def chroma_client() -> chromadb.ClientAPI:
    path = PROJECT_ROOT / load_config()["paths"]["vector_store"]
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


class DenseIndex:
    def __init__(self, embedder: Embedder, chunks: list[Chunk]):
        self.embedder = embedder
        self.chunks = chunks
        self.position = {c.chunk_id: i for i, c in enumerate(chunks)}
        self.collection = chroma_client().get_collection(collection_name(embedder.model_name))
        if self.collection.count() != len(chunks):
            raise RuntimeError(
                f"Index has {self.collection.count()} chunks but the corpus has {len(chunks)}. "
                "Rebuild it with `make index`."
            )

    @staticmethod
    def build(embedder: Embedder, chunks: list[Chunk], batch: int = 256) -> None:
        client = chroma_client()
        name = collection_name(embedder.model_name)
        if name in [c.name for c in client.list_collections()]:
            client.delete_collection(name)
        # Default distance in Chroma is L2; we want cosine (vectors are normalised).
        col = client.create_collection(name, metadata={"hnsw:space": "cosine",
                                                       "model": embedder.model_name})
        vectors = embedder.embed_documents([c.text for c in chunks], show_progress=True)
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            col.add(ids=[c.chunk_id for c in part],
                    embeddings=vectors[i:i + batch].tolist(),
                    metadatas=[{"doc_id": c.doc_id, "source_id": c.source_id} for c in part])

    def search(self, query: str, k: int = 30) -> list[tuple[int, float]]:
        """Return (chunk index, cosine similarity) pairs, best first."""
        res = self.collection.query(query_embeddings=[self.embedder.embed_query(query).tolist()],
                                    n_results=k, include=["distances"])
        return [(self.position[cid], 1.0 - dist)
                for cid, dist in zip(res["ids"][0], res["distances"][0]) if cid in self.position]
