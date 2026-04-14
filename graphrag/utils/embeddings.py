"""Utilidades para generación y gestión de embeddings."""

import math

from ..llm.embedding_client import EmbeddingClient


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calcula la similitud coseno entre dos vectores."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class EmbeddingGenerator:
    """Generador de embeddings"""

    def __init__(self) -> None:
        self.client = EmbeddingClient()

    def embed_text(self, text: str) -> list[float]:
        """Genera el embedding de un único texto."""
        return self.client.embed_single(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Genera embeddings para una lista de textos."""
        return self.client.embed(texts)

    def embed_with_prefix(
        self, text: str, prefix: str = "search_document: "
    ) -> list[float]:
        """Genera el embedding de un texto con un prefijo de tarea."""
        return self.client.embed_single(prefix + text)

    def embed_chunks(
        self, chunks: list[dict], text_key: str = "text"
    ) -> list[dict]:
        """Genera embeddings para una lista de chunks y los añade in-place."""
        texts = [chunk[text_key] for chunk in chunks]
        embeddings = self.client.embed(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb
        return chunks
