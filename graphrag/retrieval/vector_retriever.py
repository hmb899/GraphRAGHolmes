import logging
from typing import Any

from ..config import get_settings
from ..graph.neo4j_manager import Neo4jManager
from ..llm.embedding_client import EmbeddingClient
from ..llm.gemini_client import GeminiClient, ModelTier

logger = logging.getLogger(__name__)

_HYDE_SYSTEM = (
    "You are a Sherlock Holmes canon expert. "
    "Write a short passage (2-4 sentences) from a Sherlock Holmes story that would "
    "directly answer the question. Write as if quoting from the original text. "
    "Do not add explanations or preamble — output only the passage."
)


def _hyde_embed(query: str, embedding_client: EmbeddingClient) -> list[float]:
    """HyDE: embebe un pasaje hipotético de respuesta en lugar de la pregunta directa.

    Genera un texto de respuesta plausible antes de embeber para acercar el vector
    de la query al espacio semántico de los chunks reales, mejorando el recall sin
    re-indexar. Si el LLM falla, hace fallback al embedding directo de la query.
    """
    try:
        hypothetical = GeminiClient().generate(
            prompt=f"Question: {query}\n\nPassage:",
            model_tier=ModelTier.FLASH,
            system_instruction=_HYDE_SYSTEM,
            temperature=0.0,
        )
        logger.debug("HyDE passage: %s", hypothetical[:120])
        return embedding_client.embed_single(hypothetical)
    except Exception as exc:
        logger.warning("HyDE fallback to direct embedding: %s", exc)
        return embedding_client.embed_single(query)


def _build_where(filters: dict | None) -> tuple[str, dict[str, Any]]:
    """Construye la cláusula WHERE y los parámetros extra a partir de los filtros."""
    conditions: list[str] = []
    params: dict[str, Any] = {}
    if filters:
        if "story_title" in filters:
            conditions.append("s.title = $story_title")
            params["story_title"] = filters["story_title"]
        if "collection" in filters:
            conditions.append("s.collection = $collection")
            params["collection"] = filters["collection"]
    clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return clause, params


class VectorRetriever:
    """Recupera chunks relevantes usando búsqueda vectorial coseno sobre embeddings."""

    def __init__(self, neo4j_manager: Neo4jManager) -> None:
        self.neo4j = neo4j_manager
        self.embedding_client = EmbeddingClient()
        self.settings = get_settings()

    def retrieve(self, query: str, top_k: int | None = None, filters: dict | None = None,
                 use_hyde: bool = True) -> list[dict[str, Any]]:
        """Recupera los chunks más similares a la query por búsqueda vectorial pura."""
        top_k = top_k or self.settings.top_k_results
        try:
            query_embedding = (
                _hyde_embed(query, self.embedding_client)
                if use_hyde
                else self.embedding_client.embed_single(query)
            )
        except Exception as exc:
            logger.error("Error generando embedding de la query: %s", exc)
            return []

        where_clause, filter_params = _build_where(filters)
        params: dict[str, Any] = {
            "top_k": top_k,
            "query_embedding": query_embedding,
            **filter_params,
        }

        cypher = f"""
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
        YIELD node AS chunk, score
        MATCH (s:Story)-[:HAS_CHUNK]->(chunk)
        {where_clause}
        RETURN chunk.text AS text,
               chunk.id AS chunk_id,
               chunk.chunk_index AS chunk_index,
               chunk.position AS position,
               s.title AS story_title,
               s.collection AS collection,
               score
        ORDER BY score DESC
        """

        try:
            return self.neo4j.execute_query(cypher, params)
        except Exception as exc:
            logger.error("Error en búsqueda vectorial: %s", exc)
            return []

    def retrieve_with_context(self, query: str, top_k: int | None = None, filters: dict | None = None,
                              use_hyde: bool = True) -> list[dict[str, Any]]:
        """Recupera chunks con entidades mencionadas; usa OPTIONAL MATCH separados por label para evitar producto cartesiano."""
        top_k = top_k or self.settings.top_k_results
        try:
            query_embedding = (
                _hyde_embed(query, self.embedding_client)
                if use_hyde
                else self.embedding_client.embed_single(query)
            )
        except Exception as exc:
            logger.error("Error generando embedding de la query: %s", exc)
            return []

        where_clause, filter_params = _build_where(filters)
        params: dict[str, Any] = {
            "top_k": top_k,
            "query_embedding": query_embedding,
            **filter_params,
        }

        cypher = f"""
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
        YIELD node AS chunk, score
        MATCH (s:Story)-[:HAS_CHUNK]->(chunk)
        {where_clause}
        OPTIONAL MATCH (chunk)-[:MENTIONS]->(c:Character)
        WITH chunk, score, s, [x IN collect(DISTINCT c) WHERE x IS NOT NULL | {{name: x.name, occupation: x.occupation}}] AS characters
        OPTIONAL MATCH (chunk)-[:MENTIONS]->(l:Location)
        WITH chunk, score, s, characters, [x IN collect(DISTINCT l) WHERE x IS NOT NULL | {{name: x.name, type: x.type}}] AS locations
        OPTIONAL MATCH (chunk)-[:MENTIONS]->(o:Object)
        WITH chunk, score, s, characters, locations, [x IN collect(DISTINCT o) WHERE x IS NOT NULL | {{name: x.name, type: x.type}}] AS objects
        RETURN chunk.text AS text,
               chunk.id AS chunk_id,
               chunk.chunk_index AS chunk_index,
               chunk.position AS position,
               s.title AS story_title,
               s.collection AS collection,
               score,
               characters,
               locations,
               objects
        ORDER BY score DESC
        """

        try:
            return self.neo4j.execute_query(cypher, params)
        except Exception as exc:
            logger.error("Error en búsqueda vectorial con contexto: %s", exc)
            return []


# Cadena literal (no f-string) para evitar conflictos de escape con las llaves de Cypher.
_HYBRID_UNION = """
CALL () {
    CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
    YIELD node, score
    WITH collect({node: node, score: score}) AS nodes, max(score) AS max_score
    UNWIND nodes AS n
    RETURN n.node AS node,
           CASE WHEN max_score > 0 THEN n.score / max_score ELSE 0 END AS score
    UNION
    CALL db.index.fulltext.queryNodes('chunk_fulltext', $query, {limit: $top_k})
    YIELD node, score
    WITH collect({node: node, score: score}) AS nodes, max(score) AS max_score
    UNWIND nodes AS n
    RETURN n.node AS node,
           CASE WHEN max_score > 0 THEN n.score / max_score ELSE 0 END AS score
}
WITH node, max(score) AS score"""


class HybridRetriever:
    """Recupera chunks usando búsqueda híbrida (vectorial + fulltext) con scores normalizados.

    Ambos scores se normalizan dividiendo por el máximo de cada búsqueda antes
    de combinarlos. El score final es el máximo de los scores normalizados.
    """

    def __init__(self, neo4j_manager: Neo4jManager) -> None:
        self.neo4j = neo4j_manager
        self.embedding_client = EmbeddingClient()
        self.settings = get_settings()

    def retrieve(self, query: str, top_k: int | None = None, filters: dict | None = None,
                 use_hyde: bool = True) -> list[dict[str, Any]]:
        """Recupera chunks combinando búsqueda vectorial y fulltext."""
        top_k = top_k or self.settings.top_k_results
        try:
            query_embedding = (
                _hyde_embed(query, self.embedding_client)
                if use_hyde
                else self.embedding_client.embed_single(query)
            )
        except Exception as exc:
            logger.error("Error generando embedding de la query: %s", exc)
            return []

        where_clause, filter_params = _build_where(filters)
        params: dict[str, Any] = {
            "top_k": top_k,
            "query_embedding": query_embedding,
            "query": query,
            **filter_params,
        }

        cypher = _HYBRID_UNION + f"""
MATCH (s:Story)-[:HAS_CHUNK]->(node)
{where_clause}
RETURN node.text AS text,
       node.id AS chunk_id,
       node.chunk_index AS chunk_index,
       node.position AS position,
       s.title AS story_title,
       s.collection AS collection,
       score
ORDER BY score DESC
LIMIT $top_k
"""

        try:
            return self.neo4j.execute_query(cypher, params)
        except Exception as exc:
            logger.error("Error en búsqueda híbrida: %s", exc)
            return []

    def retrieve_with_context(self, query: str, top_k: int | None = None, filters: dict | None = None,
                              use_hyde: bool = True) -> list[dict[str, Any]]:
        """Recupera chunks con entidades mencionadas usando búsqueda híbrida."""
        top_k = top_k or self.settings.top_k_results
        try:
            query_embedding = (
                _hyde_embed(query, self.embedding_client)
                if use_hyde
                else self.embedding_client.embed_single(query)
            )
        except Exception as exc:
            logger.error("Error generando embedding de la query: %s", exc)
            return []

        where_clause, filter_params = _build_where(filters)
        params: dict[str, Any] = {
            "top_k": top_k,
            "query_embedding": query_embedding,
            "query": query,
            **filter_params,
        }

        cypher = _HYBRID_UNION + f"""
MATCH (s:Story)-[:HAS_CHUNK]->(node)
{where_clause}
OPTIONAL MATCH (node)-[:MENTIONS]->(c:Character)
WITH node, score, s, [x IN collect(DISTINCT c) WHERE x IS NOT NULL | {{name: x.name, occupation: x.occupation}}] AS characters
OPTIONAL MATCH (node)-[:MENTIONS]->(l:Location)
WITH node, score, s, characters, [x IN collect(DISTINCT l) WHERE x IS NOT NULL | {{name: x.name, type: x.type}}] AS locations
OPTIONAL MATCH (node)-[:MENTIONS]->(o:Object)
WITH node, score, s, characters, locations, [x IN collect(DISTINCT o) WHERE x IS NOT NULL | {{name: x.name, type: x.type}}] AS objects
RETURN node.text AS text,
       node.id AS chunk_id,
       node.chunk_index AS chunk_index,
       node.position AS position,
       s.title AS story_title,
       s.collection AS collection,
       score,
       characters,
       locations,
       objects
ORDER BY score DESC
LIMIT $top_k
"""

        try:
            return self.neo4j.execute_query(cypher, params)
        except Exception as exc:
            logger.error("Error en búsqueda híbrida con contexto: %s", exc)
            return []
