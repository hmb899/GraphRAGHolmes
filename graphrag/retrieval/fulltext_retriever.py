import logging
from typing import Any

from ..config import get_settings
from ..graph.neo4j_manager import Neo4jManager

logger = logging.getLogger(__name__)


class FullTextRetriever:
    """Recupera chunks usando el índice fulltext 'chunk_fulltext' sobre Chunk.text."""

    def __init__(self, neo4j_manager: Neo4jManager) -> None:
        self.neo4j = neo4j_manager
        self.settings = get_settings()

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Recupera chunks cuyo texto contiene las palabras clave de la query."""
        top_k = top_k or self.settings.top_k_results

        cypher = """
        CALL db.index.fulltext.queryNodes('chunk_fulltext', $query, {limit: $top_k})
        YIELD node AS chunk, score
        MATCH (s:Story)-[:HAS_CHUNK]->(chunk)
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
            return self.neo4j.execute_query(cypher, {"query": query, "top_k": top_k})
        except Exception as exc:
            logger.error("Error en búsqueda fulltext: %s", exc)
            return []
