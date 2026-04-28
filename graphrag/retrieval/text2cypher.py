"""Conversión de lenguaje natural a Cypher usando Gemini (Text2Cypher)."""

import logging
from typing import Any

from pydantic import BaseModel

from ..config import get_settings
from ..graph.neo4j_manager import Neo4jManager
from ..llm.gemini_client import GeminiClient, ModelTier

logger = logging.getLogger(__name__)


class _CypherQuery(BaseModel):
    cypher: str


class Text2CypherRetriever:
    """Convierte preguntas en lenguaje natural a queries Cypher y las ejecuta."""

    def __init__(self, neo4j_manager: Neo4jManager) -> None:
        self.neo4j = neo4j_manager
        self.client = GeminiClient()
        self.settings = get_settings()

        self.few_shot_examples: list[dict[str, str]] = [
            {
                "question": "¿En qué relatos aparece Sherlock Holmes?",
                "cypher": "MATCH (c:Character {name: 'Sherlock Holmes'})-[:APPEARS_IN]->(s:Story) RETURN s.title AS title, s.collection AS collection ORDER BY s.title",
            },
            {
                "question": "¿Cuántos personajes hay en 'A Scandal in Bohemia'?",
                "cypher": "MATCH (c:Character)-[:APPEARS_IN]->(s:Story) WHERE toLower(s.title) = toLower('A Scandal in Bohemia') RETURN count(c) AS total_characters",
            },
            {
                "question": "¿Qué crímenes investiga Holmes?",
                "cypher": "MATCH (c:Character {name: 'Sherlock Holmes'})-[:INVESTIGATES]->(cr:Crime) RETURN cr.name AS crime, cr.type AS type, cr.description AS description",
            },
            {
                "question": "¿Qué personajes se conocen entre sí?",
                "cypher": "MATCH (c1:Character)-[:KNOWS]->(c2:Character) RETURN c1.name AS character1, c2.name AS character2, c1.name + ' knows ' + c2.name AS relationship",
            },
            {
                "question": "¿Dónde vive Sherlock Holmes?",
                "cypher": "MATCH (c:Character {name: 'Sherlock Holmes'})-[:LIVES_AT]->(l:Location) RETURN l.name AS location, l.type AS type",
            },
            {
                "question": "¿Qué deducciones hace Holmes en 'A Scandal in Bohemia'?",
                "cypher": "MATCH (d:Deduction {story_title: 'A Scandal in Bohemia'}) RETURN d.observation AS observation, d.inference AS inference, d.conclusion AS conclusion",
            },
            {
                "question": "¿En qué ubicaciones ocurren crímenes?",
                "cypher": "MATCH (cr:Crime)-[:OCCURS_IN]->(l:Location) RETURN cr.name AS crime, l.name AS location, l.type AS location_type",
            },
            {
                "question": "¿Qué objetos usa Watson?",
                "cypher": "MATCH (c:Character)-[:USES]->(o:Object) WHERE toLower(c.name) CONTAINS 'watson' RETURN c.name AS character, o.name AS object, o.type AS type",
            },
            {
                "question": "¿Cuántos relatos hay por colección?",
                "cypher": "MATCH (s:Story) RETURN s.collection AS collection, count(s) AS total ORDER BY total DESC",
            },
            {
                "question": "¿Qué escenas tienen lugar en Baker Street?",
                "cypher": "MATCH (sc:Scene)-[:TAKES_PLACE_IN]->(l:Location) WHERE toLower(l.name) CONTAINS 'baker street' RETURN sc.title AS scene, sc.description AS description ORDER BY sc.sequence_order",
            },
            {
                "question": "¿Qué personajes participan en eventos del relato 'Silver Blaze'?",
                "cypher": "MATCH (c:Character)-[:PARTICIPATES_IN]->(e:Event {story_title: 'Silver Blaze'}) RETURN c.name AS character, e.name AS event, e.description AS description",
            },
            {
                "question": "¿Qué objetos se encontraron en alguna ubicación?",
                "cypher": "MATCH (o:Object)-[:FOUND_AT]->(l:Location) RETURN o.name AS object, o.type AS type, l.name AS location ORDER BY l.name",
            },
        ]

        self.terminology_mappings: dict[str, str] = {
            # Nodos
            "personaje": "Character",
            "personajes": "Character",
            "character": "Character",
            "detective": "Character (name: 'Sherlock Holmes')",
            "Holmes": "Character (name: 'Sherlock Holmes')",
            "Watson": "Character (name: 'Dr. Watson')",
            "relato": "Story",
            "historia": "Story",
            "cuento": "Story",
            "story": "Story",
            "ubicación": "Location",
            "lugar": "Location",
            "sitio": "Location",
            "crimen": "Crime",
            "delito": "Crime",
            "caso": "Crime",
            "misterio": "Crime",
            "objeto": "Object",
            "pista": "Object",
            "evidencia": "Object",
            "arma": "Object (type: 'weapon')",
            "deducción": "Deduction",
            "razonamiento": "Deduction",
            "inferencia": "Deduction",
            "escena": "Scene",
            "evento": "Event",
            "acontecimiento": "Event",
            "fragmento": "Chunk",
            # Relaciones
            "aparece en": "APPEARS_IN",
            "conoce a": "KNOWS",
            "investiga": "INVESTIGATES",
            "usa": "USES",
            "vive en": "LIVES_AT",
            "ocurre en": "OCCURS_IN",
            "participa en": "PARTICIPATES_IN",
            # Propiedades
            "colección": "collection (property of Story)",
            "nombre": "name",
            "descripción": "description",
            "tipo": "type",
            # Valores de colección
            "Adventures": "The Adventures of Sherlock Holmes",
            "Memoirs": "The Memoirs of Sherlock Holmes",
            "Return": "The Return of Sherlock Holmes",
        }

    def add_few_shot_example(self, question: str, cypher: str) -> None:
        """Añade un ejemplo few-shot a la lista."""
        self.few_shot_examples.append({"question": question, "cypher": cypher})

    def generate_cypher(self, question: str) -> str:
        """Genera una query Cypher a partir de una pregunta en lenguaje natural.

        Usa el esquema real del grafo y ejemplos few-shot para guiar la generación.
        Emplea structured_output para garantizar que el LLM devuelve Cypher puro.

        Args:
            question: Pregunta en lenguaje natural (español o inglés).

        Returns:
            Query Cypher lista para ejecutar, sin backticks ni markdown.
        """
        schema = self.neo4j.get_schema()
        schema_str = Neo4jManager.format_schema(schema)

        mappings_str = "\n".join(
            f"  '{k}' → {v}" for k, v in self.terminology_mappings.items()
        )

        examples_str = "\n\n".join(
            f"Q: {ex['question']}\nCypher: {ex['cypher']}"
            for ex in self.few_shot_examples
        )

        system_prompt = (
            "You are an expert Neo4j Cypher query generator for a knowledge graph "
            "about Sherlock Holmes stories by Arthur Conan Doyle.\n\n"
            "GRAPH SCHEMA:\n"
            f"{schema_str}\n\n"
            "TERMINOLOGY MAPPINGS (natural language → graph elements):\n"
            f"{mappings_str}\n\n"
            "FEW-SHOT EXAMPLES:\n"
            f"{examples_str}\n\n"
            "RULES:\n"
            "1. Use ONLY the node labels, relationship types, and properties shown in the schema above.\n"
            "2. Output ONLY the Cypher query in the 'cypher' field — no explanations, no markdown, no backticks.\n"
            "3. The query must be syntactically correct Neo4j Cypher.\n"
            "4. Use descriptive aliases in RETURN clauses (e.g. AS character_name, AS story_title).\n"
            "5. For text searches, prefer toLower(prop) CONTAINS toLower($value) for case-insensitive matching.\n"
            "6. Always include a RETURN clause. Never generate write queries (CREATE, MERGE, SET, DELETE).\n"
            "7. For character names, prefer toLower(c.name) CONTAINS 'partial_name' over exact matching.\n"
            "8. When counting, use count() with a meaningful alias.\n"
            "9. Add ORDER BY when the question implies a ranked or sequential result.\n"
            "10. Deduction nodes are identified by their 'observation' text (not a 'name' property).\n"
            "11. Scene nodes use 'title' as their display name; Story nodes also use 'title'.\n"
            "12. Story titles are stored with Python title-case (every word capitalised, e.g. 'A Scandal In Bohemia', 'The Adventure Of The Speckled Band'). Always match them with toLower(s.title) = toLower('...') to handle user input variations.\n"
        )

        result = self.client.structured_output(
            prompt=question,
            schema=_CypherQuery,
            model_tier=ModelTier.PRO,
            system_instruction=system_prompt,
            temperature=0.0,
        )

        cypher = result.cypher.strip()
        # Eliminar backticks de markdown si el LLM los incluye pese al structured_output
        if cypher.startswith("```"):
            lines = cypher.splitlines()
            cypher = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()
        return cypher

    def retrieve(self, question: str) -> tuple[str, list[dict[str, Any]]]:
        """Genera una query Cypher y la ejecuta contra Neo4j.

        Args:
            question: Pregunta en lenguaje natural.

        Returns:
            Tupla (cypher_query, resultados). Si la ejecución falla,
            devuelve (cypher_query, []) y registra el error.
        """
        cypher = self.generate_cypher(question)
        try:
            results = self.neo4j.execute_query(cypher)
            return cypher, results
        except Exception as exc:
            logger.error("Error ejecutando Cypher: %s\nQuery: %s", exc, cypher)
            return cypher, []

    def retrieve_with_retry(
        self,
        question: str,
        max_retries: int = 2,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Genera y ejecuta una query Cypher con reintentos ante fallos de ejecución.

        Si la query generada provoca un error en Neo4j, el error se incluye como
        feedback en el siguiente intento para que el LLM corrija la query.
        No reintenta cuando los resultados son simplemente una lista vacía legítima.

        Args:
            question: Pregunta en lenguaje natural.
            max_retries: Número máximo de reintentos tras el primer intento.

        Returns:
            Tupla (cypher_query, resultados) del intento más exitoso.
        """
        current_question = question
        last_cypher = ""

        for attempt in range(max_retries + 1):
            # Generación (puede fallar si el LLM devuelve algo inválido)
            try:
                last_cypher = self.generate_cypher(current_question)
            except Exception as exc:
                logger.warning(
                    "Intento %d/%d — error generando Cypher: %s",
                    attempt + 1, max_retries + 1, exc,
                )
                current_question = (
                    f"{question}\n\n"
                    f"[RETRY — generation error on attempt {attempt + 1}]: {exc}\n"
                    "Please generate a valid Cypher query."
                )
                continue

            # Ejecución
            try:
                results = self.neo4j.execute_query(last_cypher)
                if attempt > 0:
                    logger.info(
                        "Query corregida en el intento %d: %s", attempt + 1, last_cypher
                    )
                return last_cypher, results
            except Exception as exc:
                error_msg = str(exc)
                logger.warning(
                    "Intento %d/%d — error ejecutando query: %s\nQuery: %s",
                    attempt + 1, max_retries + 1, error_msg, last_cypher,
                )
                current_question = (
                    f"{question}\n\n"
                    f"[RETRY — attempt {attempt + 1} failed]\n"
                    f"The following query was generated but failed to execute:\n"
                    f"{last_cypher}\n\n"
                    f"Neo4j error: {error_msg}\n\n"
                    "Please generate a corrected Cypher query that avoids this error."
                )

        logger.error(
            "Todos los intentos (%d) fallaron para: '%s'", max_retries + 1, question
        )
        return last_cypher, []
