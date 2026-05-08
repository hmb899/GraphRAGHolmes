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
            {
                "question": "How many deductions does Holmes make in The Red-Headed League?",
                "cypher": "MATCH (d:Deduction) WHERE toLower(d.story_title) CONTAINS toLower('red-headed league') RETURN d.story_title AS story, count(d) AS deductions_in_story",
            },
            {
                "question": "In which stories do crimes of type murder appear?",
                "cypher": "MATCH (cr:Crime) WHERE toLower(cr.type) CONTAINS 'murder' RETURN DISTINCT cr.story_title AS story_with_murder_crime ORDER BY cr.story_title",
            },
            {
                "question": "What objects does Sherlock Holmes use?",
                "cypher": "MATCH (c:Character)-[:USES]->(o:Object) WHERE toLower(c.name) CONTAINS 'holmes' RETURN DISTINCT o.name AS object_used_by_holmes, o.type AS type ORDER BY o.name",
            },
            {
                "question": "In which stories does Watson appear?",
                "cypher": "MATCH (c:Character)-[:APPEARS_IN]->(s:Story) WHERE toLower(c.name) CONTAINS 'watson' RETURN DISTINCT s.title AS story_featuring_watson ORDER BY s.title",
            },
            {
                "question": "Which characters know Holmes AND appear in more than one story?",
                "cypher": "MATCH (c:Character)-[:KNOWS]->(h:Character {name: 'Sherlock Holmes'}), (c)-[:APPEARS_IN]->(s:Story) WITH c, count(DISTINCT s) AS story_count WHERE story_count > 1 RETURN c.name AS character_knowing_holmes, story_count ORDER BY story_count DESC",
            },
            {
                "question": "Which characters investigate crimes and appear in more than one story?",
                "cypher": "MATCH (c:Character)-[:INVESTIGATES]->(cr:Crime), (c)-[:APPEARS_IN]->(s:Story) WITH c, count(DISTINCT s) AS story_count WHERE story_count > 1 RETURN DISTINCT c.name AS character_investigates_crimes, story_count ORDER BY c.name",
            },
            {
                "question": "What is the deduction chain in The Adventure of the Speckled Band?",
                "cypher": "MATCH (d1:Deduction)-[:LEADS_TO]->(d2:Deduction) WHERE toLower(d1.story_title) CONTAINS toLower('speckled band') RETURN d1.conclusion AS deduction_leads_from, d2.conclusion AS deduction_leads_to ORDER BY d1.id",
            },
        ]

        self.terminology_mappings: dict[str, str] = {
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
            "aparece en": "APPEARS_IN",
            "conoce a": "KNOWS",
            "investiga": "INVESTIGATES",
            "usa": "USES",
            "vive en": "LIVES_AT",
            "ocurre en": "OCCURS_IN",
            "participa en": "PARTICIPATES_IN",
            "colección": "collection (property of Story)",
            "nombre": "name",
            "descripción": "description",
            "tipo": "type",
            "Adventures": "The Adventures of Sherlock Holmes",
            "Memoirs": "The Memoirs of Sherlock Holmes",
            "Return": "The Return of Sherlock Holmes",
        }

    def add_few_shot_example(self, question: str, cypher: str) -> None:
        """Añade un ejemplo few-shot a la lista."""
        self.few_shot_examples.append({"question": question, "cypher": cypher})

    def generate_cypher(self, question: str) -> str:
        """Genera una query Cypher a partir de una pregunta en lenguaje natural."""
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
            "13. Deduction and Crime nodes link to their story via a 'story_title' STRING PROPERTY, NOT via a relationship. "
            "Never write MATCH (c:Character)-[...]->(d:Deduction). "
            "Always use: MATCH (d:Deduction) WHERE toLower(d.story_title) CONTAINS toLower('...').\n"
            "14. For multi-hop queries that count appearances ('appear in more than one story', 'appear in multiple stories'), "
            "always use WITH + count(DISTINCT s) AFTER the MATCH, then filter with WHERE: "
            "MATCH (c)-[:REL1]->(), (c)-[:APPEARS_IN]->(s:Story) WITH c, count(DISTINCT s) AS n WHERE n > 1 RETURN ...\n"
            "15. NEVER use exact name matching ({name: 'Watson'}, {name: 'Dr. Watson'}) for secondary characters — "
            "their stored name may vary. Always use toLower(c.name) CONTAINS 'watson'. "
            "Exact match {name: 'Sherlock Holmes'} is only safe for Holmes himself.\n"
            "16. Use DESCRIPTIVE RETURN aliases that carry full context to the answer generator. "
            "BAD: RETURN count(d) AS total — the LLM cannot tell what 'total' refers to. "
            "GOOD: RETURN d.story_title AS story, count(d) AS deductions_in_story — the LLM sees both the story and the count. "
            "BAD: RETURN cr.story_title AS story_title — 'story_title' alone loses the crime context. "
            "GOOD: RETURN DISTINCT cr.story_title AS story_with_murder_crime — the alias preserves the filter semantics.\n"
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
        """Genera una query Cypher y la ejecuta contra Neo4j."""
        cypher = self.generate_cypher(question)
        try:
            results = self.neo4j.execute_query(cypher)
            return cypher, results
        except Exception as exc:
            logger.error("Error ejecutando Cypher: %s\nQuery: %s", exc, cypher)
            return cypher, []

    def retrieve_with_retry(self, question: str, max_retries: int = 2) -> tuple[str, list[dict[str, Any]]]:
        """Genera y ejecuta una query Cypher; si falla, reintenta con el error como feedback al LLM."""
        current_question = question
        last_cypher = ""

        for attempt in range(max_retries + 1):
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
