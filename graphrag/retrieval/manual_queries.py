"""Plantillas Cypher parametrizadas para consultas recurrentes del dominio Sherlock Holmes."""

import logging
import re
from typing import Any

from ..config import get_settings
from ..graph.neo4j_manager import Neo4jManager

logger = logging.getLogger(__name__)


class ManualRetriever:
    """Ejecuta consultas Cypher predefinidas y parametrizadas para el dominio Holmes.

    A diferencia de Text2CypherRetriever, estas queries están escritas a mano,
    probadas y garantizan resultados correctos para los patrones de pregunta más
    frecuentes. No requieren ningún LLM.
    """

    def __init__(self, neo4j_manager: Neo4jManager) -> None:
        self.neo4j = neo4j_manager
        self.settings = get_settings()
        self.queries: dict[str, dict[str, Any]] = {}
        self._register_default_queries()

    # ------------------------------------------------------------------
    # Registro de plantillas
    # ------------------------------------------------------------------

    def _register_default_queries(self) -> None:
        """Carga las 10 plantillas Cypher por defecto del dominio Sherlock Holmes."""

        self.register_query(
            name="characters_in_story",
            description="Lista todos los personajes que aparecen en un relato específico",
            cypher=(
                "MATCH (c:Character)-[:APPEARS_IN]->(s:Story) "
                "WHERE toLower(s.title) = toLower($story_title) "
                "RETURN c.name AS name, c.occupation AS occupation, c.description AS description "
                "ORDER BY c.name"
            ),
            parameters={"story_title": "Título del relato (insensible a mayúsculas)"},
            example_question="¿Qué personajes aparecen en 'A Scandal in Bohemia'?",
        )

        self.register_query(
            name="stories_of_character",
            description="Lista todos los relatos en los que aparece un personaje",
            cypher=(
                "MATCH (c:Character)-[:APPEARS_IN]->(s:Story) "
                "WHERE toLower(c.name) CONTAINS toLower($character_name) "
                "RETURN s.title AS title, s.collection AS collection "
                "ORDER BY s.title"
            ),
            parameters={"character_name": "Nombre del personaje (parcial o completo)"},
            example_question="¿En qué relatos aparece Irene Adler?",
        )

        self.register_query(
            name="recurring_characters",
            description="Personajes que aparecen en un número mínimo de relatos",
            cypher=(
                "MATCH (c:Character)-[:APPEARS_IN]->(s:Story) "
                "WITH c, count(s) AS story_count "
                "WHERE story_count >= $min_stories "
                "RETURN c.name AS name, c.occupation AS occupation, story_count "
                "ORDER BY story_count DESC"
            ),
            parameters={"min_stories": "Número mínimo de relatos (entero)"},
            example_question="¿Qué personajes aparecen en más de 2 relatos?",
        )

        self.register_query(
            name="crimes_investigated_by",
            description="Crímenes que investiga un personaje específico",
            cypher=(
                "MATCH (c:Character)-[:INVESTIGATES]->(cr:Crime) "
                "WHERE toLower(c.name) CONTAINS toLower($character_name) "
                "RETURN cr.name AS crime, cr.type AS type, "
                "cr.description AS description, cr.motive AS motive"
            ),
            parameters={"character_name": "Nombre del personaje (parcial o completo)"},
            example_question="¿Qué crímenes investiga Sherlock Holmes?",
        )

        self.register_query(
            name="character_relationships_in_story",
            description="Pares de personajes que se conocen entre sí dentro de un relato",
            cypher=(
                "MATCH (c1:Character)-[:KNOWS]->(c2:Character), "
                "(c1)-[:APPEARS_IN]->(s:Story), "
                "(c2)-[:APPEARS_IN]->(s) "
                "WHERE toLower(s.title) = toLower($story_title) "
                "RETURN c1.name AS character1, c2.name AS character2"
            ),
            parameters={"story_title": "Título del relato (insensible a mayúsculas)"},
            example_question="¿Qué personajes se conocen entre sí en 'A Scandal in Bohemia'?",
        )

        self.register_query(
            name="locations_in_story",
            description="Ubicaciones donde transcurren las escenas de un relato",
            cypher=(
                "MATCH (sc:Scene)-[:BELONGS_TO]->(s:Story), "
                "(sc)-[:TAKES_PLACE_IN]->(l:Location) "
                "WHERE toLower(s.title) = toLower($story_title) "
                "RETURN DISTINCT l.name AS location, l.type AS type, l.description AS description"
            ),
            parameters={"story_title": "Título del relato (insensible a mayúsculas)"},
            example_question="¿En qué ubicaciones transcurre 'The Red-Headed League'?",
        )

        self.register_query(
            name="deduction_chain",
            description="Cadena de deducciones de Holmes en un relato, con sus encadenamientos",
            cypher=(
                "MATCH (d:Deduction) "
                "WHERE toLower(d.story_title) = toLower($story_title) "
                "OPTIONAL MATCH (d)-[:LEADS_TO]->(d2:Deduction) "
                "RETURN d.observation AS observation, d.inference AS inference, "
                "d.conclusion AS conclusion, d.method AS method, "
                "d2.observation AS leads_to "
                "ORDER BY d.id"
            ),
            parameters={"story_title": "Título del relato (insensible a mayúsculas)"},
            example_question="¿Cuál es la cadena de deducciones en 'A Scandal in Bohemia'?",
        )

        self.register_query(
            name="objects_used_by",
            description="Objetos que utiliza un personaje",
            cypher=(
                "MATCH (c:Character)-[:USES]->(o:Object) "
                "WHERE toLower(c.name) CONTAINS toLower($character_name) "
                "RETURN o.name AS object, o.type AS type, o.description AS description"
            ),
            parameters={"character_name": "Nombre del personaje (parcial o completo)"},
            example_question="¿Qué objetos usa Sherlock Holmes?",
        )

        self.register_query(
            name="story_summary",
            description="Resumen estadístico de un relato: personajes, crímenes, escenas y deducciones",
            cypher="""
                MATCH (s:Story)
                WHERE toLower(s.title) = toLower($story_title)
                OPTIONAL MATCH (c:Character)-[:APPEARS_IN]->(s)
                WITH s, count(DISTINCT c) AS characters
                OPTIONAL MATCH (cr:Crime)
                WHERE toLower(cr.story_title) = toLower($story_title)
                WITH s, characters, count(DISTINCT cr) AS crimes
                OPTIONAL MATCH (sc:Scene)-[:BELONGS_TO]->(s)
                WITH s, characters, crimes, count(DISTINCT sc) AS scenes
                OPTIONAL MATCH (d:Deduction)
                WHERE toLower(d.story_title) = toLower($story_title)
                RETURN s.title AS title, s.collection AS collection,
                       characters, crimes, scenes, count(DISTINCT d) AS deductions
            """,
            parameters={"story_title": "Título del relato (insensible a mayúsculas)"},
            example_question="Dame un resumen de 'A Scandal in Bohemia'",
        )

        self.register_query(
            name="crimes_by_type",
            description="Agrupa todos los crímenes del corpus por tipo con su recuento",
            cypher=(
                "MATCH (cr:Crime) "
                "RETURN cr.type AS type, count(cr) AS total, collect(cr.name) AS crimes "
                "ORDER BY total DESC"
            ),
            parameters={},
            example_question="¿Qué tipos de crímenes hay y cuántos de cada uno?",
        )

    def register_query(
        self,
        name: str,
        description: str,
        cypher: str,
        parameters: dict[str, str],
        example_question: str,
    ) -> None:
        """Registra una plantilla Cypher adicional en runtime.

        Args:
            name: Identificador único de la plantilla.
            description: Descripción en lenguaje natural de qué hace la query.
            cypher: Query Cypher con parámetros del tipo $param_name.
            parameters: Dict {nombre_param: descripción} de los parámetros esperados.
            example_question: Pregunta de ejemplo que dispara esta plantilla.
        """
        self.queries[name] = {
            "name": name,
            "description": description,
            "cypher": cypher,
            "parameters": parameters,
            "example_question": example_question,
        }

    # ------------------------------------------------------------------
    # Introspección
    # ------------------------------------------------------------------

    def get_available_queries(self) -> list[dict[str, Any]]:
        """Retorna la lista de plantillas disponibles con sus metadatos.

        Útil para que el router sepa qué plantillas existen y pueda
        incluirlas en su sistema de despacho.

        Returns:
            Lista de dicts con name, description, parameters y example_question.
        """
        return [
            {
                "name": q["name"],
                "description": q["description"],
                "parameters": q["parameters"],
                "example_question": q["example_question"],
            }
            for q in self.queries.values()
        ]

    # ------------------------------------------------------------------
    # Ejecución
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_name: str,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Ejecuta una plantilla Cypher por nombre con los parámetros proporcionados.

        Args:
            query_name: Nombre de la plantilla registrada.
            parameters: Valores para los parámetros de la query ($param).

        Returns:
            Tupla (cypher_ejecutado, lista_de_resultados). Si la ejecución
            falla, retorna (cypher, []) y registra el error.

        Raises:
            ValueError: Si query_name no corresponde a ninguna plantilla registrada.
        """
        if query_name not in self.queries:
            available = ", ".join(self.queries.keys())
            raise ValueError(
                f"Plantilla desconocida: '{query_name}'. "
                f"Disponibles: {available}"
            )

        template = self.queries[query_name]
        cypher = template["cypher"]
        params = parameters or {}

        try:
            results = self.neo4j.execute_query(cypher, params)
            return cypher, results
        except Exception as exc:
            logger.error(
                "Error ejecutando plantilla '%s': %s\nParams: %s",
                query_name, exc, params,
            )
            return cypher, []

    # ------------------------------------------------------------------
    # Matching heurístico pregunta → plantilla
    # ------------------------------------------------------------------

    def find_matching_query(
        self, question: str
    ) -> tuple[str, dict[str, Any]] | None:
        """Intenta encontrar la plantilla que mejor encaja con una pregunta.

        Matching heurístico basado en keywords: no usa LLM. Extrae parámetros
        mediante regex simples (títulos entre comillas, nombres de personajes
        conocidos, números).

        Args:
            question: Pregunta en lenguaje natural (español o inglés).

        Returns:
            Tupla (query_name, parameters) si hay match con confianza suficiente,
            None en caso contrario.
        """
        q = question.lower()

        story_title = self._extract_quoted_title(question)
        character_name = self._extract_character_name(q)
        min_stories = self._extract_number(q)

        # Puntuaciones por plantilla: sumar 1 por cada keyword presente
        _TRIGGERS: dict[str, list[str]] = {
            "characters_in_story": [
                "personajes", "personaje", "characters", "character",
                "aparecen en", "appear in", "who is in", "quién aparece",
            ],
            "stories_of_character": [
                "relatos de", "historias de", "stories of", "en qué relatos",
                "in which stories", "what stories", "aparece en qué",
            ],
            "recurring_characters": [
                "recurrente", "recurring", "más de", "more than", "varios relatos",
                "multiple stories", "cuántos relatos", "how many stories",
            ],
            "crimes_investigated_by": [
                "crímenes", "crimes", "delitos", "investiga", "investigates",
                "casos de", "cases of",
            ],
            "character_relationships_in_story": [
                "conocen", "know each other", "relaciones entre", "relationships between",
                "se conocen", "quién conoce",
            ],
            "locations_in_story": [
                "ubicaciones", "locations", "lugares", "dónde transcurre",
                "where does", "escenarios",
            ],
            "deduction_chain": [
                "deducción", "deducciones", "deduction", "razonamiento", "reasoning",
                "cadena", "chain", "infiere", "concluye",
            ],
            "objects_used_by": [
                "objetos", "objects", "usa", "uses", "utiliza", "utilizes",
                "herramientas", "tools",
            ],
            "story_summary": [
                "resumen", "summary", "estadísticas", "statistics",
                "cuántos hay en", "how many in", "overview",
            ],
            "crimes_by_type": [
                "tipos de crímenes", "crime types", "tipos de delitos",
                "clases de crímenes", "categorías",
            ],
        }

        scores: dict[str, int] = {name: 0 for name in _TRIGGERS}
        for name, keywords in _TRIGGERS.items():
            for kw in keywords:
                if kw in q:
                    scores[name] += 1

        best_name = max(scores, key=lambda k: scores[k])
        best_score = scores[best_name]

        if best_score == 0:
            return None

        # Construir parámetros para la plantilla ganadora
        template_params = self.queries[best_name]["parameters"]
        extracted: dict[str, Any] = {}

        if "story_title" in template_params and story_title:
            extracted["story_title"] = story_title
        if "character_name" in template_params and character_name:
            extracted["character_name"] = character_name
        if "min_stories" in template_params:
            extracted["min_stories"] = min_stories if min_stories is not None else 2

        # Si la plantilla necesita parámetros pero no pudimos extraerlos, no hacer match
        required = set(template_params.keys())
        if required and not required.issubset(extracted.keys()):
            return None

        return best_name, extracted

    # ------------------------------------------------------------------
    # Helpers de extracción de parámetros
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_quoted_title(text: str) -> str | None:
        """Extrae el primer string entre comillas simples o dobles."""
        match = re.search(r"[\"']([\w\s\-,:]+)[\"']", text)
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_character_name(text_lower: str) -> str | None:
        """Detecta nombres canónicos de personajes en el texto normalizado."""
        # Los nombres están en orden de especificidad descendente para evitar
        # que 'holmes' capture antes que 'sherlock holmes'.
        _KNOWN_NAMES = [
            "sherlock holmes", "dr. watson", "dr watson", "irene adler",
            "professor moriarty", "moriarty", "mrs. hudson", "mrs hudson",
            "inspector lestrade", "lestrade", "mycroft holmes", "mycroft",
            "irene", "watson", "holmes",
        ]
        for name in _KNOWN_NAMES:
            if name in text_lower:
                # Devolver en Title Case para que coincida con los datos del grafo
                return name.title()
        return None

    @staticmethod
    def _extract_number(text_lower: str) -> int | None:
        """Extrae el primer entero encontrado en el texto."""
        match = re.search(r"\b(\d+)\b", text_lower)
        return int(match.group(1)) if match else None
