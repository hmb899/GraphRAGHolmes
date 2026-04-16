"""Gestión de conexión y operaciones CRUD en Neo4j."""

import logging
import re
from typing import Any

from neo4j import GraphDatabase

from ..config import get_settings

logger = logging.getLogger(__name__)


def _sanitize_id(text: str) -> str:
    """Convierte un texto en un identificador válido para Neo4j (lowercase, sin espacios)."""
    return re.sub(r'[^a-z0-9]', '_', text.lower())


def _title_tokens(title: str) -> set[str]:
    """Tokens normalizados de un título, sin puntuación."""
    return {t for t in re.sub(r"[^a-z0-9\s]", "", title.lower()).split() if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _fuzzy_match(query: str, candidates: set[str], threshold: float = 0.35) -> str | None:
    """Devuelve el candidato con mayor similitud Jaccard, o None si no supera el umbral."""
    q_tokens = _title_tokens(query)
    best_score, best = 0.0, None
    for c in candidates:
        score = _jaccard(q_tokens, _title_tokens(c))
        if score > best_score:
            best_score, best = score, c
    return best if best_score >= threshold else None


# Tipos de relación que usan Scene o Event como extremo (candidatos a fuzzy match).
_SCENE_REL_TYPES = {"PRESENT_IN", "TAKES_PLACE_IN", "FOLLOWS"}
_EVENT_REL_TYPES = {"PARTICIPATES_IN"}


# Tipos de relación que requieren nodos de cada etiqueta concreta.
_REL_SPECS: dict[str, tuple[str, str, str, str]] = {
    "APPEARS_IN":     ("Character", "Story",     "name", "title"),
    "KNOWS":          ("Character", "Character", "name", "name"),
    "INVESTIGATES":   ("Character", "Crime",     "name", "name"),
    "USES":           ("Character", "Object",    "name", "name"),
    "FOUND_AT":       ("Object",    "Location",  "name", "name"),
    "LIVES_AT":       ("Character", "Location",  "name", "name"),
    "PRESENT_IN":     ("Character", "Scene",     "name", "title"),
    "TAKES_PLACE_IN": ("Scene",     "Location",  "title", "name"),
    "FOLLOWS":        ("Scene",     "Scene",     "title", "title"),
    "BASED_ON":       ("Deduction", "Object",    "observation", "name"),
    "LEADS_TO":       ("Deduction", "Deduction", "observation", "observation"),
    "PARTICIPATES_IN":("Character", "Event",     "name", "name"),
    "OCCURS_IN":      ("Crime",     "Location",  "name", "name"),
    "MENTIONS":       ("Chunk",     "Character", "id",   "name"),
}


class Neo4jManager:
    """Gestiona la conexión y las operaciones CRUD con Neo4j."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_user, self.settings.neo4j_password),
        )

    def close(self) -> None:
        """Cierra la conexión con Neo4j."""
        self.driver.close()

    def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Ejecuta una consulta Cypher y devuelve los resultados."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def create_constraints(self) -> None:
        """Crea constraints de unicidad e índices de búsqueda para el modelo de datos."""
        constraints = [
            "CREATE CONSTRAINT story_title IF NOT EXISTS FOR (s:Story) REQUIRE s.title IS UNIQUE",
            "CREATE CONSTRAINT character_name IF NOT EXISTS FOR (c:Character) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT location_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX crime_type IF NOT EXISTS FOR (c:Crime) ON (c.type)",
            "CREATE INDEX crime_name IF NOT EXISTS FOR (c:Crime) ON (c.name)",
            "CREATE INDEX object_name IF NOT EXISTS FOR (o:Object) ON (o.name)",
            "CREATE INDEX event_name IF NOT EXISTS FOR (e:Event) ON (e.name)",
            "CREATE INDEX scene_story IF NOT EXISTS FOR (s:Scene) ON (s.sequence_order)",
            "CREATE INDEX scene_title IF NOT EXISTS FOR (s:Scene) ON (s.title)",
            "CREATE INDEX event_id IF NOT EXISTS FOR (e:Event) ON (e.id)",
            "CREATE INDEX deduction_observation IF NOT EXISTS FOR (d:Deduction) ON (d.observation)",
        ]
        for statement in constraints + indexes:
            try:
                self.execute_query(statement)
            except Exception as exc:
                logger.debug("Constraint/índice ya existe o error menor: %s", exc)

    def create_vector_index(self) -> None:
        """Crea el índice vectorial coseno y el índice fulltext sobre los chunks."""
        vector_query = f"""
        CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
        FOR (c:Chunk) ON c.embedding
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {self.settings.embedding_dimensions},
            `vector.similarity_function`: 'cosine'
        }}}}
        """
        fulltext_query = """
        CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
        FOR (c:Chunk) ON EACH [c.text]
        """
        for query, name in [(vector_query, "vectorial"), (fulltext_query, "fulltext")]:
            try:
                self.execute_query(query)
                logger.info("Índice %s creado o ya existía.", name)
            except Exception as exc:
                logger.warning("Error creando índice %s: %s", name, exc)

    def store_story(self, title: str, collection: str) -> None:
        """Crea o actualiza un nodo Story."""
        try:
            self.execute_query(
                """
                MERGE (s:Story {title: $title})
                ON CREATE SET s.collection = $collection
                """,
                {"title": title, "collection": collection},
            )
        except Exception as exc:
            logger.error("Error almacenando Story '%s': %s", title, exc)

    def store_chunk(self, chunk_data: dict, story_title: str) -> None:
        """Crea o actualiza un nodo Chunk y lo vincula a su Story."""
        chunk_id = f"{_sanitize_id(story_title)}_chunk_{chunk_data['chunk_index']}"
        chunk_data["id"] = chunk_id
        try:
            self.execute_query(
                """
                MATCH (s:Story {title: $story_title})
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text        = $text,
                    c.embedding   = $embedding,
                    c.chunk_index = $chunk_index,
                    c.token_count = $token_count,
                    c.position    = $position,
                    c.char_start  = $char_start,
                    c.char_end    = $char_end
                MERGE (s)-[:HAS_CHUNK]->(c)
                """,
                {
                    "story_title": story_title,
                    "chunk_id": chunk_id,
                    "text": chunk_data.get("text", ""),
                    "embedding": chunk_data.get("embedding", []),
                    "chunk_index": chunk_data.get("chunk_index", 0),
                    "token_count": chunk_data.get("token_count", 0),
                    "position": chunk_data.get("position", ""),
                    "char_start": chunk_data.get("char_start", 0),
                    "char_end": chunk_data.get("char_end", 0),
                },
            )
        except Exception as exc:
            logger.error("Error almacenando Chunk %s: %s", chunk_id, exc)

    def store_entities(self, entities: dict, story_title: str) -> None:
        """Almacena todas las entidades extraídas de un relato en Neo4j."""
        story_id = _sanitize_id(story_title)

        # Characters MERGE por nombre + relación APPEARS_IN
        for char in entities.get("characters", []):
            try:
                self.execute_query(
                    """
                    MERGE (c:Character {name: $name})
                    ON CREATE SET c.occupation  = $occupation,
                                  c.description = $description
                    SET c.aliases = $aliases
                    WITH c
                    MATCH (s:Story {title: $story_title})
                    MERGE (c)-[:APPEARS_IN]->(s)
                    """,
                    {
                        "name": char.get("name", ""),
                        "aliases": char.get("aliases", []),
                        "occupation": char.get("occupation", ""),
                        "description": char.get("description", ""),
                        "story_title": story_title,
                    },
                )
            except Exception as exc:
                logger.error("Error almacenando Character '%s': %s", char.get("name"), exc)

        # Locations MERGE por nombre
        for loc in entities.get("locations", []):
            try:
                self.execute_query(
                    """
                    MERGE (l:Location {name: $name})
                    SET l.type        = $type,
                        l.description = $description
                    """,
                    {
                        "name": loc.get("name", ""),
                        "type": loc.get("type", "unknown"),
                        "description": loc.get("description", ""),
                    },
                )
            except Exception as exc:
                logger.error("Error almacenando Location '%s': %s", loc.get("name"), exc)

        # Crimes → MERGE por name (único dentro de cada relato por diseño del LLM)
        for i, crime in enumerate(entities.get("crimes", [])):
            crime_name = crime.get("name", "") or f"{story_id}_crime_{i}"
            try:
                self.execute_query(
                    """
                    MERGE (cr:Crime {name: $name})
                    SET cr.type        = $type,
                        cr.description = $description,
                        cr.motive      = $motive,
                        cr.story_title = $story_title
                    """,
                    {
                        "name": crime_name,
                        "type": crime.get("type", "other"),
                        "description": crime.get("description", ""),
                        "motive": crime.get("motive", ""),
                        "story_title": story_title,
                    },
                )
            except Exception as exc:
                logger.error("Error almacenando Crime '%s': %s", crime_name, exc)

        # Objects MERGE por nombre (pueden compartirse entre relatos)
        for obj in entities.get("objects", []):
            try:
                self.execute_query(
                    """
                    MERGE (o:Object {name: $name})
                    SET o.type        = $type,
                        o.description = $description
                    """,
                    {
                        "name": obj.get("name", ""),
                        "type": obj.get("type", "other"),
                        "description": obj.get("description", ""),
                    },
                )
            except Exception as exc:
                logger.error("Error almacenando Object '%s': %s", obj.get("name"), exc)

        # Deductions CREATE con id único (no tienen nombre canónico)
        for i, ded in enumerate(entities.get("deductions", [])):
            ded_id = f"{story_id}_deduction_{i}"
            try:
                self.execute_query(
                    """
                    MERGE (d:Deduction {id: $id})
                    SET d.observation = $observation,
                        d.inference   = $inference,
                        d.conclusion  = $conclusion,
                        d.method      = $method,
                        d.story_title = $story_title
                    """,
                    {
                        "id": ded_id,
                        "observation": ded.get("observation", ""),
                        "inference": ded.get("inference", ""),
                        "conclusion": ded.get("conclusion", ""),
                        "method": ded.get("method", "logical"),
                        "story_title": story_title,
                    },
                )
            except Exception as exc:
                logger.error("Error almacenando Deduction %s: %s", ded_id, exc)

        # Scenes id único + relación BELONGS_TO con el Story
        for i, scene in enumerate(entities.get("scenes", [])):
            scene_id = f"{story_id}_scene_{i}"
            try:
                self.execute_query(
                    """
                    MERGE (sc:Scene {id: $id})
                    SET sc.title          = $title,
                        sc.description    = $description,
                        sc.sequence_order = $sequence_order
                    WITH sc
                    MATCH (s:Story {title: $story_title})
                    MERGE (sc)-[:BELONGS_TO]->(s)
                    """,
                    {
                        "id": scene_id,
                        "title": scene.get("title", ""),
                        "description": scene.get("description", ""),
                        "sequence_order": scene.get("sequence_order", i),
                        "story_title": story_title,
                    },
                )
            except Exception as exc:
                logger.error("Error almacenando Scene %s: %s", scene_id, exc)

        # Events id compuesto para evitar colisiones entre relatos
        for i, event in enumerate(entities.get("events", [])):
            event_id = f"{story_id}_event_{i}"
            try:
                self.execute_query(
                    """
                    MERGE (e:Event {id: $id})
                    SET e.name        = $name,
                        e.description = $description,
                        e.story_title = $story_title
                    """,
                    {
                        "id": event_id,
                        "name": event.get("name", ""),
                        "description": event.get("description", ""),
                        "story_title": story_title,
                    },
                )
            except Exception as exc:
                logger.error("Error almacenando Event %s: %s", event_id, exc)

    def store_relationships(
        self, relationships: list[dict], story_title: str = ""
    ) -> None:
        """Crea las relaciones entre entidades en Neo4j.

        Args:
            relationships: Lista de relaciones extraídas.
            story_title: Título del relato, usado para acotar el fuzzy matching
                         de escenas y eventos al relato correcto.
        """
        # Pre-cargar títulos de Scene y Event del relato para fuzzy matching.
        # Acotar por story_title evita que una escena de otro relato con nombre
        # similar absorba la relación equivocada.
        has_scene_rels = any(r.get("relationship_type") in _SCENE_REL_TYPES for r in relationships)
        has_event_rels = any(r.get("relationship_type") in _EVENT_REL_TYPES for r in relationships)

        scene_titles: set[str] = set()
        event_names: set[str] = set()

        if has_scene_rels:
            query = (
                "MATCH (s:Scene)-[:BELONGS_TO]->(:Story {title: $t}) RETURN s.title AS title"
                if story_title else
                "MATCH (s:Scene) WHERE s.title IS NOT NULL RETURN s.title AS title"
            )
            scene_titles = {
                row["title"]
                for row in self.execute_query(query, {"t": story_title} if story_title else {})
                if row["title"]
            }

        if has_event_rels:
            query = (
                "MATCH (e:Event {story_title: $t}) WHERE e.name IS NOT NULL RETURN e.name AS name"
                if story_title else
                "MATCH (e:Event) WHERE e.name IS NOT NULL RETURN e.name AS name"
            )
            event_names = {
                row["name"]
                for row in self.execute_query(query, {"t": story_title} if story_title else {})
                if row["name"]
            }

        for rel in relationships:
            rel_type = rel.get("relationship_type", "")
            source = rel.get("source_name", "")
            target = rel.get("target_name", "")
            props = rel.get("properties", {})

            spec = _REL_SPECS.get(rel_type)
            if spec is None:
                logger.debug("Tipo de relación desconocido '%s', omitido.", rel_type)
                continue

            src_label, tgt_label, src_key, tgt_key = spec

            # Fuzzy matching para Scene
            if rel_type in _SCENE_REL_TYPES and scene_titles:
                if src_label == "Scene" and src_key == "title":
                    matched = _fuzzy_match(source, scene_titles)
                    if matched and matched != source:
                        logger.debug("Fuzzy Scene src: %r -> %r", source, matched)
                        source = matched
                if tgt_label == "Scene" and tgt_key == "title":
                    matched = _fuzzy_match(target, scene_titles)
                    if matched and matched != target:
                        logger.debug("Fuzzy Scene tgt: %r -> %r", target, matched)
                        target = matched

            # Fuzzy matching para Event
            if rel_type in _EVENT_REL_TYPES and event_names:
                if tgt_label == "Event" and tgt_key == "name":
                    matched = _fuzzy_match(target, event_names)
                    if matched and matched != target:
                        logger.debug("Fuzzy Event tgt: %r -> %r", target, matched)
                        target = matched

            query = f"""
            MATCH (a:{src_label} {{{src_key}: $source}})
            MATCH (b:{tgt_label} {{{tgt_key}: $target}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += $props
            """
            try:
                self.execute_query(
                    query,
                    {"source": source, "target": target, "props": props},
                )
            except Exception as exc:
                logger.error(
                    "Error creando relación %s (%s -> %s): %s",
                    rel_type, source, target, exc,
                )

    def link_chunk_to_entities(
        self, chunk_id: str, entity_names: list[str]
    ) -> None:
        """Crea relaciones MENTIONS entre un Chunk y las entidades que menciona."""
        for name in entity_names:
            try:
                self.execute_query(
                    """
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (e {name: $entity_name})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    {"chunk_id": chunk_id, "entity_name": name},
                )
            except Exception as exc:
                logger.error(
                    "Error creando MENTIONS entre %s y '%s': %s",
                    chunk_id, name, exc,
                )

    def setup_database(self) -> None:
        """Crea constraints e índices. Llamar una sola vez al inicio del pipeline."""
        logger.info("Configurando base de datos Neo4j...")
        self.create_constraints()
        self.create_vector_index()
        logger.info("Base de datos configurada.")

    def clear_database(self) -> None:
        """Elimina todos los nodos y relaciones del grafo."""
        logger.warning("Borrando toda la base de datos Neo4j...")
        self.execute_query("MATCH (n) DETACH DELETE n")
        logger.info("Base de datos limpiada.")

    def get_stats(self) -> dict[str, int]:
        """Devuelve el número de nodos por etiqueta."""
        rows = self.execute_query(
            """
            MATCH (n)
            RETURN labels(n)[0] AS label, count(*) AS count
            ORDER BY count DESC
            """
        )
        return {row["label"]: row["count"] for row in rows if row["label"]}

    # ------------------------------------------------------------------
    # Schema (clase a cambiar)
    # ------------------------------------------------------------------

    def get_schema(self) -> dict[str, Any]:
        """Obtiene el esquema del grafo (nodos, propiedades y relaciones)."""
        node_props_query = """
        CALL db.schema.nodeTypeProperties()
        YIELD nodeType, propertyName, propertyTypes
        WITH nodeType, collect({property: propertyName, type: propertyTypes[0]}) as properties
        RETURN {labels: nodeType, properties: properties} AS output
        """
        rel_props_query = """
        CALL db.schema.relTypeProperties()
        YIELD relType, propertyName, propertyTypes
        WITH relType, collect({property: propertyName, type: propertyTypes[0]}) as properties
        RETURN {type: relType, properties: properties} AS output
        """
        rel_query = """
        CALL db.schema.visualization()
        YIELD nodes, relationships
        UNWIND relationships as rel
        RETURN {start: startNode(rel).name, type: type(rel), end: endNode(rel).name} AS output
        """
        try:
            node_props = self.execute_query(node_props_query)
            rel_props = self.execute_query(rel_props_query)
            relationships = self.execute_query(rel_query)
            return {
                "node_props": {
                    item["output"]["labels"]: item["output"]["properties"]
                    for item in node_props
                },
                "rel_props": {
                    item["output"]["type"]: item["output"]["properties"]
                    for item in rel_props
                },
                "relationships": [item["output"] for item in relationships],
            }
        except Exception as exc:
            logger.error("Error obteniendo schema: %s", exc)
            return {"node_props": {}, "rel_props": {}, "relationships": []}

    @staticmethod
    def format_schema(schema: dict[str, Any]) -> str:
        """Formatea el schema del grafo como texto para usar en prompts LLM.

        Args:
            schema: Dict devuelto por get_schema().

        Returns:
            String con nodos, propiedades y relaciones en formato legible.
        """
        def _fmt_props(props: list[dict]) -> str:
            return ", ".join(f"{p['property']}: {p['type']}" for p in props)

        node_lines = [
            f"{label} {{{_fmt_props(props)}}}"
            for label, props in schema["node_props"].items()
        ]
        rel_prop_lines = [
            f"{rel_type} {{{_fmt_props(props)}}}"
            for rel_type, props in schema["rel_props"].items()
        ]
        rel_lines = [
            f"(:{r['start']})-[:{r['type']}]->(:{r['end']})"
            for r in schema["relationships"]
        ]

        return "\n".join([
            "Node labels and properties:",
            "\n".join(node_lines),
            "\nRelationship types and properties:",
            "\n".join(rel_prop_lines),
            "\nThe relationships:",
            "\n".join(rel_lines),
        ])
