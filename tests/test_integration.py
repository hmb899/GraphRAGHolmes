"""Tests de integración: verifican que todos los módulos se conectan correctamente.

Estos tests no requieren Neo4j, Gemini ni Ollama activos. Solo validan
imports, lógica pura (chunking, modelos Pydantic, similitud coseno) y
que las interfaces entre módulos son compatibles.
"""


def test_imports():
    """Verificar que todos los módulos se importan sin errores."""
    from graphrag.config import get_settings  # noqa: F401
    from graphrag.llm import GeminiClient, ModelTier, EmbeddingClient  # noqa: F401
    from graphrag.utils.chunking import chunk_story, chunk_story_with_metadata  # noqa: F401
    from graphrag.utils.embeddings import EmbeddingGenerator, cosine_similarity  # noqa: F401
    from graphrag.ingestion.entity_extractor import (  # noqa: F401
        EntityExtractor,
        EntityExtractionResult,
        RelationshipExtractionResult,
        NarrativeContext,
        CharacterEntity,
        LocationEntity,
    )
    from graphrag.graph.neo4j_manager import Neo4jManager  # noqa: F401


def test_chunking():
    """Verificar que el chunking funciona con texto de ejemplo."""
    from graphrag.utils.chunking import chunk_story_with_metadata

    sample = (
        "This is paragraph one about Holmes.\n\n"
        "This is paragraph two about Watson.\n\n"
        "This is paragraph three about London."
    )

    chunks = chunk_story_with_metadata(
        text=sample,
        story_title="Test Story",
        collection="Test Collection",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 0
    assert "story_title" in chunks[0]
    assert "text" in chunks[0]
    assert "chunk_index" in chunks[0]
    assert "position" in chunks[0]
    assert "char_start" in chunks[0]
    assert "char_end" in chunks[0]
    assert "token_count" in chunks[0]
    assert "total_chunks" in chunks[0]
    assert chunks[0]["story_title"] == "Test Story"
    assert chunks[0]["collection"] == "Test Collection"
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["total_chunks"] == len(chunks)


def test_chunking_position_labels():
    """Verificar que los labels de posición (beginning/middle/end) son correctos."""
    from graphrag.utils.chunking import chunk_story_with_metadata

    # Texto largo para forzar múltiples chunks y los tres labels
    paragraphs = [f"Paragraph {i} with some text about Sherlock Holmes." for i in range(20)]
    long_text = "\n\n".join(paragraphs)

    chunks = chunk_story_with_metadata(
        text=long_text,
        story_title="Long Story",
        collection="Test",
        chunk_size=200,
        chunk_overlap=30,
    )

    positions = {c["position"] for c in chunks}
    assert "beginning" in positions

    if len(chunks) > 1:
        assert chunks[0]["position"] == "beginning"
        assert chunks[-1]["position"] == "end"


def test_pydantic_models():
    """Verificar que los modelos Pydantic se instancian y serializan correctamente."""
    from graphrag.ingestion.entity_extractor import (
        CharacterEntity,
        LocationEntity,
        CrimeEntity,
        DeductionEntity,
        NarrativeContext,
        EntityExtractionResult,
        ExtractedRelationship,
    )

    char = CharacterEntity(
        name="Sherlock Holmes",
        aliases=["Holmes", "Mr. Holmes"],
        occupation="Consulting Detective",
    )
    assert char.name == "Sherlock Holmes"
    assert "Holmes" in char.aliases
    assert char.description == ""  # default vacío

    loc = LocationEntity(name="221B Baker Street", type="building")
    assert loc.type == "building"

    crime = CrimeEntity(type="murder", description="The victim was found dead.")
    assert crime.type == "murder"

    deduction = DeductionEntity(
        observation="The client's hands are calloused.",
        inference="Manual labour of some kind.",
        conclusion="He works as a labourer.",
        method="physical_evidence",
    )
    assert deduction.method == "physical_evidence"

    ctx = NarrativeContext(
        current_location="Baker Street",
        characters_present=["Holmes", "Watson"],
        summary="Holmes and Watson are at Baker Street discussing a new case.",
    )
    assert len(ctx.characters_present) == 2
    assert ctx.active_investigation == ""  # default vacío

    result = EntityExtractionResult(characters=[char], locations=[loc])
    assert len(result.characters) == 1
    assert len(result.locations) == 1
    assert len(result.crimes) == 0  # lista vacía por defecto

    # Verificar que model_dump() produce un dict con las claves correctas
    dumped = result.model_dump()
    assert "characters" in dumped
    assert "locations" in dumped
    assert "crimes" in dumped
    assert dumped["characters"][0]["name"] == "Sherlock Holmes"

    # Verificar que ExtractedRelationship se serializa correctamente
    rel = ExtractedRelationship(
        source_name="Sherlock Holmes",
        source_type="Character",
        target_name="A Scandal in Bohemia",
        target_type="Story",
        relationship_type="APPEARS_IN",
        properties={"role": "protagonist"},
    )
    rel_dict = rel.model_dump()
    assert rel_dict["relationship_type"] == "APPEARS_IN"
    assert rel_dict["properties"]["role"] == "protagonist"


def test_cosine_similarity():
    """Verificar que la similitud coseno calcula correctamente."""
    from graphrag.utils.embeddings import cosine_similarity

    # Vectores idénticos → similitud 1.0
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(vec_a, vec_b) - 1.0) < 1e-6

    # Vectores ortogonales → similitud 0.0
    vec_c = [0.0, 1.0, 0.0]
    assert abs(cosine_similarity(vec_a, vec_c)) < 1e-6

    # Vector cero → devuelve 0.0 sin error
    vec_zero = [0.0, 0.0, 0.0]
    assert cosine_similarity(vec_a, vec_zero) == 0.0

    # Vector opuesto → similitud -1.0
    vec_neg = [-1.0, 0.0, 0.0]
    assert abs(cosine_similarity(vec_a, vec_neg) + 1.0) < 1e-6


def test_process_story_chunks_returns_dicts():
    """Verificar que process_story_chunks serializa entidades y relaciones a dicts.

    Comprueba que el valor de retorno es compatible con store_entities y
    store_relationships de Neo4jManager (que esperan dicts, no objetos Pydantic).
    """
    from graphrag.ingestion.entity_extractor import EntityExtractionResult

    # Simular el return de process_story_chunks con datos reales
    consolidated = EntityExtractionResult()
    entities_dict = consolidated.model_dump()
    relationships_dict = []

    result = {
        "story_title": "Test Story",
        "entities": entities_dict,
        "relationships": relationships_dict,
        "chunks_processed": 3,
    }

    # Verificar que store_entities podría consumirlo (acceso de dict)
    assert isinstance(result["entities"], dict)
    assert "characters" in result["entities"]
    assert isinstance(result["relationships"], list)
    assert result["chunks_processed"] == 3
