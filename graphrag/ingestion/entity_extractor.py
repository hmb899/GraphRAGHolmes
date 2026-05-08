import logging
import re

from pydantic import BaseModel, Field
from tqdm import tqdm

from ..config import get_settings
from ..llm.embedding_client import EmbeddingClient
from ..llm.gemini_client import GeminiClient, ModelTier
from ..utils.embeddings import cosine_similarity

logger = logging.getLogger(__name__)


class CharacterEntity(BaseModel):
    """Un personaje mencionado en el fragmento."""

    name: str = Field(..., description="Nombre del personaje exactamente como aparece en el texto")
    # No rellenar, se reconstruye automáticamente en entity resolution.
    aliases: list[str] = Field(
        default_factory=list,
        description="Leave empty. Do not populate. Auto-generated during entity resolution.",
    )
    occupation: str = Field(default="", description="Ocupación o rol")
    description: str = Field(
        default="", description="Descripción breve del personaje en este contexto"
    )


class LocationEntity(BaseModel):
    """Un lugar mencionado en el fragmento."""

    name: str = Field(..., description="Nombre del lugar")
    type: str = Field(
        default="unknown",
        description="Tipo: city/building/street/region/country",
    )
    description: str = Field(default="", description="Descripción del lugar")


class CrimeEntity(BaseModel):
    """Un crimen o misterio central del relato."""

    name: str = Field(
        ...,
        description="Short unique name for this crime within the story, e.g. 'Murder of John Straker', 'Disappearance of Silver Blaze', 'Theft of the Blue Carbuncle'",
    )
    type: str = Field(
        ...,
        description="Tipo: murder/theft/blackmail/fraud/disappearance/assault/other",
    )
    description: str = Field(default="", description="Descripción del crimen")
    motive: str = Field(default="", description="Motivo si se conoce")


class ObjectEntity(BaseModel):
    """Un objeto significativo para la trama."""

    name: str = Field(..., description="Nombre del objeto")
    type: str = Field(
        default="other",
        description="Tipo: weapon/document/jewel/animal/clothing/other",
    )
    description: str = Field(default="", description="Descripción y relevancia")


class DeductionEntity(BaseModel):
    """Una cadena de razonamiento deductivo de Holmes."""

    observation: str = Field(..., description="Lo que Holmes observa")
    inference: str = Field(default="", description="El razonamiento que hace")
    conclusion: str = Field(default="", description="La conclusión a la que llega")
    method: str = Field(
        default="logical",
        description="Método: physical_evidence/behavioral/forensic/logical",
    )


class SceneEntity(BaseModel):
    """Una unidad narrativa (escena) del relato."""

    title: str = Field(..., description="Título descriptivo breve de la escena")
    description: str = Field(default="", description="Qué ocurre en la escena")
    sequence_order: int = Field(
        default=0, description="Orden en la secuencia narrativa"
    )


class EventEntity(BaseModel):
    """Un evento clave de la trama."""

    name: str = Field(..., description="Nombre del evento")
    description: str = Field(default="", description="Descripción del evento")


class ExtractedRelationship(BaseModel):
    """Una relación extraída entre dos entidades del grafo."""

    source_name: str = Field(..., description="Nombre de la entidad origen")
    source_type: str = Field(
        ..., description="Tipo de la entidad origen (Character, Location, etc.)"
    )
    target_name: str = Field(..., description="Nombre de la entidad destino")
    target_type: str = Field(..., description="Tipo de la entidad destino")
    relationship_type: str = Field(
        ...,
        description="Tipo de relación (APPEARS_IN, KNOWS, INVESTIGATES, LOCATED_IN, etc.)",
    )
    properties: dict = Field(
        default_factory=dict,
        description="Propiedades de la relación (role, relationship_type, etc.)",
    )


class EntityExtractionResult(BaseModel):
    """Resultado completo de la extracción de entidades de un chunk."""

    characters: list[CharacterEntity] = Field(default_factory=list)
    locations: list[LocationEntity] = Field(default_factory=list)
    crimes: list[CrimeEntity] = Field(default_factory=list)
    objects: list[ObjectEntity] = Field(default_factory=list)
    deductions: list[DeductionEntity] = Field(default_factory=list)
    scenes: list[SceneEntity] = Field(default_factory=list)
    events: list[EventEntity] = Field(default_factory=list)


class RelationshipExtractionResult(BaseModel):
    """Resultado de la extracción de relaciones entre entidades."""

    relationships: list[ExtractedRelationship] = Field(default_factory=list)



class NarrativeContext(BaseModel):
    """Estado narrativo acumulado del relato a lo largo del sliding context."""

    current_location: str = Field(
        default="", description="Dónde se encuentran los personajes ahora"
    )
    characters_present: list[str] = Field(
        default_factory=list,
        description="Personajes actualmente en escena",
    )
    recent_events: list[str] = Field(
        default_factory=list,
        description="Últimos 3-5 eventos relevantes",
    )
    active_investigation: str = Field(
        default="", description="Caso o crimen que se está investigando"
    )
    unresolved_references: list[str] = Field(
        default_factory=list,
        description="Referencias sin resolver (pronombres, alias ambiguos)",
    )
    summary: str = Field(
        default="",
        description="Resumen narrativo de 2-3 frases del contexto hasta este punto",
    )
    known_crime_names: list[str] = Field(
        default_factory=list,
        description="Nombres exactos de crímenes ya nombrados en chunks anteriores",
    )


def _normalize_name(name: str) -> str:
    """Normaliza un nombre para comparación: lowercase y espacios simples."""
    return " ".join(name.lower().strip().split())


# Títulos y honoríficos que no forman parte del nombre propio
_TITLE_PREFIXES: frozenset[str] = frozenset({
    "mr", "mrs", "ms", "miss", "mister", "mistress", "dr", "doctor",
    "colonel", "col", "major", "major-general", "general", "captain", "capt",
    "lieutenant", "admiral", "sergeant", "inspector",
    "professor", "prof", "sir", "lady", "lord",
    "young", "old", "the", "a", "an", "née", "esq",
})


def _strip_titles(name: str) -> str:
    """Elimina títulos y honoríficos de un nombre para comparación."""
    tokens = _normalize_name(name).replace(",", "").replace(".", "").split()
    core = [t for t in tokens if t not in _TITLE_PREFIXES]
    return " ".join(core) if core else _normalize_name(name)


def _is_bare_name(original: str, stripped: str) -> bool:
    """Devuelve True si el nombre original no contiene títulos (es el nombre desnudo)."""
    bare = _normalize_name(original).replace(",", "").replace(".", "")
    return bare == stripped


# Máximo de tokens adicionales permitidos en el nombre más largo
# al comparar por suffix/prefix. Evita que nombres compuestos extraídos
# por el LLM (p. ej. "Sherlock Holmes, detective for the King") actúen
# como puentes entre grupos distintos.
_MAX_EXTRA_TOKENS = 2


def _is_word_suffix(short: str, long: str) -> bool:
    """True si 'short' coincide exactamente con los últimos tokens de 'long'
    y la diferencia de longitud no supera _MAX_EXTRA_TOKENS.

    Ejemplo: 'holmes' es suffix de 'sherlock holmes' (diff=1)
             'norton' NO es suffix de 'irene norton adler' (último token='adler')
             'holmes' NO es suffix de 'sherlock holmes detective for the king' (diff=5)
    """
    s = short.split()
    l = long.split()
    return (
        0 < len(s) <= len(l)
        and len(l) - len(s) <= _MAX_EXTRA_TOKENS
        and l[-len(s):] == s
    )


def _is_word_prefix(short: str, long: str) -> bool:
    """True si 'short' coincide exactamente con los primeros tokens de 'long'
    y la diferencia de longitud no supera _MAX_EXTRA_TOKENS.

    Ejemplo: 'king' es prefix de 'king of bohemia' (diff=2)
             'sherlock holmes' NO es prefix de 'sherlock holmes detective for...' (diff>2)
    """
    s = short.split()
    l = long.split()
    return (
        0 < len(s) <= len(l)
        and len(l) - len(s) <= _MAX_EXTRA_TOKENS
        and l[:len(s)] == s
    )


_MILITARY_RANKS: frozenset[str] = frozenset({"major-general", "general", "colonel", "admiral", "lieutenant"})
_FEMALE_PERSONAL: frozenset[str] = frozenset({"miss", "ms"})
_MALE_OR_PROFESSIONAL: frozenset[str] = frozenset({"dr", "mr", "mister", "sir", "lord", "professor", "prof",})


def _rule_based_same_character(name_a: str, name_b: str) -> bool:
    """Devuelve True si dos cadenas de nombre probablemente refieren al mismo personaje."""
    a = _strip_titles(name_a)
    b = _strip_titles(name_b)
    if not a or not b:
        return False

    # Ej: "Mrs. Watson" ≠ "Watson", "Mrs. Straker" ≠ "John Straker",
    #     "Mrs. Henry Baker" ≠ "Mr. Henry Baker"
    tokens_a = _normalize_name(name_a).replace(",", "").replace(".", "").split()
    tokens_b = _normalize_name(name_b).replace(",", "").replace(".", "").split()
    title_a = tokens_a[0] if tokens_a else ""
    title_b = tokens_b[0] if tokens_b else ""

    mrs_a = title_a == "mrs"
    mrs_b = title_b == "mrs"
    if mrs_a != mrs_b:
        return False

    # Título femenino personal (Miss/Ms) no puede coincidir con un nombre que
    # tenga título profesional/masculino (Dr./Mr./Sir...) Y además nombre de pila.
    # Evita que "Miss Roylott" (alias de Helen Stoner) se fusione con
    # "Dr. Grimesby Roylott" por compartir el apellido.
    if title_a in _FEMALE_PERSONAL and title_b in _MALE_OR_PROFESSIONAL:
        if len(b.split()) >= 2:  # b tiene nombre de pila, persona distinta
            return False
    if title_b in _FEMALE_PERSONAL and title_a in _MALE_OR_PROFESSIONAL:
        if len(a.split()) >= 2:
            return False

    if a == b:
        if (
            len(a.split()) == 1
            and not _is_bare_name(name_a, a)
            and not _is_bare_name(name_b, b)
            and _normalize_name(name_a) != _normalize_name(name_b)
        ):
            return False
        return True

    a_mil_single = len(a.split()) == 1 and title_a in _MILITARY_RANKS
    b_mil_single = len(b.split()) == 1 and title_b in _MILITARY_RANKS

    if _is_word_suffix(a, b) or _is_word_suffix(b, a):
        if (a_mil_single and len(b.split()) >= 2) or (b_mil_single and len(a.split()) >= 2):
            return False
        return True
    if _is_word_prefix(a, b) or _is_word_prefix(b, a):
        if (a_mil_single and len(b.split()) >= 2) or (b_mil_single and len(a.split()) >= 2):
            return False
        return True

    return False


def _merge_entity_group(group: list, name_attr: str = "name", prefer_proper_name: bool = False) -> object:
    """Fusiona un grupo de entidades en una sola instancia."""
    if len(group) == 1:
        return group[0]

    if prefer_proper_name:
        def _proper_name_score(name: str) -> tuple:
            stripped = _strip_titles(name)
            n_tokens = len(stripped.split()) if stripped else 0
            # Primero: más tokens de nombre real; luego: nombre stripped más largo;
            # luego: nombre original más corto (menos honoríficos superfluos)
            return (n_tokens, len(stripped), -len(name))
        canonical = max(group, key=lambda e: _proper_name_score(getattr(e, name_attr, "")))
    else:
        canonical = max(group, key=lambda e: len(getattr(e, name_attr, "")))

    updates: dict = {}

    # Construye aliases a partir de los nombres reales con que apareció
    # el personaje en distintos chunks — no del LLM.
    if hasattr(canonical, "aliases"):
        all_aliases: set[str] = set()
        for entity in group:
            all_aliases.add(getattr(entity, name_attr, ""))
        all_aliases.discard(getattr(canonical, name_attr))
        updates["aliases"] = sorted(all_aliases)

    if hasattr(canonical, "description"):
        descs = {
            getattr(e, "description", "")
            for e in group
            if getattr(e, "description", "")
        }
        if descs:
            updates["description"] = " | ".join(sorted(descs))

    return canonical.model_copy(update=updates) if updates else canonical


def _build_canonical_map(resolved: EntityExtractionResult) -> dict[str, str]:
    """Construye un mapa alias_normalizado, nombre_canónico a partir de personajes resueltos."""
    mapping: dict[str, str] = {}
    for char in resolved.characters:
        for alias in char.aliases:
            mapping[_normalize_name(alias)] = char.name
    return mapping


def _rule_based_same_location(name_a: str, name_b: str) -> bool:
    """Detecta si dos nombres de lugar son el mismo sitio por contenimiento de tokens.

    Captura casos como:
      "Baker Street" ↔ "Baker Street lodgings"
      "Baker Street" ↔ "sitting-room at Baker Street"
      "Baker Street" ↔ "221B Baker Street"
      "Oxford Street" ↔ "Oxford Street, London"
    Requiere ≥2 tokens en el nombre más corto para evitar falsos positivos
    con nombres de una sola palabra ("London" ≠ "London Bridge").
    """
    a = _normalize_name(name_a)
    b = _normalize_name(name_b)
    if not a or not b:
        return False
    if a == b:
        return True

    def _tokens(s: str) -> set[str]:
        """Tokeniza eliminando puntuación para que 'street,' == 'street'."""
        return {t for t in re.sub(r"[^a-z0-9\s]", "", s).split() if t}

    tokens_a = _tokens(a)
    tokens_b = _tokens(b)
    shorter = tokens_a if len(tokens_a) <= len(tokens_b) else tokens_b
    longer  = tokens_b if len(tokens_a) <= len(tokens_b) else tokens_a
    return len(shorter) >= 2 and shorter.issubset(longer)


def _remap_relationships(relationships: list[ExtractedRelationship], canonical_map: dict[str, str])\
        -> list[ExtractedRelationship]:
    """Actualiza source_name y target_name de las relaciones con nombres canónicos."""
    remapped = []
    for rel in relationships:
        source = canonical_map.get(_normalize_name(rel.source_name), rel.source_name)
        target = canonical_map.get(_normalize_name(rel.target_name), rel.target_name)
        if source != rel.source_name or target != rel.target_name:
            rel = rel.model_copy(update={"source_name": source, "target_name": target})
        remapped.append(rel)
    return remapped


_GENERIC_PREFIXES: tuple[str, ...] = (
  "the ", "a ", "an ",
  "my ", "your ", "his ", "her ",
  "our ", "this ", "that ",
  "young ", "old ", "elderly ", "huge ", "large ", "little ",
  "local ", "native ", "poor ", "wandering ", "half-pay ",
)
_GENERIC_EXACT: frozenset[str] = frozenset({
  "i", "we", "he", "she", "they", "you", "it",
  "him", "her", "them", "us", "me", "myself",
  "doctor", "inspector", "narrator", "gentleman",
  "companion", "stranger", "visitor", "assistant",
  "husband", "wife", "maid", "porter", "lad",
  "coachman", "landlord", "landlady",
  "sister", "brother", "aunt", "uncle", "cousin",
  "mother", "father", "daughter", "son", "widow",
  "housekeeper", "butler", "footman", "driver", "guard",
  "blacksmith", "carpenter", "gardener", "gamekeeper",
  "groom", "cook", "nurse", "clerk", "constable",
  "madam", "madame",
  "band", "crowd", "gang", "mob",
})
_GENERIC_LOC_WORDS: frozenset[str] = frozenset({
  "room", "rooms", "corridor", "chamber", "chambers", "bedroom",
  "hall", "hallway", "passage", "passageway", "landing", "staircase",
  "stairs", "stair", "floor", "wing", "block", "building", "buildings",
  "door", "window", "wall", "ceiling", "roof", "garden", "lawn",
  "courtyard", "yard", "grounds", "area", "space", "place", "spot",
  "interior", "exterior", "outside", "inside",
})
_LOC_PREFIXES_GENERIC: tuple[str, ...] = (
  "the ", "a ", "an ", "your ", "my ", "his ", "her ", "our ",
  "this ", "that ", "some ", "another ", "next ", "other ",
)


class EntityExtractor:
    """Extrae entidades y relaciones de los relatos de Sherlock Holmes.

    Uso de modelos por coste:
    - LITE: sliding context, desambiguación LLM en entity resolution
    - FLASH: extracción de entidades y relaciones (alto volumen)
    """

    # Umbrales de similitud coseno para entity resolution de entidades NO-personaje
    # (Locations, Objects, Events, Scenes).
    # Para Characters se usa matching basado en reglas (_rule_based_same_character),
    # porque los embeddings semánticos de nombres propios cortos son
    # indistinguibles entre sí y provocan fusiones erróneas.
    _MERGE_THRESHOLD = 0.92
    _AMBIGUOUS_THRESHOLD = 0.80
    _BOTH_MUST_EXIST: frozenset = frozenset({"KNOWS", "INVESTIGATES", "PRESENT_IN", "PARTICIPATES_IN", "FOLLOWS"})
    _INVESTIGATES_PRIORITY: dict = {
        "perpetrator": 0, "victim": 1, "suspect": 2,"witness": 3,
        "client": 4, "investigator": 5
    }
    _KNOWS_PRIORITY: dict = {
        "enemy": 0, "family": 1, "friend": 2, "colleague": 3,
        "client": 4, "acquaintance": 5,
    }

    def __init__(self) -> None:
        self.gemini = GeminiClient()
        self.embeddings = EmbeddingClient()
        self.settings = get_settings()

    def _update_sliding_context(self, current_context: NarrativeContext, chunk_text: str,
                                extracted_entities: EntityExtractionResult) -> NarrativeContext:
        """Actualiza el contexto narrativo acumulativo tras procesar un chunk sin llamar al LLM."""

        chars = [c.name for c in extracted_entities.characters]
        locs  = [l.name for l in extracted_entities.locations]
        events = [e.name for e in extracted_entities.events]
        crimes = [c.name for c in extracted_entities.crimes]

        recent = (current_context.recent_events + events)[-5:]

        # Acumular nombres de crímenes ya asignados (evita que cada chunk invente uno nuevo).
        # Cap en 3 para que los crímenes de fondo extraídos en chunks tempranos no
        # desplacen al crimen central cuando aparezca.
        new_crime_names = [c.name for c in extracted_entities.crimes]
        known_crimes = list(dict.fromkeys(
            current_context.known_crime_names + new_crime_names
        ))[:3]

        return NarrativeContext(
            current_location=locs[0] if locs else current_context.current_location,
            characters_present=chars if chars else current_context.characters_present,
            recent_events=recent,
            active_investigation=crimes[0] if crimes else current_context.active_investigation,
            unresolved_references=[],
            known_crime_names=known_crimes,
        )

    def extract_entities(self, chunk_text: str, context: NarrativeContext | None = None,
                         story_title: str = "") -> EntityExtractionResult:
        """Extrae entidades tipadas de un fragmento de texto."""

        context_section = ""
        if context:
            present = (
                ", ".join(context.characters_present)
                if context.characters_present
                else "unknown"
            )
            crimes_line = ""
            if context.known_crime_names:
                crimes_line = (
                    f"Crimes already named (use EXACTLY these names if the same crime "
                    f"reappears — do NOT rename them): "
                    + ", ".join(f'"{n}"' for n in context.known_crime_names)
                    + "\n"
                )
            if context.summary or context.known_crime_names:
                context_section = (
                    f"NARRATIVE CONTEXT SO FAR:\n{context.summary}\n"
                    f"Characters present: {present}\n"
                    f"Current location: {context.current_location}\n"
                    f"{crimes_line}\n"
                )

        prompt = f"""You are an expert literary analyst specializing in the Sherlock Holmes canon by Arthur Conan Doyle.

Extract all entities from the following text fragment of the story "{story_title}".

{context_section}TEXT TO ANALYZE:
{chunk_text}

Extract:
- Characters: ONLY persons with a proper name or title+surname (e.g. "Holmes", "Mr. Sherlock Holmes", "Dr. Watson", "Miss Stoner", "Percy Armitage"). Extract the name EXACTLY as it appears in this fragment. Do NOT extract unnamed characters referred to only by role, description, or relationship (e.g. do NOT extract "young lady", "huge man", "housekeeper", "sister", "old doctor", "local blacksmith", "band of gipsies", "madam"). Each character gets one entry per fragment.
- Locations: named places with their type (city/building/street/region/country)
- Crimes: ONLY crimes that Holmes was hired to investigate. There are at most 2: the original mystery (what the client reported) and an ongoing threat to the client. RULES: (a) The central crime = what the CLIENT described when asking for help — extract it immediately even if it happened years ago (e.g. "Murder of Julia Stoner"). (b) If already named in context, use that EXACT name. (c) Do NOT extract a character's past criminal history that merely establishes their dangerous nature (e.g. beating a butler years ago = character background, not a crime to extract). (d) Do NOT extract deaths or crimes that happen TO the antagonist/criminal during the investigation — these are plot resolutions, not the crimes Holmes was hired to solve (e.g. if the villain dies at the end, that is NOT a crime to extract). (e) Do NOT extract suspicions, fears, or mental states. Name format: "Murder/Attempted Murder/Theft/Disappearance of [Victim]" — never use vague adjectives or literary quotes.
- Objects: ONLY objects that are clues, weapons, or pivotal to the plot. Ignore generic furniture, clothing, and incidental items.
- Deductions: EVERY step in Holmes's reasoning process. If Holmes makes a multi-step deduction (observation A → inference B → conclusion C), extract EACH step as a separate Deduction entry. Be thorough — extract ALL observations, inferences and conclusions, not just the final one. A typical Holmes investigation should yield 4-8 deductions per story.
- Scenes: narrative units (a scene changes when location, time, or present characters change significantly)
- Events: key plot events

Be thorough with named entities. For characters, list ALL proper name variants used in the text as aliases."""

        try:
            return self.gemini.structured_output(
                prompt=prompt,
                schema=EntityExtractionResult,
                model_tier=ModelTier.FLASH,
                system_instruction=(
                    "You are a knowledge graph entity extractor. "
                    "Extract entities precisely as they appear in the text."
                ),
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("Error extrayendo entidades: %s. Devolviendo resultado vacío.", exc)
            return EntityExtractionResult()

    def extract_relationships(self, chunk_text: str, entities: EntityExtractionResult,
                              context: NarrativeContext | None = None) -> RelationshipExtractionResult:
        """Extrae relaciones entre las entidades ya identificadas."""

        character_names = [c.name for c in entities.characters]
        location_names = [loc.name for loc in entities.locations]
        crime_names = [c.name for c in entities.crimes]
        object_names = [o.name for o in entities.objects]
        deduction_obs = [d.observation for d in entities.deductions]
        scene_titles = [s.title for s in entities.scenes]
        event_names = [e.name for e in entities.events]

        deduction_block = (
            "Deductions (identified by observation text):\n"
            + "\n".join(f"  - {obs}" for obs in deduction_obs)
        ) if deduction_obs else ""

        entities_lines = [
            f"Characters: {', '.join(character_names)}" if character_names else "",
            f"Locations: {', '.join(location_names)}" if location_names else "",
            f"Crimes: {', '.join(crime_names)}" if crime_names else "",
            f"Objects: {', '.join(object_names)}" if object_names else "",
            deduction_block,
            f"Scenes: {', '.join(scene_titles)}" if scene_titles else "",
            f"Events: {', '.join(event_names)}" if event_names else "",
        ]
        entities_section = "\n".join(line for line in entities_lines if line)

        prompt = f"""You are extracting relationships for a knowledge graph about Sherlock Holmes stories.

TEXT:
{chunk_text}

ENTITIES FOUND:
{entities_section}

Extract relationships between these entities. Valid relationship types (ONLY these, no others):
- APPEARS_IN: Character → Story (with role: protagonist/antagonist/client/witness/victim)
- KNOWS: Character → Character (with relationship_type: friend/enemy/colleague/family/acquaintance)
- INVESTIGATES: Character → Crime  ← source_name = EXACT character name, target_name = EXACT crime name from the list above (with role: investigator/suspect/perpetrator/victim). MANDATORY: if the Crimes list is non-empty and a character in the text is actively involved with that crime (investigating, committing, witnessing, or being victimised), you MUST extract this relationship. Do not omit it.
- OCCURS_IN: Crime → Location  ← source_name = EXACT crime name from the list above, target_name = EXACT location name
- USES: Character → Object (with context)
- FOUND_AT: Object → Location  ← use this when an object is located somewhere (NOT "LOCATED_IN" or "FOUND_IN")
- LIVES_AT: Character → Location  ← use this when a character resides at or operates from a location (home, office, lair)
- PRESENT_IN: Character → Scene  ← source_name = EXACT character name, target_name = EXACT scene title from the list above
- TAKES_PLACE_IN: Scene → Location  ← source_name = EXACT scene title from the list above (NOT "LOCATED_IN")
- FOLLOWS: Scene → Scene  ← both source_name and target_name = EXACT scene titles from the list above
- BASED_ON: Deduction → Object or Event  (source_name = the EXACT observation text of the deduction, as listed above)
- LEADS_TO: Deduction → Deduction  (source_name and target_name = the EXACT observation text of each deduction). IMPORTANT: if this chunk contains multiple deductions that form a logical chain (one observation leads Holmes to the next), you MUST connect them with LEADS_TO. A chain of 3 deductions should have 2 LEADS_TO relations. Do not leave deductions unconnected if they clearly follow from each other.
- PARTICIPATES_IN: Character → Event  ← source_name = EXACT character name, target_name = EXACT event name from the list above

CRITICAL NAMING RULES — failure to follow these will break the knowledge graph:
- For ALL relationships involving Characters, Locations, Crimes, Scenes, and Events: source_name and target_name MUST be taken verbatim from the ENTITIES FOUND list above. Do NOT use pronouns (he/she/they/him/her), possessives (my stepfather, her sister, his colleague), generic roles (the doctor, the landlord, the stranger), or any name not explicitly listed above. If the real name is not in the list, skip the relationship entirely.
- For KNOWS: both source_name and target_name must be character names from the "Characters:" list above. Never use a pronoun or description as either endpoint.
- For LEADS_TO and BASED_ON: source_name and target_name are the observation texts of deductions (these are long phrases). Copy them as accurately as possible from the "Deductions" list above — an approximate match is acceptable for these.
- For PRESENT_IN, TAKES_PLACE_IN, FOLLOWS: use ONLY scene titles that appear verbatim in the "Scenes:" list above. Do NOT paraphrase, shorten, or invent new scene titles.
- For PARTICIPATES_IN: use ONLY event names that appear verbatim in the "Events:" list above. Do NOT paraphrase, shorten, or invent new event names.
- For INVESTIGATES and OCCURS_IN: use ONLY crime names that appear verbatim in the "Crimes:" list above. Do NOT use the crime type ("murder", "theft", etc.) — use the full crime name.
- If the text describes an action that fits a scene, event, or crime not in the list, skip that relationship rather than inventing a name.

IMPORTANT: Do NOT invent relationship types. Only use the exact types listed above.
Do not use LOCATED_IN, FOUND_IN, LIVES_IN, RESIDES_AT, OCCURRED_AT, HAPPENS_AT, or any other type not in this list.
Only extract relationships that are explicitly supported by the text. Include the specific property values (role, relationship_type, etc.) for each relationship."""

        try:
            return self.gemini.structured_output(
                prompt=prompt,
                schema=RelationshipExtractionResult,
                model_tier=ModelTier.FLASH,
                system_instruction=(
                    "You are a knowledge graph relationship extractor. "
                    "Only extract relationships explicitly supported by the text."
                ),
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("Error extrayendo relaciones: %s. Devolviendo resultado vacío.", exc)
            return RelationshipExtractionResult()


    def _llm_confirm_same_entity(self, name_a: str, name_b: str, entity_type: str) -> bool:
        """Pregunta al LLM si dos nombres corresponden a la misma entidad."""
        class _SameEntityResponse(BaseModel):
            same: bool
            canonical_name: str = ""

        prompt = (
            f"Are '{name_a}' and '{name_b}' the same {entity_type} "
            f"in the Sherlock Holmes stories? Answer with JSON."
        )
        try:
            result = self.gemini.structured_output(
                prompt=prompt,
                schema=_SameEntityResponse,
                model_tier=ModelTier.LITE,
                temperature=0.0,
            )
            return result.same
        except Exception as exc:
            logger.warning(
                "Error en LLM entity resolution para '%s'/'%s': %s", name_a, name_b, exc
            )
            return False

    def _resolve_typed_entities(self, entities: list, entity_type: str, name_attr: str = "name") -> list:
        """Resuelve duplicados en una lista de entidades del mismo tipo."""
        if not entities:
            return []

        groups: dict[str, list] = {}
        for entity in entities:
            key = _normalize_name(getattr(entity, name_attr, ""))
            groups.setdefault(key, []).append(entity)

        is_character = entity_type == "Character"
        merged = [_merge_entity_group(g, name_attr, prefer_proper_name=is_character) for g in groups.values()]

        if len(merged) <= 1:
            return merged

        to_merge: list[tuple[int, int]] = []

        if entity_type == "Character":
            # Para personajes: matching puramente basado en reglas lingüísticas.
            # Los embeddings semánticos de nombres propios cortos dan similitudes
            # altas entre personajes distintos del mismo dominio, causando que
            # todos los personajes de un relato se fusionen en uno solo.
            logger.debug("Character resolution — %d nombres únicos: %s",
                         len(merged), [getattr(e, name_attr) for e in merged])
            for i in range(len(merged)):
                for j in range(i + 1, len(merged)):
                    name_a = getattr(merged[i], name_attr)
                    name_b = getattr(merged[j], name_attr)
                    if _rule_based_same_character(name_a, name_b):
                        to_merge.append((i, j))
        else:
            already_merged: set[tuple[int, int]] = set()
            if entity_type == "Location":
                for i in range(len(merged)):
                    for j in range(i + 1, len(merged)):
                        na = getattr(merged[i], name_attr)
                        nb = getattr(merged[j], name_attr)
                        if _rule_based_same_location(na, nb):
                            to_merge.append((i, j))
                            already_merged.add((i, j))

            names = [getattr(e, name_attr) for e in merged]
            try:
                embeddings = self.embeddings.embed(names)
            except Exception as exc:
                logger.warning("Error generando embeddings para entity resolution: %s", exc)
                return merged

            rule_merged_pairs: set[tuple[int, int]] = set()
            llm_candidates: list[tuple[int, int]] = []

            # Para crímenes usamos un umbral más alto: nombres como
            # "Death of Julia" y "Death of Dr. Roylott" tienen similitud
            # semántica alta (~0.90) pero son crímenes completamente distintos.
            merge_threshold = (
                0.97 if entity_type == "Crime" else self._MERGE_THRESHOLD
            )
            ambiguous_threshold = (
                0.95 if entity_type == "Crime" else self._AMBIGUOUS_THRESHOLD
            )

            for i in range(len(merged)):
                for j in range(i + 1, len(merged)):
                    if (i, j) in already_merged:
                        continue
                    sim = cosine_similarity(embeddings[i], embeddings[j])
                    if sim > merge_threshold:
                        to_merge.append((i, j))
                        rule_merged_pairs.add((i, j))
                    elif sim > ambiguous_threshold:
                        llm_candidates.append((i, j))

            for i, j in llm_candidates:
                name_a = getattr(merged[i], name_attr)
                name_b = getattr(merged[j], name_attr)
                if self._llm_confirm_same_entity(name_a, name_b, entity_type):
                    to_merge.append((i, j))

        if not to_merge:
            return merged

        parent = list(range(len(merged)))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _first_names_in_group(root: int) -> set[str]:
            """Nombres de pila (primer token tras strip) de los miembros del grupo."""
            result = set()
            for idx in range(len(merged)):
                if _find(idx) == root:
                    name = getattr(merged[idx], name_attr, "")
                    stripped = _strip_titles(name).split()
                    if len(stripped) >= 2:
                        result.add(stripped[0])
            return result

        for i, j in to_merge:
            ri, rj = _find(i), _find(j)
            if ri != rj:
                # Para Characters: no fusionar si el grupo resultante tendría
                # dos primeros nombres distintos (ej. "Helen" y "Julia" son
                # personas distintas aunque compartan el apellido "Stoner").
                if is_character:
                    fn_i = _first_names_in_group(ri)
                    fn_j = _first_names_in_group(rj)
                    combined = fn_i | fn_j
                    if len(combined) >= 2:
                        continue  # conflicto de nombre de pila, no fusionar
                parent[rj] = ri

        root_to_group: dict[int, list] = {}
        for idx, entity in enumerate(merged):
            root_to_group.setdefault(_find(idx), []).append(entity)

        return [_merge_entity_group(g, name_attr, prefer_proper_name=is_character) for g in root_to_group.values()]

    def resolve_entities(self, all_entities: list[EntityExtractionResult]) -> EntityExtractionResult:
        """Consolida y deduplicada entidades acumuladas de todos los chunks."""

        # Descartar aliases que haya generado el LLM — se reconstruyen desde
        # los nombres reales de cada chunk en _merge_entity_group.
        characters = [
            c.model_copy(update={"aliases": []})
            for r in all_entities for c in r.characters
        ]
        locations = [e for r in all_entities for e in r.locations]
        crimes = [e for r in all_entities for e in r.crimes]
        objects = [e for r in all_entities for e in r.objects]
        deductions = [e for r in all_entities for e in r.deductions]
        scenes = [e for r in all_entities for e in r.scenes]
        events = [e for r in all_entities for e in r.events]

        def _is_generic(name: str) -> bool:
            n = _normalize_name(name)
            return any(n.startswith(p) for p in _GENERIC_PREFIXES) or n in _GENERIC_EXACT

        characters = [
            c for c in characters
            if not _is_generic(c.name)
            and len(_normalize_name(c.name)) > 1
        ]

        def _is_generic_location(name: str) -> bool:
            n = _normalize_name(name)
            for p in _LOC_PREFIXES_GENERIC:
                if n.startswith(p):
                    n = n[len(p):]
            tokens = re.sub(r"[^a-z0-9\s]", "", n).split()
            return bool(tokens) and all(t in _GENERIC_LOC_WORDS for t in tokens)

        locations = [l for l in locations if not _is_generic_location(l.name)]

        logger.info(
            "Entity resolution: %d chars, %d locs, %d objects, %d scenes, %d events",
            len(characters), len(locations), len(objects), len(scenes), len(events),
        )

        return EntityExtractionResult(
            characters=self._resolve_typed_entities(characters, "Character"),
            locations=self._resolve_typed_entities(locations, "Location"),
            crimes=self._resolve_typed_entities(crimes, "Crime"),
            objects=self._resolve_typed_entities(objects, "Object"),
            deductions=deductions,
            scenes=self._resolve_typed_entities(scenes, "Scene", name_attr="title"),
            events=self._resolve_typed_entities(events, "Event"),
        )


    def process_story_chunks(self, chunks: list[dict], story_title: str) -> dict:
        """Orquesta la extracción completa de un relato chunk a chunk."""
        context = NarrativeContext()
        all_entity_results: list[EntityExtractionResult] = []
        all_relationships: list[ExtractedRelationship] = []
        chunk_entities: list[dict] = []
        total = len(chunks)

        failed_chunks = 0

        for i, chunk in enumerate(tqdm(chunks, desc=f"Extrayendo '{story_title}'")):
            chunk_text = chunk["text"]
            logger.info("Procesando chunk %d/%d de '%s'", i + 1, total, story_title)

            entities = self.extract_entities(chunk_text, context, story_title)

            if not any([
                entities.characters, entities.locations, entities.crimes,
                entities.objects, entities.deductions, entities.scenes, entities.events,
            ]):
                failed_chunks += 1
                logger.warning(
                    "Chunk %d/%d de '%s' devolvió resultado vacío.",
                    i + 1, total, story_title,
                )

            all_entity_results.append(entities)

            # recoge nombres de entidades de este chunk para MENTIONS
            names = (
                [c.name for c in entities.characters]
                + [l.name for l in entities.locations]
                + [o.name for o in entities.objects]
            )
            chunk_entities.append({
                "chunk_id": chunk.get("id", ""),
                "entity_names": names,
            })

            relationships = self.extract_relationships(chunk_text, entities, context)
            all_relationships.extend(relationships.relationships)

            try:
                context = self._update_sliding_context(context, chunk_text, entities)
            except Exception as exc:
                logger.warning(
                    "Error actualizando contexto en chunk %d/%d: %s", i + 1, total, exc
                )

        if failed_chunks > 0:
            ratio = failed_chunks / total
            msg = (
                f"'{story_title}': {failed_chunks}/{total} chunks sin entidades ({ratio:.0%}). "
                "Posibles fallos de API."
            )
            if ratio > 0.30:
                logger.warning(" %s — considera re-extraer este relato.", msg)
            else:
                logger.warning(msg)

        logger.info("Iniciando entity resolution para '%s' (%d chunks)...", story_title, total)
        consolidated = self.resolve_entities(all_entity_results)


        canonical_map = _build_canonical_map(consolidated)
        resolved_relationships = _remap_relationships(all_relationships, canonical_map)

        # Elimina relaciones cuyo source no corresponde a ninguna entidad conocida.
        # Esto descarta relaciones con "I", "my wife", etc. que sobrevivieron
        # al chunk-level y no tienen contraparte canónica tras la entity resolution.
        known_names: set[str] = (
            {c["name"] for c in consolidated.model_dump().get("characters", [])}
            | {l["name"] for l in consolidated.model_dump().get("locations", [])}
            | {c["name"] for c in consolidated.model_dump().get("crimes", [])}
            | {o["name"] for o in consolidated.model_dump().get("objects", [])}
            | {s["title"] for s in consolidated.model_dump().get("scenes", [])}
            | {e["name"] for e in consolidated.model_dump().get("events", [])}
            | {d["observation"] for d in consolidated.model_dump().get("deductions", [])}
        )

        # LEADS_TO y BASED_ON usan textos largos de observación como claves.
        # El LLM los parafrasea ligeramente, así que se usa matching por prefijo
        # (primeros 40 chars) en vez de igualdad exacta.
        deduction_obs = {d["observation"] for d in consolidated.model_dump().get("deductions", [])}
        deduction_prefixes = {obs[:40].lower().strip() for obs in deduction_obs}

        def _deduction_known(text: str) -> bool:
            if text in deduction_obs:
                return True
            return text[:40].lower().strip() in deduction_prefixes

        before = len(resolved_relationships)
        resolved_relationships = [
            r for r in resolved_relationships
            if r.source_name in known_names
            and (
                r.relationship_type not in self._BOTH_MUST_EXIST
                or r.target_name in known_names
            )
            and (
                r.relationship_type not in {"LEADS_TO", "BASED_ON"}
                or _deduction_known(r.source_name)
            )
        ]
        dropped = before - len(resolved_relationships)
        if dropped:
            logger.info("Filtradas %d relaciones huerfanas (source o target no canónico).", dropped)

        seen: dict[tuple, ExtractedRelationship] = {}
        for rel in resolved_relationships:
            key = (rel.source_name, rel.relationship_type, rel.target_name)
            if key not in seen:
                seen[key] = rel
            else:
                existing = seen[key]
                if rel.relationship_type == "INVESTIGATES":
                    cur_role = existing.properties.get("role", "investigator")
                    new_role = rel.properties.get("role", "investigator")
                    if (self._INVESTIGATES_PRIORITY.get(new_role, 99)
                            < self._INVESTIGATES_PRIORITY.get(cur_role, 99)):
                        seen[key] = rel

        knows_best: dict[frozenset, ExtractedRelationship] = {}
        non_knows: list[ExtractedRelationship] = []
        for rel in seen.values():
            if rel.source_name == rel.target_name:
                continue
            if rel.relationship_type != "KNOWS":
                non_knows.append(rel)
            else:
                pair = frozenset({rel.source_name, rel.target_name})
                if pair not in knows_best:
                    knows_best[pair] = rel
                else:
                    existing = knows_best[pair]
                    cur_t = existing.properties.get("relationship_type", "acquaintance")
                    new_t = rel.properties.get("relationship_type", "acquaintance")
                    if self._KNOWS_PRIORITY.get(new_t, 99) < self._KNOWS_PRIORITY.get(cur_t, 99):
                        knows_best[pair] = rel

        before_dedup = len(resolved_relationships)
        resolved_relationships = non_knows + list(knows_best.values())
        deduped = before_dedup - len(resolved_relationships)
        if deduped:
            logger.info("Deduplicadas/eliminadas %d relaciones (repetidas o self-loop).", deduped)

        logger.info(
            "'%s': %d entidades consolidadas, %d relaciones extraídas.",
            story_title,
            sum([
                len(consolidated.characters),
                len(consolidated.locations),
                len(consolidated.crimes),
                len(consolidated.objects),
                len(consolidated.deductions),
                len(consolidated.scenes),
                len(consolidated.events),
            ]),
            len(resolved_relationships),
        )

        # serializar a dict para que neo4j_manager.store_entities/store_relationships
        # y el notebook puedan usar acceso de dict (.get()) en lugar de atributos Pydantic.
        return {
            "story_title": story_title,
            "entities": consolidated.model_dump(),
            "relationships": [r.model_dump() for r in resolved_relationships],
            "chunks_processed": total,
            "chunk_entities": chunk_entities,
        }

    def normalize_cross_story_entities(self, all_results: dict[str, dict]) -> dict[str, dict]:
        """Unifica nombres canónicos de Characters entre todos los relatos."""

        all_names = sorted({
            char["name"]
            for result in all_results.values()
            for char in result["entities"].get("characters", [])
        })
        if not all_names:
            return all_results

        parent = list(range(len(all_names)))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(all_names)):
            for j in range(i + 1, len(all_names)):
                if _rule_based_same_character(all_names[i], all_names[j]):
                    ri, rj = _find(i), _find(j)
                    if ri != rj:
                        parent[rj] = ri

        groups: dict[int, list[str]] = {}
        for idx, name in enumerate(all_names):
            groups.setdefault(_find(idx), []).append(name)

        name_frequency: dict[str, int] = {}
        for result in all_results.values():
            for char in result["entities"].get("characters", []):
                n = char["name"]
                name_frequency[n] = name_frequency.get(n, 0) + 1

        def _canonical_score(name: str) -> tuple:
            stripped = _strip_titles(name)
            is_bare = _normalize_name(name).replace(",", "").replace(".", "") == stripped
            token_count = len(stripped.split())
            freq = name_frequency.get(name, 0)
            return (freq, is_bare, token_count, name)

        rename_map: dict[str, str] = {}
        for group in groups.values():
            if len(group) == 1:
                continue
            canonical = max(group, key=_canonical_score)
            for name in group:
                if name != canonical:
                    rename_map[name] = canonical

        if not rename_map:
            logger.info("Cross-story resolution: sin renombrados necesarios.")
            return all_results

        logger.info(
            "Cross-story resolution: %d variantes unificadas -> %s",
            len(rename_map),
            {v: k for k, v in
             {rename_map[n]: n for n in rename_map}.items()},
        )

        updated: dict[str, dict] = {}
        for story_title, result in all_results.items():

            renamed_chars: dict[str, dict] = {}
            for char in result["entities"].get("characters", []):
                old_name = char["name"]
                new_name = rename_map.get(old_name, old_name)
                if new_name not in renamed_chars:
                    renamed_chars[new_name] = dict(char)
                    renamed_chars[new_name]["name"] = new_name

                existing = renamed_chars[new_name]
                merged_aliases: list[str] = list({
                    *existing.get("aliases", []),
                    *char.get("aliases", []),
                    *([] if old_name == new_name else [old_name]),
                })

                merged_aliases = [a for a in merged_aliases if a != new_name]
                existing["aliases"] = sorted(merged_aliases)

            new_entities = dict(result["entities"])
            new_entities["characters"] = list(renamed_chars.values())

            new_rels = []
            for rel in result.get("relationships", []):
                src = rename_map.get(rel["source_name"], rel["source_name"])
                tgt = rename_map.get(rel["target_name"], rel["target_name"])
                new_rels.append({**rel, "source_name": src, "target_name": tgt})

            new_chunk_entities = []
            for ce in result.get("chunk_entities", []):
                renamed_entity_names = [
                    rename_map.get(n, n) for n in ce.get("entity_names", [])
                ]
                new_chunk_entities.append({**ce, "entity_names": renamed_entity_names})

            updated[story_title] = {
                **result,
                "entities": new_entities,
                "relationships": new_rels,
                "chunk_entities": new_chunk_entities,
            }

        return updated
