import logging
import os
import re
import urllib.request

from tqdm import tqdm

from ..config import get_settings
from ..graph.neo4j_manager import Neo4jManager
from ..llm.embedding_client import EmbeddingClient
from ..utils.chunking import chunk_story_with_metadata

logger = logging.getLogger(__name__)


GUTENBERG_COLLECTIONS = {
    "adventures": {
        "id": 1661,
        "title": "The Adventures of Sherlock Holmes",
        "url": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    },
    "memoirs": {
        "id": 834,
        "title": "The Memoirs of Sherlock Holmes",
        "url": "https://www.gutenberg.org/cache/epub/834/pg834.txt",
    },
    "return": {
        "id": 108,
        "title": "The Return of Sherlock Holmes",
        "url": "https://www.gutenberg.org/cache/epub/108/pg108.txt",
    },
    "his_last_bow": {
        "id": 2350,
        "title": "His Last Bow",
        "url": "https://www.gutenberg.org/cache/epub/2350/pg2350.txt",
    },
    "casebook": {
        "id": 69700,
        "title": "The Case-Book of Sherlock Holmes",
        "url": "https://www.gutenberg.org/cache/epub/69700/pg69700.txt",
    },
}

PHASE1_STORIES = [
    "The Adventure of the Speckled Band",
    "The Red-Headed League",
    "A Scandal in Bohemia",
    "Silver Blaze",
    "The Final Problem",
    "The Adventure of the Dancing Men",
]

# Colección de origen de cada relato de la Fase 1
_STORY_COLLECTION_MAP: dict[str, str] = {
    "the adventure of the speckled band": "adventures",
    "the red-headed league": "adventures",
    "a scandal in bohemia": "adventures",
    "silver blaze": "memoirs",
    "the final problem": "memoirs",
    "the adventure of the dancing men": "return",
}

# Detecta cabeceras de relato en tres formatos de Gutenberg:
#   1. Numeral romano + ALL CAPS  →  "I. A SCANDAL IN BOHEMIA"  (adventures, return)
#   2. Numeral romano + Title Case →  "I. Silver Blaze"           (memoirs)
#   3. ALL CAPS sin numeral        →  "THE ADVENTURE OF THE..."   (return)
_HEADER_RE = re.compile(
    r'\n\n'
    r'('
    r'(?:(?:ADVENTURE|STORY|EPISODE) +)?[IVXLC]+\. +[A-Z][A-Z \-\'\,\.]{3,60}'
    r'|'
    r'[IVXLC]+\. +[A-Z][A-Za-z \-\'\,\.]{3,60}'
    r'|'
    r'[A-Z][A-Z \-\'\,\.]{5,60}'
    r')'
    r'\n\n',
    re.MULTILINE,
)

# Falsos positivos, líneas que coinciden con el patrón pero no son títulos de relato
_FALSE_POSITIVES: set[str] = {
    "THE END", "CONTENTS", "PREFACE", "INTRODUCTION", "NOTE", "NOTES",
    "FOREWORD", "APPENDIX", "INDEX", "GLOSSARY", "EDITOR", "DEDICATION",
    # Títulos de colección
    "THE ADVENTURES OF SHERLOCK HOLMES",
    "THE MEMOIRS OF SHERLOCK HOLMES",
    "THE RETURN OF SHERLOCK HOLMES",
    "HIS LAST BOW",
    "THE CASE-BOOK OF SHERLOCK HOLMES",
}


def _normalize(title: str) -> str:
    """Normaliza un título para comparación: lowercase y espacios simples."""
    return " ".join(title.lower().split())


def _strip_numeral_prefix(raw: str) -> str:
    """Elimina prefijos de numeral romano y normaliza a Title Case."""
    stripped = re.sub(r'^(?:(?:ADVENTURE|STORY)\s+)?[IVXLC]+\.\s+', '', raw.strip()).strip()
    return stripped.title()


class TextProcessor:
    """Descarga, limpia y fragmenta los relatos de Sherlock Holmes de Project Gutenberg"""

    def __init__(self, neo4j_manager: Neo4jManager, settings=None) -> None:
        self.neo4j = neo4j_manager
        self.settings = settings or get_settings()
        self.embedder = EmbeddingClient()

    def download_collection(self, collection_key: str) -> str:
        """Descarga una colección de Gutenberg y la almacena en disco."""
        info = GUTENBERG_COLLECTIONS[collection_key]
        os.makedirs(self.settings.data_dir, exist_ok=True)
        path = os.path.join(self.settings.data_dir, f"{collection_key}.txt")

        if os.path.exists(path):
            logger.info("Leyendo '%s' desde caché: %s", collection_key, path)
            with open(path, encoding="utf-8") as f:
                return f.read()

        logger.info("Descargando '%s' desde %s", info["title"], info["url"])
        with urllib.request.urlopen(info["url"]) as response:
            raw = response.read()

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")

        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

        logger.info("Colección '%s' guardada en %s (%d chars).", collection_key, path, len(text))
        return text

    def clean_gutenberg_text(self, text: str) -> str:
        """Elimina el header y footer estándar de Project Gutenberg."""
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG EBOOK",
            "*** START OF THIS PROJECT GUTENBERG EBOOK",
            "*END*THE SMALL PRINT",
        ]
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG EBOOK",
            "*** END OF THIS PROJECT GUTENBERG EBOOK",
            "End of Project Gutenberg",
            "End of the Project Gutenberg",
        ]

        for marker in start_markers:
            idx = text.find(marker)
            if idx != -1:
                newline = text.find("\n\n", idx)
                text = text[newline:] if newline != -1 else text[idx + len(marker):]
                break
        else:
            logger.warning("Marcador de inicio de Gutenberg no encontrado.")

        for marker in end_markers:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
                break
        else:
            logger.warning("Marcador de fin de Gutenberg no encontrado.")

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Normalizar comillas tipográficas a ASCII para que el regex de cabeceras funcione
        text = text.replace("’", "'").replace("‘", "'")
        text = text.replace("“", '"').replace("”", '"')
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def split_into_stories(self, text: str, collection_key: str) -> list[dict]:
        """Divide el texto de una colección en relatos individuales."""
        collection_name = GUTENBERG_COLLECTIONS[collection_key]["title"]
        all_matches = list(_HEADER_RE.finditer(text))

        valid_matches = []
        for m in all_matches:
            raw = m.group(1).strip()
            if raw.upper() in _FALSE_POSITIVES:
                continue
            # Descarta líneas con más de 10 palabras
            if len(raw.split()) > 10:
                continue
            valid_matches.append(m)

        if len(valid_matches) < 3:
            logger.warning(
                "Solo se detectaron %d candidatos a relato en '%s'. "
                "Puede requerir ajuste manual del parser.",
                len(valid_matches),
                collection_key,
            )

        stories: list[dict] = []
        for i, match in enumerate(valid_matches):
            title = _strip_numeral_prefix(match.group(1))
            start = match.end()
            end = valid_matches[i + 1].start() if i + 1 < len(valid_matches) else len(text)
            story_text = text[start:end].strip()

            if len(story_text) < 500:
                logger.debug(
                    "Candidato '%s' descartado: texto demasiado corto (%d chars).",
                    title,
                    len(story_text),
                )
                continue

            stories.append({
                "title": title,
                "text": story_text,
                "collection": collection_name,
            })

        logger.info("Colección '%s': %d relatos detectados.", collection_key, len(stories))
        return stories

    def process_story(self, story_title: str, story_text: str, collection: str) -> list[dict]:
        """Fragmenta un relato, genera embeddings y lo carga en Neo4j."""
        chunks = chunk_story_with_metadata(
            text=story_text,
            story_title=story_title,
            collection=collection,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        logger.info("'%s': %d chunks generados.", story_title, len(chunks))

        self.neo4j.store_story(story_title, collection)
        embeddings = self.embedder.embed([c["text"] for c in chunks])

        for chunk, embedding in tqdm(
            zip(chunks, embeddings),
            total=len(chunks),
            desc=f"Cargando '{story_title}'",
            leave=False,
        ):
            chunk["embedding"] = embedding
            self.neo4j.store_chunk(chunk, story_title)

        return chunks

    def _run_pipeline(self, target_titles: set[str] | None = None) -> dict[str, list[dict]]:
        """Ejecuta el pipeline de descarga y procesamiento."""
        results: dict[str, list[dict]] = {}

        if target_titles is not None:
            needed = {
                col for norm, col in _STORY_COLLECTION_MAP.items()
                if norm in target_titles
            }
            collection_keys = list(needed) if needed else list(GUTENBERG_COLLECTIONS)
        else:
            collection_keys = list(GUTENBERG_COLLECTIONS)

        for key in collection_keys:
            logger.info("=== Colección: %s ===", GUTENBERG_COLLECTIONS[key]["title"])
            raw = self.download_collection(key)
            clean = self.clean_gutenberg_text(raw)
            stories = self.split_into_stories(clean, key)

            for story in stories:
                norm = _normalize(story["title"])
                if target_titles is not None and norm not in target_titles:
                    continue

                logger.info("→ Procesando: '%s'", story["title"])
                try:
                    chunks = self.process_story(
                        story_title=story["title"],
                        story_text=story["text"],
                        collection=story["collection"],
                    )
                    results[story["title"]] = chunks
                except Exception as exc:
                    logger.warning("Error en '%s': %s. Continuando.", story["title"], exc)

        return results

    def process_phase1(self) -> dict[str, list[dict]]:
        """Descarga y procesa los 10 relatos de la Fase 1"""
        logger.info("Pipeline Fase 1: %d relatos", len(PHASE1_STORIES))
        targets = {_normalize(t) for t in PHASE1_STORIES}
        results = self._run_pipeline(target_titles=targets)

        found_norm = {_normalize(t) for t in results}
        missing = [t for t in PHASE1_STORIES if _normalize(t) not in found_norm]
        if missing:
            logger.warning("Relatos no encontrados: %s", missing)

        logger.info(
            "Fase 1 completada: %d/%d relatos procesados.",
            len(results),
            len(PHASE1_STORIES),
        )
        return results

    def process_stories(self, titles: list[str]) -> dict[str, list[dict]]:
        """Descarga y procesa una lista arbitraria de relatos por título."""
        logger.info("Pipeline personalizado: %d relatos", len(titles))
        targets = {_normalize(t) for t in titles}
        results = self._run_pipeline(target_titles=targets)

        found_norm = {_normalize(t) for t in results}
        missing = [t for t in titles if _normalize(t) not in found_norm]
        if missing:
            logger.warning("Relatos no encontrados: %s", missing)

        logger.info("Completado: %d/%d relatos procesados.", len(results), len(titles))
        return results

    def process_all(self) -> dict[str, list[dict]]:
        """Descarga y procesa todos los relatos de todas las colecciones."""
        logger.info("Pipeline completo: todas las colecciones")
        results = self._run_pipeline(target_titles=None)
        logger.info("Pipeline completado: %d relatos procesados.", len(results))
        return results
