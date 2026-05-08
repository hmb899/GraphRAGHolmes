from ..config import get_settings


def count_tokens(text: str) -> int:
    """Estima el número de tokens de un texto."""
    return len(text.split())


def split_into_paragraphs(text: str) -> list[str]:
    """Divide un texto en párrafos; elimina vacíos y normaliza espacios internos."""
    raw = text.split("\n\n")
    paragraphs = []
    for p in raw:
        cleaned = " ".join(p.split())
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs


def chunk_story(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None,) -> list[dict]:
    """Divide un relato en chunks respetando límites de párrafo, con overlap."""
    if chunk_size is None or chunk_overlap is None:
        settings = get_settings()
        if chunk_size is None:
            chunk_size = settings.chunk_size
        if chunk_overlap is None:
            chunk_overlap = settings.chunk_overlap

    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return []

    # Agrupa párrafos en grupos sin partir ninguno
    groups: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        separator = 1 if current else 0  # espacio entre párrafos al unir
        if current and current_len + separator + para_len > chunk_size:
            groups.append(current)
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += separator + para_len

    if current:
        groups.append(current)

    # Construye los chunks con overlap
    chunks: list[dict] = []
    char_cursor = 0  # posición en el texto original

    for idx, group in enumerate(groups):
        overlap_paragraphs: list[str] = []

        if idx > 0:
            # Toma párrafos del final del grupo anterior hasta cubrir chunk_overlap
            prev_group = groups[idx - 1]
            accumulated = 0
            for para in reversed(prev_group):
                overlap_paragraphs.insert(0, para)
                accumulated += len(para)
                if accumulated >= chunk_overlap:
                    break

        all_paragraphs = overlap_paragraphs + group
        chunk_text = " ".join(all_paragraphs)

        # char_start apunta al primer párrafo del grupo (sin overlap)
        first_para = group[0]
        char_start = text.find(first_para, char_cursor)
        if char_start == -1:
            char_start = char_cursor
        last_para = group[-1]
        char_end = text.find(last_para, char_start) + len(last_para)
        char_cursor = char_end

        chunks.append(
            {
                "text": chunk_text,
                "chunk_index": idx,
                "char_start": char_start,
                "char_end": char_end,
                "token_count": count_tokens(chunk_text),
            }
        )

    return chunks


def chunk_story_with_metadata(text: str, story_title: str, collection: str, chunk_size: int | None = None,
                              chunk_overlap: int | None = None,) -> list[dict]:
    """Wrapper de chunk_story que añade título, colección y posición narrativa."""
    chunks = chunk_story(text, chunk_size, chunk_overlap)
    total = len(chunks)

    for chunk in chunks:
        idx = chunk["chunk_index"]
        if total <= 1:
            position = "beginning"
        elif idx < max(1, round(total * 0.2)):
            position = "beginning"
        elif idx >= total - max(1, round(total * 0.2)):
            position = "end"
        else:
            position = "middle"

        chunk["story_title"] = story_title
        chunk["collection"] = collection
        chunk["position"] = position
        chunk["total_chunks"] = total

    return chunks
