import ollama
from tqdm import tqdm

from ..config import get_settings


class EmbeddingClient:
    """Cliente para generación de embeddings locales con Ollama."""

    _PROGRESS_THRESHOLD = 5  # mostrar tqdm a partir de este número de textos

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.ollama_embedding_model
        self.client = ollama.Client(host=settings.ollama_base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Genera embeddings para una lista de textos."""
        show_progress = len(texts) > self._PROGRESS_THRESHOLD
        iterator = tqdm(texts, desc="Generando embeddings") if show_progress else texts

        embeddings: list[list[float]] = []
        for text in iterator:
            response = self.client.embed(model=self._model, input=text)
            embeddings.append(response.embeddings[0])
        return embeddings

    def embed_single(self, text: str) -> list[float]:
        """Genera el embedding de un único texto."""
        return self.embed([text])[0]
