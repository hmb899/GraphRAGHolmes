from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Ruta al .env en la raíz del proyecto, independientemente del cwd
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    # Neo4j
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # Google Cloud / Vertex AI
    google_cloud_project: str
    google_cloud_location: str = "europe-west1"

    # Modelos Gemini
    gemini_flash_lite_model: str = "gemini-2.5-flash-lite"
    gemini_flash_model: str = "gemini-2.5-flash"
    gemini_pro_model: str = "gemini-2.5-pro"

    # Ollama (solo embeddings)
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"

    # Chunking
    chunk_size: int = 1500
    chunk_overlap: int = 200

    # Embeddings
    embedding_dimensions: int = 768

    # Procesamiento
    top_k_results: int = 5
    batch_size: int = 10

    # Paths
    data_dir: str = "data"
    output_dir: str = "output"

    model_config = SettingsConfigDict(env_file=str(_ENV_FILE))


@lru_cache()
def get_settings() -> Settings:
    return Settings()
