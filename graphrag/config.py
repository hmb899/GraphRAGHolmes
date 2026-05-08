from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=str(_ENV_FILE))

    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    google_cloud_project: str
    google_cloud_location: str = "europe-west1"

    gemini_flash_lite_model: str = "gemini-2.5-flash-lite"
    gemini_flash_model: str = "gemini-2.5-flash"
    gemini_pro_model: str = "gemini-2.5-pro"

    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"

    chunk_size: int = 1500
    chunk_overlap: int = 200

    embedding_dimensions: int = 768

    top_k_results: int = 10
    batch_size: int = 10

    data_dir: str = "data"
    output_dir: str = "output"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
