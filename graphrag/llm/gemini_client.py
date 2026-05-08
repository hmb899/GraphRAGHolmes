import json
import logging
import time
from enum import Enum
from typing import Any, Type, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from ..config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ModelTier(str, Enum):
    """Niveles de modelo Gemini según coste y capacidad."""

    LITE = "lite"    # Flash-Lite: sliding context, entity resolution
    FLASH = "flash"  # Flash: extracción de entidades
    PRO = "pro"      # Pro: Text2Cypher, router, agente crítico


def extract_json(response: str) -> dict[str, Any]:
    """Extrae JSON de una respuesta de texto como fallback."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        start = response.find("```json")
        if start != -1:
            start = response.find("\n", start) + 1
            end = response.find("```", start)
            if end != -1:
                return json.loads(response[start:end].strip())
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1:
            return json.loads(response[start : end + 1])
        raise ValueError("No se pudo extraer JSON de la respuesta")


class GeminiClient:
    """Cliente para generación de texto con Gemini via Vertex AI"""

    _MAX_RETRIES = 3
    _INITIAL_BACKOFF = 5  # segundos

    def __init__(self) -> None:
        settings = get_settings()
        self._model_map = {
            ModelTier.LITE: settings.gemini_flash_lite_model,
            ModelTier.FLASH: settings.gemini_flash_model,
            ModelTier.PRO: settings.gemini_pro_model,
        }
        self.client = genai.Client(
            vertexai=True,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )

    def _resolve_model(self, tier: ModelTier) -> str:
        return self._model_map[tier]

    def _call_with_retry(self, fn, *args, **kwargs):
        """Ejecuta fn con reintentos exponenciales ante errores transitorios."""
        backoff = self._INITIAL_BACKOFF
        last_exc = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                msg = str(exc)
                if any(code in msg for code in ("429", "503")):
                    logger.warning(
                        "Error transitorio (intento %d/%d): %s. Reintentando en %ds.",
                        attempt,
                        self._MAX_RETRIES,
                        msg,
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    last_exc = exc
                else:
                    raise
        raise last_exc

    def generate(self, prompt: str, model_tier: ModelTier = ModelTier.FLASH, system_instruction: str | None = None,
                 temperature: float = 0.0) -> str:
        """Genera texto libre a partir de un prompt."""
        model = self._resolve_model(model_tier)
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )
        logger.info("Generando con %s (tier=%s)", model, model_tier.value)

        response = self._call_with_retry(
            self.client.models.generate_content,
            model=model,
            contents=prompt,
            config=config,
        )
        return response.text

    def structured_output(self, prompt: str, schema: Type[T], model_tier: ModelTier = ModelTier.FLASH,
                          system_instruction: str | None = None, temperature: float = 0.0) -> T:
        """Genera una respuesta validada contra un esquema Pydantic; usa extract_json como fallback."""
        model = self._resolve_model(model_tier)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            system_instruction=system_instruction,
            temperature=temperature,
        )
        logger.info("Structured output con %s → %s", model, schema.__name__)

        response = self._call_with_retry(
            self.client.models.generate_content,
            model=model,
            contents=prompt,
            config=config,
        )

        try:
            return schema.model_validate_json(response.text)
        except Exception:
            logger.warning("model_validate_json falló, aplicando extract_json como fallback.")
            data = extract_json(response.text)
            return schema.model_validate(data)

    def generate_with_history(self, messages: list[dict[str, str]], model_tier: ModelTier = ModelTier.FLASH,
                              temperature: float = 0.0) -> str:
        """Genera una respuesta en una conversación multi-turno."""
        model = self._resolve_model(model_tier)
        contents = [
            types.Content(
                role="model" if msg["role"] == "assistant" else msg["role"],
                parts=[types.Part(text=msg["content"])],
            )
            for msg in messages
        ]
        config = types.GenerateContentConfig(temperature=temperature)
        logger.info("Multi-turno con %s (%d mensajes)", model, len(messages))

        response = self._call_with_retry(
            self.client.models.generate_content,
            model=model,
            contents=contents,
            config=config,
        )
        return response.text
