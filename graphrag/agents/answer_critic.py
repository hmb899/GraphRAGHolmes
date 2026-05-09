import logging
from typing import Any

from pydantic import BaseModel, Field

from ..llm.gemini_client import GeminiClient, ModelTier

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert answer evaluator for a Sherlock Holmes knowledge base.\n\n"
    "Your task is to evaluate an answer against a question and the retrieved context.\n\n"
    "Evaluate:\n"
    "1. COMPLETENESS: Does the answer fully address ALL parts of the question?\n"
    "   - If the question asks about multiple aspects, all must be covered.\n"
    "   - A partial answer is NOT complete.\n\n"
    "2. FAITHFULNESS: Is every claim in the answer supported by the provided context?\n"
    "   - The answer must NOT contain information not found in the context.\n"
    "   - Reasonable inferences from the context are acceptable.\n"
    "   - Fabricated details, even plausible ones, make the answer unfaithful.\n\n"
    "3. MISSING INFORMATION: If the answer is incomplete, generate specific sub-questions\n"
    "   that could be used to retrieve the missing information in another retrieval round.\n"
    "   - Sub-questions should be self-contained and specific.\n"
    "   - They should target exactly what is missing.\n"
    "   - If the answer is complete and faithful, return an empty list.\n\n"
    "Special cases:\n"
    "- If the context is empty or insufficient, and the answer acknowledges this\n"
    "  (\"No tengo información\", \"This information is not available\"), consider it\n"
    "  complete and faithful — the system correctly abstained.\n"
    "- If the question is a greeting or out-of-scope, and the answer handles it\n"
    "  appropriately, consider it complete and faithful."
)


class CritiqueResult(BaseModel):
    """Resultado estructurado de la evaluación del crítico."""

    is_complete: bool = Field(
        ...,
        description="True si la respuesta aborda todas las partes de la pregunta.",
    )
    is_faithful: bool = Field(
        ...,
        description="True si la respuesta se basa exclusivamente en el contexto proporcionado.",
    )
    missing_info: list[str] = Field(
        default_factory=list,
        description=(
            "Lista de sub-preguntas para recuperar información faltante. "
            "Vacía si la respuesta es completa."
        ),
    )
    feedback: str = Field(
        ...,
        description="Explicación breve de la evaluación.",
    )


class AnswerCritic:
    """Evalúa la completitud y fidelidad de respuestas generadas por el sistema RAG."""

    def __init__(self) -> None:
        self.client = GeminiClient()

    def critique(
        self, question: str, context: list[str], answer: str
    ) -> dict[str, Any]:
        """Evalúa si la respuesta es completa y fiel al contexto recuperado."""
        context_str = (
            "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(context))
            if context
            else "(no context retrieved)"
        )

        prompt = (
            f"Question: {question}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Answer: {answer}\n\n"
            "Evaluate this answer."
        )

        result = self.client.structured_output(
            prompt=prompt,
            schema=CritiqueResult,
            model_tier=ModelTier.FLASH,
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.0,
        )
        return result.model_dump()

    def should_retry(self, critique: dict[str, Any]) -> bool:
        """True si la respuesta es incompleta y hay sub-preguntas para relanzar retrieval."""
        return (
            not critique.get("is_complete", True)
            and bool(critique.get("missing_info"))
        )

    def get_retry_queries(self, critique: dict[str, Any]) -> list[str]:
        """Extrae las sub-preguntas del critique para relanzar retrieval."""
        return critique.get("missing_info", [])
