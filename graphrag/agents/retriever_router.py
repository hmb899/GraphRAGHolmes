"""Router agéntico que selecciona la herramienta de recuperación más adecuada."""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..config import get_settings
from ..llm.gemini_client import GeminiClient, ModelTier
from .retriever_tools import RetrieverTools

logger = logging.getLogger(__name__)


class RouterDecision(BaseModel):
    """Esquema para la decisión de enrutamiento del LLM."""

    tool: Literal[
        "vector_search",
        "fulltext_search",
        "hybrid_search",
        "text2cypher",
        "greeting",
        "out_of_scope",
    ] = Field(
        ...,
        description="Nombre de la herramienta seleccionada para manejar la consulta.",
    )
    reasoning: str = Field(
        ...,
        description="Breve explicación de por qué se eligió esta herramienta.",
    )
    query: str = Field(
        ...,
        description="La consulta reformulada u original que se pasará a la herramienta.",
    )


class RetrieverRouter:
    """Decide qué retriever usar para cada pregunta mediante Gemini Pro con structured output."""

    def __init__(self, retriever_tools: RetrieverTools) -> None:
        self.tools = retriever_tools
        self.client = GeminiClient()
        self.settings = get_settings()

    def route(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> RouterDecision:
        """Selecciona la mejor herramienta para responder la pregunta.

        Considera el historial de conversación cuando está disponible para
        desambiguar referencias anafóricas ("¿Y Watson?", "¿Cuántos hay?").

        Args:
            question: Pregunta del usuario en lenguaje natural (español o inglés).
            conversation_history: Turnos anteriores como lista de dicts
                {role: 'user'|'assistant', content: str}.

        Returns:
            RouterDecision con tool seleccionado, razonamiento y query optimizada.
        """
        conversation_history = conversation_history or []

        tool_descriptions = self.tools.get_tool_descriptions()
        tools_str = "\n".join(
            f"- {t['name']}: {t['description']}" for t in tool_descriptions
        )

        system_prompt = (
            "You are a routing assistant for a knowledge base about Sherlock Holmes "
            "stories by Arthur Conan Doyle.\n"
            "Select the single best tool for the user's question.\n\n"
            f"Available tools:\n{tools_str}\n\n"
            "ROUTING RULES — apply in order:\n\n"
            "1. greeting     — Use when the message is conversational and needs NO knowledge lookup:\n"
            "   greetings (\"Hello\", \"Hola\", \"Buenos días\"), farewells (\"Goodbye\", \"Adiós\"),\n"
            "   thanks (\"Gracias\", \"Thank you\"),\n"
            "   questions about the system (\"What can you do?\", \"¿Qué puedes hacer?\", \"¿Cómo funcionas?\").\n\n"
            "2. out_of_scope — Use when the question is clearly unrelated to Sherlock Holmes stories:\n"
            "   weather, sports, cooking, politics, current events, other fictional universes.\n"
            "   Examples: \"What is the capital of France?\", \"¿Quién ganó el Mundial?\".\n\n"
            "3. text2cypher   — Use for STRUCTURED data queries that need graph traversal:\n"
            "   - Counts: \"¿Cuántos personajes aparecen en...?\", \"How many crimes...\"\n"
            "   - Enumerations: \"Lista todos los...\", \"List all characters who...\"\n"
            "   - Specific attributes: \"¿Cuál es la ocupación de...?\"\n"
            "   - Residence/address: \"Where does X live?\", \"Where does X work?\", \"¿Dónde vive X?\" (uses LIVES_AT relationship — NOT answerable from text chunks)\n"
            "   - Location of scenes or events: \"What happens at X?\", \"What scenes take place at X?\", \"¿Qué ocurre en X?\" (uses TAKES_PLACE_IN — NOT answerable from text chunks)\n"
            "   - Relationships: \"¿A quién conoce Holmes?\", \"What crimes does X investigate?\"\n"
            "   - Aggregations: \"¿Cuántos relatos hay por colección?\"\n"
            "   - Comparisons: \"¿Qué personajes aparecen en más de un relato?\"\n\n"
            "4. vector_search — Use for DESCRIPTIVE or CONCEPTUAL questions answerable from text passages:\n"
            "   \"¿Cómo resuelve Holmes el caso?\", \"Describe the relationship between Holmes and Watson\",\n"
            "   \"¿Qué método usa Holmes para...?\", \"Explain Holmes' deduction about...\".\n\n"
            "5. fulltext_search — Use when the user wants to find EXACT mentions or specific terms:\n"
            "   \"¿Dónde se menciona 'strychnine'?\", \"Find passages about 'the red-headed league'\",\n"
            "   quoted search terms, proper nouns the user wants to locate verbatim in the text.\n\n"
            "6. hybrid_search — Use when BOTH semantic meaning AND specific keywords matter:\n"
            "   \"Holmes' analysis of the speckled band\", \"Watson's description of Irene Adler\".\n"
            "   Also use as a FALLBACK when unsure between vector_search and fulltext_search.\n\n"
            "IMPORTANT:\n"
            "- Choose greeting or out_of_scope FIRST before considering any retrieval tool.\n"
            "- Prefer text2cypher for anything countable, listable, or involving graph relationships.\n"
            "- Prefer vector_search for open-ended, descriptive, or 'how/why' questions.\n"
            "- Use fulltext_search only when the user explicitly wants keyword-exact matches.\n"
            "- Use hybrid_search when both meaning and specific terms matter, or when unsure.\n"
            "- The query field must contain a reformulated query optimized for the selected tool.\n"
            "  For text2cypher: keep it as a clear natural language question.\n"
            "  For vector/hybrid/fulltext: reformulate to maximize retrieval quality.\n"
        )

        # Incrustar el historial en el prompt del usuario para que el LLM tenga contexto
        # multi-turno. structured_output no soporta lista de mensajes, así que el
        # historial se serializa como texto plano antes de la pregunta actual.
        if conversation_history:
            history_lines = "\n".join(
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in conversation_history
            )
            prompt = (
                f"Previous conversation:\n{history_lines}\n\n"
                f"Current question: {question}"
            )
        else:
            prompt = f"Question: {question}"

        return self.client.structured_output(
            prompt=prompt,
            schema=RouterDecision,
            model_tier=ModelTier.PRO,
            system_instruction=system_prompt,
            temperature=0.0,
        )

    def retrieve(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Selecciona la herramienta apropiada y ejecuta la búsqueda.

        Args:
            question: Pregunta del usuario en lenguaje natural.
            conversation_history: Turnos anteriores de la conversación.

        Returns:
            Dict con los resultados del retriever más la clave 'routing_decision'
            con el RouterDecision que tomó la decisión de enrutamiento.
        """
        decision = self.route(question, conversation_history)

        tool_name = decision.tool
        query = decision.query or question

        logger.info(
            "Router → tool='%s', reasoning='%s'", tool_name, decision.reasoning
        )

        result = self.tools.execute_tool(tool_name, query=query)
        result["routing_decision"] = decision
        return result
