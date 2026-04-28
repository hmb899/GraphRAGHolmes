"""Herramientas de recuperación expuestas al router agéntico."""

import logging
from typing import Any, Callable, TypedDict

from ..graph.neo4j_manager import Neo4jManager
from ..retrieval.fulltext_retriever import FullTextRetriever
from ..retrieval.text2cypher import Text2CypherRetriever
from ..retrieval.vector_retriever import HybridRetriever, VectorRetriever

logger = logging.getLogger(__name__)


class ToolData(TypedDict):
    function: Callable
    description: str


class RetrieverTools:
    """Puente entre los retrievers y el router agéntico.

    Instancia todos los retrievers disponibles, expone sus descripciones
    para el prompt del router y delega la ejecución según la herramienta
    seleccionada.
    """

    _GREETING_RESPONSE = (
        "¡Hola! Soy un asistente de conocimiento especializado en las historias de "
        "Sherlock Holmes de Arthur Conan Doyle. Puedo responder preguntas sobre "
        "personajes, ubicaciones, crímenes, deducciones, objetos y eventos de los "
        "relatos del canon holmesiano."
    )

    _OUT_OF_SCOPE_RESPONSE = (
        "Esta pregunta está fuera de mi ámbito. Solo puedo responder preguntas "
        "relacionadas con las historias de Sherlock Holmes y su universo literario."
    )

    _SKILLS_RESPONSE = (
        "Puedo responder preguntas sobre los relatos de Sherlock Holmes: personajes y "
        "sus relaciones, ubicaciones, crímenes e investigaciones, deducciones y métodos "
        "de razonamiento de Holmes, objetos y pistas, escenas y eventos. Puedo buscar "
        "por similitud semántica, por palabras clave exactas, o consultar directamente "
        "el grafo de conocimiento para preguntas estructuradas."
    )

    def __init__(self, neo4j_manager: Neo4jManager) -> None:
        self.neo4j = neo4j_manager
        self.vector_retriever = VectorRetriever(neo4j_manager)
        self.hybrid_retriever = HybridRetriever(neo4j_manager)
        self.fulltext_retriever = FullTextRetriever(neo4j_manager)
        self.text2cypher = Text2CypherRetriever(neo4j_manager)
        self.custom_tools: dict[str, ToolData] = {}

    def register_custom_tool(
        self, name: str, function: Callable, description: str
    ) -> None:
        """Registra una herramienta personalizada en el sistema."""
        self.custom_tools[name] = {"function": function, "description": description}

    def get_tool_descriptions(self) -> list[dict[str, Any]]:
        """Devuelve la lista de herramientas disponibles con nombre, descripción y parámetros."""
        tools = [
            {
                "name": "greeting",
                "description": (
                    "Handle conversational messages that need no knowledge lookup: "
                    "greetings, farewells, thanks, or questions about the system's "
                    "capabilities."
                ),
                "parameters": {},
            },
            {
                "name": "out_of_scope",
                "description": (
                    "Handle questions clearly unrelated to Sherlock Holmes or his "
                    "literary universe (weather, sports, science, politics, etc.)."
                ),
                "parameters": {},
            },
            {
                "name": "skills",
                "description": (
                    "Describe the system's capabilities when the user explicitly asks "
                    "what the assistant can do or help with."
                ),
                "parameters": {},
            },
            {
                "name": "vector_search",
                "description": (
                    "Search for relevant story passages using semantic similarity. "
                    "Best for descriptive or conceptual questions: "
                    "'How does Holmes solve the case?', "
                    "'Describe the relationship between Holmes and Watson'."
                ),
                "parameters": {"query": "The search query in natural language"},
            },
            {
                "name": "fulltext_search",
                "description": (
                    "Search story passages by exact keyword matching. "
                    "Best for finding specific textual mentions: proper names, "
                    "direct quotes, or concrete terms that appear verbatim in the text."
                ),
                "parameters": {"query": "The keyword query"},
            },
            {
                "name": "hybrid_search",
                "description": (
                    "Search using both semantic similarity and keyword matching. "
                    "Best when both the meaning and specific wording matter."
                ),
                "parameters": {"query": "The search query in natural language"},
            },
            {
                "name": "text2cypher",
                "description": (
                    "Query the knowledge graph using natural language. "
                    "Best for structured questions: counts, enumerations, specific "
                    "attributes, or graph traversals — "
                    "'How many characters appear in Silver Blaze?', "
                    "'What crimes does Holmes investigate?', "
                    "'Which locations are associated with Moriarty?'."
                ),
                "parameters": {"query": "The question in natural language"},
            },
        ]

        for name, tool_info in self.custom_tools.items():
            tools.append({
                "name": name,
                "description": tool_info["description"],
                "parameters": {},
            })

        return tools

    def execute_tool(self, tool_name: str, **kwargs) -> dict[str, Any]:
        """Ejecuta la herramienta seleccionada y devuelve un dict con estructura uniforme.

        Estructura de retorno:
            {
                "tool": str,            # nombre de la herramienta usada
                "results": list,        # resultados crudos del retriever
                "context": list[str],   # textos para pasar al LLM como contexto
                "direct_response": str, # solo para greeting / out_of_scope / skills
                "cypher": str,          # solo para text2cypher
            }

        Args:
            tool_name: Nombre de la herramienta a ejecutar.
            **kwargs: Parámetros requeridos por la herramienta (p. ej. query=...).

        Returns:
            Dict con los campos descritos arriba.

        Raises:
            ValueError: Si tool_name no corresponde a ninguna herramienta registrada.
        """
        if tool_name == "greeting":
            return {
                "tool": tool_name,
                "results": [],
                "context": [],
                "direct_response": self._GREETING_RESPONSE,
            }

        if tool_name == "out_of_scope":
            return {
                "tool": tool_name,
                "results": [],
                "context": [],
                "direct_response": self._OUT_OF_SCOPE_RESPONSE,
            }

        if tool_name == "skills":
            return {
                "tool": tool_name,
                "results": [],
                "context": [],
                "direct_response": self._SKILLS_RESPONSE,
            }

        if tool_name == "vector_search":
            try:
                results = self.vector_retriever.retrieve(kwargs.get("query", ""))
            except Exception as exc:
                logger.error("Error en vector_search: %s", exc)
                results = []
            return {
                "tool": tool_name,
                "results": results,
                "context": [r["text"] for r in results],
            }

        if tool_name == "fulltext_search":
            try:
                results = self.fulltext_retriever.retrieve(kwargs.get("query", ""))
            except Exception as exc:
                logger.error("Error en fulltext_search: %s", exc)
                results = []
            return {
                "tool": tool_name,
                "results": results,
                "context": [r["text"] for r in results],
            }

        if tool_name == "hybrid_search":
            try:
                results = self.hybrid_retriever.retrieve(kwargs.get("query", ""))
            except Exception as exc:
                logger.error("Error en hybrid_search: %s", exc)
                results = []
            return {
                "tool": tool_name,
                "results": results,
                "context": [r["text"] for r in results],
            }

        if tool_name == "text2cypher":
            try:
                cypher, results = self.text2cypher.retrieve(kwargs.get("query", ""))
            except Exception as exc:
                logger.error("Error en text2cypher: %s", exc)
                cypher, results = "", []
            context = [
                "; ".join(f"{k}: {v}" for k, v in r.items()) if isinstance(r, dict) else str(r)
                for r in results
            ]
            return {
                "tool": tool_name,
                "cypher": cypher,
                "results": results,
                "context": context,
            }

        if tool_name in self.custom_tools:
            try:
                results = self.custom_tools[tool_name]["function"](**kwargs)
            except Exception as exc:
                logger.error("Error en custom tool '%s': %s", tool_name, exc)
                results = []
            return {
                "tool": tool_name,
                "results": results,
                "context": results if isinstance(results, list) else [str(results)],
            }

        raise ValueError(f"Herramienta desconocida: '{tool_name}'")
