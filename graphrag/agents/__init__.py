"""Orquestador agéntico principal del sistema GraphRAGHolmes."""

from typing import Any

from ..graph.neo4j_manager import Neo4jManager
from ..llm.gemini_client import GeminiClient, ModelTier
from .answer_critic import AnswerCritic
from .retriever_router import RetrieverRouter, RouterDecision
from .retriever_tools import RetrieverTools

_ANSWER_SYSTEM_PROMPT = (
    "You are a concise question-answering assistant specialized in Sherlock Holmes stories.\n\n"
    "STRICT RULES — follow all of them:\n"
    "1. Answer ONLY from the provided context. Do not use prior knowledge.\n"
    "2. Start your reply with the answer itself. Never begin with \"I\", \"Let me\", "
    "\"Based on\", \"The context\", or any preamble.\n"
    "3. Do not explain your reasoning or thinking process. Only state facts.\n"
    "4. Keep the answer short: one to three sentences maximum.\n"
    "5. Add an inline citation after each fact using the chunk number, "
    "e.g. \"221B Baker Street [1].\"\n"
    "6. If the context does not contain the answer, reply exactly: "
    "\"Esta información no se encuentra en la base de conocimiento.\"\n"
    "7. Answer in the same language as the question "
    "(Spanish if asked in Spanish, English if asked in English)."
)

_NO_CONTEXT_ANSWER = "Esta información no se encuentra en la base de conocimiento."

_AUTO_PASS_CRITIQUE: dict[str, Any] = {
    "is_complete": True,
    "is_faithful": True,
    "missing_info": [],
    "feedback": "Auto-approved.",
}


class AgenticRAG:
    """Orquestador del sistema agéntico de RAG para Sherlock Holmes.

    Coordina el flujo: routing → retrieval → generación → critique → retry.
    """

    def __init__(self, neo4j_manager: Neo4jManager) -> None:
        self.neo4j = neo4j_manager
        self.client = GeminiClient()
        self.tools = RetrieverTools(neo4j_manager)
        self.router = RetrieverRouter(self.tools)
        self.critic = AnswerCritic()
        self.conversation_history: list[dict[str, str]] = []

    def answer(self, question: str, max_iterations: int = 3) -> dict[str, Any]:
        """Responde una pregunta usando el flujo agéntico completo.

        Flujo por iteración:
          1. Router decide qué retriever usar.
          2. El retriever recupera contexto.
          3. El LLM genera una respuesta a partir del contexto.
          4. El Critic evalúa completitud y fidelidad.
          5. Si la respuesta no es satisfactoria, se refina la pregunta con las
             sub-preguntas del Critic y se itera de nuevo (hasta max_iterations).

        Args:
            question: Pregunta del usuario en lenguaje natural.
            max_iterations: Número máximo de iteraciones de refinamiento.

        Returns:
            Dict con keys: question, answer, iterations, final_critique,
            total_iterations.
        """
        iterations: list[dict[str, Any]] = []
        current_question = question
        answer = _NO_CONTEXT_ANSWER

        for iteration in range(max_iterations):

            # 1. Routing + retrieval
            retrieval_result = self.router.retrieve(
                current_question, self.conversation_history
            )

            # Extraer routing_decision como dict serializable
            routing_decision_obj = retrieval_result.get("routing_decision")
            routing_decision = (
                routing_decision_obj.model_dump()
                if routing_decision_obj is not None
                else {}
            )

            # 2a. Respuesta directa (greeting / out_of_scope / skills)
            if "direct_response" in retrieval_result:
                answer = retrieval_result["direct_response"]
                iterations.append({
                    "iteration": iteration + 1,
                    "question": current_question,
                    "routing_decision": routing_decision,
                    "retrieval": retrieval_result,
                    "answer": answer,
                    "critique": _AUTO_PASS_CRITIQUE,
                })
                break

            context = retrieval_result.get("context", [])

            # 2b. Sin contexto → abstención legítima (no hallucinar)
            if not context:
                answer = _NO_CONTEXT_ANSWER
                iterations.append({
                    "iteration": iteration + 1,
                    "question": current_question,
                    "routing_decision": routing_decision,
                    "retrieval": retrieval_result,
                    "answer": answer,
                    "critique": _AUTO_PASS_CRITIQUE,
                })
                break

            # 2c. Generar respuesta a partir del contexto
            answer = self._generate_answer(question, current_question, context)

            # 3. Evaluar respuesta con el Critic
            critique = self.critic.critique(current_question, context, answer)

            iterations.append({
                "iteration": iteration + 1,
                "question": current_question,
                "routing_decision": routing_decision,
                "retrieval": retrieval_result,
                "answer": answer,
                "critique": critique,
            })

            # 4. Decisión: continuar o terminar
            if critique["is_complete"] and critique["is_faithful"]:
                break

            # Refinar pregunta si hay sub-preguntas y aún quedan iteraciones
            if critique.get("missing_info") and iteration < max_iterations - 1:
                sub_questions = " ".join(critique["missing_info"])
                # Anclar siempre a la pregunta ORIGINAL para no acumular ruido
                current_question = (
                    f"{question}\n\nAdditional aspects to cover: {sub_questions}"
                )

        # Actualizar historial de conversación multi-turno
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

        return {
            "question": question,
            "answer": answer,
            "iterations": iterations,
            "final_critique": iterations[-1]["critique"] if iterations else {},
            "total_iterations": len(iterations),
        }

    def _generate_answer(
        self,
        original_question: str,
        current_question: str,
        context: list[str],
    ) -> str:
        """Genera una respuesta a partir del contexto recuperado.

        Cuando la pregunta ha sido refinada respecto a la original, incluye
        ambas en el prompt para que el LLM no pierda el foco de la pregunta
        que el usuario realmente hizo.

        Args:
            original_question: Pregunta tal como la escribió el usuario.
            current_question: Pregunta (posiblemente refinada) usada en esta iteración.
            context: Fragmentos de texto recuperados como contexto.

        Returns:
            Respuesta generada por el LLM.
        """
        context_str = "\n\n".join(
            f"[{i + 1}] {chunk}" for i, chunk in enumerate(context)
        )

        if current_question != original_question:
            prompt = (
                f"Original question: {original_question}\n"
                f"Refined question: {current_question}\n\n"
                f"Context:\n{context_str}\n\n"
                "Answer the original question using the context above."
            )
        else:
            prompt = f"Context:\n{context_str}\n\nQuestion: {original_question}"

        return self.client.generate(
            prompt=prompt,
            model_tier=ModelTier.FLASH,
            system_instruction=_ANSWER_SYSTEM_PROMPT,
            temperature=0.0,
        )

    def reset_conversation(self) -> None:
        """Reinicia el historial de conversación."""
        self.conversation_history = []


__all__ = [
    "AgenticRAG",
    "RetrieverTools",
    "RetrieverRouter",
    "RouterDecision",
    "AnswerCritic",
]
