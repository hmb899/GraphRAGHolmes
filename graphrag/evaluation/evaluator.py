import ast
import re as _re
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from ..agents import AgenticRAG
from ..graph.neo4j_manager import Neo4jManager
from ..llm.gemini_client import GeminiClient, ModelTier

_MAX_CONTEXT_CHARS = 4000  # per-call limit to avoid truncated JSON responses


def _vprint(verbose: bool, *args: Any, **kwargs: Any) -> None:
    if verbose:
        print(*args, **kwargs)


def _truncate_context(chunks: list[str], max_chars: int = _MAX_CONTEXT_CHARS) -> str:
    """Join context chunks and hard-truncate to avoid LLM token-limit JSON truncation."""
    joined = "\n".join(chunks)
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "\n[context truncated]"
    return joined


def _is_no_retrieval_needed(text: str) -> bool:
    """Returns True for meta-responses that do not require knowledge retrieval.

    Covers: KB-abstention, scope refusals, and conversational/greeting replies.
    For all of these, empty retrieved context is the *correct* system behaviour.
    """
    t = text.lower().strip()
    patterns = [
        # English patterns
        r"not in the knowledge base",
        r"not in the database",
        r"not available",
        r"not provided",
        r"cannot be determined",
        r"can't be determined",
        r"no information",
        r"insufficient information",
        r"not enough information",
        r"no relevant information",
        r"information is unavailable",
        r"outside my scope",
        r"outside the scope",
        r"i only answer",
        r"cannot answer",
        r"i am a knowledge assistant",
        r"i can answer questions about",
        r"you are welcome",
        r"goodbye",
        r"hello",
        # Spanish patterns (our system uses Spanish responses)
        r"esta información no se encuentra",
        r"fuera de mi ámbito",
        r"solo puedo responder",
        r"especializado en las historias",
        r"hola",
        r"adiós",
        r"gracias",
        r"no tengo información",
        r"no se encuentra en la base",
    ]
    return any(_re.search(p, t) for p in patterns)


def _is_abstention_answer(text: str) -> bool:
    """Returns True when the answer is a 'not in KB' meta-statement.

    These are not factual claims and should always be considered faithful.
    """
    t = text.lower().strip()
    return bool(
        _re.search(
            r"not in the knowledge base"
            r"|this information is not"
            r"|esta información no se encuentra",
            t,
        )
    )


# ── Pydantic schemas for schema-constrained LLM outputs ─────────────────────

class _AttributionResult(BaseModel):
    sentences: list[str]
    attributions: list[int]
    recall: float
    reasoning: str


class _Statements(BaseModel):
    statements: list[str]


class _Verification(BaseModel):
    verdicts: list[int]
    reasoning: list[str]


class _ClassificationItem(BaseModel):
    statement: str
    category: str
    reason: str


class _Classification(BaseModel):
    classifications: list[_ClassificationItem]
    tp_count: int
    fp_count: int
    fn_count: int


class RAGEvaluator:
    """
    Evaluates a GraphRAG pipeline following the RAGAS methodology:

    1. run_benchmark()    – executes Cypher ground truths, calls the agent,
                           records answers, contexts and latency
    2. evaluate_results() – scores each row with context_recall, faithfulness
                           and answer_correctness
    3. print_summary()    – aggregated metrics table

    Each RAGAS metric maps to the exact prompt goals described in class:
      • context_recall     : binary attribution per sentence (Yes/No)
      • faithfulness       : two-step statement decomposition + verification
      • answer_correctness : TP/FP/FN classification against ground truth
    """

    def __init__(self, rag: AgenticRAG, neo4j_manager: Neo4jManager) -> None:
        self.rag = rag
        self.neo4j = neo4j_manager
        self.client = GeminiClient()

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def load_dataset(self, csv_path: str) -> pd.DataFrame:
        """Load benchmark CSV (semicolon-delimited, columns: question, cypher)."""
        return pd.read_csv(csv_path, delimiter=";")

    def get_answer(self, question: str) -> tuple[Optional[str], list[str]]:
        """Run the agent and return (answer, retrieved_contexts).

        Returns (None, []) on failure so the benchmark loop never crashes.
        """
        try:
            result = self.rag.answer(question)
            answer = result["answer"]
            context = result["iterations"][-1]["retrieval"].get("context", [])
            return answer, context
        except Exception:
            return None, []

    # ------------------------------------------------------------------
    # RAGAS metrics
    # ------------------------------------------------------------------

    def evaluate_context_recall(
        self,
        question: str,
        ground_truth: str,
        retrieved_context: list[str],
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Robust Context Recall Evaluator

        Measures whether the retrieved context contains the information necessary
        to support the ground-truth answer.

        Handles important edge cases:
        ---------------------------------------------------------
        1. Empty retrieval + abstention ground truth
        Example:
        GT = "This information is not in the knowledge base."

        If no context was retrieved, this is considered CORRECT retrieval behavior
        and recall = 1.0

        2. Empty retrieval + factual ground truth
        Recall = 0.0

        3. Semantic matching (not exact wording)

        Returns:
            {
                "sentences": [...],
                "attributions": [0/1...],
                "reasoning": str,
                "recall": float
            }
        """
        _vprint(verbose, "\n── context_recall ──────────────────────────────────")
        _vprint(verbose, f"  Question    : {question}")
        _vprint(verbose, f"  Ground truth: {ground_truth}")
        _vprint(verbose, f"  Context chunks retrieved: {len(retrieved_context)}")

        def _has_context(ctx: list[str]) -> bool:
            if not ctx:
                return False
            return any(str(x).strip() for x in ctx)

        context_exists = _has_context(retrieved_context)
        # Prepend the question so the LLM can interpret bare-fact ground truths
        # (e.g. GT="Germany" + Q="Where was Einstein born?" → "born in Germany")
        # and so structured graph results (e.g. "{'count(p)': 2}") get question framing.
        context_str = f"[Question: {question}]\n\n" + _truncate_context(retrieved_context)

        # ---------------------------------------------------------
        # CASE A: Empty retrieval
        # ---------------------------------------------------------
        if not context_exists:
            _vprint(verbose, "\n  [Step 1] No retrieved context detected.")

            # Greetings, scope refusals, and KB-abstention responses all require
            # no retrieval — empty context is the correct system behaviour here.
            if _is_no_retrieval_needed(ground_truth):
                result = {
                    "sentences": [ground_truth],
                    "attributions": [1],
                    "reasoning": (
                        "No context was retrieved, and the ground truth is a "
                        "conversational, scope-refusal, or KB-abstention response "
                        "that requires no knowledge lookup. Recall = 1.0."
                    ),
                    "recall": 1.0,
                }

            else:
                result = {
                    "sentences": [ground_truth],
                    "attributions": [0],
                    "reasoning": (
                        "No context was retrieved, but the ground truth contains "
                        "answerable factual information. Retrieval failed to surface evidence."
                    ),
                    "recall": 0.0,
                }

            if verbose:
                for s, a in zip(result["sentences"], result["attributions"]):
                    label = "✓ attributed" if a else "✗ not found"
                    _vprint(verbose, f"    [{label}] {s}")

                _vprint(verbose, f"\n  Reasoning : {result['reasoning']}")
                _vprint(verbose, f"  Score     : {result['recall']:.3f}")
                _vprint(verbose, "────────────────────────────────────────────────────")

            return result

        # ---------------------------------------------------------
        # CASE B: Context exists -> normal attribution
        # ---------------------------------------------------------
        system_prompt = (
            "Goal: Determine whether the ground-truth answer is supported by the retrieved context.\n\n"
            "IMPORTANT — short ground truths: if the ground truth is a single word or short phrase "
            "(e.g. 'Germany', '1915', '2 person(s)'), use the question to interpret it as a full "
            "statement (e.g. question='Where was Einstein born?', GT='Germany' → "
            "'Einstein was born in Germany'). Then check if that statement is supported.\n\n"
            "IMPORTANT — structured results: the context may contain raw graph query results "
            "like {'count(p)': 2} or {'name': 'Ulm'}. Interpret these in light of the question.\n\n"
            "Return attribution = 1 if the ground truth (interpreted as above) is:\n"
            "- explicitly present in the context\n"
            "- semantically equivalent to something in the context\n"
            "- clearly implied by the context\n\n"
            "Return attribution = 0 if unsupported, contradicted, or missing.\n\n"
            "Use semantic meaning, not exact wording."
        )

        user_message = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Ground truth answer:\n{ground_truth}"
        )

        _vprint(
            verbose,
            "\n  [Step 1] Asking the LLM to attribute each ground-truth sentence to context..."
        )

        result_obj = self.client.structured_output(
            user_message,
            _AttributionResult,
            model_tier=ModelTier.FLASH,
            system_instruction=system_prompt,
            temperature=0.0,
        )

        result = result_obj.model_dump()

        # Align lists: LLM may return different lengths for sentences vs attributions.
        sentences = result.get("sentences", [])
        attributions = result.get("attributions", [])
        if len(sentences) != len(attributions):
            min_len = min(len(sentences), len(attributions))
            sentences = sentences[:min_len]
            attributions = attributions[:min_len]
            result["sentences"] = sentences
            result["attributions"] = attributions

        if attributions:
            result["recall"] = sum(attributions) / len(attributions)
        else:
            result["recall"] = 0.0

        # ---------------------------------------------------------
        # Verbose output
        # ---------------------------------------------------------
        if verbose:
            sentences = result.get("sentences", [])

            _vprint(verbose, f"\n  Ground-truth sentences ({len(sentences)}):")

            for s, a in zip(sentences, attributions):
                label = "✓ attributed" if a else "✗ not found"
                _vprint(verbose, f"    [{label}] {s}")

            _vprint(verbose, f"\n  Reasoning : {result.get('reasoning', '')}")
            _vprint(verbose, f"  Score     : {result['recall']:.3f}")
            _vprint(verbose, "────────────────────────────────────────────────────")

        return result

    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        context: list[str],
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Robust faithfulness evaluator.

        Handles two modes automatically:

        1. Retrieval-grounded mode (context exists)
        Measures whether factual claims in the answer are supported by context.

        2. No-context conversational mode
        Prevents unfair hallucination penalties for greetings, assistant-role
        statements, generic capability claims, and harmless conversational text.

        Returns:
            {
                "statements": [...],
                "verdicts": [...],     # 1 supported / 0 unsupported
                "faithfulness": float,
                "reasoning": [...]
            }
        """
        _vprint(verbose, "\n── faithfulness ────────────────────────────────────")
        _vprint(verbose, f"  Question: {question}")
        _vprint(verbose, f"  Answer  : {answer}")

        def _has_context(ctx: list[str]) -> bool:
            if not ctx:
                return False
            return any(str(x).strip() for x in ctx)

        # Prepend question so structured graph results are interpretable in context.
        context_str = f"[Question: {question}]\n\n" + _truncate_context(context)
        has_context = _has_context(context)

        # Remove inline citations like [1], [2]
        answer_clean = _re.sub(r"\s*\[\d+\]", "", answer).strip()

        # "Not in knowledge base" is a meta-statement, not a factual claim.
        # It is always faithful — verifying it against chunks would be meaningless.
        if _is_abstention_answer(answer_clean):
            _vprint(verbose, "\n  Answer is a KB-abstention — faithfulness = 1.0 by convention.")
            _vprint(verbose, "────────────────────────────────────────────────────")
            return {
                "statements": [answer_clean],
                "verdicts": [1],
                "faithfulness": 1.0,
                "reasoning": ["KB-abstention responses make no factual claim and are always faithful."],
            }

        # ---------------------------------------------------------
        # Step 1: Decompose answer into meaningful claims
        # ---------------------------------------------------------
        decompose_prompt = """
    Goal: Given a question and an answer, break the answer into meaningful factual or semantic statements.

    Rules:
    - Use concise standalone statements.
    - Remove pronouns when possible.
    - Ignore citation markers like [1], [2].
    - Do NOT oversplit greetings or conversational intros.
    - Do NOT split coordinated capability lists unless they express materially different claims.
    - Prefer 1 combined statement over many tiny redundant statements.
    """

        _vprint(verbose, "\n  [Step 1] Decomposing the answer into statements...")

        try:
            stmt_obj = self.client.structured_output(
                f"Question: {question}\nAnswer: {answer_clean}",
                _Statements,
                model_tier=ModelTier.FLASH,
                system_instruction=decompose_prompt,
                temperature=0.0,
            )
            statements = stmt_obj.statements
        except Exception as exc:
            _vprint(verbose, f"  [Step 1 ERROR] {exc}")
            return {
                "statements": [],
                "verdicts": [],
                "faithfulness": 0.0,
                "reasoning": [],
            }

        if verbose:
            _vprint(verbose, f"  Statements extracted ({len(statements)}):")
            for i, s in enumerate(statements, 1):
                _vprint(verbose, f"    {i}. {s}")

        if not statements:
            return {
                "statements": [],
                "verdicts": [],
                "faithfulness": 1.0,
                "reasoning": [],
            }

        # ---------------------------------------------------------
        # MODE A: No context supplied
        # ---------------------------------------------------------
        if not has_context:
            _vprint(verbose, "\n  [Step 2] No external context detected.")
            _vprint(verbose, "           Using conversational truthfulness mode...")

            verify_prompt = (
                "Goal: Judge whether each statement is acceptable when NO external evidence context exists.\n\n"
                "Mark verdict = 1 for:\n"
                "- greetings and farewells\n"
                "- conversational text\n"
                "- assistant role/identity statements\n"
                "- generic capability or helpfulness statements\n"
                "- scope refusals ('This question is outside my scope', 'I only answer about X')\n"
                "- harmless general descriptions\n\n"
                "Mark verdict = 0 for:\n"
                "- specific factual claims about the world requiring external evidence\n"
                "- numbers, dates, or statistics about external events\n"
                "- biographical or historical claims about specific people or events\n\n"
                "Return exactly one verdict and one reasoning item per statement."
            )

            statements_str = "\n".join(
                f"{i + 1}. {s}" for i, s in enumerate(statements)
            )

            verif_obj = self.client.structured_output(
                f"Question: {question}\n\nStatements:\n{statements_str}",
                _Verification,
                model_tier=ModelTier.FLASH,
                system_instruction=verify_prompt,
                temperature=0.0,
            )

            verdicts = verif_obj.verdicts
            reasoning = verif_obj.reasoning

        # ---------------------------------------------------------
        # MODE B: Context supplied
        # ---------------------------------------------------------
        else:
            _vprint(verbose, "\n  [Step 2] Context detected.")
            _vprint(verbose, "           Using retrieval-grounded mode...")

            verify_prompt = """
    Goal: Judge the faithfulness of each statement using the provided context and question.

    Return verdict = 1 if the statement is:
    - explicitly supported by context
    - semantically equivalent to context
    - clearly implied by context using normal reasoning

    Return verdict = 0 if the statement:
    - adds unsupported new facts
    - exaggerates certainty
    - contradicts context
    - requires speculation

    Rules:
    - Use semantic meaning, not exact wording.
    - Minor paraphrases count as supported.
    - Use the question to resolve shorthand answers.

    Return exactly one verdict and one reasoning item per statement.
    """

            statements_str = "\n".join(
                f"{i + 1}. {s}" for i, s in enumerate(statements)
            )

            verif_obj = self.client.structured_output(
                (
                    f"Question: {question}\n\n"
                    f"Context:\n{context_str}\n\n"
                    f"Statements:\n{statements_str}"
                ),
                _Verification,
                model_tier=ModelTier.FLASH,
                system_instruction=verify_prompt,
                temperature=0.0,
            )

            verdicts = verif_obj.verdicts
            reasoning = verif_obj.reasoning

        # Align: LLM may return different list lengths for verdicts vs statements.
        if len(verdicts) != len(statements):
            min_len = min(len(verdicts), len(statements))
            verdicts = verdicts[:min_len]
            reasoning = reasoning[:min_len]
            statements = statements[:min_len]

        # ---------------------------------------------------------
        # Score
        # ---------------------------------------------------------
        faithfulness_score = (
            sum(verdicts) / len(verdicts)
            if verdicts
            else 1.0
        )

        # ---------------------------------------------------------
        # Verbose output
        # ---------------------------------------------------------
        if verbose:
            _vprint(verbose, "\n  Verdicts:")
            for s, v, r in zip(statements, verdicts, reasoning):
                label = "✓ supported" if v else "✗ unsupported"
                _vprint(verbose, f"    [{label}] {s}")
                if r:
                    _vprint(verbose, f"              → {r}")

            _vprint(verbose, f"\n  Score : {faithfulness_score:.3f}")
            _vprint(verbose, "────────────────────────────────────────────────────")

        return {
            "statements": statements,
            "verdicts": verdicts,
            "faithfulness": faithfulness_score,
            "reasoning": reasoning,
        }

    def evaluate_answer_correctness(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Goal: Given a ground truth and an answer statement, analyze each statement
        and classify it into one of the following categories:

        TP (true positive): Statements present in the answer that are also directly
            supported by one or more statements in the ground truth.
        FP (false positive): Statements present in the answer but not directly
            supported by any statement in the ground truth.
        FN (false negative): Statements found in the ground truth but not present
            in the answer.

        Each statement can only belong to one of these categories.
        Provide a reason for each classification.

        verbose=True prints each step so students can follow the reasoning.
        """
        _vprint(verbose, "\n── answer_correctness ──────────────────────────────")
        _vprint(verbose, f"  Question    : {question}")
        _vprint(verbose, f"  Answer      : {answer}")
        _vprint(verbose, f"  Ground truth: {ground_truth}")

        # Meta-responses (greetings, scope refusals, KB-abstentions) are evaluated
        # as correct when both ground truth and answer belong to the same meta-category.
        # TP/FP/FN classification would penalise extra politeness or cross-language
        # paraphrases — neither of which represents a real error.
        if _is_no_retrieval_needed(ground_truth) and _is_no_retrieval_needed(answer):
            _vprint(verbose, "\n  Both answer and ground truth are meta-responses — correctness = 1.0 by convention.")
            _vprint(verbose, "────────────────────────────────────────────────────")
            return {
                "classifications": [],
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "precision": 1.0,
                "recall": 1.0,
                "answer_correctness": 1.0,
            }

        breakdown_prompt = (
            "Goal: Given a question and an answer, analyze the complexity of each sentence "
            "in the answer. Break down each sentence into one or more fully understandable "
            "statements. Ensure that no pronouns are used in any statement. "
            "Citation markers such as [1], [2] are NOT statements — ignore them entirely."
        )

        def _get_statements(text: str) -> list[str]:
            clean = _re.sub(r"\s*\[\d+\]", "", text).strip()
            obj = self.client.structured_output(
                f"Question: {question}\nText: {clean}",
                _Statements,
                model_tier=ModelTier.FLASH,
                system_instruction=breakdown_prompt,
                temperature=0.0,
            )
            return obj.statements

        _vprint(verbose, "\n  [Step 1] Decomposing the answer into statements...")
        answer_statements = _get_statements(answer)
        _vprint(verbose, f"  Answer statements ({len(answer_statements)}):")
        for i, s in enumerate(answer_statements, 1):
            _vprint(verbose, f"    {i}. {s}")

        _vprint(verbose, "\n  [Step 2] Decomposing the ground truth into statements...")
        truth_statements = _get_statements(ground_truth)
        _vprint(verbose, f"  Ground truth statements ({len(truth_statements)}):")
        for i, s in enumerate(truth_statements, 1):
            _vprint(verbose, f"    {i}. {s}")

        classify_prompt = """
Goal: Compare answer statements against ground truth statements using semantic meaning,
not exact wording.

Classify each statement:

TP (true positive):
A statement in the answer that is explicitly stated OR clearly implied by the ground truth.
Paraphrases, perspective shifts, grammatical rewrites, and equivalent restatements count as TP.

FP (false positive):
A statement in the answer that introduces materially new facts, unsupported claims,
stronger specificity, or contradictions not justified by the ground truth.

FN (false negative):
An important statement in the ground truth missing from the answer.

Rules:
- Evaluate meaning, not wording.
- "You can ask me about X" and "I can help with X" are equivalent.
- Specific examples named in the ground truth may be restated as examples.
- Minor stylistic additions should not count as FP.
- Only penalize genuinely new factual content.

tp_count, fp_count, fn_count must equal the classified totals.
Provide concise reasons.
"""

        answer_str = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(answer_statements))
        truth_str = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(truth_statements))

        _vprint(verbose, "\n  [Step 3] Classifying each statement as TP / FP / FN...")
        classif_obj = self.client.structured_output(
            (
                f"Answer statements:\n{answer_str}\n\n"
                f"Ground truth statements:\n{truth_str}"
            ),
            _Classification,
            model_tier=ModelTier.FLASH,
            system_instruction=classify_prompt,
            temperature=0.0,
        )

        # Count from the actual classification list, not from LLM-reported totals
        # (LLM-provided tp_count/fp_count/fn_count can be inconsistent with the list).
        tp = sum(1 for c in classif_obj.classifications if c.category == "TP")
        fp = sum(1 for c in classif_obj.classifications if c.category == "FP")
        fn = sum(1 for c in classif_obj.classifications if c.category == "FN")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        if verbose:
            _vprint(verbose, f"  Classifications:")
            for c in classif_obj.classifications:
                _vprint(verbose, f"    [{c.category}] {c.statement} — {c.reason}")
            _vprint(verbose, f"\n  TP={tp}  FP={fp}  FN={fn}")
            _vprint(verbose, f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
            _vprint(verbose, "────────────────────────────────────────────────────")

        return {
            "classifications": [c.model_dump() for c in classif_obj.classifications],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "answer_correctness": f1,
        }

    # ------------------------------------------------------------------
    # Pipeline: run_benchmark → evaluate_results → print_summary
    # ------------------------------------------------------------------

    def run_benchmark(self, dataset: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Execute the ground-truth Cypher queries, call the agent for each question,
        and record answers, contexts and latency (mirrors Listing 8.2).

        Input DataFrame must have columns: question, cypher
        Output DataFrame adds:  ground_truth, answer, latency, retrieved_contexts
        Pass verbose=True to print each question, its ground truth and the agent's answer.
        """
        answers: list = []
        ground_truths: list = []
        latencies: list = []
        contexts: list = []

        for i, (_, row) in enumerate(tqdm(dataset.iterrows(), total=len(dataset), desc="Processing rows"), 1):
            _vprint(verbose, f"\n{'━'*60}")
            _vprint(verbose, f"  [{i}/{len(dataset)}] {row['question']}")
            _vprint(verbose, f"{'━'*60}")

            # Execute Cypher to obtain the ground truth dynamically
            _vprint(verbose, f"  [Step 1] Executing Cypher ground truth...")
            _vprint(verbose, f"           {row['cypher']}")
            gt_records = self.neo4j.execute_query(row["cypher"])
            gt_values = []
            for r in gt_records:
                val = r.get("ground_truth") if isinstance(r, dict) else str(r)
                if val is not None and str(val) not in ("None", ""):
                    gt_values.append(str(val))
            gt_str = "; ".join(gt_values) if gt_values else "Esta información no se encuentra en la base de conocimiento."
            ground_truths.append(gt_str)
            _vprint(verbose, f"  Ground truth: {gt_str}")

            # Call the agent
            _vprint(verbose, f"\n  [Step 2] Calling the RAG agent...")
            start = datetime.now()
            try:
                answer, context = self.get_answer(row["question"])
            except Exception:
                answer, context = None, []
            elapsed = (datetime.now() - start).total_seconds()
            latencies.append(elapsed)

            _vprint(verbose, f"  Answer  : {answer}")
            _vprint(verbose, f"  Context chunks: {len(context)}")
            _vprint(verbose, f"  Latency : {elapsed:.2f}s")

            answers.append(answer)
            contexts.append(context)

        results = dataset.copy()
        results["ground_truth"] = ground_truths
        results["answer"] = answers
        results["latency"] = latencies
        results["retrieved_contexts"] = contexts
        return results

    def evaluate_results(self, results_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Apply the three RAGAS metrics to every row in the results DataFrame
        and add the scores as new columns (mirrors Listing 8.3 / 8.4).

        Missing answers are replaced with "I don't know" before scoring.
        Pass verbose=True to print the step-by-step reasoning for every row.
        """
        df = results_df.fillna("I don't know").copy()

        recall_scores: list[float] = []
        faithfulness_scores: list[float] = []
        correctness_scores: list[float] = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            if verbose:
                print(f"\n{'━'*60}")
                print(f"  Q: {row['question']}")
                print(f"{'━'*60}")

            contexts = row["retrieved_contexts"]
            if isinstance(contexts, str):
                # When loaded from CSV, lists are serialized as strings like "['a', 'b']".
                # Wrapping with [str] would produce ["['a', 'b']"] — a single fake chunk.
                try:
                    parsed = ast.literal_eval(contexts)
                    contexts = parsed if isinstance(parsed, list) else []
                except (ValueError, SyntaxError):
                    contexts = []
            if not isinstance(contexts, list):
                contexts = []

            recall = self.evaluate_context_recall(
                row["question"], row["ground_truth"], contexts, verbose=verbose
            )
            faith = self.evaluate_faithfulness(
                row["question"], row["answer"], contexts, verbose=verbose
            )
            corr = self.evaluate_answer_correctness(
                row["question"], row["answer"], row["ground_truth"], verbose=verbose
            )

            recall_scores.append(float(recall.get("recall", 0.0)))
            faithfulness_scores.append(float(faith.get("faithfulness", 0.0)))
            correctness_scores.append(float(corr.get("answer_correctness", 0.0)))

        df["context_recall"] = recall_scores
        df["faithfulness"] = faithfulness_scores
        df["answer_correctness"] = correctness_scores
        return df

    def print_summary(self, results_df: pd.DataFrame) -> None:
        """Print a benchmark summary table."""
        print("\n=== Benchmark Summary ===")
        metric_cols = ["answer_correctness", "context_recall", "faithfulness"]
        for col in metric_cols:
            if col in results_df.columns:
                print(f"{col:25s}: {results_df[col].mean():.4f}")
        if "latency" in results_df.columns:
            print(f"{'avg_latency_s':25s}: {results_df['latency'].mean():.2f}")
