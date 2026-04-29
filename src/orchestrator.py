"""Evaluation Orchestrator for coordinating the two-step evaluation pipeline.

Iterates through conversations, coordinates goal classification and achievement
evaluation for each, collects results, emits progress events, and builds the
final EvaluationReport.
"""

import time

from . import report
from .judge_llm import JudgeLLMClient, JudgeLLMError
from .models import (
    AppConfig,
    Conversation,
    ConversationEvaluation,
    EvaluationReport,
    EvaluationSuite,
    ProgressEvent,
    ProgressEventType,
)
from .progress import ProgressEmitter


class EvaluationOrchestrator:
    """Coordinates execution of the two-step evaluation pipeline.

    For each conversation:
    1. Classify which goal the customer was pursuing
    2. If classified (not 'unclassified'), evaluate goal achievement
    3. If unclassified, record as failed with explanation

    Emits progress events throughout. Continues on individual failures.
    """

    def __init__(self, config: AppConfig, progress_emitter: ProgressEmitter):
        """Initialize with app config and progress emitter.

        Args:
            config: Application configuration with Ollama connection details.
            progress_emitter: Emitter for publishing progress events.
        """
        self.config = config
        self.progress_emitter = progress_emitter

    def run_evaluation(
        self, suite: EvaluationSuite, conversations: list[Conversation]
    ) -> EvaluationReport:
        """Evaluate all conversations against the suite goals.

        Creates a JudgeLLMClient, verifies the Ollama connection, then evaluates
        each conversation through the two-step pipeline. Collects results and
        builds the final report.

        Args:
            suite: The EvaluationSuite defining the goals to evaluate against.
            conversations: The list of Conversations to evaluate.

        Returns:
            An EvaluationReport with all results and aggregated statistics.

        Raises:
            JudgeLLMError: If the Ollama connection verification fails.
        """
        start_time = time.time()

        # Create the Judge LLM client
        judge = JudgeLLMClient(
            base_url=self.config.ollama_base_url,
            model=self.config.ollama_model or "",
            timeout=self.config.llm_timeout,
        )

        # Verify Ollama connection before starting
        judge.verify_connection()

        total = len(conversations)

        # Emit evaluation_started
        self.progress_emitter.emit(
            ProgressEvent(
                event_type=ProgressEventType.EVALUATION_STARTED,
                total=total,
                message=f"Starting evaluation: {suite.name} ({total} conversations)",
            )
        )

        evaluations: list[ConversationEvaluation] = []

        for idx, conversation in enumerate(conversations, 1):
            # Emit evaluation_in_progress
            self.progress_emitter.emit(
                ProgressEvent(
                    event_type=ProgressEventType.EVALUATION_IN_PROGRESS,
                    conversation_id=conversation.id,
                    current=idx,
                    total=total,
                    message=f"Evaluating conversation {idx}/{total}: {conversation.id}",
                )
            )

            evaluation = self._evaluate_conversation(
                conversation, suite, idx, total
            )
            evaluations.append(evaluation)

            # Emit evaluation_completed
            self.progress_emitter.emit(
                ProgressEvent(
                    event_type=ProgressEventType.EVALUATION_COMPLETED,
                    conversation_id=conversation.id,
                    classified_goal=evaluation.classified_goal,
                    success=evaluation.success,
                    current=idx,
                    total=total,
                    message=(
                        f"Completed {conversation.id}: "
                        f"goal={evaluation.classified_goal}, "
                        f"success={evaluation.success}"
                    ),
                    conversation_evaluation=evaluation,
                )
            )

        # Build the report
        duration = time.time() - start_time
        evaluation_report = report.build_report(suite, evaluations, duration)

        # Emit evaluation_run_completed
        self.progress_emitter.emit(
            ProgressEvent(
                event_type=ProgressEventType.EVALUATION_RUN_COMPLETED,
                overall_success_rate=evaluation_report.overall_success_rate,
                duration_seconds=duration,
                message=(
                    f"Evaluation complete: "
                    f"{evaluation_report.overall_success_rate:.1%} success rate "
                    f"in {duration:.1f}s"
                ),
            )
        )

        return evaluation_report

    def _evaluate_conversation(
        self,
        conversation: Conversation,
        suite: EvaluationSuite,
        current: int,
        total: int,
    ) -> ConversationEvaluation:
        """Evaluate a single conversation through the two-step pipeline.

        Args:
            conversation: The conversation to evaluate.
            suite: The evaluation suite with goals.
            current: Current evaluation number (for progress).
            total: Total number of evaluations (for progress).

        Returns:
            A ConversationEvaluation with the results.
        """
        judge = JudgeLLMClient(
            base_url=self.config.ollama_base_url,
            model=self.config.ollama_model or "",
            timeout=self.config.llm_timeout,
        )

        try:
            # Step 1: Classify goal
            classification = judge.classify_goal(conversation, suite.goals)

            if classification.classified_goal == "unclassified":
                # Unclassified - record as failed, skip achievement
                return ConversationEvaluation(
                    conversation_id=conversation.id,
                    classified_goal="unclassified",
                    success=False,
                    classification_reasoning=classification.classification_reasoning,
                    achievement_explanation="Conversation did not match any defined goal",
                    conversation=conversation.messages,
                )

            # Step 2: Evaluate achievement for the classified goal
            matching_goal = next(
                (g for g in suite.goals if g.name == classification.classified_goal),
                None,
            )

            if matching_goal is None:
                # LLM returned a goal name that doesn't exist in the suite
                return ConversationEvaluation(
                    conversation_id=conversation.id,
                    classified_goal=classification.classified_goal,
                    success=False,
                    classification_reasoning=classification.classification_reasoning,
                    achievement_explanation=(
                        f"Classified goal '{classification.classified_goal}' "
                        f"not found in evaluation suite"
                    ),
                    conversation=conversation.messages,
                )

            achievement = judge.evaluate_achievement(conversation, matching_goal)

            return ConversationEvaluation(
                conversation_id=conversation.id,
                classified_goal=classification.classified_goal,
                success=achievement.success,
                classification_reasoning=classification.classification_reasoning,
                achievement_explanation=achievement.explanation,
                conversation=conversation.messages,
            )

        except JudgeLLMError as e:
            # Record as failed with error, continue to next conversation
            return ConversationEvaluation(
                conversation_id=conversation.id,
                classified_goal="unclassified",
                success=False,
                classification_reasoning="",
                achievement_explanation="",
                conversation=conversation.messages,
                error=str(e),
            )
