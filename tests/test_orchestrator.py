"""Unit tests for the EvaluationOrchestrator with mocked JudgeLLMClient."""

import queue
from unittest.mock import MagicMock, patch

import pytest

from src.judge_llm import JudgeLLMError
from src.models import (
    AppConfig,
    Conversation,
    ConversationEvaluation,
    EvaluationSuite,
    Goal,
    GoalAchievement,
    GoalClassification,
    Message,
    MessageRole,
    ProgressEvent,
    ProgressEventType,
)
from src.orchestrator import EvaluationOrchestrator
from src.progress import ProgressEmitter


# --- Fixtures ---


@pytest.fixture
def app_config():
    """Create a test AppConfig."""
    return AppConfig(
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3",
        llm_timeout=30,
    )


@pytest.fixture
def progress_emitter():
    """Create a ProgressEmitter with a subscriber for capturing events."""
    emitter = ProgressEmitter()
    return emitter


@pytest.fixture
def sample_suite():
    """Create a sample EvaluationSuite with two goals."""
    return EvaluationSuite(
        name="Test Suite",
        goals=[
            Goal(
                name="Book Appointment",
                description="Customer wants to book an appointment",
                criteria="Agent successfully schedules an appointment with date and time",
            ),
            Goal(
                name="Check Balance",
                description="Customer wants to check their account balance",
                criteria="Agent provides the current account balance",
            ),
        ],
    )


@pytest.fixture
def sample_conversations():
    """Create sample conversations for testing."""
    return [
        Conversation(
            id="conv-001",
            messages=[
                Message(role=MessageRole.CUSTOMER, content="I want to book an appointment"),
                Message(role=MessageRole.AGENT, content="Sure, when would you like?"),
                Message(role=MessageRole.CUSTOMER, content="Tomorrow at 3pm"),
                Message(role=MessageRole.AGENT, content="Done! Booked for tomorrow at 3pm."),
            ],
        ),
        Conversation(
            id="conv-002",
            messages=[
                Message(role=MessageRole.CUSTOMER, content="What is my balance?"),
                Message(role=MessageRole.AGENT, content="Your balance is $500."),
            ],
        ),
        Conversation(
            id="conv-003",
            messages=[
                Message(role=MessageRole.CUSTOMER, content="Hello, random question"),
                Message(role=MessageRole.AGENT, content="I can help with that."),
            ],
        ),
    ]


# --- Tests ---


class TestEvaluationOrchestrator:
    """Tests for EvaluationOrchestrator.run_evaluation."""

    @patch("src.orchestrator.JudgeLLMClient")
    def test_successful_evaluation_all_classified(
        self, mock_client_class, app_config, progress_emitter, sample_suite
    ):
        """Test that all conversations are evaluated when classification succeeds."""
        conversations = [
            Conversation(
                id="conv-001",
                messages=[
                    Message(role=MessageRole.CUSTOMER, content="Book appointment"),
                    Message(role=MessageRole.AGENT, content="Done!"),
                ],
            ),
            Conversation(
                id="conv-002",
                messages=[
                    Message(role=MessageRole.CUSTOMER, content="Check balance"),
                    Message(role=MessageRole.AGENT, content="$500"),
                ],
            ),
        ]

        # Mock the JudgeLLMClient instance
        mock_judge = MagicMock()
        mock_client_class.return_value = mock_judge

        # Mock classify_goal to return appropriate goals
        mock_judge.classify_goal.side_effect = [
            GoalClassification(
                classified_goal="Book Appointment",
                classification_reasoning="Customer asked to book",
            ),
            GoalClassification(
                classified_goal="Check Balance",
                classification_reasoning="Customer asked about balance",
            ),
        ]

        # Mock evaluate_achievement to return success
        mock_judge.evaluate_achievement.side_effect = [
            GoalAchievement(success=True, explanation="Appointment was booked"),
            GoalAchievement(success=True, explanation="Balance was provided"),
        ]

        orchestrator = EvaluationOrchestrator(app_config, progress_emitter)
        report = orchestrator.run_evaluation(sample_suite, conversations)

        assert report.total_conversations == 2
        assert report.total_successes == 2
        assert report.total_failures == 0
        assert report.overall_success_rate == 1.0
        assert report.unclassified_count == 0
        assert len(report.conversation_evaluations) == 2

    @patch("src.orchestrator.JudgeLLMClient")
    def test_unclassified_conversation_records_failure(
        self, mock_client_class, app_config, progress_emitter, sample_suite
    ):
        """Test that unclassified conversations are recorded as failures."""
        conversations = [
            Conversation(
                id="conv-001",
                messages=[
                    Message(role=MessageRole.CUSTOMER, content="Random question"),
                    Message(role=MessageRole.AGENT, content="I don't know"),
                ],
            ),
        ]

        mock_judge = MagicMock()
        mock_client_class.return_value = mock_judge

        mock_judge.classify_goal.return_value = GoalClassification(
            classified_goal="unclassified",
            classification_reasoning="Does not match any goal",
        )

        orchestrator = EvaluationOrchestrator(app_config, progress_emitter)
        report = orchestrator.run_evaluation(sample_suite, conversations)

        assert report.total_conversations == 1
        assert report.total_successes == 0
        assert report.total_failures == 1
        assert report.unclassified_count == 1

        evaluation = report.conversation_evaluations[0]
        assert evaluation.classified_goal == "unclassified"
        assert evaluation.success is False
        assert "did not match any defined goal" in evaluation.achievement_explanation

        # Verify evaluate_achievement was NOT called
        mock_judge.evaluate_achievement.assert_not_called()

    @patch("src.orchestrator.JudgeLLMClient")
    def test_llm_error_continues_evaluation(
        self, mock_client_class, app_config, progress_emitter, sample_suite
    ):
        """Test that LLM errors on one conversation don't stop the rest."""
        conversations = [
            Conversation(
                id="conv-001",
                messages=[
                    Message(role=MessageRole.CUSTOMER, content="Book appointment"),
                    Message(role=MessageRole.AGENT, content="Done!"),
                ],
            ),
            Conversation(
                id="conv-002",
                messages=[
                    Message(role=MessageRole.CUSTOMER, content="Check balance"),
                    Message(role=MessageRole.AGENT, content="$500"),
                ],
            ),
        ]

        mock_judge = MagicMock()
        mock_client_class.return_value = mock_judge

        # First conversation fails with LLM error, second succeeds
        mock_judge.classify_goal.side_effect = [
            JudgeLLMError("Connection timeout"),
            GoalClassification(
                classified_goal="Check Balance",
                classification_reasoning="Customer asked about balance",
            ),
        ]
        mock_judge.evaluate_achievement.return_value = GoalAchievement(
            success=True, explanation="Balance provided"
        )

        orchestrator = EvaluationOrchestrator(app_config, progress_emitter)
        report = orchestrator.run_evaluation(sample_suite, conversations)

        assert report.total_conversations == 2
        assert report.total_successes == 1
        assert report.total_failures == 1

        # First conversation should have error recorded
        eval_1 = report.conversation_evaluations[0]
        assert eval_1.success is False
        assert eval_1.error == "Connection timeout"

        # Second conversation should succeed
        eval_2 = report.conversation_evaluations[1]
        assert eval_2.success is True
        assert eval_2.error is None

    @patch("src.orchestrator.JudgeLLMClient")
    def test_achievement_error_records_failure(
        self, mock_client_class, app_config, progress_emitter, sample_suite
    ):
        """Test that errors during achievement evaluation are handled gracefully."""
        conversations = [
            Conversation(
                id="conv-001",
                messages=[
                    Message(role=MessageRole.CUSTOMER, content="Book appointment"),
                    Message(role=MessageRole.AGENT, content="Done!"),
                ],
            ),
        ]

        mock_judge = MagicMock()
        mock_client_class.return_value = mock_judge

        mock_judge.classify_goal.return_value = GoalClassification(
            classified_goal="Book Appointment",
            classification_reasoning="Customer asked to book",
        )
        mock_judge.evaluate_achievement.side_effect = JudgeLLMError(
            "Failed to parse response"
        )

        orchestrator = EvaluationOrchestrator(app_config, progress_emitter)
        report = orchestrator.run_evaluation(sample_suite, conversations)

        assert report.total_conversations == 1
        assert report.total_successes == 0
        assert report.total_failures == 1

        evaluation = report.conversation_evaluations[0]
        assert evaluation.success is False
        assert evaluation.error == "Failed to parse response"

    @patch("src.orchestrator.JudgeLLMClient")
    def test_progress_events_emitted_correctly(
        self, mock_client_class, app_config, progress_emitter, sample_suite
    ):
        """Test that progress events are emitted in the correct order."""
        conversations = [
            Conversation(
                id="conv-001",
                messages=[
                    Message(role=MessageRole.CUSTOMER, content="Book appointment"),
                    Message(role=MessageRole.AGENT, content="Done!"),
                ],
            ),
            Conversation(
                id="conv-002",
                messages=[
                    Message(role=MessageRole.CUSTOMER, content="Check balance"),
                    Message(role=MessageRole.AGENT, content="$500"),
                ],
            ),
        ]

        mock_judge = MagicMock()
        mock_client_class.return_value = mock_judge

        mock_judge.classify_goal.side_effect = [
            GoalClassification(
                classified_goal="Book Appointment",
                classification_reasoning="Booking",
            ),
            GoalClassification(
                classified_goal="Check Balance",
                classification_reasoning="Balance",
            ),
        ]
        mock_judge.evaluate_achievement.side_effect = [
            GoalAchievement(success=True, explanation="Booked"),
            GoalAchievement(success=False, explanation="Not provided"),
        ]

        # Subscribe to capture events
        event_queue = progress_emitter.subscribe()

        orchestrator = EvaluationOrchestrator(app_config, progress_emitter)
        orchestrator.run_evaluation(sample_suite, conversations)

        # Collect all events
        events = []
        while not event_queue.empty():
            events.append(event_queue.get_nowait())

        # Verify event structure
        assert events[0].event_type == ProgressEventType.EVALUATION_STARTED
        assert events[0].total == 2

        # For 2 conversations: in_progress, completed, in_progress, completed
        assert events[1].event_type == ProgressEventType.EVALUATION_IN_PROGRESS
        assert events[1].conversation_id == "conv-001"
        assert events[1].current == 1

        assert events[2].event_type == ProgressEventType.EVALUATION_COMPLETED
        assert events[2].conversation_id == "conv-001"
        assert events[2].classified_goal == "Book Appointment"
        assert events[2].success is True

        assert events[3].event_type == ProgressEventType.EVALUATION_IN_PROGRESS
        assert events[3].conversation_id == "conv-002"
        assert events[3].current == 2

        assert events[4].event_type == ProgressEventType.EVALUATION_COMPLETED
        assert events[4].conversation_id == "conv-002"
        assert events[4].classified_goal == "Check Balance"
        assert events[4].success is False

        assert events[5].event_type == ProgressEventType.EVALUATION_RUN_COMPLETED
        assert events[5].overall_success_rate == 0.5
        assert events[5].duration_seconds is not None

    @patch("src.orchestrator.JudgeLLMClient")
    def test_verify_connection_called_before_evaluation(
        self, mock_client_class, app_config, progress_emitter, sample_suite
    ):
        """Test that verify_connection is called before any evaluations."""
        conversations = [
            Conversation(
                id="conv-001",
                messages=[
                    Message(role=MessageRole.CUSTOMER, content="Hello"),
                    Message(role=MessageRole.AGENT, content="Hi"),
                ],
            ),
        ]

        mock_judge = MagicMock()
        mock_client_class.return_value = mock_judge
        mock_judge.verify_connection.side_effect = JudgeLLMError(
            "Cannot connect to Ollama at http://localhost:11434"
        )

        orchestrator = EvaluationOrchestrator(app_config, progress_emitter)

        with pytest.raises(JudgeLLMError, match="Cannot connect"):
            orchestrator.run_evaluation(sample_suite, conversations)

        # verify_connection was called
        mock_judge.verify_connection.assert_called_once()
        # classify_goal should NOT have been called
        mock_judge.classify_goal.assert_not_called()

    @patch("src.orchestrator.JudgeLLMClient")
    def test_empty_conversations_list(
        self, mock_client_class, app_config, progress_emitter, sample_suite
    ):
        """Test evaluation with no conversations produces empty report."""
        conversations = []

        mock_judge = MagicMock()
        mock_client_class.return_value = mock_judge

        orchestrator = EvaluationOrchestrator(app_config, progress_emitter)
        report = orchestrator.run_evaluation(sample_suite, conversations)

        assert report.total_conversations == 0
        assert report.total_successes == 0
        assert report.total_failures == 0
        assert report.overall_success_rate == 0.0
        assert len(report.conversation_evaluations) == 0

    @patch("src.orchestrator.JudgeLLMClient")
    def test_classified_goal_not_in_suite(
        self, mock_client_class, app_config, progress_emitter, sample_suite
    ):
        """Test handling when LLM returns a goal name not in the suite."""
        conversations = [
            Conversation(
                id="conv-001",
                messages=[
                    Message(role=MessageRole.CUSTOMER, content="Something"),
                    Message(role=MessageRole.AGENT, content="Response"),
                ],
            ),
        ]

        mock_judge = MagicMock()
        mock_client_class.return_value = mock_judge

        mock_judge.classify_goal.return_value = GoalClassification(
            classified_goal="Nonexistent Goal",
            classification_reasoning="Hallucinated goal",
        )

        orchestrator = EvaluationOrchestrator(app_config, progress_emitter)
        report = orchestrator.run_evaluation(sample_suite, conversations)

        evaluation = report.conversation_evaluations[0]
        assert evaluation.success is False
        assert "not found in evaluation suite" in evaluation.achievement_explanation
        # evaluate_achievement should NOT have been called
        mock_judge.evaluate_achievement.assert_not_called()

    @patch("src.orchestrator.JudgeLLMClient")
    def test_report_contains_correct_suite_name(
        self, mock_client_class, app_config, progress_emitter, sample_suite
    ):
        """Test that the report contains the correct suite name."""
        conversations = [
            Conversation(
                id="conv-001",
                messages=[
                    Message(role=MessageRole.CUSTOMER, content="Hello"),
                    Message(role=MessageRole.AGENT, content="Hi"),
                ],
            ),
        ]

        mock_judge = MagicMock()
        mock_client_class.return_value = mock_judge

        mock_judge.classify_goal.return_value = GoalClassification(
            classified_goal="Book Appointment",
            classification_reasoning="Booking",
        )
        mock_judge.evaluate_achievement.return_value = GoalAchievement(
            success=True, explanation="Done"
        )

        orchestrator = EvaluationOrchestrator(app_config, progress_emitter)
        report = orchestrator.run_evaluation(sample_suite, conversations)

        assert report.suite_name == "Test Suite"

    @patch("src.orchestrator.JudgeLLMClient")
    def test_conversation_messages_preserved_in_evaluation(
        self, mock_client_class, app_config, progress_emitter, sample_suite
    ):
        """Test that conversation messages are preserved in the evaluation result."""
        messages = [
            Message(role=MessageRole.CUSTOMER, content="Book appointment"),
            Message(role=MessageRole.AGENT, content="When?"),
            Message(role=MessageRole.CUSTOMER, content="Tomorrow"),
            Message(role=MessageRole.AGENT, content="Done!"),
        ]
        conversations = [
            Conversation(id="conv-001", messages=messages),
        ]

        mock_judge = MagicMock()
        mock_client_class.return_value = mock_judge

        mock_judge.classify_goal.return_value = GoalClassification(
            classified_goal="Book Appointment",
            classification_reasoning="Booking",
        )
        mock_judge.evaluate_achievement.return_value = GoalAchievement(
            success=True, explanation="Booked"
        )

        orchestrator = EvaluationOrchestrator(app_config, progress_emitter)
        report = orchestrator.run_evaluation(sample_suite, conversations)

        evaluation = report.conversation_evaluations[0]
        assert len(evaluation.conversation) == 4
        assert evaluation.conversation[0].content == "Book appointment"
        assert evaluation.conversation[3].content == "Done!"

    @patch("src.orchestrator.JudgeLLMClient")
    def test_mixed_results_correct_statistics(
        self, mock_client_class, app_config, progress_emitter, sample_suite
    ):
        """Test that mixed success/failure/unclassified produces correct stats."""
        conversations = [
            Conversation(
                id="conv-001",
                messages=[Message(role=MessageRole.CUSTOMER, content="Book")],
            ),
            Conversation(
                id="conv-002",
                messages=[Message(role=MessageRole.CUSTOMER, content="Balance")],
            ),
            Conversation(
                id="conv-003",
                messages=[Message(role=MessageRole.CUSTOMER, content="Random")],
            ),
        ]

        mock_judge = MagicMock()
        mock_client_class.return_value = mock_judge

        mock_judge.classify_goal.side_effect = [
            GoalClassification(
                classified_goal="Book Appointment",
                classification_reasoning="Booking",
            ),
            GoalClassification(
                classified_goal="Check Balance",
                classification_reasoning="Balance",
            ),
            GoalClassification(
                classified_goal="unclassified",
                classification_reasoning="No match",
            ),
        ]
        mock_judge.evaluate_achievement.side_effect = [
            GoalAchievement(success=True, explanation="Booked"),
            GoalAchievement(success=False, explanation="Not provided"),
        ]

        orchestrator = EvaluationOrchestrator(app_config, progress_emitter)
        report = orchestrator.run_evaluation(sample_suite, conversations)

        assert report.total_conversations == 3
        assert report.total_successes == 1
        assert report.total_failures == 2
        assert report.unclassified_count == 1
        assert report.overall_success_rate == pytest.approx(1 / 3)

        # Check goal summaries
        goal_names = {gs.goal_name for gs in report.goal_summaries}
        assert "Book Appointment" in goal_names
        assert "Check Balance" in goal_names

        book_summary = next(
            gs for gs in report.goal_summaries if gs.goal_name == "Book Appointment"
        )
        assert book_summary.conversations_classified == 1
        assert book_summary.successes == 1
        assert book_summary.success_rate == 1.0

        balance_summary = next(
            gs for gs in report.goal_summaries if gs.goal_name == "Check Balance"
        )
        assert balance_summary.conversations_classified == 1
        assert balance_summary.successes == 0
        assert balance_summary.success_rate == 0.0
