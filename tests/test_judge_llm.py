"""Unit tests for the Judge LLM client."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.judge_llm import JudgeLLMClient, JudgeLLMError
from src.models import (
    Conversation,
    Goal,
    GoalAchievement,
    GoalClassification,
    Message,
    MessageRole,
)


# --- Fixtures ---


@pytest.fixture
def client():
    """Create a JudgeLLMClient with default test settings."""
    return JudgeLLMClient(
        base_url="http://localhost:11434",
        model="llama3",
        timeout=60,
    )


@pytest.fixture
def sample_goals():
    """Create a list of sample goals for testing."""
    return [
        Goal(
            name="Check Balance",
            description="Customer wants to check their account balance",
            criteria="The agent provides the current account balance to the customer",
        ),
        Goal(
            name="Reset Password",
            description="Customer wants to reset their password",
            criteria="The agent successfully initiates a password reset for the customer",
        ),
        Goal(
            name="Book Appointment",
            description="Customer wants to schedule an appointment",
            criteria="The agent confirms a specific date and time for the appointment",
        ),
    ]


@pytest.fixture
def sample_conversation():
    """Create a sample conversation for testing."""
    return Conversation(
        id="conv-001",
        messages=[
            Message(role=MessageRole.AGENT, content="Hello! How can I help you today?"),
            Message(role=MessageRole.CUSTOMER, content="I'd like to check my balance."),
            Message(
                role=MessageRole.AGENT,
                content="Sure! Your current balance is $1,234.56.",
            ),
            Message(role=MessageRole.CUSTOMER, content="Thank you!"),
        ],
    )


# --- verify_connection tests ---


class TestVerifyConnection:
    """Tests for verify_connection method."""

    @patch("src.judge_llm.requests.get")
    def test_verify_connection_success(self, mock_get, client):
        """verify_connection succeeds when model is available."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3:latest"},
                {"name": "mistral:latest"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Should not raise
        client.verify_connection()
        mock_get.assert_called_once_with(
            "http://localhost:11434/api/tags", timeout=60
        )

    @patch("src.judge_llm.requests.get")
    def test_verify_connection_model_without_tag(self, mock_get, client):
        """verify_connection matches model name without :tag suffix."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "llama3:latest"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # "llama3" should match "llama3:latest"
        client.verify_connection()

    @patch("src.judge_llm.requests.get")
    def test_verify_connection_unreachable(self, mock_get, client):
        """verify_connection raises JudgeLLMError when Ollama is unreachable."""
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        with pytest.raises(JudgeLLMError) as exc_info:
            client.verify_connection()

        error_msg = str(exc_info.value)
        assert "http://localhost:11434" in error_msg
        assert "llama3" in error_msg

    @patch("src.judge_llm.requests.get")
    def test_verify_connection_model_not_found(self, mock_get, client):
        """verify_connection raises JudgeLLMError when model is not available."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "mistral:latest"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with pytest.raises(JudgeLLMError) as exc_info:
            client.verify_connection()

        error_msg = str(exc_info.value)
        assert "llama3" in error_msg
        assert "http://localhost:11434" in error_msg
        assert "mistral:latest" in error_msg

    @patch("src.judge_llm.requests.get")
    def test_verify_connection_invalid_json_response(self, mock_get, client):
        """verify_connection raises JudgeLLMError on invalid JSON response."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("err", "", 0)
        mock_get.return_value = mock_response

        with pytest.raises(JudgeLLMError) as exc_info:
            client.verify_connection()

        error_msg = str(exc_info.value)
        assert "http://localhost:11434" in error_msg
        assert "llama3" in error_msg


# --- classify_goal tests ---


class TestClassifyGoal:
    """Tests for classify_goal method."""

    @patch("src.judge_llm.requests.post")
    def test_classify_goal_prompt_contains_all_goals(
        self, mock_post, client, sample_goals, sample_conversation
    ):
        """Classification prompt includes all goal names, descriptions, and criteria."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps(
                    {
                        "classified_goal": "Check Balance",
                        "classification_reasoning": "Customer asked about balance.",
                    }
                )
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client.classify_goal(sample_conversation, sample_goals)

        # Verify the prompt sent to Ollama
        call_args = mock_post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        messages = payload["messages"]

        # System prompt should contain all goals
        system_prompt = messages[0]["content"]
        for goal in sample_goals:
            assert goal.name in system_prompt
            assert goal.description in system_prompt
            assert goal.criteria in system_prompt

    @patch("src.judge_llm.requests.post")
    def test_classify_goal_prompt_contains_full_conversation(
        self, mock_post, client, sample_goals, sample_conversation
    ):
        """Classification prompt includes the full conversation history."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps(
                    {
                        "classified_goal": "Check Balance",
                        "classification_reasoning": "Customer asked about balance.",
                    }
                )
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client.classify_goal(sample_conversation, sample_goals)

        call_args = mock_post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        messages = payload["messages"]

        # Messages should include all conversation messages (system + conversation + instruction)
        # system(1) + conversation messages(4) + instruction(1) = 6
        assert len(messages) == 6

        # Verify conversation messages are in order with correct roles
        assert messages[1]["role"] == "assistant"  # agent
        assert messages[1]["content"] == "Hello! How can I help you today?"
        assert messages[2]["role"] == "user"  # customer
        assert messages[2]["content"] == "I'd like to check my balance."
        assert messages[3]["role"] == "assistant"  # agent
        assert messages[3]["content"] == "Sure! Your current balance is $1,234.56."
        assert messages[4]["role"] == "user"  # customer
        assert messages[4]["content"] == "Thank you!"

    @patch("src.judge_llm.requests.post")
    def test_classify_goal_returns_goal_classification(
        self, mock_post, client, sample_goals, sample_conversation
    ):
        """classify_goal returns a valid GoalClassification object."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps(
                    {
                        "classified_goal": "Check Balance",
                        "classification_reasoning": "The customer explicitly asked to check their balance.",
                    }
                )
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = client.classify_goal(sample_conversation, sample_goals)

        assert isinstance(result, GoalClassification)
        assert result.classified_goal == "Check Balance"
        assert "balance" in result.classification_reasoning.lower()

    @patch("src.judge_llm.requests.post")
    def test_classify_goal_unclassified(
        self, mock_post, client, sample_goals, sample_conversation
    ):
        """classify_goal handles 'unclassified' response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps(
                    {
                        "classified_goal": "unclassified",
                        "classification_reasoning": "The conversation does not match any defined goal.",
                    }
                )
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = client.classify_goal(sample_conversation, sample_goals)

        assert result.classified_goal == "unclassified"


# --- evaluate_achievement tests ---


class TestEvaluateAchievement:
    """Tests for evaluate_achievement method."""

    @patch("src.judge_llm.requests.post")
    def test_evaluate_achievement_prompt_contains_goal(
        self, mock_post, client, sample_goals, sample_conversation
    ):
        """Achievement prompt includes the goal name, description, and criteria."""
        goal = sample_goals[0]  # Check Balance
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps(
                    {"success": True, "explanation": "Balance was provided."}
                )
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client.evaluate_achievement(sample_conversation, goal)

        call_args = mock_post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        messages = payload["messages"]

        system_prompt = messages[0]["content"]
        assert goal.name in system_prompt
        assert goal.description in system_prompt
        assert goal.criteria in system_prompt

    @patch("src.judge_llm.requests.post")
    def test_evaluate_achievement_prompt_contains_full_conversation(
        self, mock_post, client, sample_goals, sample_conversation
    ):
        """Achievement prompt includes the full conversation history."""
        goal = sample_goals[0]
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps(
                    {"success": True, "explanation": "Balance was provided."}
                )
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client.evaluate_achievement(sample_conversation, goal)

        call_args = mock_post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        messages = payload["messages"]

        # system(1) + conversation messages(4) + instruction(1) = 6
        assert len(messages) == 6

        # Verify conversation messages are in order
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hello! How can I help you today?"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "I'd like to check my balance."
        assert messages[3]["role"] == "assistant"
        assert messages[3]["content"] == "Sure! Your current balance is $1,234.56."
        assert messages[4]["role"] == "user"
        assert messages[4]["content"] == "Thank you!"

    @patch("src.judge_llm.requests.post")
    def test_evaluate_achievement_returns_goal_achievement(
        self, mock_post, client, sample_goals, sample_conversation
    ):
        """evaluate_achievement returns a valid GoalAchievement object."""
        goal = sample_goals[0]
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps(
                    {
                        "success": True,
                        "explanation": "The agent provided the account balance of $1,234.56.",
                    }
                )
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = client.evaluate_achievement(sample_conversation, goal)

        assert isinstance(result, GoalAchievement)
        assert result.success is True
        assert "balance" in result.explanation.lower()

    @patch("src.judge_llm.requests.post")
    def test_evaluate_achievement_failure(
        self, mock_post, client, sample_goals, sample_conversation
    ):
        """evaluate_achievement handles failure response."""
        goal = sample_goals[0]
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps(
                    {
                        "success": False,
                        "explanation": "The agent did not provide the balance.",
                    }
                )
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = client.evaluate_achievement(sample_conversation, goal)

        assert result.success is False


# --- Error handling tests ---


class TestErrorHandling:
    """Tests for error handling in the Judge LLM client."""

    @patch("src.judge_llm.requests.post")
    def test_connection_error_includes_url_and_model(
        self, mock_post, client, sample_goals, sample_conversation
    ):
        """Connection errors include the base URL and model name."""
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        with pytest.raises(JudgeLLMError) as exc_info:
            client.classify_goal(sample_conversation, sample_goals)

        error_msg = str(exc_info.value)
        assert "http://localhost:11434" in error_msg
        assert "llama3" in error_msg

    @patch("src.judge_llm.requests.post")
    def test_timeout_error_includes_url_and_model(
        self, mock_post, client, sample_goals, sample_conversation
    ):
        """Timeout errors include the base URL and model name."""
        mock_post.side_effect = requests.Timeout("Request timed out")

        with pytest.raises(JudgeLLMError) as exc_info:
            client.classify_goal(sample_conversation, sample_goals)

        error_msg = str(exc_info.value)
        assert "http://localhost:11434" in error_msg
        assert "llama3" in error_msg
        assert "timed out" in error_msg.lower()

    @patch("src.judge_llm.requests.post")
    def test_parse_error_on_invalid_json_classification(
        self, mock_post, client, sample_goals, sample_conversation
    ):
        """Parse errors are raised when classification response is not valid JSON."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "This is not JSON at all"}
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with pytest.raises(JudgeLLMError) as exc_info:
            client.classify_goal(sample_conversation, sample_goals)

        error_msg = str(exc_info.value)
        assert "GoalClassification" in error_msg

    @patch("src.judge_llm.requests.post")
    def test_parse_error_on_invalid_json_achievement(
        self, mock_post, client, sample_goals, sample_conversation
    ):
        """Parse errors are raised when achievement response is not valid JSON."""
        goal = sample_goals[0]
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Not valid JSON response"}
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with pytest.raises(JudgeLLMError) as exc_info:
            client.evaluate_achievement(sample_conversation, goal)

        error_msg = str(exc_info.value)
        assert "GoalAchievement" in error_msg

    @patch("src.judge_llm.requests.post")
    def test_empty_response_raises_error(
        self, mock_post, client, sample_goals, sample_conversation
    ):
        """Empty LLM response raises JudgeLLMError."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": ""}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with pytest.raises(JudgeLLMError) as exc_info:
            client.classify_goal(sample_conversation, sample_goals)

        error_msg = str(exc_info.value)
        assert "Empty response" in error_msg

    @patch("src.judge_llm.requests.post")
    def test_connection_error_on_achievement(
        self, mock_post, client, sample_goals, sample_conversation
    ):
        """Connection errors during achievement evaluation include URL and model."""
        goal = sample_goals[0]
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        with pytest.raises(JudgeLLMError) as exc_info:
            client.evaluate_achievement(sample_conversation, goal)

        error_msg = str(exc_info.value)
        assert "http://localhost:11434" in error_msg
        assert "llama3" in error_msg


# --- _extract_json tests ---


class TestExtractJson:
    """Tests for _extract_json handling of markdown fences and boolean normalization."""

    def test_plain_json_passthrough(self, client):
        """Plain JSON is returned unchanged."""
        json_str = '{"classified_goal": "Check Balance", "classification_reasoning": "test"}'
        result = client._extract_json(json_str)
        assert json.loads(result) == {
            "classified_goal": "Check Balance",
            "classification_reasoning": "test",
        }

    def test_markdown_json_fence(self, client):
        """JSON wrapped in ```json ... ``` fences is extracted."""
        text = '```json\n{"classified_goal": "Check Balance", "classification_reasoning": "test"}\n```'
        result = client._extract_json(text)
        assert json.loads(result) == {
            "classified_goal": "Check Balance",
            "classification_reasoning": "test",
        }

    def test_markdown_plain_fence(self, client):
        """JSON wrapped in ``` ... ``` fences (no language) is extracted."""
        text = '```\n{"success": true, "explanation": "done"}\n```'
        result = client._extract_json(text)
        assert json.loads(result) == {"success": True, "explanation": "done"}

    def test_boolean_normalization_true_false(self, client):
        """Python-style True/False are normalized to JSON true/false."""
        text = '{"success": True, "explanation": "done"}'
        result = client._extract_json(text)
        parsed = json.loads(result)
        assert parsed["success"] is True

    def test_boolean_normalization_uppercase(self, client):
        """Uppercase TRUE/FALSE are normalized to JSON true/false."""
        text = '{"success": TRUE, "explanation": "done"}'
        result = client._extract_json(text)
        parsed = json.loads(result)
        assert parsed["success"] is True

    def test_boolean_normalization_false(self, client):
        """Python-style False is normalized to JSON false."""
        text = '{"success": False, "explanation": "failed"}'
        result = client._extract_json(text)
        parsed = json.loads(result)
        assert parsed["success"] is False

    def test_none_normalization(self, client):
        """Python-style None is normalized to JSON null."""
        text = '{"classified_goal": None, "classification_reasoning": "test"}'
        result = client._extract_json(text)
        parsed = json.loads(result)
        assert parsed["classified_goal"] is None

    def test_null_normalization_uppercase(self, client):
        """Uppercase NULL is normalized to JSON null."""
        text = '{"classified_goal": NULL, "classification_reasoning": "test"}'
        result = client._extract_json(text)
        parsed = json.loads(result)
        assert parsed["classified_goal"] is None

    def test_whitespace_stripping(self, client):
        """Leading/trailing whitespace is stripped."""
        text = '  \n  {"success": true, "explanation": "done"}  \n  '
        result = client._extract_json(text)
        assert json.loads(result) == {"success": True, "explanation": "done"}

    def test_markdown_fence_with_boolean_normalization(self, client):
        """Markdown fences combined with boolean normalization."""
        text = '```json\n{"success": True, "explanation": "done"}\n```'
        result = client._extract_json(text)
        parsed = json.loads(result)
        assert parsed["success"] is True


# --- Response parsing tests ---


class TestResponseParsing:
    """Tests for _parse_goal_classification and _parse_goal_achievement."""

    def test_parse_goal_classification_valid(self, client):
        """Valid GoalClassification JSON is parsed correctly."""
        response = json.dumps(
            {
                "classified_goal": "Check Balance",
                "classification_reasoning": "Customer asked about their balance.",
            }
        )
        result = client._parse_goal_classification(response)
        assert isinstance(result, GoalClassification)
        assert result.classified_goal == "Check Balance"
        assert result.classification_reasoning == "Customer asked about their balance."

    def test_parse_goal_classification_with_fences(self, client):
        """GoalClassification wrapped in markdown fences is parsed."""
        response = '```json\n{"classified_goal": "Reset Password", "classification_reasoning": "Password reset requested."}\n```'
        result = client._parse_goal_classification(response)
        assert result.classified_goal == "Reset Password"

    def test_parse_goal_classification_invalid_json(self, client):
        """Invalid JSON raises JudgeLLMError for classification."""
        with pytest.raises(JudgeLLMError) as exc_info:
            client._parse_goal_classification("not json at all")
        assert "GoalClassification" in str(exc_info.value)

    def test_parse_goal_classification_missing_field(self, client):
        """Missing required field raises JudgeLLMError."""
        response = json.dumps({"classified_goal": "Check Balance"})
        with pytest.raises(JudgeLLMError) as exc_info:
            client._parse_goal_classification(response)
        assert "GoalClassification" in str(exc_info.value)

    def test_parse_goal_achievement_valid_success(self, client):
        """Valid GoalAchievement with success=true is parsed correctly."""
        response = json.dumps(
            {"success": True, "explanation": "The goal was achieved."}
        )
        result = client._parse_goal_achievement(response)
        assert isinstance(result, GoalAchievement)
        assert result.success is True
        assert result.explanation == "The goal was achieved."

    def test_parse_goal_achievement_valid_failure(self, client):
        """Valid GoalAchievement with success=false is parsed correctly."""
        response = json.dumps(
            {"success": False, "explanation": "The goal was not achieved."}
        )
        result = client._parse_goal_achievement(response)
        assert result.success is False

    def test_parse_goal_achievement_with_fences(self, client):
        """GoalAchievement wrapped in markdown fences is parsed."""
        response = '```\n{"success": true, "explanation": "Done."}\n```'
        result = client._parse_goal_achievement(response)
        assert result.success is True

    def test_parse_goal_achievement_with_boolean_normalization(self, client):
        """GoalAchievement with Python-style True is parsed correctly."""
        response = '{"success": True, "explanation": "Done."}'
        result = client._parse_goal_achievement(response)
        assert result.success is True

    def test_parse_goal_achievement_invalid_json(self, client):
        """Invalid JSON raises JudgeLLMError for achievement."""
        with pytest.raises(JudgeLLMError) as exc_info:
            client._parse_goal_achievement("garbage text")
        assert "GoalAchievement" in str(exc_info.value)

    def test_parse_goal_achievement_missing_field(self, client):
        """Missing required field raises JudgeLLMError."""
        response = json.dumps({"success": True})
        with pytest.raises(JudgeLLMError) as exc_info:
            client._parse_goal_achievement(response)
        assert "GoalAchievement" in str(exc_info.value)


# --- Initialization tests ---


class TestInit:
    """Tests for JudgeLLMClient initialization."""

    def test_strips_trailing_slash_from_url(self):
        """Trailing slash is stripped from base_url."""
        client = JudgeLLMClient(
            base_url="http://localhost:11434/", model="llama3"
        )
        assert client.base_url == "http://localhost:11434"

    def test_default_timeout(self):
        """Default timeout is 120 seconds."""
        client = JudgeLLMClient(base_url="http://localhost:11434", model="llama3")
        assert client.timeout == 120

    def test_custom_timeout(self):
        """Custom timeout is stored correctly."""
        client = JudgeLLMClient(
            base_url="http://localhost:11434", model="llama3", timeout=300
        )
        assert client.timeout == 300
