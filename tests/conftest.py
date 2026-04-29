"""Shared Hypothesis strategies and pytest fixtures for the GC Chat Log Evaluator tests.

Provides reusable strategies for generating valid/invalid model instances and
pytest fixtures for sample data, mock responses, and the Flask test client.
"""

import json
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import strategies as st

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
)


# =============================================================================
# Hypothesis Strategies
# =============================================================================


# --- Primitive strategies ---


def non_empty_text(min_size: int = 1, max_size: int = 200) -> st.SearchStrategy[str]:
    """Generate non-empty, non-whitespace-only text strings."""
    return st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "S", "Z"),
            blacklist_characters="\x00",
        ),
        min_size=min_size,
        max_size=max_size,
    ).filter(lambda s: s.strip())


def conversation_id_strategy() -> st.SearchStrategy[str]:
    """Generate valid conversation IDs (non-empty alphanumeric with dashes)."""
    return st.from_regex(r"[a-zA-Z0-9][a-zA-Z0-9\-]{0,49}", fullmatch=True)


def optional_timestamp_strategy() -> st.SearchStrategy[Optional[datetime]]:
    """Generate optional datetime timestamps."""
    return st.one_of(
        st.none(),
        st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2030, 12, 31),
        ),
    )


# --- Message strategies ---


@st.composite
def message_strategy(draw) -> Message:
    """Generate a valid Message object with random role, content, and optional timestamp."""
    role = draw(st.sampled_from(list(MessageRole)))
    content = draw(non_empty_text(min_size=1, max_size=500))
    timestamp = draw(optional_timestamp_strategy())
    return Message(role=role, content=content, timestamp=timestamp)


@st.composite
def message_without_timestamp_strategy(draw) -> Message:
    """Generate a valid Message without a timestamp (for round-trip tests)."""
    role = draw(st.sampled_from(list(MessageRole)))
    content = draw(non_empty_text(min_size=1, max_size=500))
    return Message(role=role, content=content, timestamp=None)


# --- Conversation strategies ---


@st.composite
def conversation_strategy(draw, min_messages: int = 1, max_messages: int = 10) -> Conversation:
    """Generate a valid Conversation with id and non-empty message list."""
    conv_id = draw(conversation_id_strategy())
    messages = draw(
        st.lists(message_strategy(), min_size=min_messages, max_size=max_messages)
    )
    return Conversation(id=conv_id, messages=messages)


@st.composite
def conversation_without_timestamps_strategy(
    draw, min_messages: int = 1, max_messages: int = 10
) -> Conversation:
    """Generate a valid Conversation without timestamps (for JSON round-trip tests)."""
    conv_id = draw(conversation_id_strategy())
    messages = draw(
        st.lists(
            message_without_timestamp_strategy(),
            min_size=min_messages,
            max_size=max_messages,
        )
    )
    return Conversation(id=conv_id, messages=messages)


# --- Goal strategies ---


@st.composite
def goal_strategy(draw) -> Goal:
    """Generate a valid Goal with non-empty name, description, and criteria."""
    name = draw(non_empty_text(min_size=1, max_size=100))
    description = draw(non_empty_text(min_size=1, max_size=300))
    criteria = draw(non_empty_text(min_size=1, max_size=300))
    return Goal(name=name, description=description, criteria=criteria)


# --- EvaluationSuite strategies ---


@st.composite
def evaluation_suite_strategy(draw, min_goals: int = 1, max_goals: int = 5) -> EvaluationSuite:
    """Generate a valid EvaluationSuite with a name and at least one goal."""
    name = draw(non_empty_text(min_size=1, max_size=100))
    goals = draw(st.lists(goal_strategy(), min_size=min_goals, max_size=max_goals))
    return EvaluationSuite(name=name, goals=goals)


# --- GoalClassification strategies ---


@st.composite
def goal_classification_strategy(draw) -> GoalClassification:
    """Generate a valid GoalClassification with classified_goal and reasoning."""
    classified_goal = draw(
        st.one_of(
            non_empty_text(min_size=1, max_size=100),
            st.just("unclassified"),
        )
    )
    classification_reasoning = draw(non_empty_text(min_size=1, max_size=300))
    return GoalClassification(
        classified_goal=classified_goal,
        classification_reasoning=classification_reasoning,
    )


# --- GoalAchievement strategies ---


@st.composite
def goal_achievement_strategy(draw) -> GoalAchievement:
    """Generate a valid GoalAchievement with success boolean and explanation."""
    success = draw(st.booleans())
    explanation = draw(non_empty_text(min_size=1, max_size=300))
    return GoalAchievement(success=success, explanation=explanation)


# --- ConversationEvaluation strategies ---


@st.composite
def conversation_evaluation_strategy(
    draw, goal_names: Optional[list[str]] = None
) -> ConversationEvaluation:
    """Generate a valid ConversationEvaluation object.

    If goal_names is provided, classified_goal is sampled from those names
    plus 'unclassified'. Otherwise, random text is used.
    """
    conversation_id = draw(conversation_id_strategy())

    if goal_names:
        classified_goal = draw(
            st.sampled_from(goal_names + ["unclassified"])
        )
    else:
        classified_goal = draw(
            st.one_of(
                non_empty_text(min_size=1, max_size=100),
                st.just("unclassified"),
            )
        )

    success = draw(st.booleans())
    # If unclassified, success should be False
    if classified_goal == "unclassified":
        success = False

    classification_reasoning = draw(non_empty_text(min_size=1, max_size=300))
    achievement_explanation = draw(non_empty_text(min_size=1, max_size=300))
    messages = draw(st.lists(message_strategy(), min_size=1, max_size=5))
    error = draw(st.one_of(st.none(), non_empty_text(min_size=1, max_size=100)))

    return ConversationEvaluation(
        conversation_id=conversation_id,
        classified_goal=classified_goal,
        success=success,
        classification_reasoning=classification_reasoning,
        achievement_explanation=achievement_explanation,
        conversation=messages,
        error=error,
    )


# --- AppConfig strategies ---


@st.composite
def app_config_strategy(draw) -> AppConfig:
    """Generate a valid AppConfig with various combinations of values."""
    ollama_base_url = draw(
        st.sampled_from([
            "http://localhost:11434",
            "http://192.168.1.100:11434",
            "http://ollama.local:11434",
        ])
    )
    ollama_model = draw(
        st.one_of(
            st.none(),
            st.sampled_from(["llama3", "mistral", "codellama", "gemma"]),
        )
    )
    llm_timeout = draw(st.integers(min_value=10, max_value=600))
    return AppConfig(
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        llm_timeout=llm_timeout,
    )


# --- CSV row strategies ---


@st.composite
def csv_row_strategy(draw, conversation_id: Optional[str] = None) -> dict:
    """Generate a valid Genesys Cloud CSV row as a dict.

    Returns a dict with keys matching the expected CSV columns.
    """
    conv_id = conversation_id or draw(conversation_id_strategy())
    session_id = draw(st.from_regex(r"sess-[a-z0-9]{4,8}", fullmatch=True))
    date = draw(
        st.datetimes(
            min_value=datetime(2024, 1, 1),
            max_value=datetime(2024, 12, 31),
        )
    ).strftime("%Y-%m-%dT%H:%M:%S")

    # Either utterance or prompt (or both) should be non-empty
    has_utterance = draw(st.booleans())
    has_prompt = draw(st.booleans())

    # Ensure at least one is non-empty
    if not has_utterance and not has_prompt:
        has_utterance = True

    utterance = draw(non_empty_text(min_size=1, max_size=200)) if has_utterance else ""
    prompt = draw(non_empty_text(min_size=1, max_size=200)) if has_prompt else ""

    return {
        "Conversation ID": conv_id,
        "Session ID": session_id,
        "Date": date,
        "Utterance": utterance,
        "Prompt": prompt,
        "Ask Action Number": "",
        "Ask Action Name": "",
        "Ask Action Type": "",
        "Ask Action Outcome": "",
        "Intent": "",
        "Intent Confidence": "",
        "Slots": "",
    }


# --- Invalid data strategies ---


@st.composite
def invalid_goal_dict_strategy(draw) -> dict:
    """Generate a Goal dict that is missing or has invalid fields."""
    variant = draw(st.sampled_from(["missing_name", "empty_name", "missing_description", "empty_criteria"]))

    if variant == "missing_name":
        return {"description": "desc", "criteria": "criteria"}
    elif variant == "empty_name":
        return {"name": "   ", "description": "desc", "criteria": "criteria"}
    elif variant == "missing_description":
        return {"name": "goal", "criteria": "criteria"}
    elif variant == "empty_criteria":
        return {"name": "goal", "description": "desc", "criteria": "  "}


@st.composite
def invalid_conversation_dict_strategy(draw) -> dict:
    """Generate a Conversation dict that is missing or has invalid fields."""
    variant = draw(st.sampled_from(["missing_id", "missing_messages", "empty_messages"]))

    if variant == "missing_id":
        return {"messages": [{"role": "agent", "content": "hello"}]}
    elif variant == "missing_messages":
        return {"id": "conv-1"}
    elif variant == "empty_messages":
        return {"id": "conv-1", "messages": []}


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def sample_conversation() -> Conversation:
    """A fixed sample Conversation for unit tests."""
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


@pytest.fixture
def sample_suite() -> EvaluationSuite:
    """A fixed sample EvaluationSuite for unit tests."""
    return EvaluationSuite(
        name="Test Suite",
        goals=[
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
        ],
    )


@pytest.fixture
def sample_chat_log_json() -> str:
    """A sample JSON chat log string with multiple conversations."""
    return json.dumps({
        "conversations": [
            {
                "id": "conv-001",
                "messages": [
                    {"role": "agent", "content": "Hello! How can I help?"},
                    {"role": "customer", "content": "I want to check my balance."},
                    {"role": "agent", "content": "Your balance is $500."},
                ],
            },
            {
                "id": "conv-002",
                "messages": [
                    {"role": "customer", "content": "I need to reset my password."},
                    {"role": "agent", "content": "I'll send you a reset link."},
                    {"role": "customer", "content": "Got it, thanks!"},
                ],
            },
        ]
    })


@pytest.fixture
def sample_chat_log_csv() -> str:
    """A sample CSV chat log string in Genesys Cloud format."""
    header = "Conversation ID,Session ID,Date,Utterance,Prompt,Ask Action Number,Ask Action Name,Ask Action Type,Ask Action Outcome,Intent,Intent Confidence,Slots"
    rows = [
        "conv-001,sess-1,2024-01-15T10:00:00,Hello,,,,,,,,",
        "conv-001,sess-1,2024-01-15T10:00:01,,Hi! How can I help?,,,,,,,",
        "conv-001,sess-1,2024-01-15T10:00:02,Check my balance,,,,,,,,",
        "conv-001,sess-1,2024-01-15T10:00:03,,Your balance is $500.,,,,,,,",
        "conv-002,sess-2,2024-01-15T11:00:00,Reset my password,,,,,,,,",
        "conv-002,sess-2,2024-01-15T11:00:01,,I'll send a reset link.,,,,,,,",
    ]
    return "\n".join([header] + rows) + "\n"


@pytest.fixture
def sample_evaluation_suite_yaml() -> str:
    """A sample YAML evaluation suite string."""
    return """name: Test Suite
goals:
  - name: Check Balance
    description: Customer wants to check their account balance
    criteria: The agent provides the current account balance to the customer
  - name: Reset Password
    description: Customer wants to reset their password
    criteria: The agent successfully initiates a password reset for the customer
"""


@pytest.fixture
def sample_evaluation_suite_json() -> str:
    """A sample JSON evaluation suite string."""
    return json.dumps({
        "name": "Test Suite",
        "goals": [
            {
                "name": "Check Balance",
                "description": "Customer wants to check their account balance",
                "criteria": "The agent provides the current account balance to the customer",
            },
            {
                "name": "Reset Password",
                "description": "Customer wants to reset their password",
                "criteria": "The agent successfully initiates a password reset for the customer",
            },
        ],
    })


@pytest.fixture
def mock_ollama_classify_response():
    """Factory fixture for creating mock Ollama classification responses."""
    def _make_response(goal_name: str = "Check Balance", reasoning: str = "Customer asked about balance"):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps({
                    "classified_goal": goal_name,
                    "classification_reasoning": reasoning,
                })
            }
        }
        mock_response.raise_for_status = MagicMock()
        return mock_response
    return _make_response


@pytest.fixture
def mock_ollama_achievement_response():
    """Factory fixture for creating mock Ollama achievement responses."""
    def _make_response(success: bool = True, explanation: str = "Goal was achieved"):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps({
                    "success": success,
                    "explanation": explanation,
                })
            }
        }
        mock_response.raise_for_status = MagicMock()
        return mock_response
    return _make_response


@pytest.fixture
def mock_ollama_tags_response():
    """Factory fixture for creating mock Ollama /api/tags responses."""
    def _make_response(models: Optional[list[str]] = None):
        if models is None:
            models = ["llama3:latest", "mistral:latest"]
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": m} for m in models]
        }
        mock_response.raise_for_status = MagicMock()
        return mock_response
    return _make_response


@pytest.fixture
def flask_test_client():
    """Flask test client from create_app()."""
    from src.web_app import create_app

    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
