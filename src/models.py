"""Pydantic data models for the GC Chat Log Evaluator."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# --- Evaluation Suite and Goals ---


class Goal(BaseModel):
    """A single evaluation goal with criteria for classification and achievement."""

    name: str
    description: str
    criteria: str

    @field_validator("name", "description", "criteria")
    @classmethod
    def must_be_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("field must be non-empty")
        return v


class EvaluationSuite(BaseModel):
    """A collection of goals that defines the evaluation criteria."""

    name: str
    goals: list[Goal] = Field(min_length=1)


# --- Conversation and Messages ---


class MessageRole(str, Enum):
    """Role of a message sender in a conversation."""

    AGENT = "agent"
    CUSTOMER = "customer"


class Message(BaseModel):
    """A single message in a conversation."""

    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None


class CsvMessageMetadata(BaseModel):
    """Optional metadata from Genesys Cloud CSV export columns."""

    session_id: Optional[str] = None
    ask_action_outcome: Optional[str] = None
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    slots: Optional[str] = None


class Conversation(BaseModel):
    """A single chat conversation with an ordered list of messages."""

    id: str
    messages: list[Message] = Field(min_length=1)
    metadata: Optional[dict] = None  # Preserves CSV metadata per conversation


# --- Configuration ---


class AppConfig(BaseModel):
    """Application configuration with defaults."""

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: Optional[str] = None
    llm_timeout: int = 120  # seconds


# --- Evaluation Results ---


class GoalClassification(BaseModel):
    """Result of the LLM classifying which goal a conversation was pursuing."""

    classified_goal: str  # Goal name or "unclassified"
    classification_reasoning: str


class GoalAchievement(BaseModel):
    """Result of the LLM evaluating whether the goal was achieved."""

    success: bool
    explanation: str


class ConversationEvaluation(BaseModel):
    """Combined evaluation result for a single conversation."""

    conversation_id: str
    classified_goal: str  # Goal name or "unclassified"
    success: bool
    classification_reasoning: str
    achievement_explanation: str
    conversation: list[Message]  # Full message history for JSON export
    error: Optional[str] = None  # Set if evaluation failed due to error


class GoalSummary(BaseModel):
    """Aggregated results for a single goal."""

    goal_name: str
    conversations_classified: int
    successes: int
    failures: int
    success_rate: float


class EvaluationReport(BaseModel):
    """Aggregated output of evaluating all conversations."""

    suite_name: str
    timestamp: datetime
    duration_seconds: float
    goal_summaries: list[GoalSummary]
    unclassified_count: int
    conversation_evaluations: list[ConversationEvaluation]
    total_conversations: int
    total_successes: int
    total_failures: int
    overall_success_rate: float


# --- Progress Events ---


class ProgressEventType(str, Enum):
    """Types of progress events emitted during evaluation."""

    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_IN_PROGRESS = "evaluation_in_progress"
    EVALUATION_COMPLETED = "evaluation_completed"
    EVALUATION_RUN_COMPLETED = "evaluation_run_completed"


class ProgressEvent(BaseModel):
    """A progress event emitted during evaluation."""

    event_type: ProgressEventType
    conversation_id: Optional[str] = None
    classified_goal: Optional[str] = None
    success: Optional[bool] = None
    current: Optional[int] = None  # Current evaluation number
    total: Optional[int] = None  # Total evaluations
    overall_success_rate: Optional[float] = None
    message: str
    duration_seconds: Optional[float] = None
    conversation_evaluation: Optional[ConversationEvaluation] = None  # Full result for live UI
