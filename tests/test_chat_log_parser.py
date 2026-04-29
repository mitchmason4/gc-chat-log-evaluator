"""Unit tests for the chat log parser module."""

import csv
import io
import json
from typing import Optional

import pytest
from pydantic import ValidationError

from src.chat_log_parser import (
    parse_chat_log,
    parse_chat_log_csv,
    parse_chat_log_json,
    serialize_conversations,
)
from src.models import Conversation, Message, MessageRole


class TestParseChatLogJson:
    """Tests for parse_chat_log_json function."""

    def test_single_conversation_object(self):
        """Parse a single conversation object with id and messages."""
        content = json.dumps(
            {
                "id": "conv-1",
                "messages": [
                    {"role": "agent", "content": "Hello, how can I help?"},
                    {"role": "customer", "content": "I need assistance"},
                ],
            }
        )
        result = parse_chat_log_json(content)
        assert len(result) == 1
        assert result[0].id == "conv-1"
        assert len(result[0].messages) == 2
        assert result[0].messages[0].role == MessageRole.AGENT
        assert result[0].messages[0].content == "Hello, how can I help?"
        assert result[0].messages[1].role == MessageRole.CUSTOMER
        assert result[0].messages[1].content == "I need assistance"

    def test_multi_conversation_with_wrapper(self):
        """Parse multiple conversations from a 'conversations' array wrapper."""
        content = json.dumps(
            {
                "conversations": [
                    {
                        "id": "conv-1",
                        "messages": [
                            {"role": "agent", "content": "Hello"},
                        ],
                    },
                    {
                        "id": "conv-2",
                        "messages": [
                            {"role": "customer", "content": "Hi there"},
                        ],
                    },
                ]
            }
        )
        result = parse_chat_log_json(content)
        assert len(result) == 2
        assert result[0].id == "conv-1"
        assert result[1].id == "conv-2"

    def test_bare_array_of_conversations(self):
        """Parse a bare JSON array of conversation objects."""
        content = json.dumps(
            [
                {
                    "id": "conv-1",
                    "messages": [{"role": "agent", "content": "Hello"}],
                },
                {
                    "id": "conv-2",
                    "messages": [{"role": "customer", "content": "Hi"}],
                },
            ]
        )
        result = parse_chat_log_json(content)
        assert len(result) == 2
        assert result[0].id == "conv-1"
        assert result[1].id == "conv-2"

    def test_preserves_message_timestamp(self):
        """Timestamps are preserved when present."""
        content = json.dumps(
            {
                "id": "conv-1",
                "messages": [
                    {
                        "role": "agent",
                        "content": "Hello",
                        "timestamp": "2024-01-15T10:30:00",
                    }
                ],
            }
        )
        result = parse_chat_log_json(content)
        assert result[0].messages[0].timestamp is not None

    def test_timestamp_optional(self):
        """Messages without timestamps are valid."""
        content = json.dumps(
            {
                "id": "conv-1",
                "messages": [{"role": "agent", "content": "Hello"}],
            }
        )
        result = parse_chat_log_json(content)
        assert result[0].messages[0].timestamp is None

    def test_genesys_cloud_web_messaging_format(self):
        """Parse Genesys Cloud Web Messaging format with participant/body fields."""
        content = json.dumps(
            {
                "id": "gc-conv-1",
                "messages": [
                    {"participant": "Agent", "body": "How can I help you?"},
                    {"participant": "Customer", "body": "I need to reset my password"},
                    {"participant": "Bot", "body": "Let me help with that"},
                    {"participant": "User", "body": "Thanks"},
                ],
            }
        )
        result = parse_chat_log_json(content)
        assert len(result) == 1
        assert len(result[0].messages) == 4
        assert result[0].messages[0].role == MessageRole.AGENT
        assert result[0].messages[0].content == "How can I help you?"
        assert result[0].messages[1].role == MessageRole.CUSTOMER
        assert result[0].messages[2].role == MessageRole.AGENT  # Bot maps to agent
        assert result[0].messages[3].role == MessageRole.CUSTOMER  # User maps to customer

    def test_invalid_json_raises_value_error(self):
        """Invalid JSON content raises ValueError with descriptive message."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_chat_log_json("not valid json {{{")

    def test_empty_string_raises_value_error(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_chat_log_json("")

    def test_missing_id_raises_validation_error(self):
        """Missing conversation id raises ValidationError."""
        content = json.dumps(
            {"messages": [{"role": "agent", "content": "Hello"}]}
        )
        with pytest.raises(ValidationError):
            parse_chat_log_json(content)

    def test_missing_messages_raises_validation_error(self):
        """Missing messages field raises ValidationError."""
        content = json.dumps({"id": "conv-1"})
        with pytest.raises(ValidationError):
            parse_chat_log_json(content)

    def test_empty_messages_list_raises_validation_error(self):
        """Empty messages list raises ValidationError."""
        content = json.dumps({"id": "conv-1", "messages": []})
        with pytest.raises(ValidationError):
            parse_chat_log_json(content)

    def test_empty_conversations_array_raises_validation_error(self):
        """Empty conversations array raises ValidationError."""
        content = json.dumps({"conversations": []})
        with pytest.raises(ValidationError):
            parse_chat_log_json(content)

    def test_empty_bare_array_raises_validation_error(self):
        """Empty bare array raises ValidationError."""
        content = json.dumps([])
        with pytest.raises(ValidationError):
            parse_chat_log_json(content)

    def test_invalid_role_raises_validation_error(self):
        """Invalid message role raises ValidationError."""
        content = json.dumps(
            {
                "id": "conv-1",
                "messages": [{"role": "unknown", "content": "Hello"}],
            }
        )
        with pytest.raises(ValidationError):
            parse_chat_log_json(content)

    def test_unknown_participant_raises_validation_error(self):
        """Unknown participant in GC format raises ValidationError."""
        content = json.dumps(
            {
                "id": "conv-1",
                "messages": [{"participant": "System", "body": "Hello"}],
            }
        )
        with pytest.raises(ValidationError):
            parse_chat_log_json(content)

    def test_non_dict_non_list_raises_value_error(self):
        """JSON that is neither dict nor list raises ValueError."""
        with pytest.raises(ValueError, match="must be a JSON object or array"):
            parse_chat_log_json('"just a string"')

    def test_conversations_field_not_list_raises_validation_error(self):
        """Non-list 'conversations' field raises ValidationError."""
        content = json.dumps({"conversations": "not a list"})
        with pytest.raises(ValidationError):
            parse_chat_log_json(content)

    def test_messages_field_not_list_raises_validation_error(self):
        """Non-list 'messages' field raises ValidationError."""
        content = json.dumps({"id": "conv-1", "messages": "not a list"})
        with pytest.raises(ValidationError):
            parse_chat_log_json(content)

    def test_preserves_metadata(self):
        """Metadata field is preserved when present."""
        content = json.dumps(
            {
                "id": "conv-1",
                "messages": [{"role": "agent", "content": "Hello"}],
                "metadata": {"source": "web", "session": "abc123"},
            }
        )
        result = parse_chat_log_json(content)
        assert result[0].metadata == {"source": "web", "session": "abc123"}


class TestParseChatLog:
    """Tests for the parse_chat_log dispatcher function."""

    def test_routes_to_json_parser(self):
        """Format 'json' routes to parse_chat_log_json."""
        content = json.dumps(
            {
                "id": "conv-1",
                "messages": [{"role": "agent", "content": "Hello"}],
            }
        )
        result = parse_chat_log(content, "json")
        assert len(result) == 1
        assert result[0].id == "conv-1"

    def test_routes_to_json_parser_case_insensitive(self):
        """Format routing is case-insensitive."""
        content = json.dumps(
            {
                "id": "conv-1",
                "messages": [{"role": "agent", "content": "Hello"}],
            }
        )
        result = parse_chat_log(content, "JSON")
        assert len(result) == 1

    def test_routes_to_csv_parser(self):
        """Format 'csv' routes to CSV parser."""
        csv_content = (
            "Conversation ID,Session ID,Date,Utterance,Prompt,Ask Action Number,"
            "Ask Action Name,Ask Action Type,Ask Action Outcome,Intent,Intent Confidence,Slots\n"
            "conv-1,sess-1,2024-01-15T10:00:00,Hello,,,,,,,,\n"
            "conv-1,sess-1,2024-01-15T10:00:01,,Hi there!,,,,,,,\n"
        )
        result = parse_chat_log(csv_content, "csv")
        assert len(result) == 1
        assert result[0].id == "conv-1"

    def test_unsupported_format_raises_value_error(self):
        """Unsupported format raises ValueError with descriptive message."""
        with pytest.raises(ValueError, match="Unsupported chat log format"):
            parse_chat_log("data", "xml")

    def test_unsupported_format_includes_format_name(self):
        """Error message includes the unsupported format name."""
        with pytest.raises(ValueError, match="yaml"):
            parse_chat_log("data", "yaml")


class TestParseChatLogCsv:
    """Tests for parse_chat_log_csv function."""

    def _make_csv(self, rows: list, header: Optional[list] = None) -> str:
        """Helper to build CSV content from rows."""
        if header is None:
            header = [
                "Conversation ID", "Session ID", "Date", "Utterance", "Prompt",
                "Ask Action Number", "Ask Action Name", "Ask Action Type",
                "Ask Action Outcome", "Intent", "Intent Confidence", "Slots",
            ]
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
        return output.getvalue()

    def test_basic_csv_parsing_single_conversation(self):
        """Parse a basic CSV with one conversation containing customer and agent messages."""
        csv_content = self._make_csv([
            ["conv-1", "sess-1", "2024-01-15T10:00:00", "Hello", "", "", "", "", "", "", "", ""],
            ["conv-1", "sess-1", "2024-01-15T10:00:01", "", "Hi! How can I help?", "", "", "", "", "", "", ""],
            ["conv-1", "sess-1", "2024-01-15T10:00:02", "I need help", "", "", "", "", "", "", "", ""],
        ])
        result = parse_chat_log_csv(csv_content)
        assert len(result) == 1
        conv = result[0]
        assert conv.id == "conv-1"
        assert len(conv.messages) == 3
        assert conv.messages[0].role == MessageRole.CUSTOMER
        assert conv.messages[0].content == "Hello"
        assert conv.messages[1].role == MessageRole.AGENT
        assert conv.messages[1].content == "Hi! How can I help?"
        assert conv.messages[2].role == MessageRole.CUSTOMER
        assert conv.messages[2].content == "I need help"

    def test_basic_csv_parsing_multiple_conversations(self):
        """Parse a CSV with multiple conversations grouped by Conversation ID."""
        csv_content = self._make_csv([
            ["conv-1", "sess-1", "2024-01-15T10:00:00", "Hello", "", "", "", "", "", "", "", ""],
            ["conv-1", "sess-1", "2024-01-15T10:00:01", "", "Hi!", "", "", "", "", "", "", ""],
            ["conv-2", "sess-2", "2024-01-15T11:00:00", "Help me", "", "", "", "", "", "", "", ""],
            ["conv-2", "sess-2", "2024-01-15T11:00:01", "", "Sure!", "", "", "", "", "", "", ""],
        ])
        result = parse_chat_log_csv(csv_content)
        assert len(result) == 2
        conv_ids = {c.id for c in result}
        assert conv_ids == {"conv-1", "conv-2"}

    def test_orders_messages_by_date(self):
        """Messages within a conversation are ordered by Date column."""
        csv_content = self._make_csv([
            ["conv-1", "sess-1", "2024-01-15T10:00:05", "Second", "", "", "", "", "", "", "", ""],
            ["conv-1", "sess-1", "2024-01-15T10:00:01", "First", "", "", "", "", "", "", "", ""],
            ["conv-1", "sess-1", "2024-01-15T10:00:10", "Third", "", "", "", "", "", "", "", ""],
        ])
        result = parse_chat_log_csv(csv_content)
        assert result[0].messages[0].content == "First"
        assert result[0].messages[1].content == "Second"
        assert result[0].messages[2].content == "Third"

    def test_utterance_maps_to_customer_message(self):
        """Non-empty Utterance column creates a customer message."""
        csv_content = self._make_csv([
            ["conv-1", "sess-1", "2024-01-15T10:00:00", "Customer says this", "", "", "", "", "", "", "", ""],
        ])
        result = parse_chat_log_csv(csv_content)
        assert result[0].messages[0].role == MessageRole.CUSTOMER
        assert result[0].messages[0].content == "Customer says this"

    def test_prompt_maps_to_agent_message(self):
        """Non-empty Prompt column creates an agent message."""
        csv_content = self._make_csv([
            ["conv-1", "sess-1", "2024-01-15T10:00:00", "", "Agent says this", "", "", "", "", "", "", ""],
        ])
        result = parse_chat_log_csv(csv_content)
        assert result[0].messages[0].role == MessageRole.AGENT
        assert result[0].messages[0].content == "Agent says this"

    def test_both_utterance_and_prompt_creates_two_messages(self):
        """Row with both Utterance and Prompt creates customer then agent message."""
        csv_content = self._make_csv([
            ["conv-1", "sess-1", "2024-01-15T10:00:00", "Customer text", "Agent response", "", "", "", "", "", "", ""],
        ])
        result = parse_chat_log_csv(csv_content)
        assert len(result[0].messages) == 2
        assert result[0].messages[0].role == MessageRole.CUSTOMER
        assert result[0].messages[0].content == "Customer text"
        assert result[0].messages[1].role == MessageRole.AGENT
        assert result[0].messages[1].content == "Agent response"

    def test_empty_utterance_and_prompt_skips_row(self):
        """Rows where both Utterance and Prompt are empty are skipped."""
        csv_content = self._make_csv([
            ["conv-1", "sess-1", "2024-01-15T10:00:00", "Hello", "", "", "", "", "", "", "", ""],
            ["conv-1", "sess-1", "2024-01-15T10:00:01", "", "", "", "", "", "", "", "", ""],
            ["conv-1", "sess-1", "2024-01-15T10:00:02", "", "Response", "", "", "", "", "", "", ""],
        ])
        result = parse_chat_log_csv(csv_content)
        assert len(result[0].messages) == 2
        assert result[0].messages[0].content == "Hello"
        assert result[0].messages[1].content == "Response"

    def test_noinputcollection_deduplication(self):
        """Consecutive duplicate agent prompts are deduplicated."""
        csv_content = self._make_csv([
            ["conv-1", "sess-1", "2024-01-15T10:00:00", "Hello", "", "", "", "", "", "", "", ""],
            ["conv-1", "sess-1", "2024-01-15T10:00:01", "", "Please provide your ID", "", "", "", "NoInputCollection", "", "", ""],
            ["conv-1", "sess-1", "2024-01-15T10:00:30", "", "Please provide your ID", "", "", "", "NoInputCollection", "", "", ""],
            ["conv-1", "sess-1", "2024-01-15T10:01:00", "", "Please provide your ID", "", "", "", "NoInputCollection", "", "", ""],
            ["conv-1", "sess-1", "2024-01-15T10:01:30", "12345", "", "", "", "", "", "", "", ""],
        ])
        result = parse_chat_log_csv(csv_content)
        conv = result[0]
        # Should have: customer "Hello", agent "Please provide your ID" (once), customer "12345"
        assert len(conv.messages) == 3
        assert conv.messages[0].role == MessageRole.CUSTOMER
        assert conv.messages[0].content == "Hello"
        assert conv.messages[1].role == MessageRole.AGENT
        assert conv.messages[1].content == "Please provide your ID"
        assert conv.messages[2].role == MessageRole.CUSTOMER
        assert conv.messages[2].content == "12345"

    def test_noinputcollection_different_prompts_not_deduplicated(self):
        """Different consecutive agent prompts are NOT deduplicated."""
        csv_content = self._make_csv([
            ["conv-1", "sess-1", "2024-01-15T10:00:00", "", "First prompt", "", "", "", "", "", "", ""],
            ["conv-1", "sess-1", "2024-01-15T10:00:01", "", "Second prompt", "", "", "", "", "", "", ""],
        ])
        result = parse_chat_log_csv(csv_content)
        assert len(result[0].messages) == 2
        assert result[0].messages[0].content == "First prompt"
        assert result[0].messages[1].content == "Second prompt"

    def test_noinputcollection_non_consecutive_duplicates_kept(self):
        """Non-consecutive duplicate agent prompts are kept."""
        csv_content = self._make_csv([
            ["conv-1", "sess-1", "2024-01-15T10:00:00", "", "Repeat me", "", "", "", "", "", "", ""],
            ["conv-1", "sess-1", "2024-01-15T10:00:01", "Customer reply", "", "", "", "", "", "", "", ""],
            ["conv-1", "sess-1", "2024-01-15T10:00:02", "", "Repeat me", "", "", "", "", "", "", ""],
        ])
        result = parse_chat_log_csv(csv_content)
        assert len(result[0].messages) == 3
        assert result[0].messages[0].content == "Repeat me"
        assert result[0].messages[1].content == "Customer reply"
        assert result[0].messages[2].content == "Repeat me"

    def test_metadata_preservation(self):
        """CSV metadata columns are preserved on the Conversation metadata dict."""
        csv_content = self._make_csv([
            ["conv-1", "sess-abc", "2024-01-15T10:00:00", "Hello", "", "", "", "", "Success", "greeting", "0.95", "name=John"],
        ])
        result = parse_chat_log_csv(csv_content)
        assert result[0].metadata is not None
        assert result[0].metadata["session_id"] == "sess-abc"
        assert result[0].metadata["ask_action_outcome"] == "Success"
        assert result[0].metadata["intent"] == "greeting"
        assert result[0].metadata["intent_confidence"] == 0.95
        assert result[0].metadata["slots"] == "name=John"

    def test_metadata_not_set_when_all_empty(self):
        """Metadata is None when all metadata columns are empty."""
        csv_content = self._make_csv([
            ["conv-1", "", "2024-01-15T10:00:00", "Hello", "", "", "", "", "", "", "", ""],
        ])
        result = parse_chat_log_csv(csv_content)
        assert result[0].metadata is None

    def test_missing_required_columns_raises_value_error(self):
        """CSV missing required columns raises ValueError listing missing columns."""
        csv_content = "Name,Value\nfoo,bar\n"
        with pytest.raises(ValueError, match="missing required columns"):
            parse_chat_log_csv(csv_content)

    def test_missing_required_columns_lists_which_are_missing(self):
        """Error message identifies which specific columns are missing."""
        csv_content = "Conversation ID,Date\nconv-1,2024-01-15\n"
        with pytest.raises(ValueError, match="Prompt"):
            parse_chat_log_csv(csv_content)
        with pytest.raises(ValueError, match="Utterance"):
            parse_chat_log_csv(csv_content)

    def test_empty_csv_raises_value_error(self):
        """Empty CSV content raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            parse_chat_log_csv("")

    def test_whitespace_only_csv_raises_value_error(self):
        """Whitespace-only CSV content raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            parse_chat_log_csv("   \n  \n  ")

    def test_header_only_csv_raises_value_error(self):
        """CSV with only a header row (no data) raises ValueError."""
        csv_content = "Conversation ID,Session ID,Date,Utterance,Prompt,Ask Action Number,Ask Action Name,Ask Action Type,Ask Action Outcome,Intent,Intent Confidence,Slots\n"
        with pytest.raises(ValueError, match="no conversations"):
            parse_chat_log_csv(csv_content)

    def test_preserves_timestamps(self):
        """Timestamps from the Date column are preserved on messages."""
        csv_content = self._make_csv([
            ["conv-1", "sess-1", "2024-01-15T10:30:00", "Hello", "", "", "", "", "", "", "", ""],
        ])
        result = parse_chat_log_csv(csv_content)
        assert result[0].messages[0].timestamp is not None
        assert result[0].messages[0].timestamp.hour == 10
        assert result[0].messages[0].timestamp.minute == 30

    def test_invalid_date_still_parses(self):
        """Rows with unparseable dates still produce messages (timestamp=None)."""
        csv_content = self._make_csv([
            ["conv-1", "sess-1", "not-a-date", "Hello", "", "", "", "", "", "", "", ""],
        ])
        result = parse_chat_log_csv(csv_content)
        assert result[0].messages[0].timestamp is None
        assert result[0].messages[0].content == "Hello"
