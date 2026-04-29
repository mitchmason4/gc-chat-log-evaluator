"""Chat log parser for JSON and CSV Genesys Cloud chat exports."""

import csv
import io
import json
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from .models import Conversation, Message, MessageRole


def _parse_message(data: dict) -> Message:
    """Parse a single message dict into a Message object.

    Handles both the generic format and Genesys Cloud Web Messaging format.

    Args:
        data: Dictionary with message fields.

    Returns:
        A validated Message object.

    Raises:
        ValidationError: If required fields are missing or invalid.
    """
    # Handle Genesys Cloud Web Messaging transcript format
    # which uses "participant" instead of "role" and "body" instead of "content"
    if "participant" in data and "role" not in data:
        role_mapping = {
            "agent": MessageRole.AGENT,
            "bot": MessageRole.AGENT,
            "customer": MessageRole.CUSTOMER,
            "user": MessageRole.CUSTOMER,
        }
        participant = data["participant"].lower()
        role = role_mapping.get(participant)
        if role is None:
            raise ValidationError.from_exception_data(
                title="Message",
                line_errors=[
                    {
                        "type": "value_error",
                        "loc": ("participant",),
                        "msg": f"Unknown participant role: '{data['participant']}'. Expected one of: agent, bot, customer, user",
                        "input": data["participant"],
                        "ctx": {"error": ValueError(f"Unknown participant role: '{data['participant']}'")},
                    }
                ],
            )
        content = data.get("body", data.get("content", ""))
        timestamp = data.get("timestamp")
        return Message(role=role, content=content, timestamp=timestamp)

    # Generic format: role, content, timestamp
    return Message.model_validate(data)


def _parse_conversation(data: dict) -> Conversation:
    """Parse a single conversation dict into a Conversation object.

    Args:
        data: Dictionary with conversation fields (id, messages).

    Returns:
        A validated Conversation object.

    Raises:
        ValidationError: If required fields are missing or invalid.
    """
    if "id" not in data:
        raise ValidationError.from_exception_data(
            title="Conversation",
            line_errors=[
                {
                    "type": "missing",
                    "loc": ("id",),
                    "msg": "Field required",
                    "input": data,
                }
            ],
        )

    if "messages" not in data:
        raise ValidationError.from_exception_data(
            title="Conversation",
            line_errors=[
                {
                    "type": "missing",
                    "loc": ("messages",),
                    "msg": "Field required",
                    "input": data,
                }
            ],
        )

    messages_data = data["messages"]
    if not isinstance(messages_data, list):
        raise ValidationError.from_exception_data(
            title="Conversation",
            line_errors=[
                {
                    "type": "value_error",
                    "loc": ("messages",),
                    "msg": "Messages must be a list",
                    "input": messages_data,
                    "ctx": {"error": ValueError("Messages must be a list")},
                }
            ],
        )

    messages = [_parse_message(msg) for msg in messages_data]

    return Conversation(
        id=data["id"],
        messages=messages,
        metadata=data.get("metadata"),
    )


def parse_chat_log_json(content: str) -> list[Conversation]:
    """Parse a JSON chat log into a list of Conversations.

    Supports:
    - Single conversation object: {"id": "...", "messages": [...]}
    - Multi-conversation with wrapper: {"conversations": [...]}
    - Bare array of conversations: [{"id": "...", "messages": [...]}, ...]

    Also handles Genesys Cloud Web Messaging transcript format by mapping
    participant roles and message body fields to the internal model.

    Args:
        content: Raw JSON string content of the chat log.

    Returns:
        A list of validated Conversation objects.

    Raises:
        ValueError: If the content is not valid JSON.
        ValidationError: If required fields are missing or invalid.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    return _parse_json_data(data)


def _parse_json_data(data: Any) -> list[Conversation]:
    """Parse JSON data (already deserialized) into Conversations.

    Args:
        data: Deserialized JSON data (dict or list).

    Returns:
        A list of validated Conversation objects.

    Raises:
        ValidationError: If required fields are missing or invalid.
    """
    # Case 1: Bare array of conversations
    if isinstance(data, list):
        if len(data) == 0:
            raise ValidationError.from_exception_data(
                title="ChatLog",
                line_errors=[
                    {
                        "type": "value_error",
                        "loc": ("conversations",),
                        "msg": "Chat log contains no conversations",
                        "input": data,
                        "ctx": {"error": ValueError("Chat log contains no conversations")},
                    }
                ],
            )
        return [_parse_conversation(item) for item in data]

    # Case 2: Dict - either single conversation or wrapper with "conversations" key
    if isinstance(data, dict):
        # Multi-conversation wrapper: {"conversations": [...]}
        if "conversations" in data:
            conversations_data = data["conversations"]
            if not isinstance(conversations_data, list):
                raise ValidationError.from_exception_data(
                    title="ChatLog",
                    line_errors=[
                        {
                            "type": "value_error",
                            "loc": ("conversations",),
                            "msg": "The 'conversations' field must be a list",
                            "input": conversations_data,
                            "ctx": {"error": ValueError("The 'conversations' field must be a list")},
                        }
                    ],
                )
            if len(conversations_data) == 0:
                raise ValidationError.from_exception_data(
                    title="ChatLog",
                    line_errors=[
                        {
                            "type": "value_error",
                            "loc": ("conversations",),
                            "msg": "Chat log contains no conversations",
                            "input": conversations_data,
                            "ctx": {"error": ValueError("Chat log contains no conversations")},
                        }
                    ],
                )
            return [_parse_conversation(item) for item in conversations_data]

        # Single conversation object: {"id": "...", "messages": [...]}
        return [_parse_conversation(data)]

    raise ValueError(
        "Chat log must be a JSON object or array, "
        f"got {type(data).__name__}"
    )


def parse_chat_log_csv(content: str) -> list[Conversation]:
    """Parse a Genesys Cloud CSV export into a list of Conversations.

    Groups rows by Conversation ID, orders by Date, maps Utterance to
    customer messages and Prompt to agent messages. Filters consecutive
    duplicate agent prompts (NoInputCollection retries).

    Args:
        content: Raw CSV string content.

    Returns:
        A list of validated Conversation objects.

    Raises:
        ValueError: If the CSV is malformed or missing required columns.
    """
    if not content or not content.strip():
        raise ValueError("CSV content is empty")

    required_columns = {"Conversation ID", "Date", "Utterance", "Prompt"}

    try:
        reader = csv.DictReader(io.StringIO(content))
        if reader.fieldnames is None:
            raise ValueError("CSV content is empty or has no header row")
    except csv.Error as e:
        raise ValueError(f"Malformed CSV: {e}") from e

    # Validate required columns are present
    actual_columns = set(reader.fieldnames) if reader.fieldnames else set()
    missing_columns = required_columns - actual_columns
    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: {', '.join(sorted(missing_columns))}"
        )

    # Group rows by Conversation ID
    conversations_rows: dict[str, list[dict]] = {}
    try:
        for row in reader:
            conv_id = row.get("Conversation ID", "").strip()
            if not conv_id:
                continue  # Skip rows with empty conversation ID
            if conv_id not in conversations_rows:
                conversations_rows[conv_id] = []
            conversations_rows[conv_id].append(row)
    except csv.Error as e:
        raise ValueError(f"Malformed CSV: {e}") from e

    if not conversations_rows:
        raise ValueError("CSV contains no conversations (no valid rows found)")

    # Parse each conversation
    conversations: list[Conversation] = []
    for conv_id, rows in conversations_rows.items():
        # Order rows by Date
        rows.sort(key=lambda r: r.get("Date", ""))

        # Build messages from rows
        messages: list[Message] = []
        metadata: dict = {}

        for row in rows:
            utterance = row.get("Utterance", "").strip()
            prompt = row.get("Prompt", "").strip()
            date_str = row.get("Date", "").strip()

            # Parse timestamp
            timestamp = None
            if date_str:
                try:
                    timestamp = datetime.fromisoformat(date_str)
                except (ValueError, TypeError):
                    # If date can't be parsed, leave timestamp as None
                    pass

            # Skip rows where both Utterance and Prompt are empty
            if not utterance and not prompt:
                continue

            # If Utterance is non-empty, create a customer message
            if utterance:
                messages.append(
                    Message(
                        role=MessageRole.CUSTOMER,
                        content=utterance,
                        timestamp=timestamp,
                    )
                )

            # If Prompt is non-empty, create an agent message
            if prompt:
                messages.append(
                    Message(
                        role=MessageRole.AGENT,
                        content=prompt,
                        timestamp=timestamp,
                    )
                )

            # Collect metadata from the first row that has it
            if not metadata:
                session_id = row.get("Session ID", "").strip()
                ask_action_outcome = row.get("Ask Action Outcome", "").strip()
                intent = row.get("Intent", "").strip()
                intent_confidence = row.get("Intent Confidence", "").strip()
                slots = row.get("Slots", "").strip()

                if any([session_id, ask_action_outcome, intent, intent_confidence, slots]):
                    if session_id:
                        metadata["session_id"] = session_id
                    if ask_action_outcome:
                        metadata["ask_action_outcome"] = ask_action_outcome
                    if intent:
                        metadata["intent"] = intent
                    if intent_confidence:
                        try:
                            metadata["intent_confidence"] = float(intent_confidence)
                        except (ValueError, TypeError):
                            metadata["intent_confidence"] = intent_confidence
                    if slots:
                        metadata["slots"] = slots

        # Apply NoInputCollection deduplication: filter consecutive duplicate agent messages
        messages = _deduplicate_consecutive_agent_messages(messages)

        # Skip conversations with no messages
        if not messages:
            continue

        conversations.append(
            Conversation(
                id=conv_id,
                messages=messages,
                metadata=metadata if metadata else None,
            )
        )

    if not conversations:
        raise ValueError("CSV contains no valid conversations (all conversations had empty messages)")

    return conversations


def _deduplicate_consecutive_agent_messages(messages: list[Message]) -> list[Message]:
    """Filter consecutive duplicate agent messages (NoInputCollection deduplication).

    When the agent repeats the same prompt due to NoInputCollection timeout retries,
    only the first occurrence in a consecutive sequence is kept.

    Args:
        messages: List of messages to deduplicate.

    Returns:
        A new list with consecutive duplicate agent messages removed.
    """
    if not messages:
        return messages

    result: list[Message] = [messages[0]]
    for msg in messages[1:]:
        if (
            msg.role == MessageRole.AGENT
            and result[-1].role == MessageRole.AGENT
            and msg.content == result[-1].content
        ):
            # Skip consecutive duplicate agent message
            continue
        result.append(msg)

    return result


def parse_chat_log(content: str, format: str) -> list[Conversation]:
    """Parse a chat log from string content in the specified format.

    Args:
        content: Raw string content of the chat log.
        format: The format of the content - "json" or "csv".

    Returns:
        A list of validated Conversation objects.

    Raises:
        ValueError: If the format is unsupported or content cannot be parsed.
        ValidationError: If the data fails validation.
    """
    fmt = format.lower()
    if fmt == "json":
        return parse_chat_log_json(content)
    elif fmt == "csv":
        return parse_chat_log_csv(content)
    else:
        raise ValueError(
            f"Unsupported chat log format '{format}'. Use 'json' or 'csv'"
        )


def serialize_conversations(conversations: list[Conversation]) -> str:
    """Serialize a list of Conversations back to valid JSON.

    Uses the generic conversation format with the "conversations" wrapper
    for multiple conversations, or a single object for one conversation.

    Args:
        conversations: List of Conversation objects to serialize.

    Returns:
        A valid JSON string representing the conversations.
    """
    data = [
        conv.model_dump(mode="json", exclude_none=True)
        for conv in conversations
    ]

    if len(data) == 1:
        return json.dumps(data[0], indent=2)
    else:
        return json.dumps({"conversations": data}, indent=2)
