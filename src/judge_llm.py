"""Judge LLM Client for communicating with Ollama to classify goals and evaluate achievement."""

import json
import re

import requests

from .models import Conversation, Goal, GoalAchievement, GoalClassification, MessageRole


class JudgeLLMError(Exception):
    """Raised when the Judge LLM encounters an error (connection, parsing, etc.)."""

    pass


class JudgeLLMClient:
    """Client for interacting with Ollama to classify goals and evaluate achievement."""

    def __init__(self, base_url: str, model: str, timeout: int = 120):
        """Initialize with Ollama connection details.

        Args:
            base_url: The base URL of the Ollama instance (e.g., http://localhost:11434).
            model: The name of the model to use for generation.
            timeout: HTTP request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def verify_connection(self) -> None:
        """Check Ollama is reachable and model is available via HTTP GET to /api/tags.

        Raises:
            JudgeLLMError: If Ollama is unreachable or the model is not available.
        """
        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise JudgeLLMError(
                f"Failed to connect to Ollama at {self.base_url} "
                f"for model '{self.model}': {e}"
            )

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise JudgeLLMError(
                f"Invalid response from Ollama at {self.base_url} "
                f"for model '{self.model}': {e}"
            )

        models = data.get("models", [])
        available_names = [m.get("name", "") for m in models]
        # Check both exact match and match without tag (e.g., "llama3" matches "llama3:latest")
        model_found = any(
            self.model == name or self.model == name.split(":")[0]
            for name in available_names
        )

        if not model_found:
            raise JudgeLLMError(
                f"Model '{self.model}' not found at Ollama instance {self.base_url}. "
                f"Available models: {available_names}"
            )

    def classify_goal(
        self, conversation: Conversation, goals: list[Goal]
    ) -> GoalClassification:
        """Classify which goal the customer was pursuing in the conversation.

        The prompt includes all goal names, descriptions, and criteria,
        plus the full conversation history. Returns the classified goal
        name (or 'unclassified') with reasoning.

        Args:
            conversation: The conversation to classify.
            goals: The full list of goals from the evaluation suite.

        Returns:
            GoalClassification with classified_goal and classification_reasoning.

        Raises:
            JudgeLLMError: If the LLM response cannot be parsed or the request fails.
        """
        system_prompt = self._build_classification_prompt(goals)
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for msg in conversation.messages:
            role = "assistant" if msg.role == MessageRole.AGENT else "user"
            messages.append({"role": role, "content": msg.content})

        # Add instruction to classify
        messages.append(
            {
                "role": "user",
                "content": "Which goal was the customer pursuing in this conversation? Respond with JSON only.",
            }
        )

        response_text = self._call_chat(messages)
        return self._parse_goal_classification(response_text)

    def evaluate_achievement(
        self, conversation: Conversation, goal: Goal
    ) -> GoalAchievement:
        """Evaluate whether the classified goal was achieved.

        The prompt includes the goal name, description, criteria, and
        the full conversation history. Returns success/failure with explanation.

        Args:
            conversation: The conversation to evaluate.
            goal: The classified goal to evaluate against.

        Returns:
            GoalAchievement with success boolean and explanation.

        Raises:
            JudgeLLMError: If the LLM response cannot be parsed or the request fails.
        """
        system_prompt = self._build_achievement_prompt(goal)
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for msg in conversation.messages:
            role = "assistant" if msg.role == MessageRole.AGENT else "user"
            messages.append({"role": role, "content": msg.content})

        # Add instruction to evaluate
        messages.append(
            {
                "role": "user",
                "content": "Was the goal achieved in this conversation? Respond with JSON only.",
            }
        )

        response_text = self._call_chat(messages)
        return self._parse_goal_achievement(response_text)

    def _build_classification_prompt(self, goals: list[Goal]) -> str:
        """Build the system prompt for goal classification.

        Args:
            goals: The full list of goals to include in the prompt.

        Returns:
            The system prompt string.
        """
        goals_section = ""
        for i, goal in enumerate(goals, 1):
            goals_section += (
                f"\n{i}. Name: {goal.name}\n"
                f"   Description: {goal.description}\n"
                f"   Criteria: {goal.criteria}\n"
            )

        return (
            "You are classifying which goal a customer was pursuing in a conversation.\n\n"
            "AVAILABLE GOALS:\n"
            f"{goals_section}\n"
            "INSTRUCTIONS:\n"
            "- Analyze the conversation and determine which single goal the customer was pursuing.\n"
            "- If the conversation does not clearly match any of the defined goals, classify as 'unclassified'.\n"
            "- Provide reasoning for your classification.\n\n"
            "Respond with ONLY valid JSON:\n"
            '{"classified_goal": "<goal name or unclassified>", "classification_reasoning": "<explanation>"}'
        )

    def _build_achievement_prompt(self, goal: Goal) -> str:
        """Build the system prompt for goal achievement evaluation.

        Args:
            goal: The classified goal to evaluate against.

        Returns:
            The system prompt string.
        """
        return (
            "You are evaluating whether a goal was achieved in a conversation.\n\n"
            f"GOAL: {goal.name}\n"
            f"DESCRIPTION: {goal.description}\n"
            f"CRITERIA: {goal.criteria}\n\n"
            "INSTRUCTIONS:\n"
            "- Review the conversation and determine if the goal was successfully achieved.\n"
            "- Consider the criteria when making your determination.\n"
            "- Provide a clear explanation of your reasoning.\n\n"
            "Respond with ONLY valid JSON:\n"
            '{"success": true, "explanation": "<explanation>"}\n'
            "or\n"
            '{"success": false, "explanation": "<explanation>"}'
        )

    def _call_chat(self, messages: list[dict]) -> str:
        """Call the Ollama /api/chat endpoint and return the response content.

        Args:
            messages: The messages to send to the chat API.

        Returns:
            The response content text.

        Raises:
            JudgeLLMError: If the request fails or response is invalid.
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.Timeout as e:
            raise JudgeLLMError(
                f"Request timed out calling Ollama at {self.base_url} "
                f"for model '{self.model}' (timeout={self.timeout}s): {e}"
            )
        except requests.RequestException as e:
            raise JudgeLLMError(
                f"Failed to call Ollama chat API at {self.base_url} "
                f"for model '{self.model}': {e}"
            )

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise JudgeLLMError(
                f"Invalid JSON response from Ollama at {self.base_url} "
                f"for model '{self.model}': {e}"
            )

        message = data.get("message", {})
        content = message.get("content", "")

        if not content:
            raise JudgeLLMError(
                f"Empty response from Ollama at {self.base_url} "
                f"for model '{self.model}'"
            )

        return content

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain markdown code fences or extra whitespace.

        Handles:
        - Markdown code fences (```json ... ``` or ``` ... ```)
        - Boolean normalization (True/False/TRUE/FALSE -> true/false)
        - None/NULL -> null normalization

        Args:
            text: The raw text that should contain JSON.

        Returns:
            The extracted and normalized JSON string.
        """
        text = text.strip()
        # Handle markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # Remove opening fence line
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # Remove closing fence
            text = "\n".join(lines).strip()
        # Fix case-sensitive boolean values (LLMs sometimes output TRUE/FALSE/True/False)
        text = re.sub(r"\bTRUE\b", "true", text)
        text = re.sub(r"\bFALSE\b", "false", text)
        text = re.sub(r"\bNULL\b", "null", text)
        text = re.sub(r":\s*True\b", ": true", text)
        text = re.sub(r":\s*False\b", ": false", text)
        text = re.sub(r":\s*None\b", ": null", text)
        return text

    def _parse_goal_classification(self, response_text: str) -> GoalClassification:
        """Parse a GoalClassification from LLM response text.

        Args:
            response_text: The raw text response from the LLM.

        Returns:
            A validated GoalClassification object.

        Raises:
            JudgeLLMError: If the response cannot be parsed as valid JSON or doesn't match schema.
        """
        json_str = self._extract_json(response_text)
        try:
            data = json.loads(json_str)
            return GoalClassification(**data)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise JudgeLLMError(
                f"Failed to parse GoalClassification from LLM response: {e}. "
                f"Response was: {response_text[:200]}"
            )

    def _parse_goal_achievement(self, response_text: str) -> GoalAchievement:
        """Parse a GoalAchievement from LLM response text.

        Args:
            response_text: The raw text response from the LLM.

        Returns:
            A validated GoalAchievement object.

        Raises:
            JudgeLLMError: If the response cannot be parsed as valid JSON or doesn't match schema.
        """
        json_str = self._extract_json(response_text)
        try:
            data = json.loads(json_str)
            return GoalAchievement(**data)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise JudgeLLMError(
                f"Failed to parse GoalAchievement from LLM response: {e}. "
                f"Response was: {response_text[:200]}"
            )
