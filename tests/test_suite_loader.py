"""Unit tests for the evaluation suite loader module."""

import json
import os
import tempfile

import pytest
import yaml
from pydantic import ValidationError

from src.suite_loader import (
    load_evaluation_suite,
    load_evaluation_suite_from_string,
    serialize_evaluation_suite,
    validate_evaluation_suite,
)
from src.models import EvaluationSuite, Goal


class TestLoadEvaluationSuite:
    """Tests for load_evaluation_suite (file-based loading)."""

    def _write_temp_file(self, content: str, suffix: str) -> str:
        """Write content to a temp file and return the path."""
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        )
        f.write(content)
        f.close()
        return f.name

    def test_load_yaml_file(self):
        """Load a valid YAML evaluation suite file."""
        content = yaml.dump({
            "name": "Test Suite",
            "goals": [
                {
                    "name": "Intent Classification",
                    "description": "Classify customer intent",
                    "criteria": "The agent correctly identifies the intent",
                }
            ],
        })
        path = self._write_temp_file(content, ".yaml")
        try:
            suite = load_evaluation_suite(path)
            assert suite.name == "Test Suite"
            assert len(suite.goals) == 1
            assert suite.goals[0].name == "Intent Classification"
        finally:
            os.unlink(path)

    def test_load_yml_extension(self):
        """Load a valid .yml file."""
        content = yaml.dump({
            "name": "YML Suite",
            "goals": [
                {
                    "name": "Goal A",
                    "description": "Description A",
                    "criteria": "Criteria A",
                }
            ],
        })
        path = self._write_temp_file(content, ".yml")
        try:
            suite = load_evaluation_suite(path)
            assert suite.name == "YML Suite"
        finally:
            os.unlink(path)

    def test_load_json_file(self):
        """Load a valid JSON evaluation suite file."""
        content = json.dumps({
            "name": "JSON Suite",
            "goals": [
                {
                    "name": "Goal B",
                    "description": "Description B",
                    "criteria": "Criteria B",
                }
            ],
        })
        path = self._write_temp_file(content, ".json")
        try:
            suite = load_evaluation_suite(path)
            assert suite.name == "JSON Suite"
            assert len(suite.goals) == 1
        finally:
            os.unlink(path)

    def test_file_not_found_raises_error(self):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_evaluation_suite("/nonexistent/path/suite.yaml")

    def test_unsupported_extension_raises_value_error(self):
        """Unsupported file extension raises ValueError."""
        path = self._write_temp_file("data", ".xml")
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                load_evaluation_suite(path)
        finally:
            os.unlink(path)

    def test_multiple_goals(self):
        """Load a suite with multiple goals."""
        content = yaml.dump({
            "name": "Multi-Goal Suite",
            "goals": [
                {
                    "name": "Goal 1",
                    "description": "First goal",
                    "criteria": "First criteria",
                },
                {
                    "name": "Goal 2",
                    "description": "Second goal",
                    "criteria": "Second criteria",
                },
                {
                    "name": "Goal 3",
                    "description": "Third goal",
                    "criteria": "Third criteria",
                },
            ],
        })
        path = self._write_temp_file(content, ".yaml")
        try:
            suite = load_evaluation_suite(path)
            assert len(suite.goals) == 3
            assert suite.goals[0].name == "Goal 1"
            assert suite.goals[2].name == "Goal 3"
        finally:
            os.unlink(path)


class TestLoadEvaluationSuiteFromString:
    """Tests for load_evaluation_suite_from_string (string-based loading)."""

    def test_load_from_yaml_string(self):
        """Parse a valid YAML string into an EvaluationSuite."""
        content = "name: My Suite\ngoals:\n  - name: Goal A\n    description: Desc A\n    criteria: Criteria A\n"
        suite = load_evaluation_suite_from_string(content, "yaml")
        assert suite.name == "My Suite"
        assert len(suite.goals) == 1
        assert suite.goals[0].name == "Goal A"

    def test_load_from_json_string(self):
        """Parse a valid JSON string into an EvaluationSuite."""
        content = json.dumps({
            "name": "JSON Suite",
            "goals": [
                {"name": "G1", "description": "D1", "criteria": "C1"}
            ],
        })
        suite = load_evaluation_suite_from_string(content, "json")
        assert suite.name == "JSON Suite"

    def test_format_case_insensitive(self):
        """Format parameter is case-insensitive."""
        content = json.dumps({
            "name": "Suite",
            "goals": [{"name": "G", "description": "D", "criteria": "C"}],
        })
        suite = load_evaluation_suite_from_string(content, "JSON")
        assert suite.name == "Suite"

    def test_yml_format_accepted(self):
        """Format 'yml' is accepted as YAML."""
        content = "name: Suite\ngoals:\n  - name: G\n    description: D\n    criteria: C\n"
        suite = load_evaluation_suite_from_string(content, "yml")
        assert suite.name == "Suite"

    def test_unsupported_format_raises_value_error(self):
        """Unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported format"):
            load_evaluation_suite_from_string("data", "xml")

    def test_invalid_json_raises_value_error(self):
        """Invalid JSON content raises ValueError with descriptive message."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_evaluation_suite_from_string("{not valid json", "json")

    def test_invalid_yaml_raises_value_error(self):
        """Invalid YAML content raises ValueError with descriptive message."""
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_evaluation_suite_from_string(":\n  :\n    - [invalid", "yaml")

    def test_non_dict_content_raises_value_error(self):
        """Content that parses to non-dict raises ValueError."""
        with pytest.raises(ValueError, match="must be a JSON/YAML object"):
            load_evaluation_suite_from_string('"just a string"', "json")

    def test_non_dict_yaml_raises_value_error(self):
        """YAML that parses to a list raises ValueError."""
        with pytest.raises(ValueError, match="must be a JSON/YAML object"):
            load_evaluation_suite_from_string("- item1\n- item2\n", "yaml")

    def test_missing_name_raises_validation_error(self):
        """Missing suite name raises ValidationError."""
        content = json.dumps({
            "goals": [{"name": "G", "description": "D", "criteria": "C"}]
        })
        with pytest.raises(ValidationError):
            load_evaluation_suite_from_string(content, "json")

    def test_empty_goals_list_raises_validation_error(self):
        """Empty goals list raises ValidationError."""
        content = json.dumps({"name": "Suite", "goals": []})
        with pytest.raises(ValidationError):
            load_evaluation_suite_from_string(content, "json")

    def test_missing_goals_raises_validation_error(self):
        """Missing goals field raises ValidationError."""
        content = json.dumps({"name": "Suite"})
        with pytest.raises(ValidationError):
            load_evaluation_suite_from_string(content, "json")

    def test_goal_missing_name_raises_validation_error(self):
        """Goal with missing name raises ValidationError."""
        content = json.dumps({
            "name": "Suite",
            "goals": [{"description": "D", "criteria": "C"}],
        })
        with pytest.raises(ValidationError):
            load_evaluation_suite_from_string(content, "json")

    def test_goal_missing_description_raises_validation_error(self):
        """Goal with missing description raises ValidationError."""
        content = json.dumps({
            "name": "Suite",
            "goals": [{"name": "G", "criteria": "C"}],
        })
        with pytest.raises(ValidationError):
            load_evaluation_suite_from_string(content, "json")

    def test_goal_missing_criteria_raises_validation_error(self):
        """Goal with missing criteria raises ValidationError."""
        content = json.dumps({
            "name": "Suite",
            "goals": [{"name": "G", "description": "D"}],
        })
        with pytest.raises(ValidationError):
            load_evaluation_suite_from_string(content, "json")

    def test_goal_empty_name_raises_validation_error(self):
        """Goal with empty name raises ValidationError."""
        content = json.dumps({
            "name": "Suite",
            "goals": [{"name": "", "description": "D", "criteria": "C"}],
        })
        with pytest.raises(ValidationError):
            load_evaluation_suite_from_string(content, "json")

    def test_goal_whitespace_only_name_raises_validation_error(self):
        """Goal with whitespace-only name raises ValidationError."""
        content = json.dumps({
            "name": "Suite",
            "goals": [{"name": "   ", "description": "D", "criteria": "C"}],
        })
        with pytest.raises(ValidationError):
            load_evaluation_suite_from_string(content, "json")

    def test_goal_empty_description_raises_validation_error(self):
        """Goal with empty description raises ValidationError."""
        content = json.dumps({
            "name": "Suite",
            "goals": [{"name": "G", "description": "", "criteria": "C"}],
        })
        with pytest.raises(ValidationError):
            load_evaluation_suite_from_string(content, "json")

    def test_goal_empty_criteria_raises_validation_error(self):
        """Goal with empty criteria raises ValidationError."""
        content = json.dumps({
            "name": "Suite",
            "goals": [{"name": "G", "description": "D", "criteria": ""}],
        })
        with pytest.raises(ValidationError):
            load_evaluation_suite_from_string(content, "json")


class TestValidateEvaluationSuite:
    """Tests for validate_evaluation_suite (raw dict validation)."""

    def test_valid_dict(self):
        """Valid dict produces an EvaluationSuite."""
        data = {
            "name": "Suite",
            "goals": [
                {"name": "G1", "description": "D1", "criteria": "C1"},
            ],
        }
        suite = validate_evaluation_suite(data)
        assert isinstance(suite, EvaluationSuite)
        assert suite.name == "Suite"

    def test_missing_name_raises_validation_error(self):
        """Dict without name raises ValidationError."""
        data = {
            "goals": [{"name": "G", "description": "D", "criteria": "C"}]
        }
        with pytest.raises(ValidationError):
            validate_evaluation_suite(data)

    def test_empty_goals_raises_validation_error(self):
        """Dict with empty goals raises ValidationError."""
        data = {"name": "Suite", "goals": []}
        with pytest.raises(ValidationError):
            validate_evaluation_suite(data)

    def test_invalid_goal_fields_raises_validation_error(self):
        """Dict with invalid goal fields raises ValidationError."""
        data = {
            "name": "Suite",
            "goals": [{"name": "G", "description": "", "criteria": "C"}],
        }
        with pytest.raises(ValidationError):
            validate_evaluation_suite(data)

    def test_multiple_valid_goals(self):
        """Dict with multiple valid goals produces correct EvaluationSuite."""
        data = {
            "name": "Multi",
            "goals": [
                {"name": "G1", "description": "D1", "criteria": "C1"},
                {"name": "G2", "description": "D2", "criteria": "C2"},
            ],
        }
        suite = validate_evaluation_suite(data)
        assert len(suite.goals) == 2


class TestSerializeEvaluationSuite:
    """Tests for serialize_evaluation_suite (round-trip support)."""

    def _make_suite(self) -> EvaluationSuite:
        """Create a sample EvaluationSuite for testing."""
        return EvaluationSuite(
            name="Test Suite",
            goals=[
                Goal(
                    name="Intent Classification",
                    description="The agent correctly classified the customer's intent",
                    criteria="The agent's response must contain the correct intent label.",
                ),
                Goal(
                    name="Resolution",
                    description="The agent resolved the customer's issue",
                    criteria="The conversation ends with the customer's problem solved.",
                ),
            ],
        )

    def test_serialize_to_yaml(self):
        """Serialize to YAML produces valid YAML string."""
        suite = self._make_suite()
        result = serialize_evaluation_suite(suite, "yaml")
        # Should be parseable YAML
        data = yaml.safe_load(result)
        assert data["name"] == "Test Suite"
        assert len(data["goals"]) == 2

    def test_serialize_to_json(self):
        """Serialize to JSON produces valid JSON string."""
        suite = self._make_suite()
        result = serialize_evaluation_suite(suite, "json")
        data = json.loads(result)
        assert data["name"] == "Test Suite"
        assert len(data["goals"]) == 2

    def test_default_format_is_yaml(self):
        """Default format is YAML."""
        suite = self._make_suite()
        result = serialize_evaluation_suite(suite)
        # YAML output should not start with { (JSON indicator)
        assert not result.strip().startswith("{")
        data = yaml.safe_load(result)
        assert data["name"] == "Test Suite"

    def test_unsupported_format_raises_value_error(self):
        """Unsupported format raises ValueError."""
        suite = self._make_suite()
        with pytest.raises(ValueError, match="Unsupported format"):
            serialize_evaluation_suite(suite, "xml")

    def test_yaml_round_trip(self):
        """Serialize to YAML then parse back produces equivalent suite."""
        suite = self._make_suite()
        yaml_str = serialize_evaluation_suite(suite, "yaml")
        restored = load_evaluation_suite_from_string(yaml_str, "yaml")
        assert restored.name == suite.name
        assert len(restored.goals) == len(suite.goals)
        for orig, rest in zip(suite.goals, restored.goals):
            assert orig.name == rest.name
            assert orig.description == rest.description
            assert orig.criteria == rest.criteria

    def test_json_round_trip(self):
        """Serialize to JSON then parse back produces equivalent suite."""
        suite = self._make_suite()
        json_str = serialize_evaluation_suite(suite, "json")
        restored = load_evaluation_suite_from_string(json_str, "json")
        assert restored.name == suite.name
        assert len(restored.goals) == len(suite.goals)
        for orig, rest in zip(suite.goals, restored.goals):
            assert orig.name == rest.name
            assert orig.description == rest.description
            assert orig.criteria == rest.criteria

    def test_format_case_insensitive(self):
        """Format parameter is case-insensitive."""
        suite = self._make_suite()
        result = serialize_evaluation_suite(suite, "YAML")
        data = yaml.safe_load(result)
        assert data["name"] == "Test Suite"

    def test_yml_format_accepted(self):
        """Format 'yml' is accepted as YAML."""
        suite = self._make_suite()
        result = serialize_evaluation_suite(suite, "yml")
        data = yaml.safe_load(result)
        assert data["name"] == "Test Suite"

    def test_single_goal_round_trip(self):
        """Single-goal suite round-trips correctly."""
        suite = EvaluationSuite(
            name="Single Goal Suite",
            goals=[
                Goal(
                    name="Only Goal",
                    description="The only goal",
                    criteria="Must pass this criteria",
                )
            ],
        )
        yaml_str = serialize_evaluation_suite(suite, "yaml")
        restored = load_evaluation_suite_from_string(yaml_str, "yaml")
        assert restored == suite

    def test_special_characters_in_fields(self):
        """Goals with special characters round-trip correctly."""
        suite = EvaluationSuite(
            name="Suite with 'quotes' & <special> chars",
            goals=[
                Goal(
                    name="Goal: with colon",
                    description="Description with\nnewline",
                    criteria="Criteria with \"double quotes\" and 'single'",
                )
            ],
        )
        yaml_str = serialize_evaluation_suite(suite, "yaml")
        restored = load_evaluation_suite_from_string(yaml_str, "yaml")
        assert restored.name == suite.name
        assert restored.goals[0].name == suite.goals[0].name
        assert restored.goals[0].description == suite.goals[0].description
        assert restored.goals[0].criteria == suite.goals[0].criteria
