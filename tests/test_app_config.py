"""Unit tests for app_config module."""

import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from src.app_config import (
    _load_config_file,
    _load_env_vars,
    load_app_config,
    merge_config,
    validate_required_config,
)
from src.models import AppConfig


class TestLoadConfigFile:
    """Tests for _load_config_file."""

    def test_returns_empty_dict_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GC_EVALUATOR_CONFIG_FILE", str(tmp_path / "nonexistent.yaml"))
        result = _load_config_file()
        assert result == {}

    def test_loads_yaml_config_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"ollama_base_url": "http://myhost:11434", "ollama_model": "llama3"})
        )
        monkeypatch.setenv("GC_EVALUATOR_CONFIG_FILE", str(config_file))
        result = _load_config_file()
        assert result == {"ollama_base_url": "http://myhost:11434", "ollama_model": "llama3"}

    def test_returns_empty_dict_for_non_dict_yaml(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("just a string")
        monkeypatch.setenv("GC_EVALUATOR_CONFIG_FILE", str(config_file))
        result = _load_config_file()
        assert result == {}

    def test_returns_empty_dict_for_empty_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        monkeypatch.setenv("GC_EVALUATOR_CONFIG_FILE", str(config_file))
        result = _load_config_file()
        assert result == {}

    def test_uses_default_config_yaml_path(self, monkeypatch):
        # Ensure no env var is set and default path doesn't exist
        monkeypatch.delenv("GC_EVALUATOR_CONFIG_FILE", raising=False)
        monkeypatch.chdir(tempfile.mkdtemp())
        result = _load_config_file()
        assert result == {}


class TestLoadEnvVars:
    """Tests for _load_env_vars."""

    def test_returns_empty_when_no_env_vars_set(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("GC_EVALUATOR_LLM_TIMEOUT", raising=False)
        result = _load_env_vars()
        assert result == {}

    def test_loads_ollama_base_url(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote:11434")
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("GC_EVALUATOR_LLM_TIMEOUT", raising=False)
        result = _load_env_vars()
        assert result == {"ollama_base_url": "http://remote:11434"}

    def test_loads_ollama_model(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.setenv("OLLAMA_MODEL", "mistral")
        monkeypatch.delenv("GC_EVALUATOR_LLM_TIMEOUT", raising=False)
        result = _load_env_vars()
        assert result == {"ollama_model": "mistral"}

    def test_loads_llm_timeout_as_int(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.setenv("GC_EVALUATOR_LLM_TIMEOUT", "300")
        result = _load_env_vars()
        assert result == {"llm_timeout": 300}
        assert isinstance(result["llm_timeout"], int)

    def test_loads_all_env_vars(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://host:1234")
        monkeypatch.setenv("OLLAMA_MODEL", "phi3")
        monkeypatch.setenv("GC_EVALUATOR_LLM_TIMEOUT", "60")
        result = _load_env_vars()
        assert result == {
            "ollama_base_url": "http://host:1234",
            "ollama_model": "phi3",
            "llm_timeout": 60,
        }


class TestLoadAppConfig:
    """Tests for load_app_config."""

    def test_returns_defaults_when_no_sources(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GC_EVALUATOR_CONFIG_FILE", str(tmp_path / "nonexistent.yaml"))
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("GC_EVALUATOR_LLM_TIMEOUT", raising=False)
        config = load_app_config()
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.ollama_model is None
        assert config.llm_timeout == 120

    def test_config_file_overrides_defaults(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"ollama_base_url": "http://filehost:5555", "ollama_model": "gemma"})
        )
        monkeypatch.setenv("GC_EVALUATOR_CONFIG_FILE", str(config_file))
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("GC_EVALUATOR_LLM_TIMEOUT", raising=False)
        config = load_app_config()
        assert config.ollama_base_url == "http://filehost:5555"
        assert config.ollama_model == "gemma"
        assert config.llm_timeout == 120  # default preserved

    def test_env_vars_override_config_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({
                "ollama_base_url": "http://filehost:5555",
                "ollama_model": "gemma",
                "llm_timeout": 60,
            })
        )
        monkeypatch.setenv("GC_EVALUATOR_CONFIG_FILE", str(config_file))
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://envhost:9999")
        monkeypatch.setenv("OLLAMA_MODEL", "llama3")
        monkeypatch.delenv("GC_EVALUATOR_LLM_TIMEOUT", raising=False)
        config = load_app_config()
        assert config.ollama_base_url == "http://envhost:9999"
        assert config.ollama_model == "llama3"
        assert config.llm_timeout == 60  # from file, not overridden by env

    def test_env_var_timeout_overrides_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"llm_timeout": 60}))
        monkeypatch.setenv("GC_EVALUATOR_CONFIG_FILE", str(config_file))
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.setenv("GC_EVALUATOR_LLM_TIMEOUT", "200")
        config = load_app_config()
        assert config.llm_timeout == 200


class TestMergeConfig:
    """Tests for merge_config."""

    def test_web_overrides_take_precedence(self):
        base = AppConfig(
            ollama_base_url="http://base:11434",
            ollama_model="base-model",
            llm_timeout=120,
        )
        overrides = {"ollama_model": "web-model", "ollama_base_url": "http://web:11434"}
        result = merge_config(base, overrides)
        assert result.ollama_model == "web-model"
        assert result.ollama_base_url == "http://web:11434"
        assert result.llm_timeout == 120  # unchanged

    def test_none_overrides_are_ignored(self):
        base = AppConfig(
            ollama_base_url="http://base:11434",
            ollama_model="base-model",
            llm_timeout=120,
        )
        overrides = {"ollama_model": None, "ollama_base_url": None}
        result = merge_config(base, overrides)
        assert result.ollama_model == "base-model"
        assert result.ollama_base_url == "http://base:11434"

    def test_empty_string_overrides_are_ignored(self):
        base = AppConfig(
            ollama_base_url="http://base:11434",
            ollama_model="base-model",
            llm_timeout=120,
        )
        overrides = {"ollama_model": "", "ollama_base_url": ""}
        result = merge_config(base, overrides)
        assert result.ollama_model == "base-model"
        assert result.ollama_base_url == "http://base:11434"

    def test_timeout_override_converted_to_int(self):
        base = AppConfig(llm_timeout=120)
        overrides = {"llm_timeout": "300"}
        result = merge_config(base, overrides)
        assert result.llm_timeout == 300
        assert isinstance(result.llm_timeout, int)

    def test_empty_overrides_dict_returns_same_values(self):
        base = AppConfig(
            ollama_base_url="http://base:11434",
            ollama_model="my-model",
            llm_timeout=90,
        )
        result = merge_config(base, {})
        assert result.ollama_base_url == base.ollama_base_url
        assert result.ollama_model == base.ollama_model
        assert result.llm_timeout == base.llm_timeout

    def test_unknown_keys_in_overrides_are_passed_through(self):
        """Unknown keys will cause Pydantic validation error if not valid fields."""
        base = AppConfig(ollama_model="model")
        # Only valid AppConfig fields should be passed
        overrides = {"ollama_model": "new-model"}
        result = merge_config(base, overrides)
        assert result.ollama_model == "new-model"


class TestValidateRequiredConfig:
    """Tests for validate_required_config."""

    def test_returns_empty_list_when_all_present(self):
        config = AppConfig(ollama_model="llama3")
        missing = validate_required_config(config)
        assert missing == []

    def test_returns_ollama_model_when_missing(self):
        config = AppConfig(ollama_model=None)
        missing = validate_required_config(config)
        assert missing == ["ollama_model"]

    def test_returns_ollama_model_for_default_config(self):
        config = AppConfig()
        missing = validate_required_config(config)
        assert missing == ["ollama_model"]

    def test_model_set_means_no_missing_fields(self):
        config = AppConfig(
            ollama_base_url="http://custom:11434",
            ollama_model="phi3",
            llm_timeout=60,
        )
        missing = validate_required_config(config)
        assert missing == []


class TestConfigPrecedenceIntegration:
    """Integration tests verifying the full precedence chain."""

    def test_full_precedence_chain(self, tmp_path, monkeypatch):
        """Web UI > env vars > config file > defaults."""
        # Config file sets all values
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({
                "ollama_base_url": "http://file:1111",
                "ollama_model": "file-model",
                "llm_timeout": 50,
            })
        )
        monkeypatch.setenv("GC_EVALUATOR_CONFIG_FILE", str(config_file))

        # Env vars override some values
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://env:2222")
        monkeypatch.setenv("OLLAMA_MODEL", "env-model")
        monkeypatch.delenv("GC_EVALUATOR_LLM_TIMEOUT", raising=False)

        # Load base config (env > file > defaults)
        base = load_app_config()
        assert base.ollama_base_url == "http://env:2222"
        assert base.ollama_model == "env-model"
        assert base.llm_timeout == 50  # from file

        # Web UI overrides everything
        web_overrides = {"ollama_base_url": "http://web:3333", "ollama_model": "web-model"}
        final = merge_config(base, web_overrides)
        assert final.ollama_base_url == "http://web:3333"
        assert final.ollama_model == "web-model"
        assert final.llm_timeout == 50  # still from file (not overridden by web)
