"""Tests for the setup wizard helper functions."""

import pytest

from mragent.cli.setup_wizard import filter_models_by_prefix, FAMILIES


# ── filter_models_by_prefix ──────────────────────────────────────────────────

SAMPLE_MODELS = [
    {"id": "meta/llama-3.1-8b-instruct", "owned_by": "meta"},
    {"id": "meta/llama-3.1-70b-instruct", "owned_by": "meta"},
    {"id": "qwen/qwen2.5-72b-instruct", "owned_by": "qwen"},
    {"id": "nvidia/llama-3.1-nemotron-70b-instruct", "owned_by": "nvidia"},
    {"id": "mistral/mistral-7b-instruct", "owned_by": "mistral"},
    {"id": "google/gemma-7b", "owned_by": "google"},
]


def test_filter_none_returns_all():
    """None prefix should return the full list."""
    result = filter_models_by_prefix(SAMPLE_MODELS, None)
    assert result == SAMPLE_MODELS


def test_filter_meta():
    result = filter_models_by_prefix(SAMPLE_MODELS, "meta/")
    assert len(result) == 2
    assert all(m["id"].startswith("meta/") for m in result)


def test_filter_qwen():
    result = filter_models_by_prefix(SAMPLE_MODELS, "qwen/")
    assert len(result) == 1
    assert result[0]["id"] == "qwen/qwen2.5-72b-instruct"


def test_filter_nvidia():
    result = filter_models_by_prefix(SAMPLE_MODELS, "nvidia/")
    assert len(result) == 1
    assert result[0]["id"] == "nvidia/llama-3.1-nemotron-70b-instruct"


def test_filter_unknown_prefix_returns_empty():
    result = filter_models_by_prefix(SAMPLE_MODELS, "openai/")
    assert result == []


def test_filter_empty_list():
    result = filter_models_by_prefix([], "meta/")
    assert result == []


# ── FAMILIES registry ────────────────────────────────────────────────────────

def test_families_has_all_entry():
    """FAMILIES must include an 'all' catch-all entry first."""
    assert FAMILIES[0][0] == "all"


def test_families_includes_required_providers():
    """Must list Meta, Qwen, and NVIDIA families."""
    keys = [k for k, _ in FAMILIES]
    assert "meta/" in keys
    assert "qwen/" in keys
    assert "nvidia/" in keys
