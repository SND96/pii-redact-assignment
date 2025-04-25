import json
from pathlib import Path
from unittest import mock

import pytest

from pii_redactor import PIIRedactor


@pytest.fixture(scope="module")
def redactor():
    """Use a dummy OpenAI client (no network)."""
    r = PIIRedactor(api_key="dummy", use_mistral=False)
    # Patch out the real network call
    r._redact_with_openai = lambda text: {
        "redacted_text": "[GIVENNAME1] went to [CITY].",
        "pii_entities": [
            {"label": "GIVENNAME1", "value": "Alice", "start": 0, "end": 5},
            {"label": "CITY", "value": "Paris", "start": 15, "end": 20},
        ],
    }
    return r


def test_align_redaction(redactor):
    """Test the alignment of redacted text with original text."""
    # Simple test with one entity
    original = "Bob is here."
    redacted = "[NAME] is here."
    result = redactor.align_redaction(original, redacted)
    
    assert len(result) == 1
    assert result[0]["label"] == "NAME"
    assert result[0]["value"].strip() == "Bob"


def test_recover_spans_exact_match(redactor):
    """Test exact span recovery."""
    text = "Phone: (555) 123-4567"
    pii = [{"label": "TEL", "value": "(555) 123-4567"}]
    result = redactor.recover_spans(text, pii)
    
    assert len(result) == 1
    entity = result[0]
    assert entity["label"] == "TEL"
    assert entity["value"] == "(555) 123-4567"
    assert entity["start"] == 7


def test_find_approximate_span(redactor):
    """Test approximate span finding."""
    # Simple test with exact match
    text = "The ID is AB1234"
    value = "AB1234"
    idx = redactor._find_approximate_span(text, value)
    assert idx >= 0, "Should find the exact match"
    assert text[idx:].startswith("AB1234")


def test_redact_text_wrapper(redactor):
    """Test the redaction wrapper."""
    result = redactor.redact_text("Irrelevant for the stub")
    assert result["redacted_text"] == "[GIVENNAME1] went to [CITY]."
    assert len(result["pii_entities"]) == 2


def test_process_single_file(tmp_path, redactor):
    """Test processing a single file."""
    sample = tmp_path / "sample.txt"
    sample.write_text("Alice went to Paris.")

    results = redactor.process_dataset(sample)
    assert len(results) == 1
    assert results[0]["redacted_text"] == "[GIVENNAME1] went to [CITY]."