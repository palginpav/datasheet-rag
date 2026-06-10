"""Tests for the corpus manifest schema."""

import json

import pytest
from pydantic import ValidationError

from datasheet_rag.corpus.manifest import Manifest, ManifestEntry


def entry(**overrides) -> dict:
    base = {
        "part": "LM358",
        "manufacturer": "ti",
        "url": "https://www.ti.com/lit/ds/symlink/lm358.pdf",
        "sha256": None,
        "category": "opamp",
        "pages": None,
    }
    base.update(overrides)
    return base


def test_entry_normalizes_part_number():
    e = ManifestEntry.model_validate(entry(part="  lm358 "))
    assert e.part == "LM358"


def test_entry_rejects_bad_checksum():
    with pytest.raises(ValidationError):
        ManifestEntry.model_validate(entry(sha256="not-a-digest"))


def test_entry_accepts_valid_checksum():
    digest = "a" * 64
    e = ManifestEntry.model_validate(entry(sha256=digest))
    assert e.sha256 == digest


def test_manifest_rejects_duplicate_parts():
    with pytest.raises(ValidationError, match="duplicate"):
        Manifest.model_validate({"version": 1, "entries": [entry(), entry()]})


def test_manifest_allows_same_part_different_manufacturer():
    m = Manifest.model_validate(
        {"version": 1, "entries": [entry(), entry(manufacturer="st")]}
    )
    assert len(m.entries) == 2


def test_load_ignores_schema_note(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"$schema_note": "doc", "version": 1, "entries": [entry()]}))
    m = Manifest.load(p)
    assert m.entries[0].part == "LM358"


def test_save_round_trip(tmp_path):
    m = Manifest.model_validate({"version": 1, "entries": [entry()]})
    p = tmp_path / "manifest.json"
    m.save(p)
    assert Manifest.load(p) == m
