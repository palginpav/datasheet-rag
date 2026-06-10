"""Manifest schema and loading.

The manifest is the reproducibility contract of this project: datasheet PDFs
are publicly downloadable but not redistributable, so the repository ships
only this metadata and a downloader. Anyone can rebuild the exact corpus
locally; checksums pin the document versions the results were measured on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

Manufacturer = Literal["ti", "st", "nxp", "microchip", "analog", "onsemi", "renesas", "other"]
Category = Literal["mcu", "opamp", "ldo", "interface", "adc", "dac", "power", "sensor", "other"]


class ManifestEntry(BaseModel):
    """One datasheet in the corpus."""

    part: str = Field(min_length=2, description="Manufacturer part number, e.g. 'LM358'")
    manufacturer: Manufacturer
    url: HttpUrl = Field(description="Canonical manufacturer PDF URL")
    sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Hex digest pinning the PDF version; null until first verified download",
    )
    category: Category = "other"
    pages: int | None = Field(default=None, ge=1)

    @field_validator("part")
    @classmethod
    def normalize_part(cls, v: str) -> str:
        return v.strip().upper()


class Manifest(BaseModel):
    """The corpus manifest (data/manifest.json)."""

    version: int = 1
    entries: list[ManifestEntry] = Field(default_factory=list)

    @field_validator("entries")
    @classmethod
    def no_duplicate_parts(cls, v: list[ManifestEntry]) -> list[ManifestEntry]:
        seen: set[tuple[str, str]] = set()
        for e in v:
            key = (e.manufacturer, e.part)
            if key in seen:
                raise ValueError(f"duplicate manifest entry: {e.manufacturer}/{e.part}")
            seen.add(key)
        return v

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw.pop("$schema_note", None)
        return cls.model_validate(raw)

    def save(self, path: str | Path) -> None:
        payload = self.model_dump(mode="json")
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
