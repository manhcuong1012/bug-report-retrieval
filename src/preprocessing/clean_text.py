from __future__ import annotations

import re

MULTISPACE_RE = re.compile(r"\s+")
NON_TEXT_RE = re.compile(r"[^a-z0-9._/#\-\+\s]")


def normalize_whitespace(text: str) -> str:
    return MULTISPACE_RE.sub(" ", text).strip()


def normalize_text(text: str) -> str:
    lowered = text.lower()
    cleaned = NON_TEXT_RE.sub(" ", lowered)
    return normalize_whitespace(cleaned)


def build_text_raw(summary: str, description: str) -> str:
    return normalize_whitespace(f"{summary} {description}".strip())


def build_text_clean(summary: str, description: str) -> str:
    return normalize_text(build_text_raw(summary, description))


def build_weighted_text(summary: str, description: str, summary_repeat: int = 3) -> str:
    repeated_summary = " ".join([summary] * max(summary_repeat, 1))
    return normalize_text(f"{repeated_summary} {description}".strip())
