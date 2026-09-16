"""Izrada LLM sažetka s determinističkim fallbackom i zajedničko spremanje."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from .local_llm_summary import LocalLlmClient, create_local_llm_summary
from .medical_summary import create_medical_summary


# Materijaliziraj ulazne iskaze i predaj ih LLM ekstraktoru s mogućnošću fallbacka.
def create_configured_medical_summary(
    utterances: Iterable[dict[str, Any]],
    source: dict[str, Any] | None = None,
    *,
    client: LocalLlmClient | None = None,
    fallback_on_error: bool = True,
) -> dict[str, Any]:
    utterance_list = list(utterances)
    if client is None and os.getenv("TIPUM_DISABLE_LLM", "").lower() in {"1", "true", "yes"}:
        payload = create_medical_summary(utterance_list, source=source)
        payload["generator"]["backend_selection"] = "rules_only"
        return payload
    return create_local_llm_summary(
        utterance_list,
        source=source,
        client=client,
        fallback_on_error=fallback_on_error,
    )


# Spremi odabrani sažetak u JSON te zasebno njegov summary_text.
def save_configured_medical_summary(
    utterances: Iterable[dict[str, Any]],
    json_path: Path,
    text_path: Path | None = None,
    source: dict[str, Any] | None = None,
    *,
    client: LocalLlmClient | None = None,
    fallback_on_error: bool = True,
) -> dict[str, Any]:
    payload = create_configured_medical_summary(
        utterances,
        source=source,
        client=client,
        fallback_on_error=fallback_on_error,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (text_path or json_path.with_suffix(".txt")).write_text(
        payload["summary_text"], encoding="utf-8"
    )
    return payload
