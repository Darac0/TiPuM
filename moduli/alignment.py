"""Poravnanje Whisperovih vremenskih oznaka s pyannote govornicima."""

from __future__ import annotations

import re
from typing import Any, Iterable


UNKNOWN_SPEAKER = "NEPOZNATO"


def overlapDuration(
    firstStart: float,
    firstEnd: float,
    secondStart: float,
    secondEnd: float,
) -> float:
    """Vrati trajanje presjeka dvaju zatvorenih vremenskih intervala."""
    return max(0.0, min(firstEnd, secondEnd) - max(firstStart, secondStart))


def normalizeDiarizationSegments(segments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Provjeri i sortiraj pyannote segmente u zajednički format."""
    normalized = []

    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        speaker = str(segment["speaker"])

        if end <= start:
            continue

        normalized.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker": speaker,
        })

    return sorted(normalized, key=lambda item: (item["start"], item["end"], item["speaker"]))


# Rangiraj intervale po preklapanju, udaljenosti sredine riječi i početku intervala.
def _findSpeakerInNormalized(
    start: float,
    end: float,
    normalizedSegments: list[dict[str, Any]],
) -> str:
    if not normalizedSegments:
        return UNKNOWN_SPEAKER

    start = float(start)
    end = float(end)
    if end < start:
        start, end = end, start

    midpoint = (start + end) / 2
    ranked = []

    for segment in normalizedSegments:
        overlap = overlapDuration(start, end, segment["start"], segment["end"])
        if segment["start"] <= midpoint <= segment["end"]:
            midpointDistance = 0.0
        else:
            midpointDistance = min(
                abs(midpoint - segment["start"]),
                abs(midpoint - segment["end"]),
            )

        # Najveće preklapanje, zatim najmanja udaljenost i najraniji segment.
        ranked.append((-overlap, midpointDistance, segment["start"], segment["speaker"]))

    return min(ranked)[3]


def findSpeaker(start: float, end: float, segments: Iterable[dict[str, Any]]) -> str:
    """Odaberi govornika s najvećim preklapanjem, uz najbliži segment kao fallback."""
    normalized = normalizeDiarizationSegments(segments)
    return _findSpeakerInNormalized(start, end, normalized)


def extractWhisperWords(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """Pretvori Whisper rezultat u riječi; koristi cijeli segment samo kao fallback."""
    words = []

    for segment in transcript.get("segments", []):
        segmentWords = segment.get("words") or []

        if not segmentWords:
            text = str(segment.get("text", "")).strip()
            if text:
                words.append({
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "text": text,
                })
            continue

        for word in segmentWords:
            text = str(word.get("word", "")).strip()
            if not text or "start" not in word or "end" not in word:
                continue

            item = {
                "start": float(word["start"]),
                "end": float(word["end"]),
                "text": text,
            }
            if word.get("probability") is not None:
                item["probability"] = float(word["probability"])
            words.append(item)

    return sorted(words, key=lambda item: (item["start"], item["end"]))


def alignWordsToSpeakers(
    words: Iterable[dict[str, Any]],
    diarizationSegments: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Svakoj riječi dodijeli pyannote govornika prema vremenskom preklapanju."""
    segments = normalizeDiarizationSegments(diarizationSegments)
    aligned = []

    for word in words:
        item = dict(word)
        item["start"] = float(item["start"])
        item["end"] = float(item["end"])
        item["speaker"] = _findSpeakerInNormalized(item["start"], item["end"], segments)
        aligned.append(item)

    return sorted(aligned, key=lambda item: (item["start"], item["end"]))


# Spoji tokene razmakom pa ukloni suvišne razmake uz interpunkciju.
def _joinTokens(tokens: Iterable[str]) -> str:
    text = " ".join(token.strip() for token in tokens if token.strip())
    text = re.sub(r"\s+([,.;:!?%…\)\]])", r"\1", text)
    text = re.sub(r"([\(\[])\s+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def groupWordsBySpeaker(
    alignedWords: Iterable[dict[str, Any]],
    maxGap: float = 1.5,
) -> list[dict[str, Any]]:
    """Spoji susjedne riječi istog govornika u kraće, čitljive iskaze."""
    utterances: list[dict[str, Any]] = []

    for word in sorted(alignedWords, key=lambda item: (item["start"], item["end"])):
        speaker = str(word.get("speaker", UNKNOWN_SPEAKER))
        start = float(word["start"])
        end = float(word["end"])
        text = str(word.get("text", "")).strip()

        if not text or end <= start:
            continue

        canMerge = bool(
            utterances
            and utterances[-1]["speaker"] == speaker
            and start - utterances[-1]["end"] <= maxGap
        )

        wordRecord = {"start": start, "end": end, "text": text}
        if word.get("probability") is not None:
            wordRecord["probability"] = float(word["probability"])

        if canMerge:
            current = utterances[-1]
            current["end"] = max(current["end"], end)
            current["words"].append(wordRecord)
            current["text"] = _joinTokens(item["text"] for item in current["words"])
        else:
            utterances.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
                "words": [wordRecord],
            })

    for utterance in utterances:
        utterance["start"] = round(utterance["start"], 3)
        utterance["end"] = round(utterance["end"], 3)

    return utterances


def alignTranscript(
    transcript: dict[str, Any],
    diarizationSegments: Iterable[dict[str, Any]],
    maxGap: float = 1.5,
) -> list[dict[str, Any]]:
    """Cijeli Whisper rezultat pretvori u iskaze označene govornikom."""
    words = extractWhisperWords(transcript)
    alignedWords = alignWordsToSpeakers(words, diarizationSegments)
    return groupWordsBySpeaker(alignedWords, maxGap=maxGap)
