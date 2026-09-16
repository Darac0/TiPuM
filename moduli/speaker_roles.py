"""Jezična klasifikacija uloga liječnika i pacijenta.

Pravila namjerno opisuju gramatičko lice i tip komunikacijske radnje umjesto
pojedinačnih dijagnoza ili rečenica iz evaluacijskog skupa. Time klasifikator
ostaje primjenjiv na nove medicinske razgovore.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Pattern


ROLE_CLASSIFIER_VERSION = "lingvisticka_pravila_v2"
DOCTOR = "LIJEČNIK"
PATIENT = "PACIJENT"
UNKNOWN = "NEPOZNATO"


# Jedno ponderirano jezično pravilo za prepoznavanje komunikacijske uloge.
@dataclass(frozen=True)
class _Rule:
    pattern: Pattern[str]
    weight: float
    description: str


# Kompajliraj regex uz zanemarivanje veličine slova i pridruži mu težinu.
def _rule(pattern: str, weight: float, description: str) -> _Rule:
    return _Rule(re.compile(pattern, flags=re.IGNORECASE), weight, description)


# Drugo lice množine/formalno: „trebate li”, „imate li”, „osjećate li”.
# Ne uključujemo svaku riječ koja završava na -te jer bi imenice i prilozi
# stvarali mnogo lažnih pogodaka; uzorak vrijedi samo neposredno ispred „li”.
_DOCTOR_RULES = (
    _rule(
        r"\b[\wčćđšž]*(?:ate|ete|ite)\s+li\b",
        3.5,
        "pitanje u drugom licu množine/formalnom obliku",
    ),
    _rule(
        r"\b(?:jeste|biste)\s+li\b",
        3.5,
        "nepravilno pitanje u drugom licu",
    ),
    _rule(
        r"\b(?:koliko dugo|od kada|otkad|kada (?:je|su)|gdje vas|što vas|"
        r"kako se osjećate|koliko često|kakve simptome)\b",
        2.0,
        "uzimanje anamneze",
    ),
    _rule(
        r"\b(?:preporučujem|savjetujem|predlažem|propisat ću|uputit ću|"
        r"pregledat ću|poslušat ću|izmjerit ću|provjerit ću|napravit ćemo)\b",
        3.0,
        "klinička preporuka ili radnja liječnika",
    ),
    _rule(
        r"\b(?:uzimajte|uzmite|pijte|odmarajte|odmorite|javite se|dođite|"
        r"izbjegavajte|nastavite|prestanite|pratite|pričekajte)\b",
        2.5,
        "uputa pacijentu",
    ),
    _rule(
        r"\b(?:nalaz pokazuje|pregled pokazuje|radi se o|moja preporuka|"
        r"prema nalazu|na pregledu)\b",
        1.75,
        "tumačenje nalaza",
    ),
    _rule(
        r"\b(?:vas|vam|vaš|vaša|vaše)\b[^?!.]{0,45}\?",
        1.0,
        "pitanje usmjereno drugoj osobi",
    ),
)


# Prvo lice: pravilni oblici na -am/-em te česti nepravilni oblici.
_PATIENT_RULES = (
    _rule(
        r"\b[\wčćđšž]*(?:am|em)\s+li\b",
        3.5,
        "pitanje u prvom licu jednine",
    ),
    _rule(
        r"\b(?:mogu|hoću)\s+li\b|\bću\s+li\b",
        3.5,
        "nepravilno ili pomoćno pitanje u prvom licu",
    ),
    _rule(
        r"\b(?:imam|nemam|osjećam|kašljem|kišem|povraćam|uzimam|pijem|"
        r"primjećujem|trebam|mislim)\b",
        1.5,
        "opis vlastitog stanja u prvom licu",
    ),
    _rule(
        r"\b(?:boli|peče|steže|svrbi)\s+me\b|\bvrti mi se\b|"
        r"\bteško mi je\b|\bne mogu\b|\bmeni\b",
        2.0,
        "opis vlastitog simptoma",
    ),
    _rule(
        r"\b(?:primijetio|primijetila|uzeo|uzela|bio|bila|dobio|dobila)\s+sam\b",
        1.5,
        "iskustvo pacijenta u prvom licu",
    ),
    _rule(
        r"\b(?:moj|moja|moje|moji|moju|mog|mene|mi)\b",
        0.5,
        "zamjenica prvog lica",
    ),
    _rule(
        r"^(?:da|ne|jesam|nisam|hvala|u redu)[.!? ]*$",
        0.25,
        "kratak odgovor pacijenta",
    ),
)


# Ujednači tekst prije bodovanja jezičnih obrazaca.
def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


# Zbroji težine pravila koja se pojavljuju u tekstu iskaza.
def _score_text(text: str) -> dict[str, Any]:
    normalized = _normalize(text)
    scores = {DOCTOR: 0.0, PATIENT: 0.0}
    evidence: dict[str, list[str]] = {DOCTOR: [], PATIENT: []}
    for role, rules in ((DOCTOR, _DOCTOR_RULES), (PATIENT, _PATIENT_RULES)):
        for rule in rules:
            matches = list(rule.pattern.finditer(normalized))
            if not matches:
                continue
            contribution = rule.weight * len(matches)
            scores[role] += contribution
            evidence[role].append(f"{rule.description} (+{contribution:g})")
    return {"scores": scores, "evidence": evidence}


def classifyRole(text: str) -> str:
    """Klasificiraj jedan iskaz; neriješeni rezultat ostavi nepoznatim."""
    result = _score_text(text)
    doctor_score = result["scores"][DOCTOR]
    patient_score = result["scores"][PATIENT]
    if doctor_score > patient_score:
        return DOCTOR
    if patient_score > doctor_score:
        return PATIENT
    return UNKNOWN


def buildRoleDecision(utterances: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Boduj cijeli razgovor i vrati mapu uloga s procjenom pouzdanosti.

    Kada postoje točno dva govornika, obje moguće raspodjele vrednuju se
    zajednički. Time rezultat poštuje poznatu strukturu razgovora: jedan
    liječnik i jedan pacijent. Slab prior prvog govornika služi samo za
    neriješene ili gotovo prazne razgovore i ne nadjačava jezične dokaze.
    """
    items = list(utterances)
    speakers: list[str] = []
    scores: dict[str, dict[str, float]] = {}
    evidence: dict[str, dict[str, list[str]]] = {}

    for utterance in items:
        speaker = str(utterance.get("speaker", UNKNOWN))
        if speaker not in scores:
            speakers.append(speaker)
            scores[speaker] = {DOCTOR: 0.0, PATIENT: 0.0}
            evidence[speaker] = {DOCTOR: [], PATIENT: []}
        result = _score_text(str(utterance.get("text", "")))
        for role in (DOCTOR, PATIENT):
            scores[speaker][role] += float(result["scores"][role])
            evidence[speaker][role].extend(result["evidence"][role])

    role_map: dict[str, str] = {}
    assignment_margin = 0.0
    confidence = 0.0

    if len(speakers) == 2:
        first, second = speakers
        first_doctor = scores[first][DOCTOR] + scores[second][PATIENT] + 0.35
        second_doctor = scores[first][PATIENT] + scores[second][DOCTOR]
        assignment_margin = abs(first_doctor - second_doctor)
        if first_doctor >= second_doctor:
            role_map = {first: DOCTOR, second: PATIENT}
        else:
            role_map = {first: PATIENT, second: DOCTOR}
        assignment_total = first_doctor + second_doctor
        confidence = assignment_margin / max(assignment_total, 1.0)
    else:
        for speaker in speakers:
            doctor_score = scores[speaker][DOCTOR]
            patient_score = scores[speaker][PATIENT]
            if doctor_score > patient_score:
                role_map[speaker] = DOCTOR
            elif patient_score > doctor_score:
                role_map[speaker] = PATIENT
            else:
                role_map[speaker] = speaker
        differences = [
            abs(value[DOCTOR] - value[PATIENT]) for value in scores.values()
        ]
        totals = [sum(value.values()) for value in scores.values()]
        assignment_margin = sum(differences)
        confidence = assignment_margin / max(sum(totals), 1.0)

    total_evidence = sum(sum(value.values()) for value in scores.values())
    needs_review = total_evidence < 2.0 or confidence < 0.15
    if needs_review:
        role_map = {speaker: UNKNOWN for speaker in speakers}
    return {
        "version": ROLE_CLASSIFIER_VERSION,
        "role_map": role_map,
        "confidence": round(confidence, 6),
        "assignment_margin": round(assignment_margin, 6),
        "needs_review": needs_review,
        "speaker_scores": scores,
        "evidence": evidence,
    }


def buildRoleMap(utterances: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Glasove ``SPEAKER_XX`` mapiraj na uloge prema cijelom razgovoru."""
    return buildRoleDecision(utterances)["role_map"]


def buildRoleConfusionMatrix(
    observations: Iterable[tuple[str, str, int]],
    labels: tuple[str, ...] = (DOCTOR, PATIENT, UNKNOWN),
) -> dict[str, Any]:
    """Izračunaj ponderiranu matricu zabune za referentne i predviđene uloge."""
    matrix = {
        reference: {predicted: 0 for predicted in labels}
        for reference in labels
    }
    total = correct = 0
    for reference, predicted, weight in observations:
        if reference not in matrix:
            raise ValueError(f"Nepoznata referentna uloga: {reference}")
        if predicted not in matrix[reference]:
            raise ValueError(f"Nepoznata predviđena uloga: {predicted}")
        weight = int(weight)
        if weight < 0:
            raise ValueError("Težina opažanja ne smije biti negativna.")
        matrix[reference][predicted] += weight
        total += weight
        if reference == predicted:
            correct += weight

    return {
        "labels": list(labels),
        "matrix": matrix,
        "support": {
            reference: sum(matrix[reference].values()) for reference in labels
        },
        "total_weight": total,
        "correct_weight": correct,
        "accuracy": correct / total if total else None,
    }


def mergeRoleConfusionMatrices(matrices: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Zbroji matrice iz pojedinih razgovora bez gubitka neodređenih predikcija."""
    matrices = list(matrices)
    if not matrices:
        return None
    labels = tuple(matrices[0]["labels"])
    observations: list[tuple[str, str, int]] = []
    for item in matrices:
        if tuple(item["labels"]) != labels:
            raise ValueError("Matrice zabune nemaju jednak poredak oznaka.")
        for reference in labels:
            for predicted in labels:
                observations.append(
                    (reference, predicted, int(item["matrix"][reference][predicted]))
                )
    return buildRoleConfusionMatrix(observations, labels=labels)
