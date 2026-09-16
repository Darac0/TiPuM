"""Lokalna ekstrakcija medicinskih činjenica iz strukturiranog transkripta.

Modul je namjerno ekstraktivan: svaki podatak nosi doslovni citat, ulogu i
vremensku oznaku. Ne postavlja dijagnoze i ne dodaje informacije koje nisu
izgovorene u razgovoru.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
GENERATOR_VERSION = "lokalna_pravila_v1_3"
DEFAULT_REVIEW_REASON = (
    "Automatska transkripcija i ekstrakcija mogu sadržavati pogreške; "
    "prije kliničke uporabe potreban je ljudski pregled."
)
SECTION_ORDER = (
    "patient_reported_symptoms",
    "patient_negated_symptoms",
    "patient_history",
    "measurements",
    "clinician_findings",
    "clinician_assessments",
    "plan_and_recommendations",
    "safety_instructions",
    "patient_questions",
    "clinician_questions",
)
SECTION_LABELS = {
    "patient_reported_symptoms": "Simptomi koje pacijent navodi",
    "patient_negated_symptoms": "Simptomi koje pacijent negira",
    "patient_history": "Anamneza i pridržavanje terapije",
    "measurements": "Mjerenja i brojčane vrijednosti",
    "clinician_findings": "Nalazi koje liječnik izgovara",
    "clinician_assessments": "Izgovorene procjene liječnika",
    "plan_and_recommendations": "Plan i preporuke",
    "safety_instructions": "Upute za hitno ili ponovno javljanje",
    "patient_questions": "Pitanja pacijenta",
    "clinician_questions": "Pitanja liječnika",
}

ROLE_ALIASES = {
    "DOKTOR": "LIJEČNIK",
    "LIJECNIK": "LIJEČNIK",
    "LIJEČNIK": "LIJEČNIK",
    "PACIJENT": "PACIJENT",
}

SYMPTOM_PATTERN = re.compile(
    # Ručno navedene osnove riječi obuhvaćaju dio fleksije, ali nisu medicinska ontologija.
    r"\b(?:bol\w*|temperatur\w*|kašalj\w*|kašlj\w*|grl\w*|umor\w*|"
    r"disanj\w*|diš\w*|zrak\w*|zaduh\w*|mučnin\w*|proljev\w*|nadut\w*|"
    r"slabost\w*|trnc\w*|utrnul\w*|mokren\w*|mokrać\w*|pečenj\w*|"
    r"(?:često|učestalo|stalno)\s+(?:mokri\w*|na\s+zahod)|"
    r"moram\s+(?:vrlo\s+)?često\s+na\s+zahod|"
    r"pritisak\w*|zimic\w*|glavobolj\w*|piskanj\w*|palpitacij\w*|"
    r"oticanj\w*|otečen\w*|dispnej\w*|sinkop\w*|nesvjestic\w*|"
    r"povrać\w*|krv\w*|začepljen\w*|curi\s+nos|suze\s+oči|simptom\w*|"
    r"drhtavic\w*|hladan\s+znoj|glad\w*|svijest\w*)\b",
    re.IGNORECASE,
)
NEGATION_PATTERN = re.compile(
    # Negacija se traži unutar analizirane klauze; pravilo ne razumije svu sintaksu.
    r"\b(?:ne|nemam|nema|nisam|nije|nikad|bez|ništa)\b|"
    r"\b(?:normalan|normalna|normalno|uredan|uredna|uredno)\b",
    re.IGNORECASE,
)
HISTORY_PATTERN = re.compile(
    r"\b(?:uzimam|uzeo|uzela|koristim|zaboravio|zaboravila|preskoč\w*|"
    r"alerg\w*|trudn\w*|prehran\w*|jedem|hran\w*|slan\w*|sol\w*|"
    r"puš\w*|alkohol\w*|hodam|krećem|"
    r"operacij\w*|terapij\w*)\b",
    re.IGNORECASE,
)
MEASUREMENT_PATTERN = re.compile(
    r"\b(?:tlak|mjer\w*|temperatur\w*|puls|šećer|hba1c|hemoglobin|kreatinin|"
    r"trombocit\w*|tsh|t4|frakcij\w*|otkucaj\w*|miligram\w*|mg|"
    r"milimol\w*|mikromol\w*|gram\w*|post\w*|vrijednost\w*)\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(
    r"\b(?:\d+(?:[.,]\d+)?|nula|jedan|jedna|jednu|dva|dvije|tri|"
    r"četiri|pet|šest|sedam|osam|devet|deset|jedanaest|dvanaest|"
    r"trinaest|četrnaest|petnaest|šesnaest|sedamnaest|osamnaest|"
    r"devetnaest|dvadeset|trideset|četrdeset|pedeset|šezdeset|"
    r"sedamdeset|osamdeset|devedeset|sto|dvjesto|tristo|četiristo|"
    r"petsto|tisuću|pola|pol)\b",
    re.IGNORECASE,
)
FINDING_PATTERN = re.compile(
    r"\b(?:pregled|nalaz\w*|pluća|grlo|vrijednost\w*|iznosi|pokazuje|"
    r"zvuče|crven\w*|ritam|funkcij\w*|hemoglobin|trombocit\w*|"
    r"elektrokardiogram|ultrazvuk|frakcij\w*|uredn\w*|stabiln\w*)\b",
    re.IGNORECASE,
)
ASSESSMENT_PATTERN = re.compile(
    r"\b(?:vjerojatno|najvjerojatnije|radi\s+se\s+o|riječ\s+je\s+o|"
    r"dijagnoz\w*|procjena\s+je|nalazi\s+su|iznad(?:\s+\w+){0,2}\s+cilja|"
    r"nije\s+potreban|nije\s+potrebna)\b",
    re.IGNORECASE,
)
PLAN_PATTERN = re.compile(
    r"\b(?:preporuč\w*|savjet\w*|pokušaj\w*|raspored\w*|uključ\w*|"
    r"ću|ćemo|napravit\w*|učinit\w*|"
    r"provjerit\w*|mjerite|uzimajte|pijte|izbjegavajte|nastavite|"
    r"zapišite|donesite|postavit\w*|kontrol\w*|trebala\s+bi|"
    r"potrebno\s+je|javite\s+se)\b",
    re.IGNORECASE,
)
SAFETY_PATTERN = re.compile(
    r"\b(?:odmah|hitno|potraž\w*\s+pomoć|javite\s+se\s+ako|"
    r"ne\s+možete\s+govoriti|gubitak\s+svijesti|slabost\s+jedne\s+strane|"
    r"brzo\s+pogoršava)\b",
    re.IGNORECASE,
)
CONDITIONAL_PATTERN = re.compile(
    r"\b(?:ako|ukoliko)\b|\b(?:ću|ćete)\s+se\s+javiti\b",
    re.IGNORECASE,
)
ANAPHORIC_NEGATION_PATTERN = re.compile(
    r"\b(?:ništa\s+od\s+toga|takv\w*\s+epizod\w*|sve\s+navedeno|oboje)\b",
    re.IGNORECASE,
)
ANAPHORIC_ASSESSMENT_PATTERN = re.compile(
    r"\b(?:nešto|to|ono)\b.*\biznad\b.*\bcilj\w*\b",
    re.IGNORECASE,
)
PATIENT_PLAN_COMMITMENT_PATTERN = re.compile(
    r"\b(?:zapisat\w*|donijet\w*|dostavit\w*|postavit\w*|pratit\w*|vodit\w*)\b",
    re.IGNORECASE,
)
SYMPTOM_TRIGGER_RESPONSE_PATTERN = re.compile(
    r"\b(?:najgore|jač\w*|gor\w*|pogorš\w*|nakon|uz|tijekom)\b",
    re.IGNORECASE,
)

MEDICATIONS = {
    "acetilsalicilat": "acetilsalicilatna kiselina",
    "amlodipin": "amlodipin",
    "antibiotik": "antibiotik",
    "apiksaban": "apiksaban",
    "apixaban": "apiksaban",
    "bisoprolol": "bisoprolol",
    "budezonid": "budezonid",
    "formoterol": "formoterol",
    "ibuprofen": "ibuprofen",
    "ketoprofen": "ketoprofen",
    "metformin": "metformin",
    "paracetamol": "paracetamol",
    "salbutamol": "salbutamol",
}
MEDICATION_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, MEDICATIONS), key=len, reverse=True)) + r")\w*\b",
    re.IGNORECASE,
)
NUMBER_ATOM = (
    r"(?:\d+(?:[.,]\d+)?|nula|jedan|jedna|jednu|dva|dvije|tri|četiri|"
    r"pet|šest|sedam|osam|devet|deset|dvadeset|trideset|četrdeset|"
    r"pedeset|šezdeset|sedamdeset|osamdeset|devedeset|sto|dvjesto|"
    r"tristo|četiristo|petsto|tisuću)"
)
CONTEXTUAL_MEASUREMENT_PATTERN = re.compile(
    rf"\b(?:iznad|ispod|između|oko)\s+{NUMBER_ATOM}\b|"
    rf"\b{NUMBER_ATOM}\s+(?:sekund\w*|minut\w*|sat\w*|dan\w*|tjed\w*|mjesec\w*|godin\w*|puta)\b|"
    rf"\b{NUMBER_ATOM}\s+kroz\s+{NUMBER_ATOM}\b|"
    rf"\b(?:porast\w*|padn\w*)\s+na\s+{NUMBER_ATOM}\b",
    re.IGNORECASE,
)
MEASUREMENT_TIME_PATTERN = re.compile(
    r"\b(?:ujutro|navečer|popodne|poslijepodne|noću|"
    r"prije\s+(?:doručka|ručka|večere|spavanja)|nakon\s+buđenja)\b",
    re.IGNORECASE,
)
DOSE_PATTERN = re.compile(
    rf"\b(?:od\s+)?(?P<dose>{NUMBER_ATOM}(?:\s+(?:i|cijel\w*|pol|pola|{NUMBER_ATOM})){{0,5}}"
    r"\s+(?:miligram\w*|mg|gram\w*|mililitar\w*|ml))\b",
    re.IGNORECASE,
)
FREQUENCY_PATTERN = re.compile(
    r"\b(?:(?:gotovo|ponekad|samo)\s+)?(?:"
    r"dva\s+puta\s+dnevno|tri\s+puta\s+dnevno|jednom\s+dnevno|"
    r"jednom\s+tjedno|svaki\s+dan|svakog\s+dana|ujutro\s+i\s+navečer|"
    r"ujutro|navečer|dnevno|tjedno|po\s+potrebi)\b",
    re.IGNORECASE,
)


# Prevedi dopuštene varijante naziva uloge u zajednički naziv.
def normalize_role(role: str) -> str:
    return ROLE_ALIASES.get(str(role).strip().upper(), str(role).strip().upper())


# Razdvoji tekst na rečenice za daljnju analizu sadržaja.
def split_sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in re.finditer(r"[^.!?]+[.!?]?", text) if match.group(0).strip()]


# Razdvoji dijelove rečenice kako potvrda i negacija ne bi dijelile isti dokaz.
def split_clauses(sentence: str) -> list[str]:
    return [
        clause.strip(" ,")
        for clause in re.split(
            r"\s+(?:ali|no|međutim)\s+|\s+i\s+(?=moram\b)",
            sentence,
            flags=re.IGNORECASE,
        )
        if clause.strip(" ,")
    ]


# Prepoznaj pitanje prema završnom upitniku, bez dublje gramatičke analize.
def _is_question(text: str) -> bool:
    return text.rstrip().endswith("?")


def _medical_clinician_question_context_index(
    role: str,
    utterance_index: int,
    utterances: list[dict[str, Any]],
) -> int | None:
    """Vrati neposredno prethodno liječničko pitanje ako je medicinski relevantno."""
    if role != "PACIJENT" or utterance_index <= 0:
        return None
    previous = utterances[utterance_index - 1]
    previous_text = str(previous.get("text", "")).strip()
    if (
        normalize_role(str(previous.get("role", ""))) != "LIJEČNIK"
        or not _is_question(previous_text)
    ):
        return None
    if any(
        pattern.search(previous_text)
        for pattern in (
            SYMPTOM_PATTERN,
            HISTORY_PATTERN,
            MEASUREMENT_PATTERN,
            MEDICATION_PATTERN,
        )
    ):
        return utterance_index - 1
    return None


def _is_contextual_measurement_response(
    sentence: str,
    role: str,
    utterance_index: int,
    utterances: list[dict[str, Any]],
) -> bool:
    """Prepoznaj brojčani odgovor na prethodno liječničko pitanje o stanju."""
    if role != "PACIJENT" or utterance_index <= 0:
        return False
    previous = utterances[utterance_index - 1]
    previous_role = normalize_role(str(previous.get("role", "NEPOZNATO")))
    previous_text = str(previous.get("text", "")).strip()
    if previous_role != "LIJEČNIK" or not _is_question(previous_text):
        return False
    measurement_topic = bool(MEASUREMENT_PATTERN.search(previous_text))
    symptom_topic = bool(SYMPTOM_PATTERN.search(previous_text))
    return bool(
        ((measurement_topic or symptom_topic) and CONTEXTUAL_MEASUREMENT_PATTERN.search(sentence))
        or (measurement_topic and MEASUREMENT_TIME_PATTERN.search(sentence))
    )


# Provjeri može li se kratki odgovor vezati uz prethodno pitanje o terapiji.
def _is_contextual_medication_response(
    sentence: str,
    role: str,
    utterance_index: int,
    utterances: list[dict[str, Any]],
) -> bool:
    if role != "PACIJENT" or utterance_index <= 0:
        return False
    previous = utterances[utterance_index - 1]
    previous_text = str(previous.get("text", "")).strip()
    return bool(
        normalize_role(str(previous.get("role", ""))) == "LIJEČNIK"
        and _is_question(previous_text)
        and MEDICATION_PATTERN.search(previous_text)
    )


def _is_contextual_symptom_trigger_response(
    sentence: str,
    role: str,
    question_context_index: int | None,
    utterances: list[dict[str, Any]],
) -> bool:
    """Prepoznaj opis okidača koji ima smisla samo uz liječnikovo pitanje."""
    if role != "PACIJENT" or question_context_index is None:
        return False
    question = str(utterances[question_context_index].get("text", "")).strip()
    return bool(
        SYMPTOM_PATTERN.search(question)
        and SYMPTOM_TRIGGER_RESPONSE_PATTERN.search(sentence)
        and not SYMPTOM_PATTERN.search(sentence)
    )


# Prepoznaj opis učestalosti i vrati ga kada odgovara poznatim obrascima.
def _frequency_description(text: str) -> str | None:
    frequencies: list[str] = []
    seen: set[str] = set()
    for match in FREQUENCY_PATTERN.finditer(text):
        value = match.group(0).strip()
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            frequencies.append(value)
    return "; ".join(frequencies) if frequencies else None


# Protumači kratku negaciju samo uz kontekst pitanja o simptomu.
def _is_contextual_symptom_negation(
    sentence: str,
    role: str,
    utterance_index: int,
    utterances: list[dict[str, Any]],
) -> bool:
    if (
        role != "PACIJENT"
        or utterance_index <= 0
        or not NEGATION_PATTERN.search(sentence)
        or not ANAPHORIC_NEGATION_PATTERN.search(sentence)
    ):
        return False
    previous = utterances[utterance_index - 1]
    previous_text = str(previous.get("text", "")).strip()
    return bool(
        normalize_role(str(previous.get("role", ""))) == "LIJEČNIK"
        and _is_question(previous_text)
        and SYMPTOM_PATTERN.search(previous_text)
    )


# Potraži prethodni kontekst koji objašnjava na što se brojčani odgovor odnosi.
def _find_prior_measurement_context(
    sentence: str,
    utterance_index: int,
    utterances: list[dict[str, Any]],
) -> int | None:
    if not ANAPHORIC_ASSESSMENT_PATTERN.search(sentence):
        return None
    for index in range(utterance_index - 1, max(-1, utterance_index - 4), -1):
        candidate = utterances[index]
        candidate_text = str(candidate.get("text", "")).strip()
        if (
            normalize_role(str(candidate.get("role", ""))) == "LIJEČNIK"
            and MEASUREMENT_PATTERN.search(candidate_text)
            and NUMBER_PATTERN.search(candidate_text)
        ):
            return index
    return None


# Provjeri pripada li odgovor prethodno spomenutom planu ili preporuci.
def _is_contextual_plan_response(
    sentence: str,
    role: str,
    utterance_index: int,
    utterances: list[dict[str, Any]],
) -> bool:
    if role != "LIJEČNIK" or utterance_index <= 0 or not PLAN_PATTERN.search(sentence):
        return False
    previous = utterances[utterance_index - 1]
    previous_text = str(previous.get("text", "")).strip()
    return bool(
        normalize_role(str(previous.get("role", ""))) == "PACIJENT"
        and _is_question(previous_text)
        and (MEASUREMENT_PATTERN.search(previous_text) or MEDICATION_PATTERN.search(previous_text))
    )


# Prepoznaj pacijentovu namjeru da slijedi uputu, a ne novu liječničku preporuku.
def _is_patient_plan_commitment(
    sentence: str,
    role: str,
    utterance_index: int,
    utterances: list[dict[str, Any]],
) -> bool:
    if (
        role != "PACIJENT"
        or utterance_index <= 0
        or not PATIENT_PLAN_COMMITMENT_PATTERN.search(sentence)
    ):
        return False
    previous = utterances[utterance_index - 1]
    return bool(
        normalize_role(str(previous.get("role", ""))) == "LIJEČNIK"
        and PLAN_PATTERN.search(str(previous.get("text", "")))
    )


# Sastavi dokaz s izvornim iskazom, ulogom i vremenskim oznakama.
def _evidence(utterance: dict[str, Any], utterance_index: int, quote: str) -> dict[str, Any]:
    return {
        "utterance_index": utterance_index,
        "speaker": str(utterance.get("speaker", "NEPOZNATO")),
        "role": normalize_role(str(utterance.get("role", "NEPOZNATO"))),
        "start": float(utterance.get("start", 0.0)),
        "end": float(utterance.get("end", 0.0)),
        "quote": quote,
    }


# Razlikuj pitanje, negaciju, primjenu i preporuku lijeka prema tekstu i ulozi.
def _medication_status(text: str, role: str, question: bool) -> str:
    if question:
        return "questioned"
    if re.search(
        r"\b(?:ne\s+uzimam|nisam\s+uzima\w*|nisam\s+uzeo|nije\s+potreban|bez)\b",
        text,
        re.IGNORECASE,
    ):
        return "negated"
    if role == "PACIJENT" and re.search(r"\b(?:zaborav\w*|preskoč\w*)\b", text, re.IGNORECASE):
        return "adherence_issue"
    if role == "LIJEČNIK" and re.search(r"\bpropis\w*\b", text, re.IGNORECASE):
        return "prescribed"
    if role == "LIJEČNIK" and re.search(r"\b(?:preporuč\w*|uzimajte)\b", text, re.IGNORECASE):
        return "recommended"
    if role == "PACIJENT":
        return "patient_reported"
    return "mentioned"


# Izdvoji lijekove poznatim obrascima te zadrži njihovu povezanost s iskazom.
def extract_medications(
    text: str, utterance: dict[str, Any], utterance_index: int
) -> list[dict[str, Any]]:
    matches = list(MEDICATION_PATTERN.finditer(text))
    role = normalize_role(str(utterance.get("role", "NEPOZNATO")))
    result: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        local_context = text[match.end():end]
        dose_match = DOSE_PATTERN.search(local_context)
        frequency = _frequency_description(local_context)
        stem = match.group(1).lower()
        result.append(
            {
                "name": MEDICATIONS[stem],
                "mention": match.group(0),
                "dose": dose_match.group("dose") if dose_match else None,
                "frequency": frequency,
                "status": _medication_status(text, role, _is_question(text)),
                "evidence": _evidence(utterance, utterance_index, text),
            }
        )
    return result


# Izgradi jednu strukturiranu činjenicu s kategorijom, tvrdnjom i dokazom.
def _fact(
    fact_id: str,
    assertion: str,
    text: str,
    utterance: dict[str, Any],
    utterance_index: int,
    context_evidence: dict[str, Any] | None = None,
    context_relation: str | None = None,
) -> dict[str, Any]:
    fact = {
        "id": fact_id,
        "assertion": assertion,
        "text": text,
        "evidence": _evidence(utterance, utterance_index, text),
        "extraction": "contextual_direct_quotes" if context_evidence else "direct_quote",
    }
    if context_evidence:
        fact["context_evidence"] = context_evidence
        fact["context_relation"] = context_relation or "answer_to_clinician_question"
    return fact


# Ukloni ponovljene stavke prema ključu koji definira njihovu istovjetnost.
def _deduplicate(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result = []
    for item in items:
        evidence = item["evidence"]
        key = (
            item.get("name"), item.get("assertion"), item.get("text"),
            evidence["utterance_index"], evidence["quote"],
        )
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


# Pretvori kategorije i lijekove u čitljiv tekst bez promjene strukturiranih dokaza.
def _render_summary(payload: dict[str, Any]) -> str:
    lines = [
        "AUTOMATSKI EKSTRAKTIVNI MEDICINSKI SAŽETAK — OBAVEZNA PROVJERA",
        "Sadržaj je izdvojen iz razgovora i nije dijagnoza ni zamjena za medicinsku dokumentaciju.",
    ]
    for section in SECTION_ORDER:
        facts = payload["sections"][section]
        if not facts or section == "clinician_questions":
            continue
        lines.extend(["", SECTION_LABELS[section] + ":"])
        for fact in facts:
            evidence = fact["evidence"]
            context = fact.get("context_evidence")
            if context:
                lines.append(
                    f"- [{context['role']}, {context['start']:.1f}–{context['end']:.1f} s → "
                    f"{evidence['role']}, {evidence['start']:.1f}–{evidence['end']:.1f} s] "
                    f"{context['quote']} → {evidence['quote']}"
                )
            else:
                lines.append(
                    f"- [{evidence['role']}, {evidence['start']:.1f}–{evidence['end']:.1f} s] "
                    f"{evidence['quote']}"
                )
    if payload["medications"]:
        lines.extend(["", "Spomenuti lijekovi i terapija:"])
        for medication in payload["medications"]:
            details = [medication["status"]]
            if medication["dose"]:
                details.append(f"doza: {medication['dose']}")
            if medication["frequency"]:
                details.append(f"učestalost: {medication['frequency']}")
            context = medication.get("context_evidence")
            evidence = medication["evidence"]
            if context:
                lines.append(
                    f"- {medication['name']} ({'; '.join(details)}) — "
                    f"[{context['role']}, {context['start']:.1f}–{context['end']:.1f} s → "
                    f"{evidence['role']}, {evidence['start']:.1f}–{evidence['end']:.1f} s] "
                    f"{context['quote']} → {evidence['quote']}"
                )
            else:
                lines.append(f"- {medication['name']} ({'; '.join(details)})")
    lines.extend([
        "",
        "Zaključci sustava:",
        "- Nisu generirani. Procjene iznad samo su doslovno izgovorene procjene liječnika.",
    ])
    return "\n".join(lines) + "\n"


# Provjeri strukturu sažetka i valjanost dokaza prema ulaznim iskazima.
def validate_medical_summary(payload: dict[str, Any], utterances: list[dict[str, Any]]) -> None:
    if payload.get("system_inferences"):
        raise ValueError("Ekstraktivni način ne smije sadržavati zaključke sustava.")
    evidence_items = []

    # Provjeri dodatni kontekst činjenice prema izvornom indeksu i citatu.
    def validate_context(
        context: dict[str, Any],
        evidence: dict[str, Any],
        relation: str,
    ) -> None:
        context_role = normalize_role(str(context.get("role", "")))
        context_quote = str(context.get("quote", ""))
        if int(context["utterance_index"]) >= int(evidence["utterance_index"]):
            raise ValueError("Kontekst mora prethoditi povezanom iskazu.")
        if relation == "answer_to_clinician_question":
            valid = context_role == "LIJEČNIK" and _is_question(context_quote)
        elif relation == "answer_to_patient_question":
            valid = context_role == "PACIJENT" and _is_question(context_quote)
        elif relation == "refers_to_prior_measurement":
            valid = bool(
                context_role == "LIJEČNIK"
                and MEASUREMENT_PATTERN.search(context_quote)
                and NUMBER_PATTERN.search(context_quote)
            )
        elif relation == "acknowledges_clinician_plan":
            valid = bool(context_role == "LIJEČNIK" and PLAN_PATTERN.search(context_quote))
        else:
            raise ValueError(f"Nepoznata veza konteksta: {relation}")
        if not valid:
            raise ValueError("Uloga ili vrsta konteksta ne odgovara povezanoj činjenici.")

    for facts in payload["sections"].values():
        for fact in facts:
            evidence_items.append(fact["evidence"])
            context = fact.get("context_evidence")
            if context:
                validate_context(
                    context,
                    fact["evidence"],
                    str(fact.get("context_relation", "answer_to_clinician_question")),
                )
                evidence_items.append(context)
    for medication in payload["medications"]:
        evidence_items.append(medication["evidence"])
        context = medication.get("context_evidence")
        if context:
            validate_context(
                context,
                medication["evidence"],
                str(medication.get("context_relation", "answer_to_clinician_question")),
            )
            evidence_items.append(context)
    for evidence in evidence_items:
        index = int(evidence["utterance_index"])
        if index < 0 or index >= len(utterances):
            raise ValueError("Dokaz upućuje na nepostojeći iskaz.")
        source_text = " ".join(str(utterances[index].get("text", "")).split()).casefold()
        quote = " ".join(str(evidence["quote"]).split()).casefold()
        if not quote or quote not in source_text:
            raise ValueError("Izdvojena činjenica nema doslovni dokaz u transkriptu.")
        if evidence["role"] != normalize_role(str(utterances[index].get("role", "NEPOZNATO"))):
            raise ValueError("Uloga u dokazu ne odgovara izvornom iskazu.")


def assemble_medical_summary(
    utterances: Iterable[dict[str, Any]],
    sections: dict[str, list[dict[str, Any]]],
    medications: list[dict[str, Any]],
    *,
    source: dict[str, Any] | None = None,
    generator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sastavi i provjeri zajednički izlaz iz već izdvojenih činjenica.

    Funkciju koriste i pravila i lokalni LLM. ``sections`` i ``medications``
    već moraju sadržavati dokaze kako bi ista završna provjera vrijedila za oba
    načina izdvajanja.
    """
    utterance_list = [deepcopy(item) for item in utterances]
    complete_sections = {
        name: _deduplicate(sections.get(name, [])) for name in SECTION_ORDER
    }
    fact_counter = 0
    for section_name in SECTION_ORDER:
        for fact in complete_sections[section_name]:
            fact_counter += 1
            fact["id"] = f"F{fact_counter:03d}"
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator": deepcopy(generator or {
            "name": GENERATOR_VERSION,
            "type": "local_rule_based_extractive",
            "inference_enabled": False,
        }),
        "source": deepcopy(source or {}),
        "sections": complete_sections,
        "medications": _deduplicate(medications),
        "system_inferences": [],
        "review": {
            "required": True,
            "reason": DEFAULT_REVIEW_REASON,
        },
        "statistics": {
            "utterances": len(utterance_list),
            "extracted_facts": sum(len(values) for values in complete_sections.values()),
            "medication_mentions": len(_deduplicate(medications)),
        },
    }
    payload["summary_text"] = _render_summary(payload)
    validate_medical_summary(payload, utterance_list)
    return payload


def build_medical_summary_from_extractions(
    utterances: Iterable[dict[str, Any]],
    extracted_sections: dict[str, list[dict[str, Any]]],
    extracted_medications: list[dict[str, Any]],
    *,
    source: dict[str, Any] | None = None,
    generator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pretvori strogo provjerene ekstrakcije u standardni format sažetka."""
    utterance_list = [deepcopy(item) for item in utterances]
    sections: dict[str, list[dict[str, Any]]] = {name: [] for name in SECTION_ORDER}
    counter = 0
    for section_name in SECTION_ORDER:
        for extraction in extracted_sections.get(section_name, []):
            utterance_index = int(extraction["utterance_index"])
            if utterance_index < 0 or utterance_index >= len(utterance_list):
                raise ValueError("Ekstrakcija upućuje na nepostojeći iskaz.")
            quote = str(extraction["quote"]).strip()
            counter += 1
            sections[section_name].append(
                _fact(
                    f"F{counter:03d}",
                    str(extraction["assertion"]),
                    quote,
                    utterance_list[utterance_index],
                    utterance_index,
                )
            )

    medications = []
    for extraction in extracted_medications:
        utterance_index = int(extraction["utterance_index"])
        if utterance_index < 0 or utterance_index >= len(utterance_list):
            raise ValueError("Lijek upućuje na nepostojeći iskaz.")
        quote = str(extraction["quote"]).strip()
        medications.append(
            {
                "name": str(extraction["name"]).strip(),
                "mention": str(extraction["mention"]).strip(),
                "dose": extraction.get("dose"),
                "frequency": extraction.get("frequency"),
                "status": str(extraction["status"]),
                "evidence": _evidence(
                    utterance_list[utterance_index], utterance_index, quote
                ),
            }
        )

    return assemble_medical_summary(
        utterance_list,
        sections,
        medications,
        source=source,
        generator=generator,
    )


# Prođi iskaze i jezična pravila te sastavi deterministički medicinski sažetak.
def create_medical_summary(
    utterances: Iterable[dict[str, Any]], source: dict[str, Any] | None = None
) -> dict[str, Any]:
    utterance_list = [deepcopy(item) for item in utterances]
    sections: dict[str, list[dict[str, Any]]] = {name: [] for name in SECTION_ORDER}
    medications: list[dict[str, Any]] = []
    counter = 0

    # Dodaj činjenicu u radni skup uz pripadajuću tvrdnju i opcionalni kontekst.
    def add(
        section: str,
        assertion: str,
        quote: str,
        utterance: dict[str, Any],
        index: int,
        context_index: int | None = None,
        context_relation: str | None = None,
    ) -> None:
        nonlocal counter
        counter += 1
        context_evidence = None
        if context_index is not None:
            context_utterance = utterance_list[context_index]
            context_evidence = _evidence(
                context_utterance,
                context_index,
                str(context_utterance.get("text", "")).strip(),
            )
        sections[section].append(
            _fact(
                f"F{counter:03d}",
                assertion,
                quote,
                utterance,
                index,
                context_evidence=context_evidence,
                context_relation=context_relation,
            )
        )

    for utterance_index, utterance in enumerate(utterance_list):
        role = normalize_role(str(utterance.get("role", "NEPOZNATO")))
        text = str(utterance.get("text", "")).strip()
        if not text:
            continue
        utterance_fact_start = counter
        question_context_index = _medical_clinician_question_context_index(
            role,
            utterance_index,
            utterance_list,
        )
        for sentence in split_sentences(text):
            question = _is_question(sentence)
            sentence_medications = extract_medications(sentence, utterance, utterance_index)
            medications.extend(sentence_medications)
            if question:
                section = "patient_questions" if role == "PACIJENT" else "clinician_questions"
                add(section, "questioned", sentence, utterance, utterance_index)
                continue

            contextual_measurement = _is_contextual_measurement_response(
                sentence,
                role,
                utterance_index,
                utterance_list,
            )
            contextual_medication = _is_contextual_medication_response(
                sentence,
                role,
                utterance_index,
                utterance_list,
            )
            contextual_plan_response = _is_contextual_plan_response(
                sentence,
                role,
                utterance_index,
                utterance_list,
            )
            if contextual_medication:
                previous_index = utterance_index - 1
                for medication in medications:
                    if (
                        medication["status"] == "questioned"
                        and int(medication["evidence"]["utterance_index"]) == previous_index
                    ):
                        medication["context_evidence"] = medication["evidence"]
                        medication["evidence"] = _evidence(
                            utterance,
                            utterance_index,
                            text,
                        )
                        medication["status"] = _medication_status(text, role, False)
                        medication["frequency"] = _frequency_description(text)
                        medication["extraction"] = "contextual_direct_quotes"
                        medication["context_relation"] = "answer_to_clinician_question"
            explicit_measurement = bool(
                not sentence_medications
                and MEASUREMENT_PATTERN.search(sentence)
                and NUMBER_PATTERN.search(sentence)
            )
            is_measurement = bool(
                not sentence_medications
                and (explicit_measurement or contextual_measurement)
            )
            if is_measurement:
                measurement_context_index = None
                measurement_context_relation = None
                if contextual_measurement:
                    measurement_context_index = utterance_index - 1
                    measurement_context_relation = "answer_to_clinician_question"
                elif question_context_index is not None:
                    measurement_context_index = question_context_index
                    measurement_context_relation = "answer_to_clinician_question"
                elif contextual_plan_response:
                    measurement_context_index = utterance_index - 1
                    measurement_context_relation = "answer_to_patient_question"
                add(
                    "measurements",
                    "asserted",
                    sentence,
                    utterance,
                    utterance_index,
                    context_index=measurement_context_index,
                    context_relation=measurement_context_relation,
                )

            if role == "PACIJENT":
                contextual_symptom_negation = _is_contextual_symptom_negation(
                    sentence,
                    role,
                    utterance_index,
                    utterance_list,
                )
                if contextual_symptom_negation:
                    add(
                        "patient_negated_symptoms",
                        "negated",
                        sentence,
                        utterance,
                        utterance_index,
                        context_index=utterance_index - 1,
                        context_relation="answer_to_clinician_question",
                    )
                if HISTORY_PATTERN.search(sentence):
                    history_assertion = "negated" if NEGATION_PATTERN.search(sentence) else "asserted"
                    add(
                        "patient_history",
                        history_assertion,
                        sentence,
                        utterance,
                        utterance_index,
                        context_index=question_context_index,
                        context_relation=(
                            "answer_to_clinician_question"
                            if question_context_index is not None
                            else None
                        ),
                    )
                if _is_patient_plan_commitment(
                    sentence,
                    role,
                    utterance_index,
                    utterance_list,
                ):
                    add(
                        "plan_and_recommendations",
                        "patient_acknowledged_plan",
                        sentence,
                        utterance,
                        utterance_index,
                        context_index=utterance_index - 1,
                        context_relation="acknowledges_clinician_plan",
                    )
                if SAFETY_PATTERN.search(sentence):
                    add(
                        "safety_instructions",
                        "patient_repeated_safety_instruction",
                        sentence,
                        utterance,
                        utterance_index,
                    )
                for clause in split_clauses(sentence):
                    if not SYMPTOM_PATTERN.search(clause):
                        continue
                    if CONDITIONAL_PATTERN.search(clause):
                        continue
                    # Obje vrste simptoma imaju isti dokaz; razlikuju se kategorija i tvrdnja.
                    negated = bool(NEGATION_PATTERN.search(clause))
                    add(
                        "patient_negated_symptoms" if negated else "patient_reported_symptoms",
                        "negated" if negated else "asserted",
                        clause,
                        utterance,
                        utterance_index,
                        context_index=question_context_index,
                        context_relation=(
                            "answer_to_clinician_question"
                            if question_context_index is not None
                            else None
                        ),
                    )
                if _is_contextual_symptom_trigger_response(
                    sentence,
                    role,
                    question_context_index,
                    utterance_list,
                ):
                    add(
                        "patient_reported_symptoms",
                        "asserted",
                        sentence,
                        utterance,
                        utterance_index,
                        context_index=question_context_index,
                        context_relation="answer_to_clinician_question",
                    )
            elif role == "LIJEČNIK":
                planned_action = bool(PLAN_PATTERN.search(sentence))
                if FINDING_PATTERN.search(sentence) and not planned_action:
                    add("clinician_findings", "clinician_stated", sentence, utterance, utterance_index)
                if ASSESSMENT_PATTERN.search(sentence):
                    assertion = "uncertain_clinician_assessment" if re.search(
                        r"\b(?:vjerojatno|najvjerojatnije)\b", sentence, re.IGNORECASE
                    ) else "clinician_assessment"
                    assessment_context_index = _find_prior_measurement_context(
                        sentence,
                        utterance_index,
                        utterance_list,
                    )
                    add(
                        "clinician_assessments",
                        assertion,
                        sentence,
                        utterance,
                        utterance_index,
                        context_index=assessment_context_index,
                        context_relation=(
                            "refers_to_prior_measurement"
                            if assessment_context_index is not None
                            else None
                        ),
                    )
                if PLAN_PATTERN.search(sentence):
                    add(
                        "plan_and_recommendations",
                        "planned_or_recommended",
                        sentence,
                        utterance,
                        utterance_index,
                        context_index=utterance_index - 1 if contextual_plan_response else None,
                        context_relation=(
                            "answer_to_patient_question" if contextual_plan_response else None
                        ),
                    )
                if SAFETY_PATTERN.search(sentence):
                    add("safety_instructions", "safety_instruction", sentence, utterance, utterance_index)

        # Ako odgovor na medicinsko liječničko pitanje nije uhvatilo nijedno
        # specifičnije pravilo, sačuvaj cijeli doslovni par pitanje–odgovor.
        if (
            role == "PACIJENT"
            and question_context_index is not None
            and counter == utterance_fact_start
        ):
            add(
                "patient_history",
                "answer_to_medical_question",
                text,
                utterance,
                utterance_index,
                context_index=question_context_index,
                context_relation="answer_to_clinician_question",
            )

    return assemble_medical_summary(
        utterance_list,
        sections,
        medications,
        source=source,
    )


# Izradi sažetak te spremi JSON i njegov čitljivi tekstni prikaz.
def save_medical_summary(
    utterances: Iterable[dict[str, Any]],
    json_path: Path,
    text_path: Path | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    utterance_list = list(utterances)
    payload = create_medical_summary(utterance_list, source=source)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_text_path = text_path or json_path.with_suffix(".txt")
    output_text_path.write_text(payload["summary_text"], encoding="utf-8")
    return payload
