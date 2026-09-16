"""Izrada liječničkog zapisa u trećem licu iz provjerenog medicinskog sažetka.

Ulaz ostaje ekstraktivan medicinski sažetak s dokaznim citatima. Ovaj modul
ne postavlja dijagnoze i ne izdvaja nove činjenice, nego već izdvojene činjenice
raspoređuje u oblik prikladniji liječničkom zapisu. Svaka preoblikovana stavka
zadržava vezu na izvornu kategoriju, identifikator činjenice i dokazni iskaz.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
GENERATOR_VERSION = "lokalni_lijecnicki_zapis_v1_1"
REVIEW_REASON = (
    "Automatska transkripcija, izdvajanje i jezično preoblikovanje mogu sadržavati "
    "pogreške; prije uporabe potreban je ljudski pregled."
)

PATIENT_SECTIONS = (
    "patient_reported_symptoms",
    "patient_negated_symptoms",
    "patient_history",
)
CLINICIAN_SECTIONS = (
    "clinician_findings",
    "clinician_assessments",
)
QUESTION_SECTIONS = ("patient_questions", "clinician_questions")

MEDICATION_STATUS_LABELS = {
    "prescribed": "propisano",
    "recommended": "preporučeno",
    "patient_reported": "pacijent navodi primjenu",
    "adherence_issue": "problem pridržavanja terapije",
    "negated": "negirano",
    "questioned": "spomenuto u pitanju",
    "mentioned": "spomenuto",
}


# Ukloni početne pozdrave i izdvojeno da/ne s zarezom prije preoblikovanja iskaza.
def _clean_text(text: str) -> str:
    value = " ".join(str(text).strip().split())
    value = re.sub(r"^(?:dobar dan|dobro)[,!]?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(?:da|ne)[,]\s*", "", value, flags=re.IGNORECASE)
    return value.strip()


# Smanji početno slovo, ali sačuvaj kratice i prve riječi koje sadržavaju brojeve.
def _lower_initial(text: str) -> str:
    if not text:
        return text
    first_token = text.split(maxsplit=1)[0]
    if any(character.isdigit() for character in first_token):
        return text
    if sum(character.isupper() for character in first_token) >= 2:
        return text
    return text[0].lower() + text[1:]


# Postavi veliko početno slovo samo ako tekst nije prazan.
def _upper_initial(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


# Završi narativnu rečenicu točkom; završni upitnik pretvara se u točku.
def _with_period(text: str) -> str:
    value = text.strip()
    if not value:
        return value
    if value.endswith("?"):
        value = value[:-1].rstrip()
    return value if value.endswith((".", "!")) else value + "."


# Primijeni uređena regex pravila i vrati tekst zajedno s brojem zamjena.
def _replace_words(text: str, replacements: tuple[tuple[str, str], ...]) -> tuple[str, int]:
    result = text
    count = 0
    for pattern, replacement in replacements:
        result, changed = re.subn(pattern, replacement, result, flags=re.IGNORECASE)
        count += changed
    return result, count


def _patient_clause(text: str) -> tuple[str, int]:
    """Konzervativno pretvori česte oblike prvog lica u treće lice.

    Ako nijedno sigurno pravilo nije primjenjivo, renderer koristi konstrukciju
    ``Prema izjavi pacijenta`` i ne pokušava slobodno parafrazirati sadržaj.
    """

    replacements = (
        (r"\bne mogu\b", "ne može"),
        (r"\bnisam\b", "nije"),
        (r"\bnemam\b", "nema"),
        (r"\bimam\b", "ima"),
        (r"\bosjećam\b", "osjeća"),
        (r"\buzimam\b", "uzima"),
        (r"\bkoristim\b", "koristi"),
        (r"\bpijem\b", "pije"),
        (r"\bjedem\b", "jede"),
        (r"\bhodam\b", "hoda"),
        (r"\bkašljem\b", "kašlje"),
        (r"\bčujem\b", "čuje"),
        (r"\bdišem\b", "diše"),
        (r"\bidem\b", "ide"),
        (r"\bmogu\b", "može"),
        (r"\bmoram\b", "mora"),
        (r"\bmjerim\b", "mjeri"),
        (r"\bprimjećujem\b", "primjećuje"),
        (r"\btrebam\b", "treba"),
        (r"\bpreskočim\b", "preskoči"),
        (r"\bzaboravim\b", "zaboravi"),
        (r"\bdođem\b", "dođe"),
        (r"\bću se javiti\b", "će se javiti"),
    )
    result, count = _replace_words(text, replacements)
    result, reordered = re.subn(
        r"\b(uzima|koristi|pije)\s+ga\b",
        r"ga \1",
        result,
        flags=re.IGNORECASE,
    )
    count += reordered
    result, reordered = re.subn(
        r"^Odmah\s+će\s+se\s+javiti\b",
        "će se odmah javiti",
        result,
        flags=re.IGNORECASE,
    )
    count += reordered
    return result, count


# Oblikuj pacijentov iskaz u narativ trećeg lica uz ograničena jezična pravila.
def _patient_narrative(text: str) -> str:
    cleaned = _clean_text(text)
    clause, changes = _patient_clause(cleaned)
    unsafe_indirect = bool(re.search(r"\b(?:sam|mi|me)\b", cleaned, re.IGNORECASE))
    safe_first_person = bool(
        re.search(
            r"\b(?:ne mogu|mogu|nisam|nemam|imam|osjećam|uzimam|koristim|"
            r"pijem|jedem|hodam|kašljem|čujem|dišem|idem|moram|mjerim|"
            r"primjećujem|trebam|preskočim|zaboravim|dođem|ću se javiti)\b",
            cleaned,
            re.IGNORECASE,
        )
    )
    if changes and safe_first_person and not unsafe_indirect:
        return _with_period(f"Pacijent je izjavio da {_lower_initial(clause)}")
    any_first_person = bool(
        re.search(
            r"\b(?:sam|mi|me|ću|mogu|nisam|nemam|imam|osjećam|uzimam|"
            r"koristim|pijem|jedem|hodam|kašljem|čujem|dišem|idem|moram|"
            r"mjerim|primjećujem|trebam|preskočim|zaboravim|dođem)\b",
            cleaned,
            re.IGNORECASE,
        )
    )
    if not any_first_person:
        return _with_period(f"Prema izjavi pacijenta, {_lower_initial(cleaned)}")
    return _with_period(f"Pacijent je izjavio: „{cleaned.rstrip('.')}”")


def _measurement_narrative(fact: dict[str, Any]) -> str:
    """Pretvori mjerenje i kontekst pitanja u samostalan klinički iskaz."""

    text = _clean_text(str(fact.get("text", "")))
    context = _clean_text(str((fact.get("context_evidence") or {}).get("quote", "")))

    duration = re.match(
        r"^Koliko\s+dugo\s+traje\s+(.+?)[?]?$", context, flags=re.IGNORECASE
    )
    if duration:
        subject = duration.group(1).strip().rstrip(".?")
        return _with_period(
            f"{_upper_initial(subject)} traje {_lower_initial(text)}"
        )

    context_labels = (
        (r"krvni\s+tlak", "Krvni tlak"),
        (r"u\s+koje\s+doba\s+dana.*mjerili", "Vrijeme mjerenja krvnog tlaka"),
        (r"koliko\s+je\s+bol\s+jaka", "Intenzitet boli"),
        (r"vrijednosti\s+šećera", "Vrijednosti šećera"),
        (
            r"vrijednosti\s+dva\s+sata\s+nakon\s+obroka",
            "Vrijednosti šećera dva sata nakon obroka",
        ),
        (
            r"budite\s+li\s+se\s+noću",
            "Noćni kašalj ili nedostatak zraka",
        ),
        (r"promijenilo.*disanjem", "Promjene disanja"),
    )
    for pattern, label in context_labels:
        if re.search(pattern, context, flags=re.IGNORECASE):
            return _with_period(f"{label}: {_lower_initial(text)}")

    role = str(fact.get("evidence", {}).get("role", "")).upper()
    if role == "LIJEČNIK":
        return _with_period(text)
    return _patient_narrative(text)


# Prilagodi početak zavisne rečenice za uključivanje u dulji narativ.
def _subordinate_clause(text: str) -> str:
    if re.match(r"^Vjerojatno\s+se\s+radi\s+o\b", text, re.IGNORECASE):
        return re.sub(
            r"^Vjerojatno\s+se\s+radi\s+o\b",
            "se vjerojatno radi o",
            text,
            flags=re.IGNORECASE,
        )
    most_likely = re.match(r"^Najvjerojatnije\s+(je|su)\s+(.+)$", text, re.IGNORECASE)
    if most_likely:
        return f"{most_likely.group(1).lower()} najvjerojatnije {most_likely.group(2)}"
    copula = re.match(r"^([^,]{1,55}?)\s+(je|su)\s+(.+)$", text, re.IGNORECASE)
    if copula and len(copula.group(1).split()) <= 6:
        return (
            f"{copula.group(2).lower()} {_lower_initial(copula.group(1))} "
            f"{copula.group(3)}"
        )
    return _lower_initial(text)


# Odaberi uvodnu formulaciju prema vrsti liječnikove činjenice.
def _clinician_narrative(text: str, section: str) -> str:
    cleaned = _clean_text(text)
    clause, _ = _replace_words(
        cleaned,
        (
            (r"\bVam\b", "pacijentu"),
            (r"\bVas\b", "pacijenta"),
            (r"\bVaša\b", "pacijentova"),
            (r"\bVaše\b", "pacijentovo"),
            (r"\bVaš\b", "pacijentov"),
        ),
    )
    prefix = "Liječnik je procijenio da" if section == "clinician_assessments" else "Liječnik je naveo da"
    return _with_period(f"{prefix} {_subordinate_clause(clause)}")


# Pretvori poznate oblike formalnog obraćanja u izraze prikladne trećem licu.
def _formal_to_third_person(text: str) -> str:
    replacements = (
        (r"\bne možete\b", "ne može"),
        (r"\bmožete\b", "može"),
        (r"\bdobijete\b", "dobije"),
        (r"\bpovraćate\b", "povraća"),
        (r"\buzimate\b", "uzima"),
        (r"\bmjerite\b", "mjeri"),
        (r"\bzapišite\b", "zapiše"),
        (r"\bdonesite\b", "donese"),
        (r"\bizbjegavajte\b", "izbjegava"),
        (r"\bnastavite\b", "nastavi"),
        (r"\bpijte\b", "pije"),
        (r"\buzimajte\b", "uzima"),
        (r"\bse javite\b", "se javi"),
        (r"\bjavite se\b", "se javi"),
        (r"\bVam\b", "pacijentu"),
        (r"\bVas\b", "pacijenta"),
    )
    return _replace_words(text, replacements)[0]


# Preoblikuj uputu ili plan; sigurnosne upute dobivaju zasebnu formulaciju.
def _action_narrative(text: str, *, safety: bool = False) -> str:
    cleaned = _clean_text(text)
    transformed = _formal_to_third_person(cleaned)

    start_rules = (
        (r"^Preporučujem\s+", "Liječnik je preporučio "),
        (r"^Savjetujem\s+", "Liječnik je savjetovao "),
        (r"^Prvo ćemo provjeriti\s+", "Liječnik je najprije planirao provjeriti "),
        (r"^Zato ćemo učiniti\s+", "Liječnik je zbog toga planirao učiniti "),
        (r"^Napravit ćemo\s+", "Liječnik je planirao napraviti "),
        (r"^Učinit ćemo\s+", "Liječnik je planirao učiniti "),
        (r"^Provjerit ćemo\s+", "Liječnik je planirao provjeriti "),
        (r"^Poslušat ću\s+", "Liječnik je planirao poslušati "),
        (r"^Pregledat ću\s+", "Liječnik je planirao pregledati "),
        (r"^Uputit ću\s+", "Liječnik je planirao uputiti "),
        (r"^Propisat ću\s+", "Liječnik je propisao "),
    )
    for pattern, replacement in start_rules:
        candidate, count = re.subn(pattern, replacement, transformed, flags=re.IGNORECASE)
        if count:
            return _with_period(candidate)

    drink = re.match(r"^Pijte\s+(.+?)[.]?$", cleaned, re.IGNORECASE)
    if drink:
        return _with_period(
            "Liječnik je preporučio pacijentu da pije "
            + _lower_initial(drink.group(1))
        )

    movement = re.match(
        r"^(?:Ne,\s*)?lagano se krećete,\s*izbjegavajte dizanje tereta i "
        r"nemojte dugo ostajati u istom položaju[.]?$",
        cleaned,
        re.IGNORECASE,
    )
    if movement:
        return _with_period(
            "Liječnik je preporučio lagano kretanje, izbjegavanje dizanja "
            "tereta i dugotrajnog ostajanja u istom položaju"
        )

    immediate = re.match(r"^Odmah\s+se javi\s+ako\s+(.+?)[.]?$", transformed, re.IGNORECASE)
    if immediate:
        return _with_period(
            "Liječnik je uputio pacijenta da se odmah javi ako "
            + _lower_initial(immediate.group(1))
        )

    conditional = re.match(
        r"^Ako\s+(.+?),\s*se javi\s+(odmah|ponovo)[.]?$",
        transformed,
        re.IGNORECASE,
    )
    if conditional:
        return _with_period(
            f"Liječnik je uputio pacijenta da se {conditional.group(2).lower()} javi ako "
            f"{_lower_initial(conditional.group(1))}"
        )

    urgent_condition = re.match(r"^Odmah,\s*ako\s+(.+?)[.]?$", transformed, re.IGNORECASE)
    if urgent_condition:
        return _with_period(
            "Liječnik je dao uputu za hitno javljanje ako "
            + _lower_initial(urgent_condition.group(1))
        )

    if safety:
        return _with_period(f"Liječnik je dao sigurnosnu uputu: „{cleaned.rstrip('.')}”")
    return _with_period(f"Liječnik je naveo plan ili preporuku: „{cleaned.rstrip('.')}”")


# Sastavi zapis o lijeku iz već izdvojenih podataka i statusa njegove uporabe.
def _medication_narrative(medication: dict[str, Any]) -> str:
    name = str(medication.get("name", "")).strip()
    status = str(medication.get("status", "mentioned")).strip()
    dose = medication.get("dose")
    frequency = medication.get("frequency")
    details = ""
    if dose:
        details += f" u dozi {dose}"
    if frequency:
        details += f", {frequency}"

    evidence = medication.get("evidence", {})
    role = str(evidence.get("role", "")).upper()
    if status == "prescribed":
        text = f"Liječnik je propisao terapiju: {name}{details}"
    elif status == "recommended":
        text = f"Liječnik je preporučio terapiju: {name}{details}"
    elif status == "patient_reported":
        text = f"Pacijent je izjavio da uzima terapiju: {name}{details}"
    elif status == "adherence_issue":
        text = f"Zabilježen je problem s pridržavanjem sljedeće terapije: {name}{details}"
    elif status == "negated" and role == "PACIJENT":
        text = f"Pacijent je izjavio da ne uzima terapiju: {name}{details}"
    elif status == "negated":
        text = f"Liječnik je naveo da terapija nije indicirana ili primijenjena: {name}{details}"
    elif status == "questioned" and role == "PACIJENT":
        text = f"Pacijent je pitao o terapiji: {name}{details}"
    elif status == "questioned":
        text = f"Liječnik je pitao uzima li pacijent terapiju: {name}{details}"
    else:
        text = f"U razgovoru je spomenuta terapija: {name}{details}"
    return _with_period(text)


# Sačuvaj vezu narativne stavke s kategorijom, identifikatorom i dokazom činjenice.
def _fact_reference(section: str, fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "section": section,
        "fact_id": str(fact.get("id", "")),
        "assertion": str(fact.get("assertion", "")),
        "source_text": str(fact.get("text", "")),
        "evidence": deepcopy(fact.get("evidence", {})),
        "context_evidence": deepcopy(fact.get("context_evidence")),
    }


def _fact_key(fact: dict[str, Any]) -> tuple[int, str]:
    """Usporedi činjenice po istom iskazu i ujednačenom tekstu, ne samo sadržaju."""
    return (
        int(fact.get("evidence", {}).get("utterance_index", -1)),
        " ".join(str(fact.get("text", "")).casefold().split()),
    )


# Dodaj narativnu stavku uz dokaz iz sažetka kako se izvor ne bi izgubio.
def _append_fact_entry(
    target: list[dict[str, Any]],
    section: str,
    fact: dict[str, Any],
    narrative: str,
) -> None:
    evidence = fact.get("evidence", {})
    key = _fact_key(fact)
    for item in target:
        if tuple(item["deduplication_key"]) == key:
            item["source_references"].append(_fact_reference(section, fact))
            return
    target.append(
        {
            "narrative_text": narrative,
            "start": float(evidence.get("start", 0.0)),
            "end": float(evidence.get("end", 0.0)),
            "role": str(evidence.get("role", "NEPOZNATO")),
            "source_references": [_fact_reference(section, fact)],
            "deduplication_key": list(key),
        }
    )


_EXAMINATION_VERB_PATTERN = re.compile(
    r"\b(?:posluš|pogled|pregled|provjer|naprav|učin)", re.IGNORECASE
)
_FUTURE_FINDING_PATTERN = re.compile(
    r"\b(?:planira|planirano|provjerit|pregledat|poslušat|pogledat|"
    r"napravit|učinit|bit će|će biti)\b",
    re.IGNORECASE,
)
_EXAMINATION_STOP_WORDS = {
    "ćemo",
    "dobro",
    "najprije",
    "prvo",
    "zatim",
    "poslije",
    "pregled",
    "nalaz",
    "osnovni",
}


# Ujednači zapis pojma za usporedbu spominjanja pregleda.
def _clinical_term_key(token: str) -> str:
    value = token.casefold()
    if len(value) >= 7:
        return value[:6]
    return value


# Izdvoji prepoznate termine pregleda za povezivanje nalaza i planiranih radnji.
def _examination_terms(text: str) -> set[str]:
    relevant = re.split(
        r"\b(?:te\s+razgovarati|prije\s+početka)\b",
        _clean_text(text),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    terms: set[str] = set()
    for token in re.findall(r"[a-zA-ZčćđšžČĆĐŠŽ]+", relevant):
        lowered = token.casefold()
        if len(lowered) < 4 or lowered in _EXAMINATION_STOP_WORDS:
            continue
        if _EXAMINATION_VERB_PATTERN.match(lowered):
            continue
        terms.add(_clinical_term_key(lowered))
    return terms


def _completed_examination_entry(
    plan_fact: dict[str, Any],
    possible_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Nađi kasniji nalaz koji potvrđuje da je najavljeni pregled obavljen."""

    plan_text = str(plan_fact.get("text", ""))
    if not _EXAMINATION_VERB_PATTERN.search(plan_text):
        return None
    plan_terms = _examination_terms(plan_text)
    if not plan_terms:
        return None
    plan_start = float(plan_fact.get("evidence", {}).get("start", 0.0))

    for entry in possible_results:
        if float(entry.get("start", 0.0)) < plan_start:
            continue
        result_text = " ".join(
            str(reference.get("source_text", ""))
            for reference in entry.get("source_references", [])
            if reference.get("section")
            in {"measurements", "clinician_findings", "clinician_assessments"}
        )
        if not result_text or _FUTURE_FINDING_PATTERN.search(result_text):
            continue
        result_terms = _examination_terms(result_text)
        coverage = len(plan_terms & result_terms) / len(plan_terms)
        if coverage >= 0.8:
            return entry
    return None


# Grupiraj zapise lijekova da ponovljeni navodi ne stvaraju suvišan narativ.
def _group_medications(medications: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    indexes: dict[tuple[str, str, str, str], int] = {}
    for index, medication in enumerate(medications):
        if str(medication.get("status", "mentioned")) == "questioned":
            continue
        key = (
            str(medication.get("name", "")).casefold(),
            str(medication.get("status", "mentioned")),
            str(medication.get("dose") or "").casefold(),
            str(medication.get("frequency") or "").casefold(),
        )
        reference = {
            "medication_index": index,
            "evidence": deepcopy(medication.get("evidence", {})),
            "context_evidence": deepcopy(medication.get("context_evidence")),
        }
        if key in indexes:
            grouped[indexes[key]]["source_references"].append(reference)
            continue
        indexes[key] = len(grouped)
        grouped.append(
            {
                "name": str(medication.get("name", "")),
                "status": str(medication.get("status", "mentioned")),
                "status_label": MEDICATION_STATUS_LABELS.get(
                    str(medication.get("status", "mentioned")), "spomenuto"
                ),
                "dose": medication.get("dose"),
                "frequency": medication.get("frequency"),
                "narrative_text": _medication_narrative(medication),
                "source_references": [reference],
            }
        )
    return grouped


# Dohvati dokazni tekst narativne stavke za daljnje preoblikovanje.
def _source_text(entry: dict[str, Any]) -> str:
    references = entry.get("source_references", [])
    if not references:
        return ""
    return str(references[0].get("source_text", ""))


# Skrati ponavljajuće uvode u narativu pacijenta.
def _compact_patient_text(text: str) -> str:
    value = str(text).strip()
    value = re.sub(r"^Pacijent je izjavio da\s+", "Navodi da ", value)
    value = re.sub(r"^Prema izjavi pacijenta,\s+", "", value)
    value = re.sub(r"^Pacijent je izjavio:\s+", "Navodi: ", value)
    return _with_period(_upper_initial(value))


# Ujednači kratki prikaz mjerenja u sastavljenom zapisu.
def _compact_measurement_text(text: str) -> str:
    value = str(text).strip()
    value = re.sub(r"^Pacijent je izjavio da\s+", "", value)
    value = re.sub(r"^Prema izjavi pacijenta,\s+", "", value)
    value = re.sub(r"^Liječnik je naveo da\s+", "", value)
    return _with_period(_upper_initial(value))


# Skrati uvodne izraze za radnje pri povezivanju narativnih stavki.
def _compact_action_text(text: str) -> str:
    value = str(text).strip()
    replacements = (
        (r"^Liječnik je zbog toga planirao\s+", "Planirano je "),
        (r"^Liječnik je najprije planirao\s+", "Najprije je planirano "),
        (r"^Liječnik je planirao\s+", "Planirano je "),
        (r"^Liječnik je preporučio pacijentu da\s+", "Preporučeno je da "),
        (r"^Liječnik je preporučio\s+", "Preporuka: "),
        (r"^Liječnik je savjetovao\s+", "Savjet: "),
        (r"^Liječnik je uputio pacijenta da\s+", "Uputa je da "),
        (r"^Liječnik je dao uputu za\s+", "Uputa za "),
        (r"^Liječnik je dao sigurnosnu uputu:\s+", "Sigurnosna uputa: "),
        (r"^Liječnik je naveo plan ili preporuku:\s+", "Plan ili preporuka: "),
    )
    for pattern, replacement in replacements:
        candidate, count = re.subn(pattern, replacement, value, flags=re.IGNORECASE)
        if count:
            return _with_period(candidate)
    return _with_period(value)


# Sastavi čitljivi liječnički zapis iz njegovih strukturiranih odjeljaka.
def _render_record(payload: dict[str, Any]) -> str:
    lines = [
        "AUTOMATSKI LIJEČNIČKI ZAPIS U TREĆEM LICU — OBAVEZNA PROVJERA",
        "Zapis je izveden isključivo iz izdvojenih činjenica razgovora i nije nova dijagnoza.",
        "",
        "ANAMNEZA I NAVODI PACIJENTA",
    ]
    patient_entries = payload["sections"]["patient_record"]
    if patient_entries:
        lines.append(
            " ".join(_compact_patient_text(item["narrative_text"]) for item in patient_entries)
        )
    else:
        lines.append("Nisu izdvojeni navodi pacijenta.")

    lines.extend(["", "MJERENJA"])
    measurements = payload["sections"]["measurements"]
    if measurements:
        for item in measurements:
            text = _compact_measurement_text(item["narrative_text"])
            lines.append(f"- {text}")
    else:
        lines.append("- Nisu izdvojena mjerenja ni brojčane vrijednosti.")

    lines.extend(["", "NALAZI I PROCJENA LIJEČNIKA"])
    clinician_entries = payload["sections"]["clinician_record"]
    if clinician_entries:
        lines.append(
            " ".join(
                _with_period(_clean_text(_source_text(item)))
                for item in clinician_entries
            )
        )
    else:
        lines.append("Nisu izdvojeni nalazi ni procjene liječnika.")

    lines.extend(["", "TERAPIJA I LIJEKOVI"])
    if payload["therapy_and_medications"]:
        for item in payload["therapy_and_medications"]:
            details = [item["status_label"]]
            if item.get("dose"):
                details.append(f"doza: {item['dose']}")
            if item.get("frequency"):
                details.append(f"učestalost: {item['frequency']}")
            lines.append(f"- {item['name']} — {'; '.join(details)}.")
    else:
        lines.append("- Nisu izdvojeni lijekovi ni terapija.")

    lines.extend(["", "AKCIJE, PLAN I PREPORUKE LIJEČNIKA"])
    actions = payload["sections"]["clinician_actions"]
    if actions:
        lines.extend(f"- {_compact_action_text(item['narrative_text'])}" for item in actions)
    else:
        lines.append("- Nisu izdvojene liječnikove akcije ni preporuke.")

    lines.extend(
        [
            "",
            "NAPOMENA",
            "Automatski zapis mora se usporediti s izvornim transkriptom prije uporabe.",
        ]
    )
    return "\n".join(lines) + "\n"


# Provjeri da se stavke liječničkog zapisa mogu povezati s ulaznim sažetkom.
def validate_clinical_record(record: dict[str, Any], summary: dict[str, Any]) -> None:
    if record.get("system_inferences"):
        raise ValueError("Liječnički zapis ne smije sadržavati zaključke sustava.")
    expected_fact_items: dict[tuple[str, str], dict[str, Any]] = {}
    for section, facts in summary.get("sections", {}).items():
        if section in QUESTION_SECTIONS:
            continue
        for fact in facts:
            key = (section, str(fact.get("id", "")))
            expected_fact_items[key] = fact
    expected_facts = set(expected_fact_items)

    actual_facts: list[tuple[str, str]] = []
    for section_entries in record.get("sections", {}).values():
        for entry in section_entries:
            if not str(entry.get("narrative_text", "")).strip():
                raise ValueError("Stavka liječničkog zapisa nema tekst.")
            for reference in entry.get("source_references", []):
                key = (str(reference["section"]), str(reference["fact_id"]))
                source_fact = expected_fact_items.get(key)
                if source_fact is None:
                    raise ValueError("Liječnički zapis upućuje na nepostojeću činjenicu.")
                if (
                    reference.get("source_text") != source_fact.get("text")
                    or reference.get("assertion") != source_fact.get("assertion")
                    or reference.get("evidence") != source_fact.get("evidence")
                    or reference.get("context_evidence") != source_fact.get("context_evidence")
                ):
                    raise ValueError("Dokaz u liječničkom zapisu ne odgovara izvornoj činjenici.")
                actual_facts.append(key)
    if set(actual_facts) != expected_facts or len(actual_facts) != len(expected_facts):
        raise ValueError("Liječnički zapis ne pokriva svaku izvornu činjenicu točno jednom.")

    expected_medications = {
        index
        for index, medication in enumerate(summary.get("medications", []))
        if str(medication.get("status", "mentioned")) != "questioned"
    }
    actual_medications: list[int] = []
    for item in record.get("therapy_and_medications", []):
        if not str(item.get("narrative_text", "")).strip():
            raise ValueError("Terapijski zapis nema tekst.")
        for reference in item.get("source_references", []):
            medication_index = int(reference["medication_index"])
            if medication_index not in expected_medications:
                raise ValueError("Terapijski zapis upućuje na nepostojeći lijek.")
            source_medication = summary["medications"][medication_index]
            if (
                reference.get("evidence") != source_medication.get("evidence")
                or reference.get("context_evidence")
                != source_medication.get("context_evidence")
            ):
                raise ValueError("Dokaz terapijskog zapisa ne odgovara izvornom lijeku.")
            actual_medications.append(medication_index)
    if set(actual_medications) != expected_medications or len(actual_medications) != len(expected_medications):
        raise ValueError("Terapijski dio ne pokriva svaki izvorni zapis točno jednom.")

    if record.get("record_text") != _render_record(record):
        raise ValueError("Čitljivi liječnički zapis ne odgovara strukturiranom sadržaju.")


def create_clinical_record(summary: dict[str, Any]) -> dict[str, Any]:
    """Pretvori provjereni medicinski sažetak u liječnički zapis trećeg lica."""

    sections = summary.get("sections")
    if not isinstance(sections, dict):
        raise ValueError("Nedostaju strukturirane sekcije medicinskog sažetka.")

    patient_record: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    clinician_record: list[dict[str, Any]] = []
    clinician_actions: list[dict[str, Any]] = []
    completed_examinations = 0

    for section in PATIENT_SECTIONS:
        for fact in sections.get(section, []):
            _append_fact_entry(
                patient_record,
                section,
                fact,
                _patient_narrative(str(fact.get("text", ""))),
            )

    for fact in sections.get("measurements", []):
        evidence = fact.get("evidence", {})
        role = str(evidence.get("role", "")).upper()
        fact_key = _fact_key(fact)
        action_keys = {
            _fact_key(action)
            for action in sections.get("plan_and_recommendations", [])
        }
        if fact_key in action_keys:
            target = clinician_actions if role == "LIJEČNIK" else patient_record
            narrative = (
                _action_narrative(str(fact.get("text", "")))
                if role == "LIJEČNIK"
                else _patient_narrative(str(fact.get("text", "")))
            )
            _append_fact_entry(target, "measurements", fact, narrative)
            continue
        narrative = _measurement_narrative(fact)
        _append_fact_entry(measurements, "measurements", fact, narrative)

    for section in CLINICIAN_SECTIONS:
        for fact in sections.get(section, []):
            fact_key = _fact_key(fact)
            measurement_keys = {
                tuple(item["deduplication_key"]) for item in measurements
            }
            if fact_key in measurement_keys:
                _append_fact_entry(
                    measurements,
                    section,
                    fact,
                    _clinician_narrative(str(fact.get("text", "")), section),
                )
                continue
            _append_fact_entry(
                clinician_record,
                section,
                fact,
                _clinician_narrative(str(fact.get("text", "")), section),
            )

    for section in ("plan_and_recommendations", "safety_instructions"):
        for fact in sections.get(section, []):
            role = str(fact.get("evidence", {}).get("role", "")).upper()
            if role == "LIJEČNIK":
                completed_entry = (
                    _completed_examination_entry(
                        fact, [*measurements, *clinician_record]
                    )
                    if section == "plan_and_recommendations"
                    else None
                )
                if completed_entry is not None:
                    completed_entry["source_references"].append(
                        _fact_reference(section, fact)
                    )
                    completed_examinations += 1
                    continue
                _append_fact_entry(
                    clinician_actions,
                    section,
                    fact,
                    _action_narrative(
                        str(fact.get("text", "")),
                        safety=section == "safety_instructions",
                    ),
                )
            else:
                _append_fact_entry(
                    patient_record,
                    section,
                    fact,
                    _patient_narrative(str(fact.get("text", ""))),
                )

    for entries in (patient_record, measurements, clinician_record, clinician_actions):
        entries.sort(key=lambda item: (item["start"], item["end"]))
        for item in entries:
            item.pop("deduplication_key", None)

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "document_type": "third_person_clinical_record",
        "generator": {
            "name": GENERATOR_VERSION,
            "type": "local_rule_based_reformulation",
            "inference_enabled": False,
        },
        "source": {
            **deepcopy(summary.get("source", {})),
            "medical_summary_generator": deepcopy(summary.get("generator", {})),
        },
        "sections": {
            "patient_record": patient_record,
            "measurements": measurements,
            "clinician_record": clinician_record,
            "clinician_actions": clinician_actions,
        },
        "therapy_and_medications": _group_medications(
            list(summary.get("medications", []))
        ),
        "excluded_questions": {
            section: len(sections.get(section, [])) for section in QUESTION_SECTIONS
        },
        "excluded_medication_questions": sum(
            1
            for medication in summary.get("medications", [])
            if str(medication.get("status", "mentioned")) == "questioned"
        ),
        "system_inferences": [],
        "review": {"required": True, "reason": REVIEW_REASON},
    }
    record["statistics"] = {
        "source_facts": sum(
            len(values)
            for name, values in sections.items()
            if name not in QUESTION_SECTIONS
        ),
        "record_entries": sum(len(values) for values in record["sections"].values()),
        "medication_mentions": len(summary.get("medications", [])),
        "medication_entries": len(record["therapy_and_medications"]),
        "excluded_questions": sum(record["excluded_questions"].values()),
        "excluded_medication_questions": record["excluded_medication_questions"],
        "completed_examinations": completed_examinations,
    }
    record["record_text"] = _render_record(record)
    validate_clinical_record(record, summary)
    return record


# Izradi i spremi strukturirani zapis te čitljivu TXT verziju.
def save_clinical_record(
    summary: dict[str, Any],
    json_path: Path,
    text_path: Path | None = None,
) -> dict[str, Any]:
    record = create_clinical_record(summary)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (text_path or json_path.with_suffix(".txt")).write_text(
        record["record_text"], encoding="utf-8"
    )
    return record
