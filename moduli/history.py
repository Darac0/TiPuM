"""Ručno uređivanje transkripta uz trajnu i provjerljivu povijest revizija."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from .correction import learnCorrections
from .utils import transcriptPath
from .revision_outputs import prepare_revision_outputs


HISTORY_SCHEMA_VERSION = 2


def historyPath(transcript_path: Path) -> Path:
    """Vrati putanju povijesti vezanu uz zadani transkript."""
    return transcript_path.with_name(transcript_path.stem + "_povijest.json")


# Izračunaj otisak teksta za usporedbu prethodne i nove verzije; otisak nije enkripcija.
def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Zapiši privremenu datoteku u istu mapu pa je zamijeni tek nakon dovršenog pisanja.
def _atomicWriteText(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _wordDiff(old: str, new: str) -> list[dict]:
    """Izradi potpuni trag dodanih i uklonjenih riječi, uključujući više redaka."""
    old_words = old.split()
    new_words = new.split()
    matcher = SequenceMatcher(None, old_words, new_words)
    difference: list[dict] = []

    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in {"replace", "delete"}:
            difference.extend(
                {"type": "removed", "text": word, "index": index}
                for index, word in enumerate(old_words[old_start:old_end], old_start)
            )
        if tag in {"replace", "insert"}:
            difference.extend(
                {"type": "added", "text": word, "index": index}
                for index, word in enumerate(new_words[new_start:new_end], new_start)
            )
    return difference


# Pripremi početnu povijest u memoriji s jednakim izvornim i trenutačnim tekstom.
def _newHistory(original: str) -> dict:
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "originalni_tekst": original,
        "trenutni_tekst": original,
        "promjene": [],
    }


def loadHistory(transcript_path: Path) -> dict:
    """Učitaj povijest ili pripremi početno stanje bez zapisivanja na disk."""
    transcript_path = Path(transcript_path)
    _recoverRevision(transcript_path)
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transkript ne postoji: {transcript_path}")

    history_path = historyPath(transcript_path)
    if not history_path.exists():
        return _newHistory(transcript_path.read_text(encoding="utf-8"))

    data = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(data.get("promjene"), list):
        raise ValueError("Neispravan format povijesti: 'promjene' mora biti popis.")
    if "originalni_tekst" not in data or "trenutni_tekst" not in data:
        raise ValueError("Neispravan format povijesti: nedostaje tekstualno stanje.")
    data.setdefault("schema_version", 1)
    return data


def saveRevision(
    transcript_path: Path,
    new_text: str,
    note: str = "Ručna korekcija",
    author: str = "lokalni-korisnik",
    timestamp: str | None = None,
    learn: bool = True,
) -> dict | None:
    """Spremi novu verziju i revizijski zapis; vrati ``None`` ako nema promjene."""
    transcript_path = Path(transcript_path)
    data = loadHistory(transcript_path)
    old_text = data["trenutni_tekst"]
    if old_text == new_text:
        return None

    difference = _wordDiff(old_text, new_text)
    revision = {
        "revision": len(data["promjene"]) + 1,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "autor": author,
        "napomena": note,
        "prethodni_sha256": _sha256(old_text),
        "novi_sha256": _sha256(new_text),
        "prethodni_tekst": old_text,
        "novi_tekst": new_text,
        "razlika": difference,
    }
    data["schema_version"] = HISTORY_SCHEMA_VERSION
    data["promjene"].append(revision)
    data["trenutni_tekst"] = new_text

    outputs = prepare_revision_outputs(transcript_path, new_text, revision['revision'])
    outputs[transcript_path] = new_text
    outputs[historyPath(transcript_path)] = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    _commitRevision(transcript_path, outputs)
    if learn:
        learnCorrections(difference)
    return revision


def _journalPath(transcript_path: Path) -> Path:
    """Označi nedovršeno zapisivanje cijelog skupa datoteka."""
    return transcript_path.with_name(transcript_path.stem + '_revizija_u_tijeku.json')


def _recoverRevision(transcript_path: Path) -> None:
    """Vrati prethodni skup nakon prekinutog zapisivanja revizije."""
    journal = _journalPath(transcript_path)
    if not journal.exists():
        return
    originals = json.loads(journal.read_text(encoding='utf-8'))
    for name in originals:
        if Path(name).name != name or name in {'.', '..'}:
            raise ValueError('Neispravna putanja u dnevniku revizije.')
    for name, content in originals.items():
        target = transcript_path.parent / name
        if content is None:
            target.unlink(missing_ok=True)
        else:
            _atomicWriteText(target, content)
    journal.unlink()


def _commitRevision(transcript_path: Path, outputs: dict[Path, str]) -> None:
    """Zapiši oporavljivu reviziju; dnevnik se uklanja tek nakon uspjeha svih datoteka."""
    originals = {}
    for path in outputs:
        if path.exists():
            with path.open(encoding='utf-8', newline='') as stream:
                originals[path.name] = stream.read()
        else:
            originals[path.name] = None
    _atomicWriteText(_journalPath(transcript_path), json.dumps(originals, ensure_ascii=False))
    try:
        for path, content in outputs.items():
            _atomicWriteText(path, content)
    except BaseException:
        _recoverRevision(transcript_path)
        raise
    _journalPath(transcript_path).unlink()


def restoreRevision(
    transcript_path: Path,
    revision: int,
    author: str = "lokalni-korisnik",
    timestamp: str | None = None,
) -> dict:
    """Vrati stanje nakon odabrane revizije; nula znači izvorni ASR tekst."""
    data = loadHistory(Path(transcript_path))
    if revision < 0 or revision > len(data["promjene"]):
        raise IndexError("Tražena revizija ne postoji.")
    target = (
        data["originalni_tekst"]
        if revision == 0
        else data["promjene"][revision - 1].get("novi_tekst")
    )
    if target is None:
        raise ValueError("Stara povijest ne sadrži snimku teksta potrebnu za povrat.")
    saved = saveRevision(
        Path(transcript_path),
        target,
        note=f"Povrat na reviziju {revision}",
        author=author,
        timestamp=timestamp,
        learn=False,
    )
    if saved is None:
        raise ValueError("Transkript se već nalazi na traženoj reviziji.")
    return saved


# Prikaži postojeće revizije i razlike riječi za broj transkripta koji korisnik unese.
def show():
    t_number = input("Broj transkripta: ")
    transcript_path = transcriptPath(t_number)
    try:
        data = loadHistory(transcript_path)
    except FileNotFoundError:
        print("Transkript ne postoji.")
        return
    if not data["promjene"]:
        print("Datoteka nije mijenjana.")
        return

    for change in data["promjene"]:
        print(f"\nPromjena {change.get('revision', '?')}")
        print("Vrijeme:", change["timestamp"])
        print("Autor:", change.get("autor", "nepoznat"))
        print("Napomena:", change["napomena"])
        for difference in change["razlika"]:
            marker = "+" if difference["type"] == "added" else "-"
            print(f"{marker} {difference['text']}")


# Otvori TXT u Notepadu pa zabilježi spremljene promjene kroz sustav revizija.
def edit(note="Ručna korekcija", author="lokalni-korisnik"):
    t_number = input("Broj transkripta: ")
    transcript_path = transcriptPath(t_number)
    if not transcript_path.exists():
        print("Transkript ne postoji.")
        return

    old_text = loadHistory(transcript_path)["trenutni_tekst"]
    draft = transcript_path.with_name(transcript_path.stem + '_radna_kopija.txt')
    if not draft.exists():
        _atomicWriteText(draft, old_text)
    subprocess.run(["notepad.exe", str(draft)], check=True)
    new_text = draft.read_text(encoding="utf-8")
    # Nevaljana izmjena ostaje u radnoj kopiji; izvorni skup ostaje netaknut.
    revision = saveRevision(transcript_path, new_text, note=note, author=author)
    draft.unlink()
    print("Nema promjena." if revision is None else "Promjene spremljene.")
