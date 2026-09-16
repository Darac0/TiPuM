# Orkestracija obrade: čuva izvorni tekst, korekcije i izvedene zapise.
import json
import sys
import warnings
from pathlib import Path

import whisper

# Sakrij samo obavijesti o sporijim median i DTW implementacijama, bez promjene obrade.
warnings.filterwarnings(
    "ignore",
    message=(
        r"^Failed to launch Triton kernels, likely due to missing CUDA toolkit; "
        r"falling back to a slower (?:median kernel|DTW) implementation\.\.\.$"
    ),
    category=UserWarning,
    module=r"^whisper\.timing$",
)

from .config import PROJECT_ROOT, WHISPER_MODEL
from .alignment import alignTranscript
from .diarization import diarizeAudio
from .correction import correctTextWithTrace
from .speaker_roles import buildRoleMap
from .summary_service import save_configured_medical_summary
from .clinical_record import save_clinical_record


def _printStatus(message):
    """Ispiši status bez rušenja obrade na starijoj Windows kodnoj stranici."""
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        safe_message = str(message).encode(encoding, errors="replace").decode(encoding)
        print(safe_message)


def _displayPath(path):
    """Prikaži projektnu putanju relativno, a vanjsku putanju ne mijenjaj."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)

# Učitaj konfigurirani Whisper model koji se ponovno koristi za više snimki.
def loadTModel():
    print(f"Učitavanje Whisper modela: {WHISPER_MODEL}.")
    return whisper.load_model(WHISPER_MODEL)

def transcriptionOutputPaths(tPath):
    """Return every file produced by one transcription run."""
    target = Path(tPath)
    return (
        target,
        target.with_name(f"{target.stem}_strukturirano.json"),
        target.with_name(f"{target.stem}_medicinski_sazetak.json"),
        target.with_name(f"{target.stem}_medicinski_sazetak.txt"),
        target.with_name(f"{target.stem}_lijecnicki_zapis.json"),
        target.with_name(f"{target.stem}_lijecnicki_zapis.txt"),
    )

def validateTranscriptionPaths(aPath, tPath):
    """Reject a missing audio input or any output that would be overwritten."""
    audio = Path(aPath)
    if not audio.is_file():
        raise FileNotFoundError(f"Snimka nije pronađena: {audio}")
    existing = [path for path in transcriptionOutputPaths(tPath) if path.exists()]
    if existing:
        raise FileExistsError(
            "Transkripcija ne smije prepisati postojeće izlaze: "
            + ", ".join(str(path) for path in existing)
        )

# Poveži provjeru putanja, modele, uloge, korekcije i izradu šest izlaznih datoteka.
def transcribeAudio(aPath, tPath, model, dModel):
    validateTranscriptionPaths(aPath, tPath)
    _printStatus("Transkripcija...")
    usesCuda = next(model.parameters()).is_cuda
    # Polovična preciznost primjenjuje se na CUDA modelu; riječi nose vlastita vremena.
    transcript = model.transcribe(
        str(aPath),
        fp16=usesCuda,
        language="hr",
        word_timestamps=True,
        # None skriva Whisperov zasebni progress bar; aplikacija ispisuje faze.
        verbose=None,
    )

    _printStatus("Diarizacija...")
    segments = diarizeAudio(aPath, dModel)
    # Govornik se određuje iz zvuka, a komunikacijska uloga tek iz poravnatog teksta.
    utterances = alignTranscript(transcript, segments)
    roleMap = buildRoleMap(utterances)

    lines = []
    structuredTranscript = []
    _printStatus("Spremanje...")

    for utterance in utterances:
        # Izvorni tekst ostaje uz ispravljeni da se svaka automatska zamjena može pratiti.
        speaker = utterance["speaker"]
        role = roleMap.get(speaker, speaker)
        rawText = utterance["text"]
        text, corrections = correctTextWithTrace(rawText)

        lines.append(f"{role}: {text}")
        structuredTranscript.append({
            **utterance,
            "role": role,
            "raw_text": rawText,
            "text": text,
            "corrections": corrections,
        })

    transcriptText = "\n".join(lines)
    # TXT služi čitanju, JSON zadržava vremena, uloge i trag korekcija.
    tPath.parent.mkdir(parents=True, exist_ok=True)
    with open(tPath, "w", encoding="utf-8") as f:
        f.write(transcriptText)

    structuredPath = tPath.with_name(f"{tPath.stem}_strukturirano.json")
    with open(structuredPath, "w", encoding="utf-8") as f:
        json.dump(
            {
                "audio": str(aPath),
                "model": WHISPER_MODEL,
                "language": "hr",
                "corrections_applied": sum(
                    len(item["corrections"]) for item in structuredTranscript
                ),
                "utterances": structuredTranscript,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    medicalSummaryPath = tPath.with_name(f"{tPath.stem}_medicinski_sazetak.json")
    medicalSummaryTextPath = tPath.with_name(f"{tPath.stem}_medicinski_sazetak.txt")
    summaryPayload = save_configured_medical_summary(
        structuredTranscript,
        medicalSummaryPath,
        medicalSummaryTextPath,
        source={
            "audio": str(aPath),
            "structured_transcript": str(structuredPath),
            "model": WHISPER_MODEL,
            "language": "hr",
        },
    )

    clinicalRecordPath = tPath.with_name(f"{tPath.stem}_lijecnicki_zapis.json")
    clinicalRecordTextPath = tPath.with_name(f"{tPath.stem}_lijecnicki_zapis.txt")
    save_clinical_record(
        summaryPayload,
        clinicalRecordPath,
        clinicalRecordTextPath,
    )

    output_count = len(transcriptionOutputPaths(tPath))
    _printStatus(
        f"Gotovo: {output_count} datoteka u {_displayPath(tPath.parent)}/ "
        f"(osnova: {tPath.stem})."
    )
    # Podaci o LLM fallbacku ostaju u JSON-u, bez dodatnog ispisa u konzolu.
    return structuredTranscript
