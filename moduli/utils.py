# Dodjela broja i izrada izlaznih putanja; nema pokretanja modela.
import re
from .config import RECORDING_DIR, TRANSCRIPT_DIR, RPATTERN, TPATTERN


OUTPUT_FOLDER_PATTERN = re.compile(r"^(\d+)_snimka$")


def transcriptPath(number):
    """Vrati glavni transkript unutar mape koja pripada jednoj snimci."""
    number = recordingNumber(number)
    name = f"{number}_snimka"
    return TRANSCRIPT_DIR / name / f"{name}.txt"


def recordingNumber(value):
    """Prihvati samo pozitivni cijeli broj, bez dijelova putanje."""
    text = str(value).strip()
    if not re.fullmatch(r"[0-9]+", text) or int(text) < 1:
        raise ValueError("Broj snimke mora biti pozitivan cijeli broj.")
    return int(text)


def availableTranscriptPath(number):
    """Zadrži broj izvora i odbij postojeću mapu ili stari samostalni izlaz."""
    number = recordingNumber(number)
    target = transcriptPath(number)
    legacy = TRANSCRIPT_DIR / target.name
    if target.parent.exists() or legacy.exists():
        raise FileExistsError(f"Već postoji izlaz za snimku broj {number}.")
    return target

# Vrati broj veći od svih postojećih snimki i izlaza kako se novi rezultat ne bi prepisao.
def getNumber():
    RECORDING_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    highestRec = 0
    
    for file in RECORDING_DIR.iterdir():
        if not file.is_file():
            continue
        match = RPATTERN.match(file.name)
        if match:
            number = int(match.group(1))
            highestRec = max(highestRec, number)
    
    highestTran = 0
    for item in TRANSCRIPT_DIR.iterdir():
        if item.is_dir():
            match = OUTPUT_FOLDER_PATTERN.match(item.name)
        elif item.is_file():
            # Podrška za transkripte spremljene prije uvođenja mapa po snimci.
            match = TPATTERN.match(item.name)
        else:
            match = None
        if match:
            number = int(match.group(1))
            highestTran = max(highestTran, number)
            
    return max(highestRec, highestTran) + 1
