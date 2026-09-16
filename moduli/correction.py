# Eksplicitne tekstne zamjene i učenje kratkih zamjena iz ručnih revizija.
import re
from pathlib import Path

CORRECTIONS_DIR = Path(__file__).resolve().parent / "ispravci"
CORRECTIONS_PATH = CORRECTIONS_DIR / "Ispravci.txt"

# Čitaj parove pogrešno,ispravno; preskoči prazne retke, komentare i neispravne parove.
def loadCorrections(file_name):
    path = CORRECTIONS_DIR / file_name
    corrections = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split(",", maxsplit=1)
            if len(parts) != 2:
                continue

            wrong = parts[0].strip()
            correct = parts[1].strip()

            if wrong and correct:
                corrections[wrong] = correct

    return corrections

CORRECTIONS = loadCorrections("Ispravci.txt")
CONTEXTUAL = loadCorrections("Kontekstualni.txt")
FORMAL = loadCorrections("Formalnosti.txt")

# Primijeni dulje zamjene prije kraćih, uz granice riječi i zapis svake zamjene.
def _applyCorrections(text, corrections, source, changes):
    for wrong, correct in sorted(corrections.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = rf"(?<!\w){re.escape(wrong)}(?!\w)"

        # Zabilježi pogođeni izvorni tekst prije vraćanja njegove zamjene regex mehanizmu.
        def replace(match):
            changes.append({
                "source": source,
                "original": match.group(0),
                "replacement": correct,
            })
            return correct

        text = re.sub(pattern, replace, text, flags=re.UNICODE)

    return text

def correctTextWithTrace(text):
    """Primijeni samo eksplicitna pravila i vrati trag svake izmjene."""
    changes = []
    text = _applyCorrections(text, CONTEXTUAL, "contextual", changes)
    text = _applyCorrections(text, CORRECTIONS, "learned", changes)
    text = _applyCorrections(text, FORMAL, "formal", changes)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)

    return text.strip(), changes

# Vrati samo ispravljeni tekst kada pozivatelj ne treba trag izmjena.
def correctText(text):
    corrected, _ = correctTextWithTrace(text)
    return corrected

# Ukloni interpunkciju i pretvori slova u mala za usporedbu riječi.
def normalizeWord(word):
    return re.sub(r"[^\w]", "", word, flags=re.UNICODE).lower()

# Ukloni interpunkciju, a razmake i slova sačuvaj.
def removePunctuation(text):
    return re.sub(r"[^\w\s]", "", text, flags=re.UNICODE).strip()

# Provjeri ostaju li isti nizovi riječi nakon uklanjanja interpunkcije i veličine slova.
def onlyPunctuation(old, new):
    old = [normalizeWord(w) for w in old]
    new = [normalizeWord(w) for w in new]
    
    return old == new

# Spremi kratki par zamjena u rječnik i datoteku, u varijanti s malim i velikim početnim slovom.
def saveCorrection(wrong, correct):
    wrong = wrong.strip()
    correct = correct.strip()
    
    if not wrong or not correct:
        return
    
    if len(wrong.split()) > 3 or len(correct.split()) > 3:
        return
    
    pairs = [
        (wrong[0].lower() + wrong[1:], correct[0].lower() + correct[1:]), 
        (wrong[0].capitalize() + wrong[1:], correct[0].capitalize() + correct[1:])
    ]
    existing = set()
    
    with open(CORRECTIONS_PATH, "r", encoding="utf-8") as f:
        existing = set(line.strip() for line in f if line.strip())
        
    with open(CORRECTIONS_PATH, "a", encoding="utf-8") as f:
        for w, c in pairs:
            line = f"{w},{c}"
            
            if line not in existing:
                f.write(line + "\n")
                existing.add(line)

            CORRECTIONS[w] = c

# Poveži uklonjene i dodane riječi iz razlike verzija u kandidate za nova pravila.
def extractCorrections(diffList):
    corrections, removed, added = [], [], []
    
    # Zaključi trenutačni par uklonjenih i dodanih riječi te isprazni radne popise.
    def flush():
        nonlocal removed, added
        
        if not removed or not added:
            removed = []
            added = []
            return
            
        wrong = " ".join(removed).strip()
        correct = " ".join(added).strip()
            
        if not onlyPunctuation(wrong.split(), correct.split()):
            corrections.append((wrong, correct))
            
        removed = []
        added = []
        
    for item in diffList:
        if item["type"] == "removed":
            if added:
                flush()
            
            removed.append(item["text"])
        
        elif item["type"] == "added":
            added.append(item["text"])
    
    flush()
    
    return corrections

# Pretvori ručne izmjene u kratke zamjene i dodaj ih trajnom rječniku.
def learnCorrections(difference):
    corrections = extractCorrections(difference)
    
    for wrong, correct in corrections:
        wrong = removePunctuation(wrong)
        correct = removePunctuation(correct)
        
        print("Spremam: ", wrong, " pod ", correct)
        
        saveCorrection(wrong, correct)
