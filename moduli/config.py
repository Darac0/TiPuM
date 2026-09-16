# Zajedničke putanje i postavke. Token i opcije modela dolaze iz varijabli okruženja.
import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECORDING_DIR = PROJECT_ROOT / "recordings"
# Izlazi i povijest ostaju u projektu, uz main.py, neovisno o radnoj mapi.
OUTPUTS_DIR = PROJECT_ROOT / "output"
TRANSCRIPT_DIR = OUTPUTS_DIR


RPATTERN = re.compile(r"^(\d+)_.*\.wav$") # Uzorak za snimke, naziv mora biti oblika broj_naziv.wav
TPATTERN = re.compile(r"^(\d+)_.*\.txt$") # Uzorak za transkripte, naziv mora biti oblika broj_naziv.txt

SAMPLE_RATE = 44100 # Standard
CHANNELS = 2

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "turbo")
HF_TOKEN = os.getenv("HF_TOKEN")

LOCAL_LLM_URL = os.getenv(
    "LOCAL_LLM_URL", "http://127.0.0.1:8081/v1/chat/completions"
).strip()
LOCAL_LLM_MODEL = os.getenv(
    "LOCAL_LLM_MODEL", "Qwen/Qwen3-4B-GGUF:Q4_K_M"
).strip()
LOCAL_LLM_TIMEOUT_SECONDS = float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "180"))
