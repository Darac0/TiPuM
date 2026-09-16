# Priprema zvuka i pokretanje modela za vremensko razlikovanje govornika.
import os
import warnings
from pathlib import Path

import numpy as np

_ffmpegBin = Path(os.getenv("FFMPEG_BIN", r"C:\ffmpeg\bin"))
_ffmpegDllHandle = None
if hasattr(os, "add_dll_directory") and _ffmpegBin.is_dir():
    _ffmpegDllHandle = os.add_dll_directory(str(_ffmpegBin))

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from scipy.io.wavfile import read
from pyannote.audio import Pipeline
from pyannote.audio.utils.reproducibility import ReproducibilityWarning

warnings.filterwarnings("ignore", category=ReproducibilityWarning)
# Sakrij samo poznato upozorenje o varijanci kratkih pyannote segmenata.
warnings.filterwarnings(
    "ignore",
    message=r"std\(\): degrees of freedom is <= 0\..*",
    category=UserWarning,
    module=r"pyannote\.audio\.models\.blocks\.pooling",
)

from .alignment import normalizeDiarizationSegments


def resolveDiarizationDevice(device=None):
    """Odaberi uređaj za pyannote uz mogućnost eksplicitnog CPU načina rada."""

    requested = str(device or os.getenv("PYANNOTE_DEVICE", "auto")).strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Zatražen je pyannote CUDA uređaj, ali instalirani PyTorch nema "
            "CUDA podršku. Pokrenite postavi_okruzenje.cmd ili instalirajte "
            "requirements/torch-cuda.txt."
        )
    if requested != "cpu" and not requested.startswith("cuda"):
        raise ValueError("PYANNOTE_DEVICE mora biti auto, cpu ili cuda.")
    return torch.device(requested)


# Učitaj Community-1 iz predmemorije ili izvora te ga premjesti na odabrani uređaj.
def loadDModel(token=None, device=None):
    selected_device = resolveDiarizationDevice(device)
    print(f"Učitavanje diarization modela na {selected_device}.")
    options = {"token": token} if token else {}

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            **options,
        )
    except Exception as error:
        raise RuntimeError(
            "Pyannote model nije moguće učitati. Ako još nije lokalno preuzet, "
            "slijedite postupak preuzimanja iz README-a."
        ) from error
    pipeline.to(selected_device)
    return pipeline

# Pretvori WAV u normalizirani mono tensor i vrati exclusive intervale govornika.
def diarizeAudio(aPath, model, num_speakers = 2):
    sample_rate, audio = read(aPath)

    if np.issubdtype(audio.dtype, np.integer):
        limits = np.iinfo(audio.dtype)
        scale = float(max(abs(limits.min), limits.max))
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)
    
    # ako je stereo, prebaci u mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1, dtype=np.float32)

    audio = np.clip(audio, -1.0, 1.0)
    waveform = torch.from_numpy(audio)
    
    waveform = waveform.unsqueeze(0) # kompatibilnost s pyannote
    
    diarization = model(
        {
            "waveform": waveform,
            "sample_rate": sample_rate
        },
        num_speakers = num_speakers
    )
    exclusive = getattr(diarization, "exclusive_speaker_diarization", None)
    if exclusive is None:
        raise RuntimeError("Community-1 rezultat ne sadrži exclusive diarizaciju.")

    segments = []
    
    for turn, _, speaker in exclusive.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
        
    return normalizeDiarizationSegments(segments)
