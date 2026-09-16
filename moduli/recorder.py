# Snimanje mikrofona preko sounddevice i spremanje WAV uzoraka preko scipy.
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

from .config import SAMPLE_RATE, CHANNELS

# Skupljaj blokove zvuka iz mikrofona do pritiska Enter i spoji ih po vremenskoj osi.
def record() -> np.ndarray:
    recording = []
    
    # Kopiraj blok jer audio biblioteka ponovno koristi ulazni memorijski međuspremnik.
    def callback(indata, frames, time, status):
        if status:
            print(status)
        recording.append(indata.copy())
    
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=callback
    ):
        print("Snimanje je počelo. Pritisnite ENTER za zaustavljanje.")
        input()
    
    if not recording or not any(block.size for block in recording):
        raise ValueError("Nisu primljeni zvučni uzorci. Provjerite mikrofon i pokušajte ponovno.")
    return np.concatenate(recording, axis = 0)

# Spremi uzorke u WAV uz frekvenciju definiranu konfiguracijom.
def saveAudio(path, audio):
    write(path, SAMPLE_RATE, audio)
