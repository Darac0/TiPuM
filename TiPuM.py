# Ulazna konzolna aplikacija: učitavanje modela, snimanje, obrada i povijest.
import logging


class _TritonFlopFilter(logging.Filter):
    """Sakrij samo obavijest o neobaveznom Triton brojaču operacija."""

    def filter(self, record):
        """Propusti sve ostale poruke, uključujući pogreške."""
        return not (
            record.levelno == logging.WARNING
            and record.getMessage()
            == "triton not found; flop counting will not work for triton kernels"
        )


# Filter mora biti postavljen prije nego što audio moduli učitaju PyTorch.
logging.getLogger("torch.utils.flop_counter").addFilter(_TritonFlopFilter())

from moduli.config import RECORDING_DIR, HF_TOKEN

from moduli.utils import getNumber, availableTranscriptPath, recordingNumber
from sounddevice import PortAudioError
from moduli.transcriber import loadTModel, transcribeAudio
from moduli.diarization import loadDModel
from moduli.history import edit, show
from moduli.recorder import record, saveAudio


# Učitaj oba velika modela jednom i ponovno ih koristi za svaku obradu u izborniku.
def loadModels():
    return loadTModel(), loadDModel(HF_TOKEN)


# Dodijeli slobodan broj, snimi mikrofon u WAV pa pokreni cijeli postupak obrade.
def recordAndTranscribe(tModel, dModel):
    number = getNumber()
    target = availableTranscriptPath(number)
    audioPath = RECORDING_DIR / f"{number}_snimka.wav"
    saveAudio(audioPath, record())
    print(f"Snimka spremljena pod: recordings/{audioPath.name}")
    transcribeAudio(audioPath, target, tModel, dModel)

# Modeli se učitavaju tek pri prvoj audio obradi; odabir nula završava aplikaciju.
def main():
    # Uređivanje povijesti ne zahtijeva učitavanje velikih audio modela.
    models = None
    
    while True:
        print("\n--- MENU ---")
        print("1 - Nova snimka")
        print("2 - Transkript iz gotove snimke")
        print("3 - Uredi transkript")
        print("4 - Prikaži povijest")
        print("0 - Izlaz")

        try:
            choice = input("Odabir: ")
            if choice == "0":
                break
            if choice == "2":
                source = recordingNumber(input("Unesite broj snimke: "))
                aPath = RECORDING_DIR / f"{source}_snimka.wav"
                if not aPath.is_file():
                    raise FileNotFoundError(f"Snimka ne postoji: {aPath.name}")
                target = availableTranscriptPath(source)
                if models is None:
                    models = loadModels()
                transcribeAudio(aPath, target, *models)
            elif choice == "1":
                if models is None:
                    models = loadModels()
                recordAndTranscribe(*models)
            elif choice == "3":
                edit()
            elif choice == "4":
                show()
            else:
                print("Odaberite broj od 0 do 4.")
        except (EOFError, KeyboardInterrupt):
            print("\nIzlaz iz aplikacije.")
            break
        except (OSError, ValueError, RuntimeError, PortAudioError) as error:
            print(f"Radnja nije dovršena: {error}")

if __name__=="__main__":
    main()
