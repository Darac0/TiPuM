# TiPuM

Studentski prototip za transkripciju razgovora liječnika i pacijenta na hrvatskom jeziku. Lokalno prepoznaje govor, razlikuje govornike i izrađuje medicinski sažetak te zapis u trećem licu.

## Prvo postavljanje

Potreban je Python 3.12 i FFmpeg. NVIDIA grafička kartica ubrzava obradu.

Instalacija koristi fiksirane verzije za Windows i Python 3.12: `requirements/torch-cuda.txt` za PyTorch te `requirements/runtime.txt` s ograničenjima iz `locked-windows-py312.txt` za ostale ovisnosti. FFmpeg mora biti dostupan na PATH-u; nije Python paket. Provjera instalacije pokreće se automatski, a može se ponoviti naredbom `.\.venv\Scripts\python.exe -X utf8 provjeri_instalaciju.py`. Ne provjerava preuzete modele ni mikrofon.

Na drugom računalu ponovno izradite `.venv` instalacijskim programom; nemojte kopirati postojeće virtualno okruženje.

1. Pokrenite `postavi_okruzenje.cmd` za instalaciju ovisnosti.
2. Na stranici modela `pyannote/speaker-diarization-community-1` prijavite se na Hugging Face i prihvatite uvjete pristupa.
3. Izradite Hugging Face token s ovlasti za čitanje.
4. Otvorite PowerShell u mapi `app` i jednokratno preuzmite modele:

```powershell
$env:HF_TOKEN = "hf_vas_token"
.\.venv\Scripts\python.exe -c "import os; from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization-community-1', token=os.environ['HF_TOKEN'])"
.\.venv\Scripts\python.exe -c "import whisper; whisper.load_model('turbo')"
Remove-Item Env:HF_TOKEN
```

Modeli se spremaju u predmemoriju korisničkog računa, izvan projektne mape. Token se ne upisuje u datoteke projekta. Na drugom računalu modele treba ponovno preuzeti.

Lokalni Qwen model za LLM sažetak preuzima se zasebno:

```powershell
.\tools\llm\pripremi_lokalni_llm.cmd
```

Ako Qwen nije dostupan ili njegov izlaz ne prođe provjeru, aplikacija koristi deterministički sažetak.

Ako nedostaje Qwen ili njegov poslužitelj ne uspije krenuti, pokretač ispisuje upozorenje i nastavlja bez LLM-a. Za namjerno pokretanje bez njega postavite `$env:TIPUM_DISABLE_LLM = "1"` prije pokretanja; uklonite postavku s `Remove-Item Env:TIPUM_DISABLE_LLM` kada ga ponovno želite koristiti.

## Pokretanje

Nakon što su modeli preuzeti, token više nije potreban. Jedna naredba pokreće lokalni LLM i aplikaciju:

```powershell
.\pokreni.cmd
```

Provjere se pokreću naredbom `..\development\testiraj.cmd`.

`pokreni.cmd` automatski pokreće Qwen ako već ne radi i gasi ga pri izlasku iz aplikacije. LLM ima najviše tri pokušaja izrade sažetka. Svaki izdvojeni podatak mora biti vezan uz izvorni iskaz, a LLM ne smije izostaviti podatke koje je pronašao deterministički postupak. Ako provjera ne uspije, sprema se deterministički sažetak.

## Mape

| Mapa | Sadržaj |
|---|---|
| `moduli/` | Kod aplikacije |
| `recordings/` | Ulazne WAV snimke |
| `../output/<broj>_snimka/` | Izlazi i povijest jedne obrade |
| `../development/tests/` | Testne snimke |
| `docs/` | Korisničke upute |
| `requirements/` | Python ovisnosti |
| `tools/` | Qwen model i `llama.cpp` |

`.venv/` je lokalno Python okruženje. Ulaz u program je `main.py`, a osnovne postavke nalaze se u `moduli/config.py`.

## Dokumentacija

- [Korisničke upute](docs/KORISNICKA_DOKUMENTACIJA.md)

Izlaz je automatski nacrt koji treba provjeriti prema transkriptu i snimci. Prototip nije klinički validiran. Snimke, medicinski tekst i pristupne tokene ne treba javno dijeliti.
