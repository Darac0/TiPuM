"""Provjeri ovisnosti bez preuzimanja modela ili pokretanja snimanja."""
import importlib
import shutil
import sys
from importlib.metadata import version


def main():
    """Prijavi dostupnost audio biblioteka i vanjskog FFmpega."""
    errors = []
    if sys.version_info[:2] != (3, 12):
        errors.append('Očekivan je Python 3.12.')
    for module, distribution in [('numpy','numpy'), ('scipy','scipy'),
            ('sounddevice','sounddevice'), ('whisper','openai-whisper'),
            ('torch','torch'), ('torchaudio','torchaudio'),
            ('moduli.diarization','pyannote.audio')]:
        try:
            importlib.import_module(module)
            print(f'{distribution}: {version(distribution)}')
        except Exception as error:
            errors.append(f'{module}: {error}')
    if not shutil.which('ffmpeg'):
        errors.append('FFmpeg nije na PATH-u. Instalirajte FFmpeg i dodajte njegovu bin mapu u PATH.')
    for error in errors:
        print(error)
    if not errors:
        print('Ovisnosti su dostupne. Modeli se preuzimaju zasebno prema README-u.')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
