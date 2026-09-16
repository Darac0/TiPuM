@echo off
rem Instalacijski ulaz: Python 3.12, lokalno okruzenje i ovisnosti aplikacije.
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Greska: Python Launcher "py" nije pronaden.
    echo Instalirajte Python 3.12 i ukljucite Python Launcher.
    exit /b 1
)

py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)" >nul 2>nul
if errorlevel 1 (
    echo Greska: Python 3.12 nije dostupan kroz naredbu py -3.12.
    py -0p
    exit /b 1
)

rem Postojece okruzenje ostaje sacuvano; novo se stvara samo kada nedostaje.
if not exist ".venv\Scripts\python.exe" (
    echo Izrada virtualnog okruzenja s Pythonom 3.12...
    py -3.12 -m venv .venv
    if errorlevel 1 exit /b 1
)

echo Nadogradnja pip-a...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo Instalacija PyTorcha s CUDA 13.0 podrskom...
".venv\Scripts\python.exe" -m pip install --upgrade -r requirements\torch-cuda.txt
if errorlevel 1 exit /b 1

echo Instalacija ostalih ovisnosti aplikacije...
".venv\Scripts\python.exe" -m pip install -r requirements\runtime.txt
if errorlevel 1 exit /b 1

echo Provjera ovisnosti i uvoza...
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -X utf8 provjeri_instalaciju.py
if errorlevel 1 exit /b 1

echo.
echo Okruzenje je spremno. Sljedeci korak: ..\development\testiraj.cmd
endlocal
