@echo off
rem Pokreni PowerShell skriptu i vrati njezin izlazni kod pozivatelju.
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0pokreni.ps1"
exit /b %errorlevel%
