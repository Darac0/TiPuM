@echo off
rem Pokreni pripremu lokalnog modela i proslijedi opcije poput -ForceDownload.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0pripremi_lokalni_llm.ps1" %*
pause
