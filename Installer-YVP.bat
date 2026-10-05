@echo off
rem Double-cliquez sur ce fichier pour installer YenguiVideoPrinter (YVP).
chcp 65001 >nul
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows\install.ps1"
if errorlevel 1 (
  echo.
  echo L'installation a echoue. Relancez ce fichier : il reprendra ou il s'est arrete.
)
echo.
pause
