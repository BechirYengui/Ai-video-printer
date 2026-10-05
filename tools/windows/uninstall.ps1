# Retire les raccourcis YVP (Bureau + menu Démarrer). Le dossier du projet
# n'est pas touché : supprimez-le à la main pour tout effacer.
$ErrorActionPreference = 'Continue'
Remove-Item (Join-Path ([Environment]::GetFolderPath('Desktop')) 'YenguiVideoPrinter.lnk') -ErrorAction SilentlyContinue
Remove-Item (Join-Path ([Environment]::GetFolderPath('Programs')) 'YenguiVideoPrinter.lnk') -ErrorAction SilentlyContinue
Remove-Item (Join-Path $env:LOCALAPPDATA 'YVP') -Recurse -Force -ErrorAction SilentlyContinue
Write-Host 'Raccourcis YVP supprimés. Pensez à le désépingler de la barre des tâches si besoin.'
