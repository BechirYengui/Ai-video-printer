# Installation de YenguiVideoPrinter (YVP) sous Windows 11.
# - Python + dépendances via uv (dans .venv du projet)
# - Ollama portable dans .\ollama (pas de service, rien au démarrage de Windows)
# - modèle IA dans .\ollama\models
# - raccourcis Bureau + menu Démarrer avec l'icône YVP
# Relançable sans risque : chaque étape déjà faite est sautée.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest est bien plus rapide sans barre
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root   = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Model  = if ($env:MPT_OLLAMA_MODEL) { $env:MPT_OLLAMA_MODEL } else { 'qwen3:8b' }
$OllamaDir = Join-Path $Root 'ollama'
$Ollama = Join-Path $OllamaDir 'ollama.exe'
Set-Location $Root

function Step($msg) { Write-Host ''; Write-Host ">> $msg" -ForegroundColor Cyan }

Write-Host "Installation de YenguiVideoPrinter dans : $Root" -ForegroundColor Magenta
if ($Root -match 'OneDrive') {
    Write-Host "Attention : le dossier est dans OneDrive. Mieux vaut le placer ailleurs (ex. C:\YVP)." -ForegroundColor Yellow
}

# 1. uv (gestionnaire Python) -------------------------------------------------
Step 'uv (gestionnaire Python)'
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = Join-Path $env:USERPROFILE '.local\bin\uv.exe' }
if (-not (Test-Path $uv)) {
    Write-Host 'Téléchargement de uv…'
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}
if (-not (Test-Path $uv)) { throw "uv n'a pas pu être installé." }

# 2. Python + dépendances -----------------------------------------------------
Step 'Python et dépendances (quelques minutes la première fois)'
& $uv sync --frozen --no-dev --python 3.11   # même version que sur le PC Linux
if ($LASTEXITCODE -ne 0) { throw 'uv sync a échoué.' }

# 3. Configuration ------------------------------------------------------------
if (-not (Test-Path (Join-Path $Root 'config.toml'))) {
    Copy-Item (Join-Path $Root 'config.example.toml') (Join-Path $Root 'config.toml')
    Write-Host 'config.toml créé depuis config.example.toml.'
}

# 4. Ollama portable ----------------------------------------------------------
Step 'Ollama (moteur IA local)'
if (-not (Test-Path $Ollama)) {
    $zip = Join-Path $env:TEMP 'ollama-windows-amd64.zip'
    Write-Host 'Téléchargement d''Ollama (~1,5 Go)…'
    & curl.exe -fL --retry 5 -C - -o $zip 'https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip'
    if ($LASTEXITCODE -ne 0) { throw "Téléchargement d'Ollama échoué." }
    New-Item -ItemType Directory -Force -Path $OllamaDir | Out-Null
    & tar.exe -xf $zip -C $OllamaDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Ollama)) { throw "Extraction d'Ollama échouée." }
    Remove-Item $zip -ErrorAction SilentlyContinue
}

# 5. Modèle IA ----------------------------------------------------------------
Step "Modèle IA $Model (~5 Go la première fois)"
$env:OLLAMA_MODELS = Join-Path $OllamaDir 'models'
$env:OLLAMA_HOST = '127.0.0.1:11434'
$name, $tag = $Model.Split(':'); if (-not $tag) { $tag = 'latest' }
$manifest = Join-Path $env:OLLAMA_MODELS "manifests\registry.ollama.ai\library\$name\$tag"
if (Test-Path $manifest) {
    Write-Host 'Modèle déjà présent.'
} else {
    $started = $null
    try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:11434/api/version' | Out-Null }
    catch {
        $started = Start-Process -FilePath $Ollama -ArgumentList 'serve' -WindowStyle Hidden -PassThru
        Start-Sleep -Seconds 5
    }
    try {
        & $Ollama pull $Model
        if ($LASTEXITCODE -ne 0) { throw "Téléchargement du modèle $Model échoué (relancez l'installation, il reprendra)." }
    } finally {
        if ($started) { Start-Process taskkill.exe -ArgumentList "/PID $($started.Id) /T /F" -WindowStyle Hidden -Wait }
    }
}

# 6. Raccourcis ---------------------------------------------------------------
Step 'Raccourcis Bureau et menu Démarrer'
$launcher = Join-Path $PSScriptRoot 'yvp-launcher.ps1'
$icon = Join-Path $PSScriptRoot 'yvp.ico'
$shell = New-Object -ComObject WScript.Shell
$targets = @(
    (Join-Path ([Environment]::GetFolderPath('Desktop')) 'YenguiVideoPrinter.lnk'),
    (Join-Path ([Environment]::GetFolderPath('Programs')) 'YenguiVideoPrinter.lnk')
)
foreach ($t in $targets) {
    $lnk = $shell.CreateShortcut($t)
    # conhost --headless : lance PowerShell sans fenêtre noire.
    $lnk.TargetPath = Join-Path $env:WINDIR 'System32\conhost.exe'
    $lnk.Arguments = "--headless powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$launcher`""
    $lnk.WorkingDirectory = $Root
    $lnk.IconLocation = "$icon,0"
    $lnk.Description = 'YenguiVideoPrinter — démarre au clic, s''arrête à la fermeture'
    $lnk.Save()
}

Write-Host ''
Write-Host 'Installation terminée !' -ForegroundColor Green
Write-Host 'Pour l''épingler à la barre des tâches : menu Démarrer > YenguiVideoPrinter > clic droit > Épingler à la barre des tâches.'
