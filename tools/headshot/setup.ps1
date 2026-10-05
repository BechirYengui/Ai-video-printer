# YVP : installe l'environnement « Photo de profil LinkedIn » sous Windows
# (PyTorch CUDA, diffusers, PhotoMaker V2) dans .headshot-venv, à part de
# l'application : plusieurs Go que seuls les PC avec une carte NVIDIA utilisent.
# Même contenu que setup.sh. Pas d'insightface (il faudrait Visual C++) :
# tools\headshot\faces.py fait la même détection avec onnxruntime.
# Les modèles (≈ 8 Go) se téléchargent ensuite à la première génération.
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Venv = if ($env:YVP_HEADSHOT_VENV) { $env:YVP_HEADSHOT_VENV } else { Join-Path $Root '.headshot-venv' }
$Python = Join-Path $Venv 'Scripts\python.exe'
$env:UV_HTTP_TIMEOUT = '300'; $env:UV_HTTP_RETRIES = '8'; $env:UV_CONCURRENT_DOWNLOADS = '2'

# uv : celui de l'installation de YVP (install.ps1), même s'il n'est pas dans le PATH.
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = Join-Path $env:USERPROFILE '.local\bin\uv.exe' }
if (-not (Test-Path $uv)) { Write-Output 'uv introuvable : relancez Installer-YVP.bat'; exit 1 }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Output 'git introuvable (nécessaire pour PhotoMaker) : installez Git pour Windows'; exit 1
}

Remove-Item (Join-Path $Venv '.yvp-ready') -ErrorAction SilentlyContinue
if (-not (Test-Path $Python)) { & $uv venv -q --python 3.11 $Venv }
# Connexion instable : on retente, uv reprend depuis son cache.
foreach ($attempt in 1..4) {
    Write-Output ">> installation, tentative $attempt"
    & $uv pip install -q --python $Python 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -eq 0) {
        & $uv pip install -q --python $Python numpy pillow 'diffusers>=0.30' transformers accelerate `
            safetensors peft omegaconf einops huggingface_hub 'onnxruntime-gpu==1.23.2' `
            opencv-python-headless requests 'git+https://github.com/TencentARC/PhotoMaker.git'
    }
    if ($LASTEXITCODE -eq 0) {
        & $Python (Join-Path $Root 'tools\headshot\worker.py') --check
        if ($LASTEXITCODE -eq 0) {
            New-Item -ItemType File -Force -Path (Join-Path $Venv '.yvp-ready') | Out-Null
            Write-Output ">> environnement prêt : $Venv"
            exit 0
        }
    }
    Start-Sleep -Seconds 10
}
Write-Output '>> installation impossible (voir les erreurs ci-dessus)'
exit 1
