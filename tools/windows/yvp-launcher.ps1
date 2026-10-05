# Lanceur Windows de YenguiVideoPrinter (YVP).
# Démarre Ollama + la webui, ouvre une fenêtre dédiée, et arrête tout dès que
# cette fenêtre est fermée. Rien ne tourne en dehors de ces moments-là.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms, System.Drawing

$Root    = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$State   = Join-Path $env:LOCALAPPDATA 'YVP'
$BrowserProfile = Join-Path $State 'browser-profile'
$UrlFile = Join-Path $State 'url'
$Logs    = Join-Path $Root 'logs'
$Icon    = Join-Path $PSScriptRoot 'yvp.ico'
$Python  = Join-Path $Root '.venv\Scripts\python.exe'
$Ollama  = Join-Path $Root 'ollama\ollama.exe'
New-Item -ItemType Directory -Force -Path $State, $Logs | Out-Null

function Show-Error($msg) {
    [System.Windows.Forms.MessageBox]::Show($msg, 'YVP', 'OK', 'Error') | Out-Null
}

function Find-Browser {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
        "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
        "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
        "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
    )
    foreach ($c in $candidates) { if ($c -and (Test-Path $c)) { return $c } }
    return $null
}

function Open-Window($browser, $url) {
    Start-Process -FilePath $browser -PassThru -ArgumentList @(
        "--user-data-dir=`"$BrowserProfile`"", '--no-first-run', '--no-default-browser-check', '--disable-features=Translate', "--app=$url"
    )
}

function Test-Url($url) {
    try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 $url | Out-Null; return $true } catch { return $false }
}

function Stop-Tree($proc) {
    if ($proc -and -not $proc.HasExited) { Start-Process taskkill.exe -ArgumentList "/PID $($proc.Id) /T /F" -WindowStyle Hidden -Wait }
}

$browser = Find-Browser
if (-not $browser) { Show-Error "Microsoft Edge ou Google Chrome est introuvable."; exit 1 }
if (-not (Test-Path $Python)) { Show-Error "YVP n'est pas installé : lancez d'abord Installer-YVP.bat."; exit 1 }

# Une seule instance : si YVP tourne déjà, on rouvre juste une fenêtre.
$mutex = New-Object System.Threading.Mutex($false, 'Local\YVP-Launcher')
if (-not $mutex.WaitOne(0)) {
    $url = Get-Content $UrlFile -ErrorAction SilentlyContinue
    if ($url) { Open-Window $browser $url | Out-Null }
    else { [System.Windows.Forms.MessageBox]::Show('YVP est en train de démarrer, patientez…', 'YVP') | Out-Null }
    exit 0
}

# Petite fenêtre « démarrage en cours » pour que l'on sache que le clic a marché.
$splash = New-Object System.Windows.Forms.Form
$splash.Text = 'YVP'; $splash.Width = 360; $splash.Height = 120
$splash.StartPosition = 'CenterScreen'; $splash.FormBorderStyle = 'FixedDialog'
$splash.MaximizeBox = $false; $splash.MinimizeBox = $false; $splash.TopMost = $true
if (Test-Path $Icon) { $splash.Icon = New-Object System.Drawing.Icon($Icon) }
$label = New-Object System.Windows.Forms.Label
$label.Text = 'Démarrage de YenguiVideoPrinter…'; $label.AutoSize = $false
$label.Dock = 'Fill'; $label.TextAlign = 'MiddleCenter'
$label.Font = New-Object System.Drawing.Font('Segoe UI', 11)
$splash.Controls.Add($label); $splash.Show(); [System.Windows.Forms.Application]::DoEvents()

$ollamaProc = $null; $webProc = $null
try {
    # --- Ollama (local au projet, modèles dans ollama\models) ---
    $env:OLLAMA_MODELS = Join-Path $Root 'ollama\models'
    $env:OLLAMA_HOST = '127.0.0.1:11434'
    if (-not $env:OLLAMA_KEEP_ALIVE) { $env:OLLAMA_KEEP_ALIVE = '10m' }
    if (-not (Test-Url 'http://127.0.0.1:11434/api/version')) {
        if (-not (Test-Path $Ollama)) { throw "Ollama est introuvable ($Ollama). Relancez Installer-YVP.bat." }
        $ollamaProc = Start-Process -FilePath $Ollama -ArgumentList 'serve' -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $Logs 'ollama.out.log') -RedirectStandardError (Join-Path $Logs 'ollama.log')
        $ok = $false
        for ($i = 0; $i -lt 60; $i++) {
            if (Test-Url 'http://127.0.0.1:11434/api/version') { $ok = $true; break }
            if ($ollamaProc.HasExited) { break }
            Start-Sleep -Milliseconds 1000; [System.Windows.Forms.Application]::DoEvents()
        }
        if (-not $ok) { throw "Ollama n'a pas démarré (voir logs\ollama.log)." }
    }

    # --- WebUI Streamlit sur le premier port libre 8501-8599 ---
    $port = $null
    foreach ($p in 8501..8599) {
        $l = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $p)
        try { $l.Start(); $l.Stop(); $port = $p; break } catch { }
    }
    if (-not $port) { throw 'Aucun port libre entre 8501 et 8599.' }
    $url = "http://127.0.0.1:$port"
    $env:PYTHONPATH = $Root
    $env:PYTHONIOENCODING = 'utf-8'
    # Jamais de traduction automatique (titre « Imprimante vidéo Yengui ») : dans le HTML servi.
    & $Python (Join-Path $Root 'tools\desktop\notranslate.py') 2>$null
    $webProc = Start-Process -FilePath $Python -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $Logs 'desktop.log') -RedirectStandardError (Join-Path $Logs 'desktop.err.log') `
        -ArgumentList @('-m', 'streamlit', 'run', 'webui\Main.py',
            '--server.address=127.0.0.1', "--server.port=$port", '--server.headless=true',
            '--browser.serverAddress=127.0.0.1', '--browser.gatherUsageStats=False',
            '--client.toolbarMode=minimal', '--logger.hideWelcomeMessage=True',
            '--server.showEmailPrompt=False', '--server.enableCORS=True')
    $ok = $false
    for ($i = 0; $i -lt 300; $i++) {
        if (Test-Url "$url/_stcore/health") { $ok = $true; break }
        if ($webProc.HasExited) { break }
        Start-Sleep -Milliseconds 1000; [System.Windows.Forms.Application]::DoEvents()
    }
    if (-not $ok) { throw "L'interface n'a pas démarré (voir logs\desktop.err.log)." }
    Set-Content -Path $UrlFile -Value $url -Encoding ASCII

    $splash.Close()
    # « Toujours traduire » cliqué un jour passe avant --disable-features=Translate.
    & $Python (Join-Path $Root 'tools\desktop\browser_prefs.py') $BrowserProfile 2>$null
    $win = Open-Window $browser $url

    # On s'arrête quand la fenêtre est fermée : plus aucune page connectée à
    # l'interface pendant 15 s (fiable même si Edge garde un processus en
    # arrière-plan), ou navigateur fermé avant même d'avoir affiché la page.
    $seen = $false; $idle = 0; $waited = 0
    while ($true) {
        Start-Sleep -Seconds 3
        if ($webProc.HasExited) { break }
        # Navigateur fermé sans jamais avoir affiché la page (30 s de marge,
        # au cas où Edge aurait passé la main à un processus déjà ouvert).
        if (-not $seen) { $waited += 3; if ($win.HasExited -and $waited -ge 30) { break } }
        $conns = @(Get-NetTCPConnection -LocalPort $port -State Established -ErrorAction SilentlyContinue)
        if ($conns.Count -gt 0) { $seen = $true; $idle = 0 }
        elseif ($seen) { $idle += 3; if ($idle -ge 15) { break } }
    }
}
catch {
    $splash.Close()
    Show-Error $_.Exception.Message
}
finally {
    Stop-Tree $webProc
    Stop-Tree $ollamaProc   # seulement si c'est nous qui l'avons lancé
    Remove-Item $UrlFile -ErrorAction SilentlyContinue
    $mutex.ReleaseMutex()
}
