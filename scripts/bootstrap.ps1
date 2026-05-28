# Bootstrap the Datum dev environment on native Windows.
# Mirror of scripts/bootstrap_wsl.sh. Run from PowerShell, not cmd.
#
# This script does five things:
#   1. checks Python 3.11 is reachable
#   2. installs uv if missing
#   3. creates .venv and syncs pinned deps
#   4. (optional) downloads the sample match
#   5. import-smoke-tests the package

$ErrorActionPreference = 'Stop'

# $ErrorActionPreference does NOT catch native-command non-zero exit codes.
# This helper does. Without it the script will happily continue past a
# broken `uv sync` and print "bootstrap ok" on a half-built venv.
function Invoke-Checked {
    param([string]$Stage, [scriptblock]$Block)
    & $Block
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[$Stage] FAILED (exit $LASTEXITCODE). bootstrap halted." -ForegroundColor Red
        exit 1
    }
}

# Resolve repo root from the script's own location.
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $RepoRoot

Write-Host "[datum] repo root: $RepoRoot"

# Steer uv's cache to the same drive as the repo. The default location on
# C: causes 30,000-file cross-drive copies (notably torch headers) that
# Windows answers with "os error 1450: insufficient system resources".
$RepoDrive = (Split-Path -Qualifier $RepoRoot)
if (-not $env:UV_CACHE_DIR) {
    $env:UV_CACHE_DIR = Join-Path $RepoDrive 'uv-cache'
    Write-Host "[datum] UV_CACHE_DIR set to $env:UV_CACHE_DIR (same drive as repo)"
}

# 1. Python 3.11. uv can fetch it, but flagging missing python early is friendlier.
function Test-Command($name) {
    $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

if (-not (Test-Command 'py') -and -not (Test-Command 'python')) {
    Write-Host "[1/5] no python found on PATH. install via 'winget install Python.Python.3.11'"
    Write-Host "       (uv can also fetch it; continuing.)"
} else {
    Write-Host "[1/5] python launcher present"
}

# 2. uv. The whole toolchain assumes it.
if (-not (Test-Command 'uv')) {
    Write-Host "[2/5] installing uv"
    Invoke-RestMethod -Uri 'https://astral.sh/uv/install.ps1' | Invoke-Expression
    # Refresh PATH for this session so subsequent calls see uv.
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + $env:Path
} else {
    $uvVersion = (uv --version)
    Write-Host "[2/5] uv present: $uvVersion"
}

# Pin python through uv. If 3.11 is already there, this is a no-op.
Invoke-Checked 'python' { uv python install 3.11 | Out-Null }

# 3. venv + deps.
Write-Host "[3/5] creating .venv and installing pinned deps"
Invoke-Checked 'venv'  { uv venv --python 3.11 }
Invoke-Checked 'sync'  { uv sync --extra dev --extra api --extra index --link-mode=copy }

# 4. Sample match. Placeholder until SAMPLE_URL is wired up.
$Sample = Join-Path $RepoRoot 'data\samples\sample_match.mp4'
if (-not (Test-Path $Sample)) {
    Write-Host "[4/5] sample match not present (skipping. Wire up SAMPLE_URL before release.)"
} else {
    Write-Host "[4/5] sample match already present"
}

# 5. Smoke. Import every heavyweight dep, not just the package shell. The
# previous version of this check imported only `datum`, which is a version
# string and would pass on a venv where torch failed to install.
Write-Host "[5/5] smoke check"
$SmokeScript = @'
import importlib, sys
mods = ["datum", "numpy", "cv2", "torch", "av", "pydantic", "typer"]
failed = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"    {m:10s} ok")
    except Exception as e:
        print(f"    {m:10s} FAIL: {e}")
        failed.append(m)
if failed:
    sys.exit(2)
'@
Invoke-Checked 'smoke' { uv run python -c $SmokeScript }

Write-Host ""
Write-Host "bootstrap ok"
Write-Host "next: activate with '.\.venv\Scripts\Activate.ps1' or use 'uv run ...'"
Write-Host ""
Write-Host "if torch.cuda.is_available() returns False, reinstall torch from the CUDA index:"
Write-Host "    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
