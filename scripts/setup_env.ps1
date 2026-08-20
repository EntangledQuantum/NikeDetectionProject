# Create an isolated Nike Detection environment OUTSIDE the repo (Windows).
# Usage (PowerShell):
#   .\scripts\setup_env.ps1
#   .\scripts\setup_env.ps1 -EnvDir "C:\venvs\nike-detection"

param(
    [string]$EnvDir = "$env:USERPROFILE\.venvs\nike-detection",
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"

Write-Host "==> Nike Detection — Windows environment setup" -ForegroundColor Cyan
Write-Host "    Project: $ProjectRoot"
Write-Host "    Env dir: $EnvDir"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python not found. Install Python 3.10+ from https://www.python.org/downloads/ and check 'Add to PATH'."
}

# Prefer uv when available (fast, reliable on Windows)
$useUv = $false
if (Get-Command uv -ErrorAction SilentlyContinue) {
    $useUv = $true
    Write-Host "==> Using uv to create venv" -ForegroundColor Green
    uv venv $EnvDir --python (python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
} else {
    Write-Host "==> uv not found; using built-in venv" -ForegroundColor Yellow
    python -m venv $EnvDir
}

$activate = Join-Path $EnvDir "Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    throw "Virtual environment was not created at $EnvDir"
}

. $activate
python -m pip install --upgrade pip wheel

if ($useUv) {
    uv pip install -r (Join-Path $ProjectRoot "requirements.txt")
} else {
    pip install -r (Join-Path $ProjectRoot "requirements.txt")
}

Write-Host ""
Write-Host "Done. Activate and run:" -ForegroundColor Green
Write-Host "  . `"$activate`""
Write-Host "  cd `"$ProjectRoot`""
Write-Host "  python -m nike_detection -i data\blackStripe.tiff --only stripe_misalignment --no-vis"
