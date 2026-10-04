# groundplan setup for Windows (PowerShell). Usage: powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "installing uv ..."
    python -m pip install --user uv
    $env:PATH = "$env:APPDATA\Python\Python39\Scripts;$env:APPDATA\Python\Python310\Scripts;$env:APPDATA\Python\Python311\Scripts;$env:APPDATA\Python\Python312\Scripts;$env:PATH"
}

# keep the environment out of synced folders (OneDrive / Dropbox re-upload thousands of files)
if ($repo -match "OneDrive|Dropbox|iCloud" -and -not $env:UV_PROJECT_ENVIRONMENT) {
    $env:UV_PROJECT_ENVIRONMENT = "$env:USERPROFILE\.venvs\groundplan"
    [Environment]::SetEnvironmentVariable("UV_PROJECT_ENVIRONMENT", $env:UV_PROJECT_ENVIRONMENT, "User")
    Write-Host "environment placed at $env:UV_PROJECT_ENVIRONMENT (outside the synced folder)"
}

uv sync --extra models --extra video
uv run python scripts/fetch_models.py
uv run pytest -q -m "not slow and not sample"
Write-Host "ready: uv run groundplan run <capture>"
