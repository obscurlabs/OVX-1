# Dot-source this before working:  . .\env.ps1
#
# Redirects every model/package cache into .\.cache so that deleting this
# folder after submission removes 100% of the project's disk footprint.
# Without this, ~10GB of model weights silently accumulate in C:\Users\yaksh\.cache.

$ProjectRoot = $PSScriptRoot

# Hugging Face model + dataset cache (the big one: ~1.5GB models, ~3-6GB dataset)
$env:HF_HOME = Join-Path $ProjectRoot ".cache\hf"
$env:HF_HUB_CACHE = Join-Path $ProjectRoot ".cache\hf\hub"

# torch.hub weights
$env:TORCH_HOME = Join-Path $ProjectRoot ".cache\torch"

# uv's package cache
$env:UV_CACHE_DIR = Join-Path $ProjectRoot ".cache\uv"

# ffmpeg, since winget did not refresh PATH
$FfmpegDir = "C:\Users\yaksh\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin"
if (Test-Path $FfmpegDir) { $env:PATH = "$FfmpegDir;$env:PATH" }

# Activate the venv if present
$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) { & $VenvActivate }

Write-Host "voicerag env ready" -ForegroundColor Green
Write-Host "  HF_HOME     = $env:HF_HOME"
Write-Host "  UV_CACHE_DIR= $env:UV_CACHE_DIR"
Write-Host "  python      = $((Get-Command python -ErrorAction SilentlyContinue).Source)"
