# One-time setup on ton-618 (Windows, NVIDIA GPU). Run from the repo root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\ton618_setup.ps1
# Optional: -Cuda cu126 if the driver is too old for CUDA 12.8 wheels.
param([string]$Cuda = "cu128")
$ErrorActionPreference = "Stop"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found; install it with: winget install --id=astral-sh.uv -e"
    exit 1
}

uv venv .venv --python 3.11
.\.venv\Scripts\Activate.ps1
uv pip install torch --index-url "https://download.pytorch.org/whl/$Cuda"
uv pip install -e ".[train]"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -m unittest tests.test_normalize
