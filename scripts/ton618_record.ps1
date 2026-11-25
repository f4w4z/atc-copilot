# Record ADS-B around an airport continuously (OpenSky by default). Start detached with:
#   powershell -ExecutionPolicy Bypass -File scripts\ton618_launch.ps1 -Script scripts\ton618_record.ps1 -Name record
# Output: data\adsb\<airport>\<date>.jsonl, log in logs\record.log.
param([string]$Airport = "LKPR", [string]$ExtraArgs = "")
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force logs | Out-Null
$argv = @("-m", "atc.record_adsb", "--airport", $Airport) + ($ExtraArgs -split " " | Where-Object { $_ })
& .\.venv\Scripts\python.exe @argv *>> logs\record.log
