# Week 1 on ton-618: baselines, then the first fine-tune, then the fine-tuned model's WER.
# Run from the repo root after ton618_setup.ps1, either directly:
#   powershell -ExecutionPolicy Bypass -File scripts\ton618_week1.ps1
# or detached from an SSH session with scripts\ton618_launch.ps1.
# Everything is logged to logs\week1.log; per-step logs go to logs\*.log.
param(
    [string]$FinetuneBase = "openai/whisper-small",
    [string]$RunName = "small-v1",
    [int]$BatchSize = 16,
    [switch]$SkipBaselines
)
# Native tools write progress to stderr; in Windows PowerShell 5.1 that becomes an error
# record, so never use "Stop" here. Failures are detected from $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TQDM_DISABLE = "1"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force logs, results | Out-Null
$py = ".\.venv\Scripts\python.exe"

function Step([string]$name, [string]$log, [string[]]$pyArgs) {
    $t0 = Get-Date
    Add-Content logs\week1.log "[$($t0.ToString('s'))] START $name"
    & $py @pyArgs *> "logs\$log.log"
    $code = $LASTEXITCODE
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Add-Content logs\week1.log "[$((Get-Date).ToString('s'))] END   $name exit=$code ($mins min)"
    return $code
}

$baselines = @(
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large-v3-turbo",
    "openai/whisper-large-v3",
    "jlvdoorn/whisper-large-v3-atco2-asr",
    "jlvdoorn/whisper-large-v3-atco2-asr-atcosim"
)

Add-Content logs\week1.log "[$((Get-Date).ToString('s'))] week1 started, pid $PID"

if (-not $SkipBaselines) {
    foreach ($m in $baselines) {
        $tag = $m -replace "/", "__"
        if (Test-Path "results\$tag.json") {
            Add-Content logs\week1.log "skip $m (results\$tag.json exists)"
            continue
        }
        Step "baseline $m" "eval_$tag" @("-m", "atc.eval_wer", "--model", $m, "--batch-size", $BatchSize) | Out-Null
    }
}

$code = Step "finetune $FinetuneBase -> runs\$RunName" "finetune_$RunName" @(
    "-m", "atc.finetune", "--model", $FinetuneBase, "--out", "runs\$RunName",
    "--batch-size", $BatchSize, "--report-to", "tensorboard")
if ($code -eq 0) {
    Step "eval runs\$RunName" "eval_$RunName" @(
        "-m", "atc.eval_wer", "--model", "runs\$RunName", "--tag", $RunName, "--batch-size", $BatchSize) | Out-Null
}

& $py -m atc.report *> results\report.md
Add-Content logs\week1.log "[$((Get-Date).ToString('s'))] week1 done"
