# Week 1 on ton-618: baselines, then the first fine-tune, then the fine-tuned model's WER.
# Run from the repo root after ton618_setup.ps1:
#   powershell -ExecutionPolicy Bypass -File scripts\ton618_week1.ps1
# Logs go to logs\. Results land in results\ (summaries are committed; per-utterance jsonl are not).
param(
    [string]$FinetuneBase = "openai/whisper-small",
    [string]$RunName = "small-v1",
    [int]$BatchSize = 16,
    [switch]$SkipBaselines
)
$ErrorActionPreference = "Stop"
.\.venv\Scripts\Activate.ps1
New-Item -ItemType Directory -Force logs, results | Out-Null

$baselines = @(
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large-v3-turbo",
    "openai/whisper-large-v3",
    "jlvdoorn/whisper-large-v3-atco2-asr",
    "jlvdoorn/whisper-large-v3-atco2-asr-atcosim"
)

if (-not $SkipBaselines) {
    foreach ($m in $baselines) {
        $tag = $m -replace "/", "__"
        Write-Host "== baseline $m"
        python -m atc.eval_wer --model $m --batch-size $BatchSize 2>&1 | Tee-Object "logs\eval_$tag.log"
    }
}

Write-Host "== fine-tune $FinetuneBase -> runs\$RunName"
python -m atc.finetune --model $FinetuneBase --out "runs\$RunName" --batch-size $BatchSize --report-to tensorboard 2>&1 |
    Tee-Object "logs\finetune_$RunName.log"

Write-Host "== eval runs\$RunName"
python -m atc.eval_wer --model "runs\$RunName" --tag $RunName --batch-size $BatchSize 2>&1 |
    Tee-Object "logs\eval_$RunName.log"

python -m atc.report
