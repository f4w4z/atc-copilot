# ATC Copilot

Transcribe ATC radio, parse clearances, and check them against ADS-B. Plan and references: [docs/plan.md](docs/plan.md).

Current stage: **Week 1, offline speech-to-text** (baselines and first fine-tune).

## Layout

| Path | What it does |
|---|---|
| `atc/normalize.py` | Puts references and Whisper output in one spoken form ("FL100" -> "flight level one hundred") so WER is comparable |
| `atc/data.py` | Loads ATCO2 1h, UWB-ATCC, ATCOSIM from Hugging Face at 16 kHz |
| `atc/eval_wer.py` | WER of any Whisper checkpoint on the test sets |
| `atc/finetune.py` | Fine-tunes Whisper on the public corpora |
| `atc/report.py` | Collects `results/*.json` into one table |
| `atc/parse.py`, `atc/callsign.py`, `atc/numbers.py` | Rule-based clearance parser: transcript -> speaker, ICAO callsign, instructions |
| `atc/readback.py`, `atc/eval_readback.py` | Readback checker on parsed instructions, evaluated on UWB-ATCC with injected errors |
| `atc/adsb.py` | OpenSky and readsb sources, track store |
| `atc/link.py` | Matches a heard callsign to a tracked aircraft, tolerating misheard digits |
| `atc/conformance.py` | Did the aircraft follow the altitude or heading clearance (QNH and magnetic variation corrected)? |
| `atc/loop.py` | The full loop: transmission -> parse -> link -> readback and conformance events |
| `atc/record_adsb.py` | Records ADS-B around an airport to daily JSONL |
| `scripts/ton618_*.ps1` | Setup and Week 1 run on the training box (Windows + NVIDIA) |

## Local (Mac)

```bash
python3 -m venv .venv && .venv/bin/pip install torch -e .
```

```bash
.venv/bin/python -m unittest tests.test_normalize
```

```bash
.venv/bin/python -m atc.eval_wer --model openai/whisper-tiny --max-samples 32 --tag smoke-tiny
```

## Training box (ton-618)

Reach it over Tailscale (`ssh ton-618`). Copy the code across with `scripts/sync_to_ton618.sh` (tar over SSH, since Windows has no rsync), then in PowerShell on ton-618:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\ton618_setup.ps1
powershell -ExecutionPolicy Bypass -File scripts\ton618_launch.ps1
```

`ton618_launch.ps1` starts `ton618_week1.ps1` detached, so it survives the SSH session closing. Progress is in `logs\week1.log`; stop it with `Stop-Process -Id (Get-Content logs\week1.pid)`. Re-running skips baselines that already have results.

The Week 1 script evaluates six baselines (stock Whisper small to large-v3, and both Whisper-ATC models), fine-tunes `whisper-small` for 4,000 steps on UWB-ATCC + ATCOSIM, evaluates the result, and prints the comparison table.

## Notes on the numbers

- Whisper-ATC models were likely trained on ATCO2 and ATCOSIM, so their scores there are flagged `*`. UWB-ATCC test is the fair comparison for them.
- The fine-tune trains on UWB-ATCC and ATCOSIM; ATCO2 1h is never trained on, so it is the out-of-domain check.
- None of these are the demo airport. The headline number will come from the hand-labelled set (Week 2).
- UWB-ATCC is CC BY-NC-SA 4.0: non-commercial use only.
