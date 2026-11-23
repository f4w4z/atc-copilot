"""Measure word error rate of a Whisper checkpoint on ATC test sets.

    python -m atc.eval_wer --model openai/whisper-small
    python -m atc.eval_wer --model runs/small-v1 --datasets uwb_atcc:test --max-samples 200

Writes results/<tag>.json (summary) and results/<tag>.<corpus>.jsonl (every utterance),
and prints a table. WER is computed after `atc.normalize.normalize` on both sides;
`wer_raw` (lowercase and punctuation only) is reported to show how much the normaliser matters.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import jiwer
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from atc import data
from atc.normalize import normalize
from atc.whisper_utils import generate_kwargs, pick_device, pick_dtype

DEFAULT_SETS = ["atco2_1h:test", "uwb_atcc:test", "atcosim:test"]


def raw_clean(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def transcribe(model, processor, ds, device, dtype, batch_size: int, num_beams: int):
    kw = generate_kwargs(model, num_beams=num_beams)
    hyps: list[str] = []
    for start in range(0, len(ds), batch_size):
        batch = ds[start : start + batch_size]
        arrays = [a["array"] for a in batch["audio"]]
        feats = processor.feature_extractor(
            arrays, sampling_rate=data.SAMPLING_RATE, return_tensors="pt"
        ).input_features.to(device, dtype)
        with torch.inference_mode():
            out = model.generate(feats, **kw)
        hyps += processor.batch_decode(out, skip_special_tokens=True)
        print(f"  {min(start + batch_size, len(ds))}/{len(ds)}", end="\r", flush=True)
    print()
    return hyps


def score(refs: list[str], hyps: list[str]) -> dict:
    norm_r = [normalize(r) for r in refs]
    norm_h = [normalize(h) for h in hyps]
    # jiwer rejects empty references; keep pairs with a non-empty reference.
    keep = [i for i, r in enumerate(norm_r) if r]
    m = jiwer.process_words([norm_r[i] for i in keep], [norm_h[i] or "<empty>" for i in keep])
    raw_r = [raw_clean(refs[i]) for i in keep]
    raw_h = [raw_clean(hyps[i]) or "<empty>" for i in keep]
    return {
        "wer": m.wer,
        "wer_raw": jiwer.wer(raw_r, raw_h),
        "substitutions": m.substitutions,
        "deletions": m.deletions,
        "insertions": m.insertions,
        "ref_words": sum(len(r.split()) for r in (norm_r[i] for i in keep)),
        "utterances": len(keep),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="hub id or local checkpoint dir")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_SETS)
    p.add_argument("--max-samples", type=int, default=None, help="fixed random subset per set")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--tag", default=None, help="results file name; defaults to model name")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    tag = args.tag or args.model.rstrip("/").replace("/", "__")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model} on {device} ({dtype})")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    summary = {"model": args.model, "device": str(device), "dtype": str(dtype),
               "num_beams": args.num_beams, "max_samples": args.max_samples, "sets": {}}
    overlap = data.KNOWN_OVERLAP.get(args.model, set())

    for spec in args.datasets:
        name, split = data.parse_spec(spec)
        ds = data.load(spec, max_samples=args.max_samples)
        print(f"{spec}: {len(ds)} utterances")
        t0 = time.time()
        hyps = transcribe(model, processor, ds, device, dtype, args.batch_size, args.num_beams)
        elapsed = time.time() - t0
        refs = list(ds["text"])
        s = score(refs, hyps)
        audio_s = float(sum(ds["duration"])) if "duration" in ds.column_names else None
        s.update(seconds=elapsed, audio_seconds=audio_s,
                 rtf=(elapsed / audio_s) if audio_s else None,
                 possible_train_overlap=name in overlap)
        summary["sets"][spec] = s
        with open(out_dir / f"{tag}.{name}_{split}.jsonl", "w", encoding="utf-8") as f:
            for row_id, r, h in zip(ds["id"], refs, hyps):
                f.write(json.dumps({"id": row_id, "ref": r, "hyp": h,
                                    "ref_norm": normalize(r), "hyp_norm": normalize(h)}) + "\n")

    with open(out_dir / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n| set | WER | raw WER | utts | RTF |\n|---|---|---|---|---|")
    for spec, s in summary["sets"].items():
        flag = " *" if s["possible_train_overlap"] else ""
        rtf = f"{s['rtf']:.3f}" if s["rtf"] else "-"
        print(f"| {spec}{flag} | {s['wer']:.1%} | {s['wer_raw']:.1%} | {s['utterances']} | {rtf} |")
    if overlap:
        print("* model may have been trained on this corpus")


if __name__ == "__main__":
    main()
