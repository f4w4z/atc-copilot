"""Collect results/*.json into one markdown table: python -m atc.report"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(results_dir: str = "results") -> None:
    rows = []
    sets: list[str] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        if "sets" not in s:
            continue  # not a WER summary (e.g. readback_eval.json)
        rows.append((path.stem, s))
        for k in s["sets"]:
            if k not in sets:
                sets.append(k)
    if not rows:
        print("no results yet")
        return
    print("| model | " + " | ".join(sets) + " |")
    print("|---" * (len(sets) + 1) + "|")
    for tag, s in rows:
        cells = []
        for k in sets:
            r = s["sets"].get(k)
            if r is None:
                cells.append("-")
            else:
                note = "*" if r.get("possible_train_overlap") else ""
                sub = f" (n={r['utterances']})" if s.get("max_samples") else ""
                cells.append(f"{r['wer']:.1%}{note}{sub}")
        print(f"| {s['model']} | " + " | ".join(cells) + " |")
    print("\nWER after atc.normalize on both sides. * = model may have trained on this corpus.")


if __name__ == "__main__":
    main(*sys.argv[1:])
