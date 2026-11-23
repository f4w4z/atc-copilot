"""Evaluate speaker roles, parsing coverage and readback checking on UWB-ATCC transcripts.

    python -m atc.eval_readback

UWB-ATCC ids carry the speaker (_AT controller, _PI pilot) and the position in the
recording, so controller -> pilot pairs come from the data itself. Real readback
errors are rare, so detection is measured by injecting digit errors into readbacks
that currently match. The flag rate on unmodified pairs is an upper bound on the
false alarm rate (some flags will be real errors or transcript truncations).
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

from atc.callsign import same_aircraft
from atc.numbers import DIGIT
from atc.parse import parse
from atc.readback import CHECKED, check, flagged

ID_RE = re.compile(r"^uwb-atcc_(?P<rec>.+)_(?P<start>\d{6})_(?P<end>\d{6})_(?P<role>[A-Z]+)$")
MAX_GAP_S = 15.0
DIGIT_WORDS = list(DIGIT)


def load_uwb() -> list[dict]:
    rows = []
    for split in ("train", "test"):
        ds = load_dataset("Jzuluaga/uwb_atcc", split=split).remove_columns(["audio"])
        for i, t in zip(ds["id"], ds["text"]):
            m = ID_RE.match(i)
            if m:
                rows.append({"id": i, "text": t, "rec": m["rec"], "start": int(m["start"]) / 100,
                             "end": int(m["end"]) / 100, "role": m["role"]})
    rows.sort(key=lambda r: (r["rec"], r["start"]))
    return rows


def mutate(words: list[str], start: int, end: int, rng: random.Random) -> list[str] | None:
    """Replace one digit word inside [start, end) with a different digit."""
    idx = [k for k in range(start, end) if words[k] in DIGIT]
    if not idx:
        return None
    k = rng.choice(idx)
    new = list(words)
    new[k] = rng.choice([d for d in DIGIT_WORDS if d != words[k]])
    return new


def main(out_dir: str = "results", seed: int = 0) -> None:
    rows = load_uwb()
    parsed = [parse(r["text"]) for r in rows]
    rng = random.Random(seed)

    # Speaker role vs labels.
    role_map = {"AT": "controller", "PI": "pilot"}
    conf = Counter()
    for r, p in zip(rows, parsed):
        if r["role"] in role_map:
            conf[(role_map[r["role"]], p.speaker)] += 1
    decided = sum(v for (t, s), v in conf.items() if s != "unknown")
    correct = sum(v for (t, s), v in conf.items() if t == s)
    total = sum(conf.values())

    # Parsing coverage on controller turns.
    ctrl = [p for r, p in zip(rows, parsed) if r["role"] == "AT"]
    with_inst = sum(1 for p in ctrl if any(i.type in CHECKED for i in p.instructions))
    types = Counter(i.type for p in ctrl for i in p.instructions)

    # Controller -> pilot pairs.
    pairs = []
    for k in range(len(rows) - 1):
        a, b = rows[k], rows[k + 1]
        pa, pb = parsed[k], parsed[k + 1]
        if a["role"] != "AT" or b["role"] != "PI" or a["rec"] != b["rec"]:
            continue
        if b["start"] - a["end"] > MAX_GAP_S:
            continue
        if not any(i.type in CHECKED for i in pa.instructions):
            continue
        if pa.callsign and pb.callsign and not same_aircraft(pa.callsign, pb.callsign):
            continue
        pairs.append((k, k + 1))

    status = Counter()
    by_type = defaultdict(Counter)
    flagged_pairs = []
    for a, b in pairs:
        fs = check(parsed[a], parsed[b])
        for f in fs:
            status[f.status] += 1
            by_type[f.type][f.status] += 1
        if flagged(fs):
            flagged_pairs.append((a, b, fs))
    compared = [(a, b) for a, b in pairs if any(f.status != "missing" for f in check(parsed[a], parsed[b]))]

    # Error injection into matching readbacks.
    inj = defaultdict(Counter)
    examples = []
    for a, b in compared:
        fs = check(parsed[a], parsed[b])
        if flagged(fs):
            continue
        pb = parsed[b]
        matched_types = {f.type for f in fs if f.status == "match"}
        cands = [x for x in pb.instructions if x.type in matched_types and x.digits]
        if not cands:
            continue
        target = rng.choice(cands)
        new_words = mutate(pb.words, target.start, target.end, rng)
        if new_words is None:
            continue
        mutated = parse(" ".join(new_words))
        hit = any(f.type == target.type and f.status == "mismatch" for f in check(parsed[a], mutated))
        inj[target.type]["n"] += 1
        inj[target.type]["detected"] += hit
        if not hit and len(examples) < 15:
            examples.append({"controller": parsed[a].text, "readback": pb.text,
                             "mutated": " ".join(new_words), "type": target.type})

    inj_n = sum(c["n"] for c in inj.values())
    inj_hit = sum(c["detected"] for c in inj.values())
    summary = {
        "utterances": len(rows),
        "speaker_role": {"labelled": total, "decided": decided, "accuracy_all": correct / total,
                         "accuracy_when_decided": correct / max(decided, 1),
                         "confusion": {f"{t}->{s}": v for (t, s), v in sorted(conf.items())}},
        "controller_turns": len(ctrl),
        "controller_turns_with_checkable_instruction": with_inst,
        "instruction_types": dict(types.most_common()),
        "pairs": len(pairs),
        "pairs_with_compared_value": len(compared),
        "findings": dict(status),
        "findings_by_type": {t: dict(c) for t, c in by_type.items()},
        "natural_flag_rate": len(flagged_pairs) / max(len(compared), 1),
        "injection": {"n": inj_n, "detection_rate": inj_hit / max(inj_n, 1),
                      "by_type": {t: {"n": c["n"], "detection_rate": c["detected"] / c["n"]} for t, c in inj.items()},
                      "missed_examples": examples},
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "readback_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out / "readback_flags.md", "w", encoding="utf-8") as f:
        f.write("# Flagged readbacks (UWB-ATCC, reference transcripts)\n\nReview by hand: real error, transcript issue, or parser error?\n\n")
        for a, b, fs in flagged_pairs[:100]:
            bad = [x for x in fs if x.status == "mismatch"]
            f.write(f"- **ATC:** {parsed[a].text}\n  **PI:** {parsed[b].text}\n  "
                    + "; ".join(f"{x.type}: {x.said} vs {x.readback}" for x in bad) + "\n")

    print(json.dumps({k: v for k, v in summary.items() if k != "injection"}, indent=2))
    print("injection:", json.dumps({k: v for k, v in summary["injection"].items() if k != "missed_examples"}, indent=2))


if __name__ == "__main__":
    main()
