"""Compare a controller instruction with the pilot's readback, at the level of parsed
instructions rather than words (HAAWAII found word-level comparison doesn't work).
"""

from __future__ import annotations

from dataclasses import dataclass

from atc.parse import Instruction, Parsed

CHECKED = ("altitude", "heading", "speed", "frequency", "squawk", "qnh", "runway", "direct")


@dataclass
class Finding:
    type: str
    said: object
    readback: object
    status: str  # "match", "mismatch", "partial", "missing"


def values_match(a: Instruction, b: Instruction) -> bool:
    if a.type == "frequency":
        # Pilots often read back only the last digits: 121.9 -> "one nine".
        return a.value == b.value or (len(b.digits) >= 2 and a.digits.endswith(b.digits))
    if a.type == "runway":
        an, bn = str(a.value), str(b.value)
        return an == bn or an.rstrip("LRC") == bn.rstrip("LRC") and not (an[-1:].isalpha() and bn[-1:].isalpha())
    if a.type == "direct":
        x, y = str(a.value), str(b.value)
        return len(min(x, y, key=len)) >= 3 and (x.startswith(y) or y.startswith(x))
    if a.type == "heading":
        return a.value % 360 == b.value % 360
    return a.value == b.value


def check(controller: Parsed, pilot: Parsed) -> list[Finding]:
    findings: list[Finding] = []
    for inst in controller.instructions:
        if inst.type not in CHECKED:
            continue
        rb = [x for x in pilot.instructions if x.type == inst.type]
        if not rb:
            findings.append(Finding(inst.type, inst.value, None, "missing"))
        elif any(values_match(inst, x) for x in rb):
            findings.append(Finding(inst.type, inst.value, next(x.value for x in rb if values_match(inst, x)), "match"))
        elif any(x.digits and inst.digits and (inst.digits.startswith(x.digits) or x.digits.startswith(inst.digits)) for x in rb):
            # One side cut short (segment edge, or speaker stopped early): not evidence of an error.
            findings.append(Finding(inst.type, inst.value, rb[0].value, "partial"))
        else:
            findings.append(Finding(inst.type, inst.value, rb[0].value, "mismatch"))
    return findings


def flagged(findings: list[Finding]) -> bool:
    return any(f.status == "mismatch" for f in findings)
