"""Link a callsign heard on the radio to an aircraft tracked on ADS-B.

The transcript is often off by a digit or a letter, and controllers shorten
registrations, so candidates are ranked rather than matched exactly:
exact > abbreviation of the same aircraft > one edit away > airline + flight-number prefix.
"""

from __future__ import annotations

from dataclasses import dataclass

from atc.callsign import same_aircraft


@dataclass
class Candidate:
    callsign: str
    icao24: str
    score: float
    reason: str


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def candidates(heard: str, tracked: dict[str, str], limit: int = 3) -> list[Candidate]:
    """Rank tracked aircraft (callsign -> icao24) against a heard callsign."""
    heard = heard.upper()
    out = []
    for cs, icao24 in tracked.items():
        if cs == heard:
            out.append(Candidate(cs, icao24, 1.0, "exact"))
        elif same_aircraft(cs, heard):
            out.append(Candidate(cs, icao24, 0.8, "abbreviated"))
        else:
            d = _edit_distance(cs, heard)
            same_airline = len(cs) > 3 and len(heard) > 3 and cs[:3] == heard[:3]
            if d == 1 and (same_airline or len(heard) >= 5):
                out.append(Candidate(cs, icao24, 0.6, "one edit"))
            elif d == 2 and same_airline and len(heard) >= 6:
                out.append(Candidate(cs, icao24, 0.3, "two edits"))
    out.sort(key=lambda c: -c.score)
    return out[:limit]


def link(heard: str | None, tracked: dict[str, str]) -> Candidate | None:
    """Best candidate, only if it is unambiguous."""
    if not heard:
        return None
    cs = candidates(heard, tracked)
    if not cs:
        return None
    if len(cs) > 1 and cs[1].score == cs[0].score:
        return None
    return cs[0]
