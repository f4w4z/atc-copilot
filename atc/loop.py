"""The end-to-end loop: transmission -> parse -> link to ADS-B -> readback and conformance checks.

    mon = Monitor(store, qnh_hpa=1014, mag_var_deg=5)
    events = mon.on_transmission(t, "csa one delta zulu descend flight level one hundred")
    events += mon.poll(now)          # call periodically as ADS-B arrives

Each event says what was heard, which aircraft it was linked to, and what the
aircraft did. Clearances stay open until they conform, get flagged, or time out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from atc.adsb import TrackStore
from atc.conformance import CHECKS, Verdict, check_altitude, check_heading
from atc.link import link
from atc.parse import Instruction, Parsed, parse
from atc.readback import check as readback_check

READBACK_WINDOW_S = 20
CLEARANCE_TIMEOUT_S = 600


@dataclass
class Event:
    t: float
    kind: str              # "transmission", "readback_mismatch", "nonconformance", "conformed", "unlinked"
    callsign: str | None
    icao24: str | None
    text: str
    detail: dict = field(default_factory=dict)


@dataclass
class OpenClearance:
    t: float
    parsed: Parsed
    inst: Instruction
    callsign: str
    icao24: str
    last: Verdict | None = None


class Monitor:
    def __init__(self, store: TrackStore, qnh_hpa: float | None = None, mag_var_deg: float = 0.0):
        self.store = store
        self.qnh_hpa = qnh_hpa
        self.mag_var_deg = mag_var_deg
        self.open: list[OpenClearance] = []
        self.last_controller: tuple[float, Parsed] | None = None

    def on_transmission(self, t: float, text: str) -> list[Event]:
        p = parse(text)
        cand = link(p.callsign, self.store.callsigns())
        ev = [Event(t, "transmission", cand.callsign if cand else p.callsign, cand.icao24 if cand else None, p.text,
                    {"speaker": p.speaker, "instructions": [(i.type, i.value) for i in p.instructions],
                     "link": cand.reason if cand else None})]

        if p.speaker == "pilot" and self.last_controller and t - self.last_controller[0] <= READBACK_WINDOW_S:
            ctrl = self.last_controller[1]
            same = not p.callsign or not ctrl.callsign or link(p.callsign, {ctrl.callsign: ""}) is not None
            if same:
                bad = [f for f in readback_check(ctrl, p) if f.status == "mismatch"]
                if bad:
                    ev.append(Event(t, "readback_mismatch", ctrl.callsign, None, p.text,
                                    {"controller": ctrl.text, "findings": [(f.type, f.said, f.readback) for f in bad]}))

        if p.speaker != "pilot":
            self.last_controller = (t, p)
            checkable = [i for i in p.instructions if i.type in CHECKS]
            if checkable and cand is None and p.callsign:
                ev.append(Event(t, "unlinked", p.callsign, None, p.text, {"tracked": len(self.store.callsigns())}))
            if cand:
                for inst in checkable:
                    self.open.append(OpenClearance(t, p, inst, cand.callsign, cand.icao24))
        return ev

    def poll(self, now: float) -> list[Event]:
        events = []
        still_open = []
        for c in self.open:
            pts = self.store.track(c.icao24, t0=c.t - 120)
            if c.inst.type == "altitude":
                v = check_altitude(c.inst, c.t, pts, now, qnh_hpa=self.qnh_hpa)
            else:
                v = check_heading(c.inst, c.t, pts, now, mag_var_deg=self.mag_var_deg)
            c.last = v
            info = {"instruction": (c.inst.type, c.inst.value, c.inst.detail), "status": v.status,
                    "why": v.detail, **v.evidence, "heard": c.parsed.text}
            if v.flagged:
                events.append(Event(now, "nonconformance", c.callsign, c.icao24, c.parsed.text, info))
            elif v.status == "conforming":
                events.append(Event(now, "conformed", c.callsign, c.icao24, c.parsed.text, info))
            elif now - c.t < CLEARANCE_TIMEOUT_S:
                still_open.append(c)
        self.open = still_open
        return events
