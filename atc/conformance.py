"""Does the aircraft do what it was cleared to do?

Altitude: ADS-B barometric altitude is pressure altitude (1013.25 hPa). Flight
levels compare directly; altitudes in feet are on local QNH, so the target is
converted to pressure altitude first (about 27 ft per hPa near the ground).
Heading: ADS-B gives true ground track; clearances are magnetic headings, and wind
adds drift, so headings compare with the magnetic variation applied and a tolerance.
If the aircraft broadcasts its selected altitude / heading (own receiver), that is
checked first: a wrong selection shows up seconds after the clearance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from atc.adsb import TrackPoint
from atc.parse import Instruction

FT_PER_HPA = 27.0
STD_HPA = 1013.25


@dataclass
class Verdict:
    status: str   # conforming, pending, not_started, wrong_direction, levelled_wrong, selected_mismatch, no_data
    detail: str = ""
    evidence: dict = field(default_factory=dict)

    @property
    def flagged(self) -> bool:
        return self.status in ("not_started", "wrong_direction", "levelled_wrong", "selected_mismatch")


def target_pressure_alt(inst: Instruction, qnh_hpa: float | None) -> tuple[float, str]:
    if "FL" in inst.detail:
        return float(inst.value), "flight level, no correction"
    if qnh_hpa is None:
        return float(inst.value), "altitude on QNH, no QNH available: uncorrected"
    return float(inst.value) - (qnh_hpa - STD_HPA) * FT_PER_HPA, f"altitude corrected for QNH {qnh_hpa:.0f}"


def _before(pts: list[TrackPoint], t: float) -> TrackPoint | None:
    prior = [p for p in pts if p.t <= t and p.alt_baro_ft is not None]
    return prior[-1] if prior else None


def check_altitude(inst: Instruction, t_clear: float, pts: list[TrackPoint], now: float,
                   qnh_hpa: float | None = None, start_window_s: float = 60, tol_ft: float = 250,
                   level_rate_fpm: float = 200) -> Verdict:
    target, note = target_pressure_alt(inst, qnh_hpa)
    pts = [p for p in pts if p.alt_baro_ft is not None]
    p0 = _before(pts, t_clear) or next((p for p in pts if p.t >= t_clear), None)
    if p0 is None:
        return Verdict("no_data", "no altitude data for this aircraft")
    after = [p for p in pts if p.t > t_clear]
    ev = {"target_pressure_alt_ft": round(target), "alt_at_clearance_ft": round(p0.alt_baro_ft), "note": note}

    # Selected altitude (MCP/FCU value is the dialled number: 24000 for FL240, 5000 for 5000 ft).
    sel = [p for p in after if p.sel_alt_ft is not None and p.t - t_clear >= 20]
    if sel:
        last_sel = sel[-1].sel_alt_ft
        ev["selected_alt_ft"] = last_sel
        if abs(last_sel - float(inst.value)) > 100 and sel[-1].t - t_clear >= 30:
            return Verdict("selected_mismatch", f"selected {last_sel:.0f} ft, cleared {inst.value}", ev)

    need = target - p0.alt_baro_ft
    direction = 0 if abs(need) <= tol_ft else (1 if need > 0 else -1)
    if not after:
        return Verdict("pending", "no track updates since the clearance yet", ev)
    cur = after[-1]
    ev["alt_now_ft"] = round(cur.alt_baro_ft)

    if abs(cur.alt_baro_ft - target) <= tol_ft:
        return Verdict("conforming", "at the cleared level", ev)

    moved = cur.alt_baro_ft - p0.alt_baro_ft
    rate = cur.vrate_fpm or 0.0
    if direction != 0 and moved * direction < -300:
        return Verdict("wrong_direction", f"moved {moved:+.0f} ft, needed {need:+.0f} ft", ev)

    # Levelled off somewhere other than the target?
    recent = [p for p in after if cur.t - p.t <= 30]
    level = len(recent) >= 2 and all(abs(p.vrate_fpm or 0) < level_rate_fpm for p in recent) \
        and max(p.alt_baro_ft for p in recent) - min(p.alt_baro_ft for p in recent) < 100
    if level and abs(moved) > tol_ft:
        return Verdict("levelled_wrong", f"level at {cur.alt_baro_ft:.0f} ft, target {target:.0f} ft", ev)

    started = moved * direction > 100 or rate * direction > 300
    if direction != 0 and not started and now - t_clear > start_window_s:
        return Verdict("not_started", f"no {'climb' if direction > 0 else 'descent'} {now - t_clear:.0f} s after clearance", ev)
    return Verdict("pending", "on the way" if started else "waiting for the aircraft to respond", ev)


def _ang(a: float, b: float) -> float:
    """Signed smallest difference b - a in degrees, in (-180, 180]."""
    return (b - a + 180) % 360 - 180


def check_heading(inst: Instruction, t_clear: float, pts: list[TrackPoint], now: float,
                  mag_var_deg: float = 0.0, start_window_s: float = 60, tol_deg: float = 20) -> Verdict:
    target_mag = float(inst.value) % 360
    target_true = (target_mag + mag_var_deg) % 360
    pts = [p for p in pts if p.track_deg is not None and not p.on_ground]
    prior = [p for p in pts if p.t <= t_clear]
    after = [p for p in pts if p.t > t_clear]
    if not prior and not after:
        return Verdict("no_data", "no track data for this aircraft")
    p0 = prior[-1] if prior else after[0]
    ev = {"target_true_deg": round(target_true), "track_at_clearance_deg": round(p0.track_deg)}

    sel = [p for p in after if p.sel_hdg_deg is not None and p.t - t_clear >= 30]
    if sel:
        ev["selected_hdg_deg"] = sel[-1].sel_hdg_deg
        if abs(_ang(sel[-1].sel_hdg_deg, target_mag)) > 5:
            return Verdict("selected_mismatch", f"selected heading {sel[-1].sel_hdg_deg:.0f}, cleared {target_mag:.0f}", ev)

    if not after:
        return Verdict("pending", "no track updates since the clearance yet", ev)
    cur = after[-1]
    ev["track_now_deg"] = round(cur.track_deg)
    if abs(_ang(cur.track_deg, target_true)) <= tol_deg:
        return Verdict("conforming", "on the cleared heading (within wind tolerance)", ev)

    turned = _ang(p0.track_deg, cur.track_deg)
    commanded = {"left": -1, "right": 1}.get(inst.detail, 0)
    if commanded and turned * commanded < -30:
        return Verdict("wrong_direction", f"turned {turned:+.0f} deg, cleared {inst.detail}", ev)
    if abs(turned) < 10 and now - t_clear > start_window_s:
        return Verdict("not_started", f"no turn {now - t_clear:.0f} s after clearance", ev)
    return Verdict("pending", "turning" if abs(turned) >= 10 else "waiting for the aircraft to respond", ev)


CHECKS = {"altitude": check_altitude, "heading": check_heading}
