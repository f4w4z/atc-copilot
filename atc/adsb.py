"""ADS-B sources and an in-memory track store.

Two sources produce the same `TrackPoint`:
- OpenSky REST (`/states/all` with a bounding box). Anonymous: 400 requests/day,
  10 s resolution. With a free account (OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET,
  OAuth2 client credentials): 4,000/day. No selected altitude/heading/QNH.
- A local readsb / dump1090 `aircraft.json` (own receiver): ~1 s updates and, where
  aircraft broadcast them, selected altitude, selected heading and altimeter setting.

Units in TrackPoint: feet, knots, degrees, feet per minute, unix seconds.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass

M_TO_FT = 3.28084
MS_TO_KT = 1.943844
MS_TO_FPM = 196.8504

OPENSKY_STATES = "https://opensky-network.org/api/states/all"
OPENSKY_TOKEN = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"

AIRPORTS = {
    # ICAO: (lat, lon, field elevation ft)
    "LKPR": (50.1008, 14.2600, 1247),  # Prague, matches the ATCO2 / UWB-ATCC corpora
    "LKTB": (49.1513, 16.6944, 778),   # Brno
}


@dataclass
class TrackPoint:
    t: float                      # unix time of the position
    icao24: str
    callsign: str | None
    lat: float | None
    lon: float | None
    alt_baro_ft: float | None     # pressure altitude (1013.25 hPa reference)
    alt_geo_ft: float | None
    on_ground: bool
    gs_kt: float | None
    track_deg: float | None       # ground track, true
    vrate_fpm: float | None
    squawk: str | None = None
    sel_alt_ft: float | None = None   # autopilot selected altitude (own receiver only)
    sel_hdg_deg: float | None = None
    qnh_hpa: float | None = None
    source: str = ""


def bbox(lat: float, lon: float, radius_nm: float) -> tuple[float, float, float, float]:
    dlat = radius_nm / 60.0
    dlon = radius_nm / (60.0 * max(math.cos(math.radians(lat)), 0.01))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def _clean_callsign(cs) -> str | None:
    cs = (cs or "").strip().upper()
    return cs or None


def parse_opensky(payload: dict) -> list[TrackPoint]:
    out = []
    for s in payload.get("states") or []:
        def f(i, scale=1.0):
            return None if s[i] is None else s[i] * scale
        out.append(TrackPoint(
            t=float(s[3] or s[4] or payload.get("time")), icao24=s[0], callsign=_clean_callsign(s[1]),
            lat=s[6], lon=s[5], alt_baro_ft=f(7, M_TO_FT), alt_geo_ft=f(13, M_TO_FT),
            on_ground=bool(s[8]), gs_kt=f(9, MS_TO_KT), track_deg=s[10], vrate_fpm=f(11, MS_TO_FPM),
            squawk=s[14], source="opensky"))
    return out


def parse_readsb(payload: dict) -> list[TrackPoint]:
    """readsb / dump1090 aircraft.json. alt_baro is "ground" when on the ground."""
    now = float(payload.get("now", time.time()))
    out = []
    for a in payload.get("aircraft", []):
        if "lat" not in a and "alt_baro" not in a:
            continue
        alt = a.get("alt_baro")
        on_ground = alt == "ground"
        out.append(TrackPoint(
            t=now - float(a.get("seen_pos", a.get("seen", 0)) or 0), icao24=a.get("hex", "").lstrip("~"),
            callsign=_clean_callsign(a.get("flight")), lat=a.get("lat"), lon=a.get("lon"),
            alt_baro_ft=None if on_ground or alt is None else float(alt),
            alt_geo_ft=a.get("alt_geom"), on_ground=on_ground, gs_kt=a.get("gs"),
            track_deg=a.get("track"), vrate_fpm=a.get("baro_rate", a.get("geom_rate")),
            squawk=a.get("squawk"), sel_alt_ft=a.get("nav_altitude_mcp"),
            sel_hdg_deg=a.get("nav_heading"), qnh_hpa=a.get("nav_qnh"), source="readsb"))
    return out


class OpenSky:
    def __init__(self, client_id: str | None = None, client_secret: str | None = None, timeout: float = 30):
        self.client_id = client_id or os.environ.get("OPENSKY_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("OPENSKY_CLIENT_SECRET")
        self.timeout = timeout
        self._token: str | None = None
        self._token_exp = 0.0
        self.remaining: int | None = None

    @property
    def authenticated(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _auth_header(self) -> dict:
        if not self.authenticated:
            return {}
        if time.time() > self._token_exp - 60:
            body = urllib.parse.urlencode({"grant_type": "client_credentials", "client_id": self.client_id,
                                           "client_secret": self.client_secret}).encode()
            with urllib.request.urlopen(urllib.request.Request(OPENSKY_TOKEN, data=body), timeout=self.timeout) as r:
                tok = json.load(r)
            self._token = tok["access_token"]
            self._token_exp = time.time() + float(tok.get("expires_in", 1800))
        return {"Authorization": f"Bearer {self._token}"}

    def states(self, lamin: float, lomin: float, lamax: float, lomax: float) -> tuple[float, list[TrackPoint]]:
        q = urllib.parse.urlencode({"lamin": lamin, "lomin": lomin, "lamax": lamax, "lomax": lomax})
        req = urllib.request.Request(f"{OPENSKY_STATES}?{q}", headers=self._auth_header())
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            rem = r.headers.get("X-Rate-Limit-Remaining")
            self.remaining = int(rem) if rem is not None else None
            payload = json.load(r)
        return float(payload["time"]), parse_opensky(payload)


class TrackStore:
    """Recent points per aircraft, indexed by icao24 and callsign."""

    def __init__(self, max_age_s: float = 1800):
        self.max_age_s = max_age_s
        self.points: dict[str, list[TrackPoint]] = defaultdict(list)
        self.callsign_of: dict[str, str] = {}

    def add(self, pts: list[TrackPoint]) -> None:
        for p in pts:
            series = self.points[p.icao24]
            if series and series[-1].t >= p.t:
                continue  # duplicate or out of order
            series.append(p)
            if p.callsign:
                self.callsign_of[p.icao24] = p.callsign

    def prune(self, now: float) -> None:
        for k in list(self.points):
            self.points[k] = [p for p in self.points[k] if now - p.t <= self.max_age_s]
            if not self.points[k]:
                del self.points[k]
                self.callsign_of.pop(k, None)

    def callsigns(self) -> dict[str, str]:
        """callsign -> icao24 for aircraft currently tracked."""
        return {cs: k for k, cs in self.callsign_of.items() if k in self.points}

    def track(self, icao24: str, t0: float | None = None, t1: float | None = None) -> list[TrackPoint]:
        return [p for p in self.points.get(icao24, [])
                if (t0 is None or p.t >= t0) and (t1 is None or p.t <= t1)]

    @staticmethod
    def to_json(p: TrackPoint) -> str:
        return json.dumps(asdict(p), separators=(",", ":"))
