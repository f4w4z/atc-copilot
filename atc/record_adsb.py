"""Record ADS-B around an airport to daily JSONL files, for building and replaying the loop.

    python -m atc.record_adsb --airport LKPR                       # OpenSky, anonymous
    python -m atc.record_adsb --airport LKPR --interval 30          # with OPENSKY_CLIENT_ID/SECRET set
    python -m atc.record_adsb --airport LKPR --readsb http://pi.local/tar1090/data/aircraft.json --interval 1

Writes data/adsb/<airport>/<YYYY-MM-DD>.jsonl, one TrackPoint per line (UTC dates).
Anonymous OpenSky allows 400 requests/day, so the default interval is 240 s; the
recorder slows down further if the remaining quota runs low.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from atc.adsb import AIRPORTS, OpenSky, TrackStore, bbox, parse_readsb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--airport", default="LKPR", choices=sorted(AIRPORTS))
    p.add_argument("--radius-nm", type=float, default=40)
    p.add_argument("--interval", type=float, default=None, help="seconds between polls")
    p.add_argument("--readsb", default=None, help="URL or path of a readsb aircraft.json")
    p.add_argument("--out-dir", default="data/adsb")
    p.add_argument("--max-polls", type=int, default=None, help="stop after N polls (testing)")
    args = p.parse_args()

    lat, lon, _ = AIRPORTS[args.airport]
    box = bbox(lat, lon, args.radius_nm)
    out_dir = Path(args.out_dir) / args.airport
    out_dir.mkdir(parents=True, exist_ok=True)
    sky = None if args.readsb else OpenSky()
    interval = args.interval or (1.0 if args.readsb else (30.0 if sky.authenticated else 240.0))
    print(f"recording {args.airport} r={args.radius_nm} nm every {interval:.0f} s from "
          f"{'readsb' if args.readsb else 'OpenSky (' + ('authenticated' if sky.authenticated else 'anonymous') + ')'}",
          flush=True)

    seen = TrackStore(max_age_s=3600)
    polls = 0
    while args.max_polls is None or polls < args.max_polls:
        t0 = time.time()
        try:
            if args.readsb:
                src = args.readsb
                if src.startswith("http"):
                    with urllib.request.urlopen(src, timeout=10) as r:
                        payload = json.load(r)
                else:
                    payload = json.loads(Path(src).read_text())
                pts = parse_readsb(payload)
            else:
                _, pts = sky.states(*box)
            before = sum(len(v) for v in seen.points.values())
            seen.add(pts)
            fresh = sum(len(v) for v in seen.points.values()) - before
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            with open(out_dir / f"{day}.jsonl", "a", encoding="utf-8") as f:
                for pt in pts:
                    f.write(TrackStore.to_json(pt) + "\n")
            seen.prune(t0)
            rem = "" if sky is None or sky.remaining is None else f" quota left {sky.remaining}"
            print(f"{datetime.now(timezone.utc):%H:%M:%S} {len(pts)} aircraft, {fresh} new points{rem}", flush=True)
            if sky is not None and sky.remaining is not None and sky.remaining < 20:
                interval = max(interval, 900.0)
        except Exception as e:  # network hiccups should not stop a multi-day recording
            print(f"{datetime.now(timezone.utc):%H:%M:%S} error: {e}", flush=True)
        polls += 1
        if args.max_polls is not None and polls >= args.max_polls:
            break
        time.sleep(max(0.0, interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
