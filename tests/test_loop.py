import unittest

from atc.adsb import TrackPoint, TrackStore, parse_opensky, parse_readsb
from atc.conformance import check_altitude, check_heading
from atc.link import link
from atc.loop import Monitor
from atc.parse import Instruction

T0 = 1_000_000.0


def pt(t, alt, vrate=0.0, trk=90.0, cs="CSA1DZ", icao="49d0aa", sel=None, sel_hdg=None):
    return TrackPoint(T0 + t, icao, cs, 50.0, 14.0, alt, alt, False, 250.0, trk, vrate,
                      sel_alt_ft=sel, sel_hdg_deg=sel_hdg)


def descent(target=10000, start=14000, rate=-1500, t_end=240, dt=10, delay=10):
    out, alt = [], start
    for t in range(-30, t_end + 1, dt):
        v = 0.0 if t < delay or alt <= target else rate
        out.append(pt(t, alt, v))
        if t >= delay and alt > target:
            alt = max(target, alt + rate * dt / 60)
    return out


FL100 = Instruction("altitude", 10000, "descend FL")


class TestSources(unittest.TestCase):
    def test_opensky(self):
        payload = {"time": 1795585890, "states": [
            ["471d72", "WZZ5XE  ", "Hungary", 1795585890, 1795585890, 14.93, 50.08, 2872.74,
             False, 138.61, 300.56, -13.98, None, 2895.6, "1000", False, 0]]}
        p = parse_opensky(payload)[0]
        self.assertEqual(p.callsign, "WZZ5XE")
        self.assertAlmostEqual(p.alt_baro_ft, 9425, delta=2)
        self.assertAlmostEqual(p.vrate_fpm, -2752, delta=2)

    def test_readsb(self):
        payload = {"now": 100.0, "aircraft": [
            {"hex": "49d0aa", "flight": "CSA1DZ ", "alt_baro": 14000, "baro_rate": -1200, "lat": 50, "lon": 14,
             "track": 90, "gs": 280, "nav_altitude_mcp": 10016, "nav_heading": 270.0, "nav_qnh": 1014.2, "seen_pos": 0.5},
            {"hex": "49d0ab", "alt_baro": "ground", "lat": 50.1, "lon": 14.26}]}
        a, b = parse_readsb(payload)
        self.assertEqual((a.callsign, a.sel_alt_ft, a.qnh_hpa), ("CSA1DZ", 10016, 1014.2))
        self.assertTrue(b.on_ground)


class TestLink(unittest.TestCase):
    tracked = {"CSA1DZ": "a", "DLH4703": "b", "RYR92BQ": "c", "OKFAO": "d"}

    def test_exact_and_fuzzy(self):
        self.assertEqual(link("CSA1DZ", self.tracked).icao24, "a")
        self.assertEqual(link("DLH4708", self.tracked).icao24, "b")     # one digit misheard
        self.assertEqual(link("FAO", self.tracked).icao24, "d")         # abbreviated registration
        self.assertIsNone(link("AFR123", self.tracked))


class TestConformance(unittest.TestCase):
    def test_conforming_descent(self):
        v = check_altitude(FL100, T0, descent(), now=T0 + 240)
        self.assertEqual(v.status, "conforming")

    def test_on_the_way(self):
        v = check_altitude(FL100, T0, descent(t_end=60), now=T0 + 60)
        self.assertEqual(v.status, "pending")

    def test_not_started(self):
        v = check_altitude(FL100, T0, [pt(t, 14000) for t in range(-30, 121, 10)], now=T0 + 120)
        self.assertEqual(v.status, "not_started")

    def test_wrong_direction(self):
        pts = [pt(t, 14000 + max(0, t) * 10, 600) for t in range(-30, 121, 10)]
        self.assertEqual(check_altitude(FL100, T0, pts, now=T0 + 120).status, "wrong_direction")

    def test_levelled_wrong(self):
        pts = descent(target=11000)  # levels at FL110 instead of FL100
        self.assertEqual(check_altitude(FL100, T0, pts, now=T0 + 240).status, "levelled_wrong")

    def test_selected_altitude_mismatch(self):
        pts = [pt(t, 14000, sel=11000 if t > 0 else 14000) for t in range(-30, 61, 10)]
        self.assertEqual(check_altitude(FL100, T0, pts, now=T0 + 60).status, "selected_mismatch")

    def test_qnh_correction(self):
        # 3000 ft on QNH 1033 is ~2466 ft pressure altitude.
        inst = Instruction("altitude", 3000, "descend ft")
        pts = [pt(t, 2470) for t in range(0, 61, 10)]
        self.assertEqual(check_altitude(inst, T0 - 1, pts, now=T0 + 60, qnh_hpa=1033).status, "conforming")

    def test_heading(self):
        inst = Instruction("heading", 270, "right")
        right_turn = [pt(t, 5000, trk=(90 + max(0, t) * 3) % 360) for t in range(-10, 61, 10)]
        self.assertEqual(check_heading(inst, T0, right_turn, now=T0 + 60).status, "conforming")
        left_wrong = Instruction("heading", 180, "right")
        left_turn = [pt(t, 5000, trk=(90 - max(0, t) * 1.0) % 360) for t in range(-10, 61, 10)]
        self.assertEqual(check_heading(left_wrong, T0, left_turn, now=T0 + 60).status, "wrong_direction")


class TestMonitor(unittest.TestCase):
    def test_full_loop(self):
        store = TrackStore()
        store.add(descent(target=11000))  # aircraft levels at FL110
        mon = Monitor(store)
        ev = mon.on_transmission(T0, "csa one delta zulu descend flight level one hundred")
        self.assertEqual(ev[0].icao24, "49d0aa")
        ev += mon.on_transmission(T0 + 4, "descending flight level one one zero csa one delta zulu")
        self.assertIn("readback_mismatch", [e.kind for e in ev])
        flags = mon.poll(T0 + 240)
        self.assertEqual([(e.kind, e.detail["status"]) for e in flags], [("nonconformance", "levelled_wrong")])


if __name__ == "__main__":
    unittest.main()
