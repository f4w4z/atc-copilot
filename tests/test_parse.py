import unittest

from atc.parse import parse
from atc.readback import check, flagged


def inst(text):
    return [(i.type, i.value) for i in parse(text).instructions]


class TestParse(unittest.TestCase):
    def test_altitude_and_callsign(self):
        p = parse("csa one delta zulu descend flight level one hundred")
        self.assertEqual((p.speaker, p.callsign), ("controller", "CSA1DZ"))
        self.assertEqual(inst("csa one delta zulu descend flight level one hundred"), [("altitude", 10000)])
        self.assertEqual(inst("initial climb five thousand feet"), [("altitude", 5000)])

    def test_pilot_readback(self):
        p = parse("descending flight level one hundred free speed csa one delta zulu")
        self.assertEqual((p.speaker, p.callsign), ("pilot", "CSA1DZ"))

    def test_heading(self):
        self.assertEqual(inst("ryan air nine two bravo quebec turn right heading zero nine zero"), [("heading", 90)])
        self.assertEqual(inst("nine zero degrees ryan air nine two bravo quebec"), [("heading", 90)])
        self.assertEqual(inst("runway three one cleared for takeoff wind one six zero degrees five knots"),
                         [("runway", "31"), ("takeoff", None)])

    def test_frequency_squawk_qnh(self):
        self.assertEqual(inst("csa seven three one contact ruzyne ground one two one decimal nine"), [("frequency", 121.9)])
        self.assertEqual(inst("squawk seven thousand"), [("squawk", "7000")])
        self.assertEqual(inst("qnh one zero one four"), [("qnh", 1014)])

    def test_flight_level_runs_into_callsign(self):
        self.assertEqual(inst("descend flight level two nine zero four six echo"), [("altitude", 29000)])
        self.assertEqual(inst("flight level two nine zero two thousand or greater"), [("altitude", 29000)])


class TestReadback(unittest.TestCase):
    def pair(self, atc, pilot):
        return check(parse(atc), parse(pilot))

    def test_match(self):
        self.assertFalse(flagged(self.pair("csa one delta zulu descend flight level one hundred",
                                           "descending flight level one hundred csa one delta zulu")))

    def test_mismatch(self):
        fs = self.pair("csa one delta zulu descend flight level one hundred",
                       "descending flight level one one zero csa one delta zulu")
        self.assertTrue(flagged(fs))

    def test_truncated_readback_is_partial(self):
        fs = self.pair("climb flight level three four zero", "climbing flight level three")
        self.assertEqual([f.status for f in fs], ["partial"])

    def test_abbreviated_frequency(self):
        self.assertFalse(flagged(self.pair("contact ruzyne ground one two one decimal nine", "one two one decimal nine")))


if __name__ == "__main__":
    unittest.main()
