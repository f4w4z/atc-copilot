import unittest

from atc.normalize import normalize


class TestNormalize(unittest.TestCase):
    def check(self, hyp: str, ref: str) -> None:
        self.assertEqual(normalize(hyp), normalize(ref))

    def test_reference_is_stable(self):
        ref = "csa one delta zulu descend flight level one hundred no speed restrictions"
        self.assertEqual(normalize(ref), ref)

    def test_flight_level(self):
        self.check("CSA1DZ, descend FL100.", "csa one delta zulu descend flight level one hundred")
        self.check("climb FL240", "climb flight level two four zero")
        self.check("climb flight level 240", "climb flight level two four zero")

    def test_altitudes_and_squawk(self):
        self.check("initial climb 5000 feet, squawk 1000", "initial climb five thousand feet and squawk one thousand".replace(" and", ""))
        self.check("climb 12000", "climb one two thousand")
        self.check("descend 2500 ft", "descend two thousand five hundred ft")
        self.check("squawk 7421", "squawk seven four two one")
        self.check("squawk 0421", "squawk zero four two one")

    def test_frequency(self):
        self.check("contact 124.35", "contact one two four decimal three five")
        self.check("contact one two four point three five", "contact one two four decimal three five")

    def test_runway(self):
        self.check("runway 27L", "runway two seven left")
        self.check("runway 06", "runway zero six")

    def test_alphabet_variants(self):
        self.check("Alpha Juliet niner", "alfa juliett nine")
        self.check("N123AB taxi", "november one two three alfa bravo taxi")

    def test_repeats(self):
        self.check("oscar kilo triple hotel", "oscar kilo hotel hotel hotel")
        self.check("double two", "two two")

    def test_fillers_and_tags(self):
        self.check("uh, roger [hes] thank you", "roger thank you")
        self.check("Okay.", "ok")

    def test_thousands_separator(self):
        self.check("descend 2,000 feet", "descend two thousand feet")
        self.assertEqual(normalize("$60,000"), "six zero thousand")

    def test_contractions_and_spelled_letters(self):
        self.assertEqual(normalize("I'm sorry"), "im sorry")
        self.check("talk to K-T-S", "talk to kilo tango sierra")
        self.check("goodbye", "good bye")
        self.check("X-ray", "xray")

    def test_reference_artifacts_untouched(self):
        for ref in ["oscar kilo foxtrot alfa oscar taxi to holding point runway two seven",
                    "praha radar nine zero whiskey level t", "that s ok", "k oscar zulu"]:
            self.assertEqual(normalize(ref), ref)

    def test_qnh(self):
        self.check("QNH 1014", "qnh one zero one four")


if __name__ == "__main__":
    unittest.main()
