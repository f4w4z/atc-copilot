"""Find spoken callsigns and map them to ICAO form.

"lufthansa four seven zero three" -> DLH4703
"csa one delta zulu"              -> CSA1DZ
"oscar kilo foxtrot alfa oscar"   -> OKFAO (registration)

The telephony table covers the airlines that are frequent in the public corpora
(mostly European, 1990s-2020s). Unknown telephony words are not guessed; the live
system will match against callsigns actually seen on ADS-B instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from atc.numbers import DIGIT

NATO = {
    "alfa": "A", "bravo": "B", "charlie": "C", "delta": "D", "echo": "E", "foxtrot": "F",
    "golf": "G", "hotel": "H", "india": "I", "juliett": "J", "kilo": "K", "lima": "L",
    "mike": "M", "november": "N", "oscar": "O", "papa": "P", "quebec": "Q", "romeo": "R",
    "sierra": "S", "tango": "T", "uniform": "U", "victor": "V", "whiskey": "W",
    "xray": "X", "yankee": "Y", "zulu": "Z",
}

# Spoken telephony -> ICAO airline designator.
TELEPHONY = {
    "lufthansa": "DLH", "csa": "CSA", "alitalia": "AZA", "sky travel": "TVS",
    "air france": "AFR", "sabena": "SAB", "austrian": "AUA", "swiss air": "SWR",
    "swiss": "SWR", "speed bird": "BAW", "wizz air": "WZZ", "klm": "KLM",
    "aero lloyd": "AEF", "air berlin": "BER", "hapag lloyd": "HLF", "lot": "LOT",
    "aero flot": "AFL", "aeroflot": "AFL", "easy": "EZY", "nor shuttle": "NAX",
    "ryan air": "RYR", "ryanair": "RYR", "transwede": "TWE", "air malta": "AMC",
    "turkish": "THY", "german wings": "GWI", "britannia": "BAL", "crossair": "CRX",
    "tarom": "ROT", "emirates": "UAE", "el al": "ELY", "scandinavian": "SAS",
    "cimber": "CIM", "sunwing": "SWG", "condor": "CFG", "europa": "AEA",
    "gulf air": "GFA", "saudia": "SVA", "finn air": "FIN", "finnair": "FIN",
    "malaysian": "MAS", "sobelair": "SLR", "ltu": "LTU", "lux air": "LGL",
    "shamrock": "EIN", "iberia": "IBE", "midland": "BMA", "trans avia": "TRA",
    "transavia": "TRA", "belgian airforce": "BAF", "malev": "MAH", "reach": "RCH",
    "adria": "ADR", "qatari": "QTR", "united": "UAL", "air inter": "ITF",
    "olympic": "OAL", "euro wings": "EWG", "eurowings": "EWG", "bel avia": "BRU",
    "sun express": "SXS", "germania": "GMI", "algerie": "DAH", "donavia": "DNV",
    "smart lynx": "ART", "delta": "DAL", "american": "AAL",
    "brickyard": "RPA", "cactus": "AWE", "southwest": "SWA", "jet blue": "JBU",
    "speedbird": "BAW",
}
_MAX_TEL_WORDS = max(len(k.split()) for k in TELEPHONY)
_MAX_SUFFIX = 6
_MIN_REG_LETTERS = 3


@dataclass(frozen=True)
class Callsign:
    icao: str        # DLH4703, OKFAO
    spoken: str      # words as heard
    start: int       # word span [start, end)
    end: int
    kind: str        # "airline" or "registration"


def _suffix(words: list[str], j: int) -> tuple[str, int]:
    out = ""
    k = j
    while k < len(words) and k - j < _MAX_SUFFIX:
        w = words[k]
        if w in DIGIT:
            out += str(DIGIT[w])
        elif w in NATO:
            out += NATO[w]
        else:
            break
        k += 1
    return out, k


def read_callsign(words: list[str], i: int) -> Callsign | None:
    """Read a callsign starting exactly at words[i]."""
    # Airline: telephony (longest match first) followed by a flight-number suffix.
    for L in range(_MAX_TEL_WORDS, 0, -1):
        tel = " ".join(words[i : i + L])
        if len(words[i : i + L]) == L and tel in TELEPHONY:
            suffix, k = _suffix(words, i + L)
            if suffix and suffix[0].isdigit():
                return Callsign(TELEPHONY[tel] + suffix, " ".join(words[i:k]), i, k, "airline")
    # Registration: a run of >= 3 phonetic letters, optionally followed by digits.
    k = i
    letters = ""
    while k < len(words) and words[k] in NATO and len(letters) < 6:
        letters += NATO[words[k]]
        k += 1
    if len(letters) >= _MIN_REG_LETTERS:
        return Callsign(letters, " ".join(words[i:k]), i, k, "registration")
    return None


def find_callsigns(words: list[str]) -> list[Callsign]:
    """All non-overlapping callsigns, left to right."""
    out: list[Callsign] = []
    i = 0
    while i < len(words):
        cs = read_callsign(words, i)
        if cs:
            out.append(cs)
            i = cs.end
        else:
            i += 1
    return out


def same_aircraft(a: str, b: str) -> bool:
    """Callsigns match if equal, or one is the abbreviated form of the other.

    Pilots and controllers shorten registrations to the first and last letters
    (OKFAO -> OAO, OKFAO -> FAO), and airline suffixes are sometimes cut short.
    """
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    if len(short) < 2:
        return False
    return long_.endswith(short) or (long_[0] == short[0] and long_.endswith(short[1:]))
