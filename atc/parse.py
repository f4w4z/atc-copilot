"""Rule-based clearance parser: transcript -> {speaker, callsign, instructions}.

    >>> parse("csa one delta zulu descend flight level one hundred").instructions
    [Instruction(type='altitude', value=10000, ...)]

Works on normalised spoken text (see atc.normalize), so it takes reference
transcripts and Whisper output alike. Values are kept in comparable units:
altitude in feet, heading in degrees, frequency in MHz, squawk/QNH as digit strings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from atc.callsign import Callsign, find_callsigns
from atc.normalize import normalize
from atc.numbers import Number, find_number, read_number

ALT_VERBS = {"climb": "climb", "climbing": "climb", "descend": "descend", "descending": "descend",
             "maintain": "maintain", "maintaining": "maintain", "stop": "maintain"}
STATION_WORDS = {"tower", "radar", "ground", "approach", "director", "control", "delivery",
                 "information", "center", "arrival", "departure", "apron"}
CONTACT_VERBS = {"contact", "call", "monitor"}
SKIP = {"to", "and", "level", "flight", "of", "now", "the", "your", "on", "climb", "descend",
        "maintain", "altitude", "please", "via", "immediately", "expedite"}


@dataclass
class Instruction:
    type: str                 # altitude, heading, speed, frequency, squawk, qnh, runway, direct, takeoff, land, line_up
    value: object = None      # feet / degrees / knots / MHz / "7421" / "27L" / waypoint
    detail: str = ""          # climb/descend/maintain, left/right, FL/ft
    digits: str = ""          # spoken digits, for abbreviated readbacks
    start: int = 0            # word span of the value
    end: int = 0


@dataclass
class Parsed:
    text: str
    words: list[str]
    speaker: str                      # "controller", "pilot", "unknown"
    callsign: str | None
    callsigns: list[Callsign] = field(default_factory=list)
    instructions: list[Instruction] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("words")
        d["callsigns"] = [c.icao for c in self.callsigns]
        return d


def _altitude(words: list[str], i: int, num: Number, verb: str) -> Instruction:
    """Interpret a number as an altitude given the words around it."""
    before = words[max(0, num.start - 2) : num.start]
    after = words[num.end : num.end + 1]
    if "level" in before:
        return Instruction("altitude", num.int_value * 100, f"{verb} FL".strip(), num.digits, num.start, num.end)
    if after == ["feet"] or num.int_value >= 1000:
        return Instruction("altitude", num.int_value, f"{verb} ft".strip(), num.digits, num.start, num.end)
    # Bare small numbers after an altitude verb are flight levels in Europe ("climb one two zero").
    return Instruction("altitude", num.int_value * 100, f"{verb} FL".strip(), num.digits, num.start, num.end)


def parse_instructions(words: list[str], skip_spans: list[tuple[int, int]] = ()) -> list[Instruction]:
    out: list[Instruction] = []
    used: set[int] = set()
    for a, b in skip_spans:
        used.update(range(a, b))
    n = len(words)

    def take(inst: Instruction | None) -> None:
        if inst is None or any(k in used for k in range(inst.start, inst.end)):
            return
        used.update(range(inst.start, inst.end))
        out.append(inst)

    for i, w in enumerate(words):
        prev = words[i - 1] if i else ""
        nxt = words[i + 1] if i + 1 < n else ""

        if w in ALT_VERBS and not (w == "stop" and nxt not in ("climb", "descent", "descend")):
            num = find_number(words, i + 1, 4, SKIP)
            if num and num.start not in used:
                take(_altitude(words, i, num, ALT_VERBS[w]))
        elif w == "level" and prev in ("flight", "") or (w == "level" and prev not in ALT_VERBS):
            num = read_number(words, i + 1)
            if num and prev != "passing" and words[max(0, i - 2)] != "passing":
                take(Instruction("altitude", num.int_value * 100, "FL", num.digits, num.start, num.end))
        elif w == "turn" and nxt in ("left", "right"):
            num = find_number(words, i + 2, 2, {"to", "heading", "onto"})
            if num and num.int_value <= 360:
                take(Instruction("heading", num.int_value, nxt, num.digits, num.start, num.end))
        elif w == "heading":
            num = find_number(words, i + 1, 1, {"of"})
            if num and num.int_value <= 360:
                turn = prev if prev in ("left", "right") else (words[i - 2] if i > 1 and words[i - 2] in ("left", "right") else "")
                take(Instruction("heading", num.int_value, turn, num.digits, num.start, num.end))
        elif w == "degrees" and "wind" not in words[max(0, i - 5) : i]:
            # Abbreviated heading readback: "nine zero degrees".
            k = i
            while k > 0 and words[k - 1] in ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"):
                k -= 1
            num = read_number(words, k) if k < i else None
            if num and num.end == i and num.int_value <= 360:
                take(Instruction("heading", num.int_value, "", num.digits, num.start, num.end))
        elif w == "speed" or (w in ("reduce", "increase") and nxt != "speed"):
            num = find_number(words, i + 1, 3, {"to", "speed", "of", "minimum", "maximum"})
            if num and 100 <= num.int_value <= 400:
                take(Instruction("speed", num.int_value, "", num.digits, num.start, num.end))
        elif w in CONTACT_VERBS:
            for k in range(i + 1, min(n, i + 6)):
                num = read_number(words, k)
                if num:
                    if num.decimal:
                        take(Instruction("frequency", round(num.value, 3), "", num.digits, num.start, num.end))
                    break
        elif w == "squawk":
            num = read_number(words, i + 1)
            if num and len(num.digits) <= 4:
                take(Instruction("squawk", num.digits.zfill(4), "", num.digits, num.start, num.end))
        elif w == "qnh":
            num = read_number(words, i + 1)
            if num and 3 <= len(num.digits) <= 4:
                take(Instruction("qnh", num.int_value, "", num.digits, num.start, num.end))
        elif w == "runway":
            num = read_number(words, i + 1)
            if num and num.int_value <= 36 and not num.decimal:
                side = words[num.end] if num.end < n else ""
                suffix = {"left": "L", "right": "R", "center": "C"}.get(side, "")
                end = num.end + (1 if suffix else 0)
                take(Instruction("runway", f"{num.int_value:02d}{suffix}", "", num.digits, num.start, end))
        elif w == "direct" or (w == "proceed" and nxt not in ("direct",)):
            k = i + 1
            while k < n and words[k] in ("to", "direct", "present", "position"):
                k += 1
            if k < n and words[k] not in ("heading", "runway", "and", "via") and read_number(words, k) is None:
                take(Instruction("direct", words[k], "", "", k, k + 1))

        # Clearances without a value of their own.
        two = " ".join(words[i : i + 3])
        if two.startswith("cleared for takeoff") or two.startswith("cleared for take off"):
            out.append(Instruction("takeoff", None, "", "", i, i + 3))
        elif two.startswith("cleared to land"):
            out.append(Instruction("land", None, "", "", i, i + 3))
        elif two.startswith("line up"):
            out.append(Instruction("line_up", None, "", "", i, i + 2))

    # Frequencies said without "contact" (pilot readbacks): decimals in the airband range.
    for i in range(n):
        if i in used:
            continue
        num = read_number(words, i)
        if num and num.decimal and 118 <= num.value < 137 and (i == 0 or words[i - 1] not in
                                                                ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")):
            take(Instruction("frequency", round(num.value, 3), "", num.digits, num.start, num.end))

    out.sort(key=lambda x: x.start)
    return out


def speaker_role(words: list[str], callsigns: list[Callsign]) -> str:
    """Controllers lead with the callsign; pilots usually end with it or address a station first."""
    if not words:
        return "unknown"
    lead = 1 if words[0] in ("and", "ok", "roger") else 0
    starts = bool(callsigns) and callsigns[0].start <= lead
    tail = len(words)
    while tail > 0 and words[tail - 1] in ("bye", "good", "thank", "you", "thanks", "day", "ciao", "servus"):
        tail -= 1
    ends = bool(callsigns) and callsigns[-1].end >= tail and callsigns[-1].start > 0
    addresses_station = len(words) > 1 and words[1] in STATION_WORDS and lead == 0
    if addresses_station:
        return "pilot"
    if starts and not ends:
        return "controller"
    if ends and not starts:
        return "pilot"
    if starts and ends and len(callsigns) == 1:
        return "controller"
    return "unknown"


def parse(text: str) -> Parsed:
    norm = normalize(text)
    words = norm.split()
    callsigns = find_callsigns(words)
    instructions = parse_instructions(words, [(c.start, c.end) for c in callsigns])
    role = speaker_role(words, callsigns)
    if role == "controller" and callsigns:
        cs = callsigns[0].icao
    elif callsigns:
        cs = callsigns[-1].icao
    else:
        cs = None
    return Parsed(norm, words, role, cs, callsigns, instructions)
