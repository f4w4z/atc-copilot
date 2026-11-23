"""Text normalisation for ATC transcripts.

References in ATCO2, UWB-ATCC and ATCOSIM are lowercase spoken forms
("descend flight level one hundred", "squawk one thousand"). Whisper tends to
write numerals and abbreviations ("FL100", "squawk 1000", "27L"). WER is only
meaningful if both sides go through the same normaliser, so every evaluation in
this repo uses `normalize()` on references and hypotheses alike.

Canonical form: lowercase spoken words, ICAO alphabet spellings (alfa,
juliett), digits spoken one by one except round hundreds and thousands
("one hundred", "five thousand"), "decimal" for the frequency point.
"""

from __future__ import annotations

import re

# Applied to both sides after tokenising. Maps variants onto one spelling.
VARIANTS = {
    "alpha": "alfa",
    "juliet": "juliett",
    "niner": "nine",
    "tree": "three",
    "fife": "five",
    "okay": "ok",
    "speedbird": "speed bird",
    "rwy": "runway",
    "hdg": "heading",
    "qnh's": "qnh",
    "centre": "center",
    "goodbye": "bye",
    "good-bye": "bye",
}

# Two-word expansions applied to single tokens.
EXPANSIONS = {
    "fl": ["flight", "level"],
    "ils": ["ils"],
}

# Multi-word variants, applied last.
PHRASES = [("good bye", "bye")]

FILLERS = {"uh", "um", "ah", "eh", "er", "hm", "hmm", "mhm", "erm"}

DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

NATO = {
    "a": "alfa", "b": "bravo", "c": "charlie", "d": "delta", "e": "echo",
    "f": "foxtrot", "g": "golf", "h": "hotel", "i": "india", "j": "juliett",
    "k": "kilo", "l": "lima", "m": "mike", "n": "november", "o": "oscar",
    "p": "papa", "q": "quebec", "r": "romeo", "s": "sierra", "t": "tango",
    "u": "uniform", "v": "victor", "w": "whiskey", "x": "xray", "y": "yankee",
    "z": "zulu",
}

RUNWAY_SUFFIX = {"l": "left", "r": "right", "c": "center"}

_BRACKETED = re.compile(r"\[[^\]]*\]|<[^>]*>|\([^)]*\)")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
_PUNCT = re.compile(r"[^\w\s']")
_ALNUM_SPLIT = re.compile(r"\d+|[a-z]+")


def _say_digits(digits: str) -> list[str]:
    return [DIGITS[int(d)] for d in digits]


def _say_integer(num: str) -> list[str]:
    """Spoken form of a run of digits, following ATC conventions."""
    if len(num) > 1 and num.startswith("0"):
        return _say_digits(num)  # squawks, headings: "zero niner zero"
    if re.fullmatch(r"[1-9]0{3}", num):
        return [DIGITS[int(num[0])], "thousand"]
    if re.fullmatch(r"[1-9]\d0{3}", num):
        return _say_digits(num[:2]) + ["thousand"]  # 12000 -> one two thousand
    if re.fullmatch(r"[1-9]\d?[1-9]00", num) and len(num) == 4:
        # 2500 -> two thousand five hundred
        return [DIGITS[int(num[0])], "thousand", DIGITS[int(num[1])], "hundred"]
    if re.fullmatch(r"[1-9]00", num):
        return [DIGITS[int(num[0])], "hundred"]
    return _say_digits(num)


def _expand_token(tok: str) -> list[str]:
    if tok in VARIANTS:
        return [VARIANTS[tok]]
    if tok in EXPANSIONS:
        return list(EXPANSIONS[tok])
    if tok.isalpha():
        return [tok]
    if tok.isdigit():
        return _say_integer(tok)
    # Mixed letters and digits: FL240, 27L, DLH4AB, 124.35 (already split on '.')
    parts = _ALNUM_SPLIT.findall(tok)
    out: list[str] = []
    for i, part in enumerate(parts):
        if part.isdigit():
            out += _say_integer(part)
        elif part == "fl" and i == 0:
            out += ["flight", "level"]
        elif i == 0 and len(part) == 3 and len(parts) > 1:
            out.append(part)  # ICAO airline designator: CSA1DZ -> csa one delta zulu
        elif len(part) == 1 and i > 0 and parts[i - 1].isdigit() and part in RUNWAY_SUFFIX and i == len(parts) - 1:
            out.append(RUNWAY_SUFFIX[part])
        else:
            out += [NATO.get(ch, ch) for ch in part]
    return out


def _spell_letter_runs(words: list[str]) -> list[str]:
    """'k t s' -> 'kilo tango sierra'. Lone letters stay: in the references they are
    mostly truncated words ("level t") or split contractions ("that s ok")."""
    def is_letter(w: str) -> bool:
        return len(w) == 1 and w in NATO and w not in ("a", "i")

    out = list(words)
    i = 0
    while i < len(words):
        j = i
        while j < len(words) and is_letter(words[j]):
            j += 1
        if j - i >= 2:
            out[i:j] = [NATO[w] for w in words[i:j]]
        i = max(j, i + 1)
    return out


def _decimal_points(words: list[str]) -> list[str]:
    """'one two four point three' -> '... decimal three'; 'holding point' stays."""
    return [
        "decimal" if w == "point" and 0 < i < len(words) - 1
        and words[i - 1] in DIGITS and words[i + 1] in DIGITS else w
        for i, w in enumerate(words)
    ]


def _expand_repeats(words: list[str]) -> list[str]:
    """'triple hotel' -> 'hotel hotel hotel', 'double two' -> 'two two'."""
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in ("double", "triple") and i + 1 < len(words):
            n = 2 if w == "double" else 3
            out += [words[i + 1]] * n
            i += 2
            continue
        out.append(w)
        i += 1
    return out


def normalize(text: str) -> str:
    """Normalise an ATC transcript to the canonical spoken form."""
    text = text.lower()
    text = _BRACKETED.sub(" ", text)
    text = re.sub(r"\bx[- ]?ray\b", "xray", text)
    text = _THOUSANDS.sub("", text)  # 2,000 -> 2000
    text = _DECIMAL.sub(" decimal ", text)
    text = text.replace("'", "").replace("\u2019", "")  # i'm -> im, on both sides
    text = text.replace("-", " ").replace("_", " ")
    text = _PUNCT.sub(" ", text)
    words: list[str] = []
    for tok in text.split():
        if not tok or tok in FILLERS:
            continue
        words += _expand_token(tok)
    words = [VARIANTS.get(w, w) for w in words]
    words = _spell_letter_runs(words)
    words = _decimal_points(words)
    words = _expand_repeats(words)
    out = " ".join(words)
    for a, b in PHRASES:
        out = re.sub(rf"\b{a}\b", b, out)
    return out
