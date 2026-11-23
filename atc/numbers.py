"""Read spoken ATC numbers from normalised word lists.

Input is the output of `atc.normalize.normalize(...).split()`: digits are words
("one two one decimal nine"), round values use "hundred"/"thousand".
"""

from __future__ import annotations

from dataclasses import dataclass

DIGIT = {w: i for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])}


@dataclass(frozen=True)
class Number:
    value: float          # numeric value: 1014, 5000, 121.9
    digits: str           # digit string as spoken, for suffix comparison: "1014", "5000", "1219"
    start: int            # word span [start, end) in the input list
    end: int
    decimal: bool = False

    @property
    def int_value(self) -> int:
        return int(round(self.value))


def is_digit(word: str) -> bool:
    return word in DIGIT


def read_number(words: list[str], i: int) -> Number | None:
    """Read a number starting exactly at words[i], or return None.

    Handles digit-by-digit ("two four zero"), "N thousand [M hundred]", "N hundred",
    and decimals ("one two one decimal nine").
    """
    n = len(words)
    j = i
    digits = ""
    while j < n and words[j] in DIGIT:
        digits += str(DIGIT[words[j]])
        j += 1
    if not digits:
        return None

    # "N thousand" / "N hundred" bind only to a short leading run: in
    # "flight level two nine zero two thousand" the 290 and 2000 are separate numbers.
    if j < n and words[j] in ("thousand", "hundred") and len(digits) > 2:
        j -= 1 if len(digits) > 1 else 0
        digits = digits[:-1]
        return Number(int(digits), digits, i, j)

    if j < n and words[j] == "thousand":
        value = int(digits) * 1000
        j += 1
        if j + 1 < n and words[j] in DIGIT and words[j + 1] == "hundred":
            value += DIGIT[words[j]] * 100
            j += 2
        return Number(value, str(value), i, j)

    if j < n and words[j] == "hundred":
        value = int(digits) * 100
        return Number(value, str(value), i, j + 1)

    if j + 1 < n and words[j] == "decimal" and words[j + 1] in DIGIT:
        j += 1
        frac = ""
        while j < n and words[j] in DIGIT:
            frac += str(DIGIT[words[j]])
            j += 1
        return Number(float(f"{int(digits)}.{frac}"), digits + frac, i, j, decimal=True)

    return Number(int(digits), digits, i, j)


def find_number(words: list[str], i: int, max_skip: int, skip: set[str]) -> Number | None:
    """Read the first number at or after i, skipping up to `max_skip` filler words from `skip`."""
    for k in range(i, min(len(words), i + max_skip + 1)):
        if words[k] in DIGIT:
            return read_number(words, k)
        if words[k] not in skip:
            return None
    return None
