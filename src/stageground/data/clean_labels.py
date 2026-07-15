"""Canonicalization of TNM staging labels.

Applied IDENTICALLY to gold labels and model output (DESIGN §4.1).
Folds AJCC subtypes to the major category before scoring.
"""
import re

_TOKEN = re.compile(r"^(?P<letter>[TNM])(?P<rest>.*)$")


def canonicalize(raw: str) -> str:
    """Fold a raw AJCC value to its major category.

    'T3a' -> 'T3', 'T1b1` -> 'T1', 'N2c' -> 'N2', 'M1A'ㄴ -> 'M1
    Preserve TX/NX/MX and TO/NO/MO as-is.
    """
    s = raw.strip().upper()
    s = s.removeprefix("P") # 'pT2' -> 'T2'; harmless if absent

    m = _TOKEN.match(s)
    if not m:
        raise ValueError(f"unrecognized stage token: {raw!r}")
    
    letter, rest = m.group("letter"), m.group("rest")

    # X = officially intermediate -> keep
    if rest == "X":
        return f"{letter}X"

    # first digit defines the major category (0 included)
    digit_match = re.match(r"\d", rest)
    if not digit_match:
        raise ValueError(f"no stage digit in: {raw!r}")

    return f"{letter}{digit_match.group()}"