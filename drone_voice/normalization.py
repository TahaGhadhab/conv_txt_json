"""Deterministic normalization of Arabic transcriptions ("normalization v1").

Rules, applied in this order:

1. Unicode NFC.
2. Remove Arabic diacritics: harakat U+064B..U+0652 (fathatan .. sukun, shadda
   included) and superscript alef U+0670.
3. Remove tatweel / kashida U+0640 (elongation character).
4. Remove invisible direction marks LRM U+200E, RLM U+200F, ALM U+061C and BOM U+FEFF.
5. Collapse every run of whitespace into a single space and trim both ends.
6. Remove sentence punctuation at the very end only: ``.`` ``!`` ``?`` ``؟``
   (repeated or separated by spaces).
7. Unicode NFC again, so that applying the function twice changes nothing.

Everything else is kept on purpose: letters (hamza/alef variants, ta marbuta,
alef maqsura are NOT unified), digits, words, and punctuation inside the sentence.
A sentence such as ``تقدم، ارجع`` therefore keeps its comma and cannot match a
single command.
"""

from __future__ import annotations

import unicodedata

NORMALIZATION_VERSION = 1

_DIACRITICS = frozenset(chr(c) for c in range(0x064B, 0x0653)) | {"ٰ"}
_TATWEEL = "ـ"
_INVISIBLE_MARKS = frozenset({"‎", "‏", "؜", "﻿"})
_REMOVED = _DIACRITICS | {_TATWEEL} | _INVISIBLE_MARKS

TRAILING_PUNCTUATION = ".!?؟"


def normalize_text(text: str) -> str:
    """Return the normalized form of ``text``. Raises ``TypeError`` for non-strings."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    s = unicodedata.normalize("NFC", text)
    s = "".join(ch for ch in s if ch not in _REMOVED)
    s = " ".join(s.split())
    s = s.rstrip(TRAILING_PUNCTUATION + " ")
    return unicodedata.normalize("NFC", s)
