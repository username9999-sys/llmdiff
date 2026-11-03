"""Unicode statistics dimensions: scripts, blocks, normalization, and encoding stats."""

from __future__ import annotations

import re
import unicodedata

from llmdiff.core.checks import CheckContext

__all__ = [
    "AsciiRatioCheck",
    "NonAsciiCountCheck",
    "LatinScriptRatioCheck",
    "CjkCharCountCheck",
    "ArabicScriptCountCheck",
    "CyrillicScriptCountCheck",
    "EmojiCodepointCountCheck",
    "CombiningMarkCountCheck",
    "NormalizedFormMatchCheck",
    "NfcNfdRatioCheck",
    "CodepointRangeDiversityCheck",
    "PrivateUseCountCheck",
    "ZeroWidthCountCheck",
    "BidiControlCountCheck",
    "SurrogatePairCountCheck",
    "GraphemeEstimateCheck",
    "ScriptDiversityCheck",
    "UppercaseUnicodeCountCheck",
    "TitlecaseUnicodeCountCheck",
    "DecimalDigitUnicodeCountCheck",
]

_MAX = 200_000
_ZW_RE = re.compile(r"[​‌‍⁠⁡⁢⁣⁤]")
_BIDI_RE = re.compile(r"[‎‏‪-‮⁦-⁩]")


def _s(text: str) -> str:
    return text[:_MAX]


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _script_of(ch: str) -> str:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "UNKNOWN"
    if name.startswith("LATIN"):
        return "latin"
    if name.startswith(("CJK", "HIRAGANA", "KATAKANA")):
        return "cjk"
    if name.startswith("ARABIC"):
        return "arabic"
    if name.startswith("CYRILLIC"):
        return "cyrillic"
    if name.startswith("GREEK"):
        return "greek"
    if name.startswith("HEBREW"):
        return "hebrew"
    if name.startswith("DEVANAGARI"):
        return "devanagari"
    if name.startswith("THAI"):
        return "thai"
    return "other"


class AsciiRatioCheck:
    @property
    def name(self) -> str:
        return "ascii_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            return sum(1 for c in s if ord(c) < 128) / len(s) if s else 1.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class NonAsciiCountCheck:
    @property
    def name(self) -> str:
        return "nonascii_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for c in _s(text) if ord(c) >= 128)

        return _count_match(count(context.baseline_output), count(output))


class LatinScriptRatioCheck:
    @property
    def name(self) -> str:
        return "latin_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            chars = [c for c in _s(text) if c.isalpha()]
            if not chars:
                return 0.0
            return sum(1 for c in chars if _script_of(c) == "latin") / len(chars)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class CjkCharCountCheck:
    @property
    def name(self) -> str:
        return "cjk_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for c in _s(text) if _script_of(c) == "cjk")

        return _count_match(count(context.baseline_output), count(output))


class ArabicScriptCountCheck:
    @property
    def name(self) -> str:
        return "arabic_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for c in _s(text) if _script_of(c) == "arabic")

        return _count_match(count(context.baseline_output), count(output))


class CyrillicScriptCountCheck:
    @property
    def name(self) -> str:
        return "cyrillic_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for c in _s(text) if _script_of(c) == "cyrillic")

        return _count_match(count(context.baseline_output), count(output))


class EmojiCodepointCountCheck:
    @property
    def name(self) -> str:
        return "emoji_codepoints"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for c in _s(text) if ord(c) >= 0x1F000)

        return _count_match(count(context.baseline_output), count(output))


class CombiningMarkCountCheck:
    @property
    def name(self) -> str:
        return "combining_marks"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for c in _s(text) if unicodedata.combining(c))

        return _count_match(count(context.baseline_output), count(output))


class NormalizedFormMatchCheck:
    """1.0 when NFC-normalized forms of baseline and output are identical."""

    @property
    def name(self) -> str:
        return "nfc_match"

    def score(self, output: str, context: CheckContext) -> float:
        a = unicodedata.normalize("NFC", _s(context.baseline_output))
        b = unicodedata.normalize("NFC", _s(output))
        return 1.0 if a == b else 0.0


class NfcNfdRatioCheck:
    """1.0 when the ratio of chars whose NFC != NFD matches the baseline."""

    @property
    def name(self) -> str:
        return "nfc_nfd_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            if not s:
                return 0.0
            n = sum(
                1
                for c in s
                if unicodedata.normalize("NFC", c) != unicodedata.normalize("NFD", c)
            )
            return n / len(s)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class CodepointRangeDiversityCheck:
    """Number of distinct 0x1000-wide codepoint blocks touched."""

    @property
    def name(self) -> str:
        return "cp_block_diversity"

    def score(self, output: str, context: CheckContext) -> float:
        def blocks(text: str) -> int:
            return len({ord(c) // 0x1000 for c in _s(text)})

        return _count_match(blocks(context.baseline_output), blocks(output))


class PrivateUseCountCheck:
    @property
    def name(self) -> str:
        return "private_use_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for c in _s(text) if unicodedata.category(c) == "Co")

        return _count_match(count(context.baseline_output), count(output))


class ZeroWidthCountCheck:
    @property
    def name(self) -> str:
        return "zero_width_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return len(_ZW_RE.findall(_s(text)))

        return _count_match(count(context.baseline_output), count(output))


class BidiControlCountCheck:
    @property
    def name(self) -> str:
        return "bidi_control_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return len(_BIDI_RE.findall(_s(text)))

        return _count_match(count(context.baseline_output), count(output))


class SurrogatePairCountCheck:
    """Count of codepoints above the BMP (which need surrogate pairs in UTF-16)."""

    @property
    def name(self) -> str:
        return "astral_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for c in _s(text) if ord(c) > 0xFFFF)

        return _count_match(count(context.baseline_output), count(output))


class GraphemeEstimateCheck:
    """Approximate grapheme cluster count (base char + combining marks)."""

    @property
    def name(self) -> str:
        return "grapheme_est"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            total = 0
            for c in _s(text):
                if not unicodedata.combining(c):
                    total += 1
            return total

        return _count_match(count(context.baseline_output), count(output))


class ScriptDiversityCheck:
    """Number of distinct scripts present."""

    @property
    def name(self) -> str:
        return "script_diversity"

    def score(self, output: str, context: CheckContext) -> float:
        def diversity(text: str) -> int:
            return len({_script_of(c) for c in _s(text) if c.isalpha()})

        return _count_match(diversity(context.baseline_output), diversity(output))


class UppercaseUnicodeCountCheck:
    @property
    def name(self) -> str:
        return "unicode_upper_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for c in _s(text) if c.isupper() and ord(c) >= 128)

        return _count_match(count(context.baseline_output), count(output))


class TitlecaseUnicodeCountCheck:
    @property
    def name(self) -> str:
        return "unicode_title_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for c in _s(text) if unicodedata.category(c) == "Lt")

        return _count_match(count(context.baseline_output), count(output))


class DecimalDigitUnicodeCountCheck:
    @property
    def name(self) -> str:
        return "unicode_digit_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for c in _s(text) if unicodedata.category(c) == "Nd" and ord(c) >= 128)

        return _count_match(count(context.baseline_output), count(output))
