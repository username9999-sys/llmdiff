"""Natural-language detection dimensions: language guess, script, and locale hints."""

from __future__ import annotations

import re
import unicodedata

from llmdiff.core.checks import CheckContext

__all__ = [
    "DetectedLanguageMatchCheck",
    "ScriptMatchCheck",
    "RtlPresenceCheck",
    "StopwordRatioCheck",
    "DiacriticRatioCheck",
    "VowelRatioLangCheck",
    "CommonWordCoverageCheck",
    "FunctionWordRatioCheck",
    "AvgWordLengthLangCheck",
    "CharNgramLangScoreCheck",
    "DigitScriptCountCheck",
    "LatinExtendedCountCheck",
    "GreekCountCheck",
    "HebrewCountCheck",
    "ArabicCountCheck",
    "DevanagariCountCheck",
    "ThaiCountCheck",
    "HangulCountCheck",
    "HiraganaCountCheck",
    "KatakanaCountCheck",
]

_MAX = 200_000
_WORD_RE = re.compile(r"[A-Za-zÀ-ɏ一-鿿가-힯ぁ-んァ-ン]+")

_STOPWORDS = {
    "en": {"the", "and", "is", "of", "to", "in", "that", "it", "a", "for"},
    "id": {"yang", "dan", "di", "itu", "dengan", "untuk", "pada", "ini", "dari", "tidak"},
    "de": {"der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich"},
    "fr": {"le", "la", "de", "et", "les", "des", "en", "un", "une", "est"},
    "es": {"el", "la", "de", "que", "y", "en", "un", "una", "los", "las"},
    "pt": {"de", "a", "o", "que", "e", "do", "da", "em", "um", "para"},
    "ja": {"の", "に", "は", "を", "た", "が", "で", "て", "と", "し"},
    "zh": {"的", "是", "在", "我", "有", "和", "就", "不", "人", "都"},
}

_FUNCTION_WORDS = {
    "en": {"the", "a", "an", "of", "to", "in", "and", "or", "but", "if", "is", "are", "was"},
    "id": {"yang", "di", "ke", "dari", "untuk", "dengan", "pada", "atau", "dan", "jika"},
    "de": {"der", "die", "das", "und", "oder", "aber", "wenn", "ist", "sind", "von", "zu"},
    "fr": {"le", "la", "les", "de", "et", "ou", "mais", "si", "est", "sont", "un", "une"},
}

_LANG_CHAR_HINTS: dict[str, str] = {
    "greek": "GREEK",
    "hebrew": "HEBREW",
    "arabic": "ARABIC",
    "devanagari": "DEVANAGARI",
    "thai": "THAI",
    "hangul": "HANGUL",
    "hiragana": "HIRAGANA",
    "katakana": "KATAKANA",
}


def _s(text: str) -> str:
    return text[:_MAX]


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(_s(text))]


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


def _detect_lang(text: str) -> str:
    ws = _words(text)
    if not ws:
        return "unknown"
    scores: dict[str, int] = {}
    for lang, stops in _STOPWORDS.items():
        scores[lang] = sum(1 for w in ws if w in stops)
    best = max(scores.values())
    return max(scores, key=lambda k: scores[k]) if best > 0 else "unknown"


def _script(text: str) -> str:
    counts: dict[str, int] = {}
    for ch in _s(text):
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        for key, prefix in _LANG_CHAR_HINTS.items():
            if name.startswith(prefix):
                counts[key] = counts.get(key, 0) + 1
        if name.startswith("LATIN"):
            counts["latin"] = counts.get("latin", 0) + 1
        elif name.startswith("CJK"):
            counts["cjk"] = counts.get("cjk", 0) + 1
    if not counts:
        return "unknown"
    return max(counts, key=lambda k: counts[k])


class DetectedLanguageMatchCheck:
    """1.0 when the stopword-based language guess matches the baseline's."""

    @property
    def name(self) -> str:
        return "lang_detect"

    def score(self, output: str, context: CheckContext) -> float:
        base, cand = _detect_lang(context.baseline_output), _detect_lang(output)
        return 1.0 if base == cand else 0.0


class ScriptMatchCheck:
    @property
    def name(self) -> str:
        return "script_match"

    def score(self, output: str, context: CheckContext) -> float:
        base, cand = _script(context.baseline_output), _script(output)
        return 1.0 if base == cand else 0.0


class RtlPresenceCheck:
    """1.0 when RTL-script presence matches the baseline."""

    @property
    def name(self) -> str:
        return "rtl_presence"

    def score(self, output: str, context: CheckContext) -> float:
        def has_rtl(text: str) -> bool:
            return any(
                unicodedata.bidirectional(ch) in ("R", "AL", "AN") for ch in _s(text)
            )

        return 1.0 if has_rtl(context.baseline_output) == has_rtl(output) else 0.0


class StopwordRatioCheck:
    @property
    def name(self) -> str:
        return "stopword_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            if not ws:
                return 0.0
            all_stops = set().union(*_STOPWORDS.values())
            return sum(1 for w in ws if w in all_stops) / len(ws)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class DiacriticRatioCheck:
    @property
    def name(self) -> str:
        return "diacritic_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            letters = [c for c in s if c.isalpha()]
            if not letters:
                return 0.0
            return sum(1 for c in letters if ord(c) >= 128) / len(letters)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class VowelRatioLangCheck:
    @property
    def name(self) -> str:
        return "vowel_ratio_lang"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            letters = [c for c in _s(text) if c.isalpha() and ord(c) < 128]
            if not letters:
                return 0.0
            return sum(1 for c in letters if c.lower() in "aeiou") / len(letters)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class CommonWordCoverageCheck:
    """Coverage of the 10 most common English words."""

    @property
    def name(self) -> str:
        return "common_word_cov"

    def score(self, output: str, context: CheckContext) -> float:
        common = _STOPWORDS["en"]

        def coverage(text: str) -> float:
            ws = _words(text)
            if not ws:
                return 0.0
            return sum(1 for w in ws if w in common) / len(ws)

        return _ratio_match(coverage(context.baseline_output), coverage(output))


class FunctionWordRatioCheck:
    @property
    def name(self) -> str:
        return "function_word_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            if not ws:
                return 0.0
            all_fw = set().union(*_FUNCTION_WORDS.values())
            return sum(1 for w in ws if w in all_fw) / len(ws)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class AvgWordLengthLangCheck:
    @property
    def name(self) -> str:
        return "avg_word_len_lang"

    def score(self, output: str, context: CheckContext) -> float:
        def avg(text: str) -> float:
            ws = _words(text)
            return sum(len(w) for w in ws) / len(ws) if ws else 0.0

        base, cand = avg(context.baseline_output), avg(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class CharNgramLangScoreCheck:
    """Character trigram overlap with the baseline (language fingerprint)."""

    @property
    def name(self) -> str:
        return "char_trigram_lang"

    def score(self, output: str, context: CheckContext) -> float:
        def trigrams(text: str) -> set[str]:
            s = _s(text).lower()
            return {s[i : i + 3] for i in range(len(s) - 2)}

        a, b = trigrams(context.baseline_output), trigrams(output)
        if not a and not b:
            return 1.0
        union = a | b
        return len(a & b) / len(union) if union else 1.0


def _script_count(text: str, prefix: str) -> int:
    n = 0
    for ch in _s(text):
        if not ch.isalpha():
            continue
        try:
            if unicodedata.name(ch).startswith(prefix):
                n += 1
        except ValueError:
            continue
    return n


class DigitScriptCountCheck:
    @property
    def name(self) -> str:
        return "digit_script_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for c in _s(text) if unicodedata.category(c) == "Nd" and ord(c) >= 128)

        return _count_match(count(context.baseline_output), count(output))


class LatinExtendedCountCheck:
    @property
    def name(self) -> str:
        return "latin_extended_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _script_count(context.baseline_output, "LATIN"),
            _script_count(output, "LATIN"),
        )


class GreekCountCheck:
    @property
    def name(self) -> str:
        return "greek_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _script_count(context.baseline_output, "GREEK"),
            _script_count(output, "GREEK"),
        )


class HebrewCountCheck:
    @property
    def name(self) -> str:
        return "hebrew_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _script_count(context.baseline_output, "HEBREW"),
            _script_count(output, "HEBREW"),
        )


class ArabicCountCheck:
    @property
    def name(self) -> str:
        return "arabic_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _script_count(context.baseline_output, "ARABIC"),
            _script_count(output, "ARABIC"),
        )


class DevanagariCountCheck:
    @property
    def name(self) -> str:
        return "devanagari_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _script_count(context.baseline_output, "DEVANAGARI"),
            _script_count(output, "DEVANAGARI"),
        )


class ThaiCountCheck:
    @property
    def name(self) -> str:
        return "thai_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _script_count(context.baseline_output, "THAI"),
            _script_count(output, "THAI"),
        )


class HangulCountCheck:
    @property
    def name(self) -> str:
        return "hangul_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _script_count(context.baseline_output, "HANGUL"),
            _script_count(output, "HANGUL"),
        )


class HiraganaCountCheck:
    @property
    def name(self) -> str:
        return "hiragana_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _script_count(context.baseline_output, "HIRAGANA"),
            _script_count(output, "HIRAGANA"),
        )


class KatakanaCountCheck:
    @property
    def name(self) -> str:
        return "katakana_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _script_count(context.baseline_output, "KATAKANA"),
            _script_count(output, "KATAKANA"),
        )
