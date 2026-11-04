"""Deep CSV dimensions: quoting, delimiters, header shape, and cell stats."""

from __future__ import annotations

import csv
import io
import math
import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "CsvProperParseCheck",
    "CsvRowWidthConsistencyCheck",
    "CsvQuotedFieldRatioCheck",
    "CsvDelimiterDetectCheck",
    "CsvHeaderNameJaccardCheck",
    "CsvNumericColumnRatioCheck",
    "CsvEmptyCellCountCheck",
    "CsvDuplicateRowCountCheck",
    "CsvCellLengthStdCheck",
    "CsvMaxCellLengthCheck",
    "CsvAvgCellLengthCheck",
    "CsvColumnCountDeepCheck",
    "CsvRowCountDeepCheck",
    "CsvSemicolonDelimCheck",
    "CsvTabDelimCheck",
    "CsvPipeDelimCheck",
    "CsvBomPresenceCheck",
    "CsvTrailingNewlineCheck",
    "CsvCommentLineCountCheck",
    "CsvTypeInferenceMatchCheck",
]

_MAX = 200_000


def _s(text: str) -> str:
    return text[:_MAX]


def _rows(text: str, delim: str = ",") -> list[list[str]]:
    try:
        return [row for row in csv.reader(io.StringIO(_s(text)), delimiter=delim) if row]
    except csv.Error:
        return []


def _detect_delim(text: str) -> str:
    s = _s(text)
    counts = {ch: s.count(ch) for ch in (",", ";", "\t", "|")}
    return max(counts, key=lambda k: counts[k])


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


class CsvProperParseCheck:
    """1.0 when the output parses cleanly with the detected delimiter iff baseline does."""

    @property
    def name(self) -> str:
        return "csv_parse_ok"

    def score(self, output: str, context: CheckContext) -> float:
        def ok(text: str) -> bool:
            return len(_rows(text, _detect_delim(text))) > 0

        return 1.0 if ok(context.baseline_output) == ok(output) else 0.0


class CsvRowWidthConsistencyCheck:
    """Fraction of rows whose width equals the header width."""

    @property
    def name(self) -> str:
        return "csv_row_width_consistency"

    def score(self, output: str, context: CheckContext) -> float:
        def consistency(text: str) -> float:
            rows = _rows(text, _detect_delim(text))
            if len(rows) < 2:
                return 1.0
            w = len(rows[0])
            return sum(1 for r in rows if len(r) == w) / len(rows)

        return _ratio_match(consistency(context.baseline_output), consistency(output))


class CsvQuotedFieldRatioCheck:
    @property
    def name(self) -> str:
        return "csv_quoted_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            cells = re.findall(r'"[^"]*"', s)
            total = max(1, s.count(_detect_delim(text)) + 1)
            return len(cells) / total

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class CsvDelimiterDetectCheck:
    """1.0 when the detected dominant delimiter matches the baseline's."""

    @property
    def name(self) -> str:
        return "csv_delim_detect"

    def score(self, output: str, context: CheckContext) -> float:
        base = _detect_delim(context.baseline_output)
        cand = _detect_delim(output)
        return 1.0 if base == cand else 0.0


class CsvHeaderNameJaccardCheck:
    @property
    def name(self) -> str:
        return "csv_header_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        def header(text: str) -> set[str]:
            rows = _rows(text, _detect_delim(text))
            return {c.strip().lower() for c in rows[0]} if rows else set()

        return _jaccard(header(context.baseline_output), header(output))


class CsvNumericColumnRatioCheck:
    """Ratio of cells that look numeric."""

    @property
    def name(self) -> str:
        return "csv_numeric_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            rows = _rows(text, _detect_delim(text))
            cells = [c for r in rows for c in r]
            if not cells:
                return 0.0
            numeric = sum(1 for c in cells if re.fullmatch(r"-?\d+(?:\.\d+)?", c.strip()))
            return numeric / len(cells)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class CsvEmptyCellCountCheck:
    @property
    def name(self) -> str:
        return "csv_empty_cells"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            rows = _rows(text, _detect_delim(text))
            return sum(1 for r in rows for c in r if not c.strip())

        return _count_match(count(context.baseline_output), count(output))


class CsvDuplicateRowCountCheck:
    @property
    def name(self) -> str:
        return "csv_dup_rows"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            rows = _rows(text, _detect_delim(text))
            seen: set[tuple] = set()
            dups = 0
            for r in rows:
                t = tuple(c.strip() for c in r)
                if t in seen:
                    dups += 1
                else:
                    seen.add(t)
            return dups

        return _count_match(count(context.baseline_output), count(output))


class CsvCellLengthStdCheck:
    @property
    def name(self) -> str:
        return "csv_cell_len_std"

    def score(self, output: str, context: CheckContext) -> float:
        def std(text: str) -> float:
            rows = _rows(text, _detect_delim(text))
            lens = [len(c) for r in rows for c in r]
            if len(lens) < 2:
                return 0.0
            mean = sum(lens) / len(lens)
            return math.sqrt(sum((x - mean) ** 2 for x in lens) / len(lens))

        base, cand = std(context.baseline_output), std(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class CsvMaxCellLengthCheck:
    @property
    def name(self) -> str:
        return "csv_max_cell_len"

    def score(self, output: str, context: CheckContext) -> float:
        def mx(text: str) -> int:
            rows = _rows(text, _detect_delim(text))
            return max((len(c) for r in rows for c in r), default=0)

        return _count_match(mx(context.baseline_output), mx(output))


class CsvAvgCellLengthCheck:
    @property
    def name(self) -> str:
        return "csv_avg_cell_len"

    def score(self, output: str, context: CheckContext) -> float:
        def avg(text: str) -> float:
            rows = _rows(text, _detect_delim(text))
            cells = [c for r in rows for c in r]
            return sum(len(c) for c in cells) / len(cells) if cells else 0.0

        base, cand = avg(context.baseline_output), avg(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class CsvColumnCountDeepCheck:
    @property
    def name(self) -> str:
        return "csv_col_count_deep"

    def score(self, output: str, context: CheckContext) -> float:
        def cols(text: str) -> int:
            rows = _rows(text, _detect_delim(text))
            return max((len(r) for r in rows), default=0)

        return _count_match(cols(context.baseline_output), cols(output))


class CsvRowCountDeepCheck:
    @property
    def name(self) -> str:
        return "csv_row_count_deep"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_rows(context.baseline_output, _detect_delim(context.baseline_output))),
            len(_rows(output, _detect_delim(output))),
        )


class CsvSemicolonDelimCheck:
    @property
    def name(self) -> str:
        return "csv_semicolon"

    def score(self, output: str, context: CheckContext) -> float:
        base = _detect_delim(context.baseline_output)
        cand = _detect_delim(output)
        return 1.0 if base == ";" and cand == ";" else 0.0


class CsvTabDelimCheck:
    @property
    def name(self) -> str:
        return "csv_tab"

    def score(self, output: str, context: CheckContext) -> float:
        base = _detect_delim(context.baseline_output)
        cand = _detect_delim(output)
        return 1.0 if base == "\t" and cand == "\t" else 0.0


class CsvPipeDelimCheck:
    @property
    def name(self) -> str:
        return "csv_pipe"

    def score(self, output: str, context: CheckContext) -> float:
        base = _detect_delim(context.baseline_output)
        cand = _detect_delim(output)
        return 1.0 if base == "|" and cand == "|" else 0.0


class CsvBomPresenceCheck:
    @property
    def name(self) -> str:
        return "csv_bom"

    def score(self, output: str, context: CheckContext) -> float:
        def has_bom(text: str) -> bool:
            return _s(text).startswith("﻿")

        return 1.0 if has_bom(context.baseline_output) == has_bom(output) else 0.0


class CsvTrailingNewlineCheck:
    @property
    def name(self) -> str:
        return "csv_trailing_nl"

    def score(self, output: str, context: CheckContext) -> float:
        def ends_nl(text: str) -> bool:
            return _s(text).endswith("\n")

        return 1.0 if ends_nl(context.baseline_output) == ends_nl(output) else 0.0


class CsvCommentLineCountCheck:
    @property
    def name(self) -> str:
        return "csv_comment_lines"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"^\s*#", re.MULTILINE)
        return _count_match(
            len(pat.findall(_s(context.baseline_output))),
            len(pat.findall(_s(output))),
        )


class CsvTypeInferenceMatchCheck:
    """Fraction of header columns whose inferred type (num/str) matches the baseline."""

    @property
    def name(self) -> str:
        return "csv_type_inference"

    def score(self, output: str, context: CheckContext) -> float:
        def types(text: str) -> dict[str, str]:
            rows = _rows(text, _detect_delim(text))
            if len(rows) < 2:
                return {}
            header, data = rows[0], rows[1:]
            out = {}
            for i, h in enumerate(header):
                vals = [r[i].strip() for r in data if i < len(r) and r[i].strip()]
                is_num = bool(vals) and all(
                    re.fullmatch(r"-?\d+(?:\.\d+)?", v) for v in vals
                )
                out[h.strip().lower()] = "num" if is_num else "str"
            return out

        a, b = types(context.baseline_output), types(output)
        keys = set(a) | set(b)
        if not keys:
            return 1.0
        match = sum(1 for k in keys if a.get(k) == b.get(k))
        return match / len(keys)
