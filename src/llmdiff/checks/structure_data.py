"""Structured-data dimensions: JSON/YAML/CSV shape and key-level comparison."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from llmdiff.core.checks import CheckContext

__all__ = [
    "JsonKeySetCheck",
    "JsonDepthCheck",
    "JsonArrayLengthCheck",
    "JsonTypeSignatureCheck",
    "JsonValueTypeMatchCheck",
    "JsonNestedKeyCountCheck",
    "JsonNumberSumCheck",
    "JsonStringLengthSumCheck",
    "JsonBoolCountCheck",
    "JsonNullCountCheck",
    "YamlKeyCountCheck",
    "YamlDepthCheck",
    "YamlListItemCountCheck",
    "CsvColumnCountCheck",
    "CsvRowCountCheck",
    "CsvHeaderMatchCheck",
    "CsvCellTypeRatioCheck",
    "CsvDelimiterMatchCheck",
    "CsvQuotedCellRatioCheck",
    "KeyOrderPreservedCheck",
]

_MAX = 200_000


def _s(text: str) -> str:
    return text[:_MAX]


def _try_json(text: str) -> object:
    try:
        return json.loads(_s(text))
    except (json.JSONDecodeError, ValueError):
        return None


def _walk_keys(obj: object, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            full = f"{prefix}.{k}" if prefix else str(k)
            keys.add(full)
            keys |= _walk_keys(v, full)
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        for i, v in enumerate(obj):
            keys |= _walk_keys(v, f"{prefix}[{i}]")
    return keys


def _depth(obj: object) -> int:
    if isinstance(obj, Mapping):
        return 1 + max((_depth(v) for v in obj.values()), default=0)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return 1 + max((_depth(v) for v in obj), default=0)
    return 0


def _type_sig(obj: object) -> str:
    if isinstance(obj, Mapping):
        pairs = sorted(obj.items(), key=lambda kv: str(kv[0]))
        inner = ",".join(f"{k}:{_type_sig(v)}" for k, v in pairs)
        return "{" + inner + "}"
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return "[" + ",".join(_type_sig(v) for v in obj[:50]) + "]"
    if isinstance(obj, bool):
        return "bool"
    if isinstance(obj, (int, float)):
        return "num"
    if obj is None:
        return "null"
    return "str"


def _count_types(obj: object) -> dict[str, int]:
    counts = {"num": 0, "str": 0, "bool": 0, "null": 0, "list": 0, "dict": 0}
    if isinstance(obj, Mapping):
        counts["dict"] += 1
        for v in obj.values():
            for k, c in _count_types(v).items():
                counts[k] += c
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        counts["list"] += 1
        for v in obj:
            for k, c in _count_types(v).items():
                counts[k] += c
    elif isinstance(obj, bool):
        counts["bool"] += 1
    elif isinstance(obj, (int, float)):
        counts["num"] += 1
    elif obj is None:
        counts["null"] += 1
    else:
        counts["str"] += 1
    return counts


def _num_sum(obj: object) -> float:
    total = 0.0
    if isinstance(obj, Mapping):
        for v in obj.values():
            total += _num_sum(v)
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        for v in obj:
            total += _num_sum(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        total += float(obj)
    return total


def _str_len_sum(obj: object) -> int:
    total = 0
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            total += len(str(k)) + _str_len_sum(v)
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        for v in obj:
            total += _str_len_sum(v)
    elif isinstance(obj, str):
        total += len(obj)
    return total


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class JsonKeySetCheck:
    """Jaccard similarity of the full nested key set between baseline and output."""

    @property
    def name(self) -> str:
        return "json_key_set"

    def score(self, output: str, context: CheckContext) -> float:
        base_obj, cand_obj = _try_json(context.baseline_output), _try_json(output)
        if base_obj is None or cand_obj is None:
            return 1.0 if (base_obj is None) == (cand_obj is None) else 0.0
        return _jaccard(_walk_keys(base_obj), _walk_keys(cand_obj))


class JsonDepthCheck:
    @property
    def name(self) -> str:
        return "json_depth"

    def score(self, output: str, context: CheckContext) -> float:
        base_obj, cand_obj = _try_json(context.baseline_output), _try_json(output)
        if base_obj is None or cand_obj is None:
            return 1.0 if (base_obj is None) == (cand_obj is None) else 0.0
        return _count_match(_depth(base_obj), _depth(cand_obj))


class JsonArrayLengthCheck:
    """1.0 when the total number of array elements matches the baseline."""

    @property
    def name(self) -> str:
        return "json_array_len"

    def score(self, output: str, context: CheckContext) -> float:
        def count(obj: object) -> int:
            n = 0
            if isinstance(obj, Mapping):
                for v in obj.values():
                    n += count(v)
            elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
                n += len(obj)
                for v in obj:
                    n += count(v)
            return n

        base_obj, cand_obj = _try_json(context.baseline_output), _try_json(output)
        if base_obj is None or cand_obj is None:
            return 1.0 if (base_obj is None) == (cand_obj is None) else 0.0
        return _count_match(count(base_obj), count(cand_obj))


class JsonTypeSignatureCheck:
    """1.0 when the recursive type signature matches the baseline."""

    @property
    def name(self) -> str:
        return "json_type_sig"

    def score(self, output: str, context: CheckContext) -> float:
        base_obj, cand_obj = _try_json(context.baseline_output), _try_json(output)
        if base_obj is None or cand_obj is None:
            return 1.0 if (base_obj is None) == (cand_obj is None) else 0.0
        return 1.0 if _type_sig(base_obj) == _type_sig(cand_obj) else 0.0


class JsonValueTypeMatchCheck:
    """Fraction of leaf values whose JSON type matches the baseline at the same key."""

    @property
    def name(self) -> str:
        return "json_value_types"

    def score(self, output: str, context: CheckContext) -> float:
        base_obj, cand_obj = _try_json(context.baseline_output), _try_json(output)
        if base_obj is None or cand_obj is None:
            return 1.0 if (base_obj is None) == (cand_obj is None) else 0.0

        def leaves(obj: object, prefix: str = "") -> dict[str, str]:
            out: dict[str, str] = {}
            if isinstance(obj, Mapping):
                for k, v in obj.items():
                    full = f"{prefix}.{k}" if prefix else str(k)
                    out.update(leaves(v, full))
            elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
                for i, v in enumerate(obj):
                    out.update(leaves(v, f"{prefix}[{i}]"))
            else:
                out[prefix] = _type_sig(obj)
            return out

        base_leaves, cand_leaves = leaves(base_obj), leaves(cand_obj)
        keys = set(base_leaves) | set(cand_leaves)
        if not keys:
            return 1.0
        match = sum(1 for k in keys if base_leaves.get(k) == cand_leaves.get(k))
        return match / len(keys)


class JsonNestedKeyCountCheck:
    @property
    def name(self) -> str:
        return "json_nested_keys"

    def score(self, output: str, context: CheckContext) -> float:
        base_obj, cand_obj = _try_json(context.baseline_output), _try_json(output)
        if base_obj is None or cand_obj is None:
            return 1.0 if (base_obj is None) == (cand_obj is None) else 0.0
        return _count_match(len(_walk_keys(base_obj)), len(_walk_keys(cand_obj)))


class JsonNumberSumCheck:
    """1.0 when the sum of all numeric leaves matches the baseline."""

    @property
    def name(self) -> str:
        return "json_num_sum"

    def score(self, output: str, context: CheckContext) -> float:
        base_obj, cand_obj = _try_json(context.baseline_output), _try_json(output)
        if base_obj is None or cand_obj is None:
            return 1.0 if (base_obj is None) == (cand_obj is None) else 0.0
        base, cand = _num_sum(base_obj), _num_sum(cand_obj)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / max(abs(base), 1e-9))


class JsonStringLengthSumCheck:
    @property
    def name(self) -> str:
        return "json_str_len_sum"

    def score(self, output: str, context: CheckContext) -> float:
        base_obj, cand_obj = _try_json(context.baseline_output), _try_json(output)
        if base_obj is None or cand_obj is None:
            return 1.0 if (base_obj is None) == (cand_obj is None) else 0.0
        return _count_match(_str_len_sum(base_obj), _str_len_sum(cand_obj))


class JsonBoolCountCheck:
    @property
    def name(self) -> str:
        return "json_bool_count"

    def score(self, output: str, context: CheckContext) -> float:
        base_obj, cand_obj = _try_json(context.baseline_output), _try_json(output)
        if base_obj is None or cand_obj is None:
            return 1.0 if (base_obj is None) == (cand_obj is None) else 0.0
        return _count_match(_count_types(base_obj)["bool"], _count_types(cand_obj)["bool"])


class JsonNullCountCheck:
    @property
    def name(self) -> str:
        return "json_null_count"

    def score(self, output: str, context: CheckContext) -> float:
        base_obj, cand_obj = _try_json(context.baseline_output), _try_json(output)
        if base_obj is None or cand_obj is None:
            return 1.0 if (base_obj is None) == (cand_obj is None) else 0.0
        return _count_match(_count_types(base_obj)["null"], _count_types(cand_obj)["null"])


def _yaml_lines(text: str) -> list[str]:
    return [ln for ln in _s(text).splitlines() if ln.strip() and not ln.strip().startswith("#")]


def _yaml_key_count(text: str) -> int:
    return sum(1 for ln in _yaml_lines(text) if re.match(r"^\s*[\w.-]+\s*:", ln))


def _yaml_depth(text: str) -> int:
    best = 0
    for ln in _yaml_lines(text):
        indent = len(ln) - len(ln.lstrip())
        best = max(best, indent // 2 + 1)
    return best


def _yaml_list_items(text: str) -> int:
    return sum(1 for ln in _yaml_lines(text) if re.match(r"^\s*-\s", ln))


class YamlKeyCountCheck:
    @property
    def name(self) -> str:
        return "yaml_key_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(_yaml_key_count(context.baseline_output), _yaml_key_count(output))


class YamlDepthCheck:
    @property
    def name(self) -> str:
        return "yaml_depth"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(_yaml_depth(context.baseline_output), _yaml_depth(output))


class YamlListItemCountCheck:
    @property
    def name(self) -> str:
        return "yaml_list_items"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(_yaml_list_items(context.baseline_output), _yaml_list_items(output))


def _csv_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for ln in _s(text).splitlines():
        if ln.strip():
            rows.append([c.strip() for c in ln.split(",")])
    return rows


class CsvColumnCountCheck:
    @property
    def name(self) -> str:
        return "csv_col_count"

    def score(self, output: str, context: CheckContext) -> float:
        def cols(text: str) -> int:
            rows = _csv_rows(text)
            return max((len(r) for r in rows), default=0)

        return _count_match(cols(context.baseline_output), cols(output))


class CsvRowCountCheck:
    @property
    def name(self) -> str:
        return "csv_row_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_csv_rows(context.baseline_output)),
            len(_csv_rows(output)),
        )


class CsvHeaderMatchCheck:
    """1.0 when the first CSV row (header) matches the baseline header."""

    @property
    def name(self) -> str:
        return "csv_header"

    def score(self, output: str, context: CheckContext) -> float:
        base_rows, cand_rows = _csv_rows(context.baseline_output), _csv_rows(output)
        if not base_rows or not cand_rows:
            return 1.0 if not base_rows and not cand_rows else 0.0
        return _jaccard(set(base_rows[0]), set(cand_rows[0]))


class CsvCellTypeRatioCheck:
    """1.0 when the ratio of numeric-looking cells matches the baseline."""

    @property
    def name(self) -> str:
        return "csv_numeric_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            rows = _csv_rows(text)
            cells = [c for r in rows for c in r]
            if not cells:
                return 0.0
            numeric = sum(1 for c in cells if re.fullmatch(r"-?\d+(?:\.\d+)?", c))
            return numeric / len(cells)

        base, cand = ratio(context.baseline_output), ratio(output)
        if abs(base - cand) <= 0.05:
            return 1.0
        return max(0.0, 1.0 - abs(cand - base) / max(abs(base), 1e-9))


class CsvDelimiterMatchCheck:
    """1.0 when the dominant delimiter matches the baseline (comma vs semicolon vs tab)."""

    @property
    def name(self) -> str:
        return "csv_delimiter"

    def score(self, output: str, context: CheckContext) -> float:
        def dominant(text: str) -> str:
            s = _s(text)
            counts = {ch: s.count(ch) for ch in (",", ";", "\t", "|")}
            return max(counts, key=lambda k: counts[k])

        base, cand = dominant(context.baseline_output), dominant(output)
        return 1.0 if base == cand else 0.0


class CsvQuotedCellRatioCheck:
    @property
    def name(self) -> str:
        return "csv_quoted_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            cells = re.findall(r'"[^"]*"', s)
            total = len(re.findall(r"[,;\t]", s)) + 1
            if total == 0:
                return 0.0
            return len(cells) / total

        base, cand = ratio(context.baseline_output), ratio(output)
        if abs(base - cand) <= 0.05:
            return 1.0
        return max(0.0, 1.0 - abs(cand - base) / max(abs(base), 1e-9))


class KeyOrderPreservedCheck:
    """1.0 when the top-level JSON key order matches the baseline."""

    @property
    def name(self) -> str:
        return "json_key_order"

    def score(self, output: str, context: CheckContext) -> float:
        base_obj, cand_obj = _try_json(context.baseline_output), _try_json(output)
        if not isinstance(base_obj, Mapping) or not isinstance(cand_obj, Mapping):
            return 1.0
        base_keys, cand_keys = list(base_obj.keys()), list(cand_obj.keys())
        if not base_keys:
            return 1.0
        match = sum(1 for a, b in zip(base_keys, cand_keys, strict=False) if a == b)
        return match / len(base_keys)
