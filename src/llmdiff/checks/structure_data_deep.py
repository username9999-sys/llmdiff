"""Deep structured-data dimensions: nested JSON paths, arrays, and key ordering."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from llmdiff.core.checks import CheckContext

__all__ = [
    "JsonPathSetCheck",
    "JsonLeafCountCheck",
    "JsonMaxArrayLenCheck",
    "JsonMinArrayLenCheck",
    "JsonAvgArrayLenCheck",
    "JsonEmptyArrayCountCheck",
    "JsonEmptyObjectCountCheck",
    "JsonKeyOrderMatchCheck",
    "JsonTopLevelKeySetCheck",
    "JsonNestedDepthStatsCheck",
    "JsonStringLeafCountCheck",
    "JsonNumberLeafCountCheck",
    "JsonBoolLeafCountCheck",
    "JsonNullLeafCountCheck",
    "JsonArrayLeafCountCheck",
    "JsonObjectLeafCountCheck",
    "JsonDuplicateKeyRiskCheck",
    "JsonUnicodeEscapeCountCheck",
    "JsonPrettyPrintCheck",
    "JsonKeyLengthStatsCheck",
]

_MAX = 200_000


def _s(text: str) -> str:
    return text[:_MAX]


def _try_json(text: str) -> object:
    try:
        return json.loads(_s(text))
    except (json.JSONDecodeError, ValueError):
        return None


def _walk(obj: object, prefix: str = "") -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            full = f"{prefix}.{k}" if prefix else str(k)
            out.append((full, v))
            out.extend(_walk(v, full))
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        for i, v in enumerate(obj):
            out.extend(_walk(v, f"{prefix}[{i}]"))
    else:
        out.append((prefix, obj))
    return out


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


class JsonPathSetCheck:
    """Jaccard similarity of full JSON paths (keys + array indices)."""

    @property
    def name(self) -> str:
        return "json_path_set"

    def score(self, output: str, context: CheckContext) -> float:
        a_obj, b_obj = _try_json(context.baseline_output), _try_json(output)
        if a_obj is None or b_obj is None:
            return 1.0 if (a_obj is None) == (b_obj is None) else 0.0
        return _jaccard({p for p, _ in _walk(a_obj)}, {p for p, _ in _walk(b_obj)})


class JsonLeafCountCheck:
    @property
    def name(self) -> str:
        return "json_leaf_count"

    def score(self, output: str, context: CheckContext) -> float:
        a_obj, b_obj = _try_json(context.baseline_output), _try_json(output)
        if a_obj is None or b_obj is None:
            return 1.0 if (a_obj is None) == (b_obj is None) else 0.0
        return _count_match(len(_walk(a_obj)), len(_walk(b_obj)))


def _array_lens(obj: object) -> list[int]:
    lens: list[int] = []
    if isinstance(obj, Mapping):
        for v in obj.values():
            lens.extend(_array_lens(v))
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        lens.append(len(obj))
        for v in obj:
            lens.extend(_array_lens(v))
    return lens


class JsonMaxArrayLenCheck:
    @property
    def name(self) -> str:
        return "json_max_array"

    def score(self, output: str, context: CheckContext) -> float:
        a_obj, b_obj = _try_json(context.baseline_output), _try_json(output)
        if a_obj is None or b_obj is None:
            return 1.0
        return _count_match(
            max(_array_lens(a_obj), default=0),
            max(_array_lens(b_obj), default=0),
        )


class JsonMinArrayLenCheck:
    @property
    def name(self) -> str:
        return "json_min_array"

    def score(self, output: str, context: CheckContext) -> float:
        a_obj, b_obj = _try_json(context.baseline_output), _try_json(output)
        if a_obj is None or b_obj is None:
            return 1.0
        la, lb = _array_lens(a_obj), _array_lens(b_obj)
        return _count_match(min(la, default=0), min(lb, default=0))


class JsonAvgArrayLenCheck:
    @property
    def name(self) -> str:
        return "json_avg_array"

    def score(self, output: str, context: CheckContext) -> float:
        a_obj, b_obj = _try_json(context.baseline_output), _try_json(output)
        if a_obj is None or b_obj is None:
            return 1.0
        la, lb = _array_lens(a_obj), _array_lens(b_obj)
        base = sum(la) / len(la) if la else 0.0
        cand = sum(lb) / len(lb) if lb else 0.0
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class JsonEmptyArrayCountCheck:
    @property
    def name(self) -> str:
        return "json_empty_arrays"

    def score(self, output: str, context: CheckContext) -> float:
        a_obj, b_obj = _try_json(context.baseline_output), _try_json(output)
        if a_obj is None or b_obj is None:
            return 1.0
        return _count_match(
            sum(1 for n in _array_lens(a_obj) if n == 0),
            sum(1 for n in _array_lens(b_obj) if n == 0),
        )


def _empty_objects(obj: object) -> int:
    n = 0
    if isinstance(obj, Mapping):
        if not obj:
            n += 1
        for v in obj.values():
            n += _empty_objects(v)
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        for v in obj:
            n += _empty_objects(v)
    return n


class JsonEmptyObjectCountCheck:
    @property
    def name(self) -> str:
        return "json_empty_objects"

    def score(self, output: str, context: CheckContext) -> float:
        a_obj, b_obj = _try_json(context.baseline_output), _try_json(output)
        if a_obj is None or b_obj is None:
            return 1.0
        return _count_match(_empty_objects(a_obj), _empty_objects(b_obj))


class JsonKeyOrderMatchCheck:
    """1.0 when the top-level key order matches the baseline."""

    @property
    def name(self) -> str:
        return "json_top_key_order"

    def score(self, output: str, context: CheckContext) -> float:
        a_obj, b_obj = _try_json(context.baseline_output), _try_json(output)
        if not isinstance(a_obj, Mapping) or not isinstance(b_obj, Mapping):
            return 1.0
        a_keys, b_keys = list(a_obj.keys()), list(b_obj.keys())
        if not a_keys:
            return 1.0
        match = sum(1 for x, y in zip(a_keys, b_keys, strict=False) if x == y)
        return match / len(a_keys)


class JsonTopLevelKeySetCheck:
    @property
    def name(self) -> str:
        return "json_top_keys"

    def score(self, output: str, context: CheckContext) -> float:
        a_obj, b_obj = _try_json(context.baseline_output), _try_json(output)
        if not isinstance(a_obj, Mapping) or not isinstance(b_obj, Mapping):
            return 1.0
        return _jaccard(set(a_obj.keys()), set(b_obj.keys()))


class JsonNestedDepthStatsCheck:
    """1.0 when the max nesting depth matches the baseline."""

    @property
    def name(self) -> str:
        return "json_depth_stats"

    def score(self, output: str, context: CheckContext) -> float:
        def depth(obj: object) -> int:
            if isinstance(obj, Mapping):
                return 1 + max((depth(v) for v in obj.values()), default=0)
            if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
                return 1 + max((depth(v) for v in obj), default=0)
            return 0

        a_obj, b_obj = _try_json(context.baseline_output), _try_json(output)
        if a_obj is None or b_obj is None:
            return 1.0
        return _count_match(depth(a_obj), depth(b_obj))


def _leaf_type_counts(obj: object) -> dict[str, int]:
    counts = {"str": 0, "num": 0, "bool": 0, "null": 0, "array": 0, "object": 0}
    if isinstance(obj, Mapping):
        counts["object"] += 1
        for v in obj.values():
            for k, c in _leaf_type_counts(v).items():
                counts[k] += c
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        counts["array"] += 1
        for v in obj:
            for k, c in _leaf_type_counts(v).items():
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


class JsonStringLeafCountCheck:
    @property
    def name(self) -> str:
        return "json_str_leaves"

    def score(self, output: str, context: CheckContext) -> float:
        a_obj, b_obj = _try_json(context.baseline_output), _try_json(output)
        if a_obj is None or b_obj is None:
            return 1.0
        return _count_match(_leaf_type_counts(a_obj)["str"], _leaf_type_counts(b_obj)["str"])


class JsonNumberLeafCountCheck:
    @property
    def name(self) -> str:
        return "json_num_leaves"

    def score(self, output: str, context: CheckContext) -> float:
        a_obj, b_obj = _try_json(context.baseline_output), _try_json(output)
        if a_obj is None or b_obj is None:
            return 1.0
        return _count_match(_leaf_type_counts(a_obj)["num"], _leaf_type_counts(b_obj)["num"])


class JsonBoolLeafCountCheck:
    @property
    def name(self) -> str:
        return "json_bool_leaves"

    def score(self, output: str, context: CheckContext) -> float:
        a_obj, b_obj = _try_json(context.baseline_output), _try_json(output)
        if a_obj is None or b_obj is None:
            return 1.0
        return _count_match(_leaf_type_counts(a_obj)["bool"], _leaf_type_counts(b_obj)["bool"])


class JsonNullLeafCountCheck:
    @property
    def name(self) -> str:
        return "json_null_leaves"

    def score(self, output: str, context: CheckContext) -> float:
        a_obj, b_obj = _try_json(context.baseline_output), _try_json(output)
        if a_obj is None or b_obj is None:
            return 1.0
        return _count_match(_leaf_type_counts(a_obj)["null"], _leaf_type_counts(b_obj)["null"])


class JsonArrayLeafCountCheck:
    @property
    def name(self) -> str:
        return "json_array_leaves"

    def score(self, output: str, context: CheckContext) -> float:
        a_obj, b_obj = _try_json(context.baseline_output), _try_json(output)
        if a_obj is None or b_obj is None:
            return 1.0
        return _count_match(_leaf_type_counts(a_obj)["array"], _leaf_type_counts(b_obj)["array"])


class JsonObjectLeafCountCheck:
    @property
    def name(self) -> str:
        return "json_object_leaves"

    def score(self, output: str, context: CheckContext) -> float:
        a_obj, b_obj = _try_json(context.baseline_output), _try_json(output)
        if a_obj is None or b_obj is None:
            return 1.0
        return _count_match(
            _leaf_type_counts(a_obj)["object"], _leaf_type_counts(b_obj)["object"]
        )


class JsonDuplicateKeyRiskCheck:
    """Count of raw duplicate keys in the JSON text (parse would silently drop)."""

    @property
    def name(self) -> str:
        return "json_dup_key_risk"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            keys = re.findall(r'"([^"]+)"\s*:', _s(text))
            seen: set[str] = set()
            dups = 0
            for k in keys:
                if k in seen:
                    dups += 1
                else:
                    seen.add(k)
            return dups

        return _count_match(count(context.baseline_output), count(output))


class JsonUnicodeEscapeCountCheck:
    @property
    def name(self) -> str:
        return "json_unicode_escapes"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"\\u[0-9a-fA-F]{4}")
        return _count_match(
            len(pat.findall(_s(context.baseline_output))),
            len(pat.findall(_s(output))),
        )


class JsonPrettyPrintCheck:
    """1.0 when pretty-printed (multi-line with indent) presence matches the baseline."""

    @property
    def name(self) -> str:
        return "json_pretty"

    def score(self, output: str, context: CheckContext) -> float:
        def is_pretty(text: str) -> bool:
            s = _s(text)
            return "\n  " in s or "\n\t" in s

        return 1.0 if is_pretty(context.baseline_output) == is_pretty(output) else 0.0


class JsonKeyLengthStatsCheck:
    """Average length of JSON keys."""

    @property
    def name(self) -> str:
        return "json_key_len_avg"

    def score(self, output: str, context: CheckContext) -> float:
        def avg(text: str) -> float:
            keys = re.findall(r'"([^"]+)"\s*:', _s(text))
            return sum(len(k) for k in keys) / len(keys) if keys else 0.0

        base, cand = avg(context.baseline_output), avg(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)
