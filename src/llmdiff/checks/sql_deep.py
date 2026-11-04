"""Deep SQL dimensions: clauses, joins, subqueries, and DML shapes."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "SqlFromCountCheck",
    "SqlGroupByCountCheck",
    "SqlOrderByCountCheck",
    "SqlHavingCountCheck",
    "SqlLimitCountCheck",
    "SqlUnionCountCheck",
    "SqlSubqueryCountCheck",
    "SqlCteCountCheck",
    "SqlInsertCountCheck",
    "SqlUpdateCountCheck",
    "SqlDeleteCountCheck",
    "SqlCreateCountCheck",
    "SqlDropCountCheck",
    "SqlAlterCountCheck",
    "SqlCaseCountCheck",
    "SqlDistinctCountCheck",
    "SqlAggregateFuncCountCheck",
    "SqlStringFuncCountCheck",
    "SqlDateFuncCountCheck",
    "SqlSemicolonTermCountCheck",
]

_MAX = 200_000
_FENCE_RE = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)

_KEYWORDS: dict[str, re.Pattern[str]] = {
    "from": re.compile(r"\bfrom\b", re.IGNORECASE),
    "group_by": re.compile(r"\bgroup\s+by\b", re.IGNORECASE),
    "order_by": re.compile(r"\border\s+by\b", re.IGNORECASE),
    "having": re.compile(r"\bhaving\b", re.IGNORECASE),
    "limit": re.compile(r"\blimit\b", re.IGNORECASE),
    "union": re.compile(r"\bunion\s+(?:all\s+)?\b", re.IGNORECASE),
    "subquery": re.compile(r"\(\s*select\b", re.IGNORECASE),
    "cte": re.compile(r"\bwith\b[^;]*?\bas\b", re.IGNORECASE | re.DOTALL),
    "insert": re.compile(r"\binsert\s+into\b", re.IGNORECASE),
    "update": re.compile(r"\bupdate\b", re.IGNORECASE),
    "delete": re.compile(r"\bdelete\s+from\b", re.IGNORECASE),
    "create": re.compile(r"\bcreate\s+(?:table|index|view|database)\b", re.IGNORECASE),
    "drop": re.compile(r"\bdrop\s+(?:table|index|view|database)\b", re.IGNORECASE),
    "alter": re.compile(r"\balter\s+table\b", re.IGNORECASE),
    "case": re.compile(r"\bcase\b", re.IGNORECASE),
    "distinct": re.compile(r"\bdistinct\b", re.IGNORECASE),
    "agg": re.compile(r"\b(?:count|sum|avg|min|max)\s*\(", re.IGNORECASE),
    "strfunc": re.compile(
        r"\b(?:concat|substr|substring|upper|lower|trim|replace|length|char_length)\s*\(",
        re.IGNORECASE,
    ),
    "datefunc": re.compile(
        r"\b(?:now|current_date|current_timestamp|date_trunc|extract|date_add|datediff)\s*\(",
        re.IGNORECASE,
    ),
}
_SEMI_RE = re.compile(r";\s*$", re.MULTILINE)


def _s(text: str) -> str:
    return text[:_MAX]


def _code(text: str) -> str:
    blocks = [m.group(1) for m in _FENCE_RE.finditer(_s(text))]
    return "\n".join(blocks) if blocks else _s(text)


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _kw_count(text: str, key: str) -> int:
    return len(_KEYWORDS[key].findall(_code(text)))


class SqlFromCountCheck:
    @property
    def name(self) -> str:
        return "sql_from_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(_kw_count(context.baseline_output, "from"), _kw_count(output, "from"))


class SqlGroupByCountCheck:
    @property
    def name(self) -> str:
        return "sql_groupby_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _kw_count(context.baseline_output, "group_by"),
            _kw_count(output, "group_by"),
        )


class SqlOrderByCountCheck:
    @property
    def name(self) -> str:
        return "sql_orderby_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _kw_count(context.baseline_output, "order_by"),
            _kw_count(output, "order_by"),
        )


class SqlHavingCountCheck:
    @property
    def name(self) -> str:
        return "sql_having_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _kw_count(context.baseline_output, "having"),
            _kw_count(output, "having"),
        )


class SqlLimitCountCheck:
    @property
    def name(self) -> str:
        return "sql_limit_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _kw_count(context.baseline_output, "limit"),
            _kw_count(output, "limit"),
        )


class SqlUnionCountCheck:
    @property
    def name(self) -> str:
        return "sql_union_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _kw_count(context.baseline_output, "union"),
            _kw_count(output, "union"),
        )


class SqlSubqueryCountCheck:
    @property
    def name(self) -> str:
        return "sql_subquery_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _kw_count(context.baseline_output, "subquery"),
            _kw_count(output, "subquery"),
        )


class SqlCteCountCheck:
    @property
    def name(self) -> str:
        return "sql_cte_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _kw_count(context.baseline_output, "cte"),
            _kw_count(output, "cte"),
        )


class SqlInsertCountCheck:
    @property
    def name(self) -> str:
        return "sql_insert_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _kw_count(context.baseline_output, "insert"),
            _kw_count(output, "insert"),
        )


class SqlUpdateCountCheck:
    @property
    def name(self) -> str:
        return "sql_update_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _kw_count(context.baseline_output, "update"),
            _kw_count(output, "update"),
        )


class SqlDeleteCountCheck:
    @property
    def name(self) -> str:
        return "sql_delete_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _kw_count(context.baseline_output, "delete"),
            _kw_count(output, "delete"),
        )


class SqlCreateCountCheck:
    @property
    def name(self) -> str:
        return "sql_create_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _kw_count(context.baseline_output, "create"),
            _kw_count(output, "create"),
        )


class SqlDropCountCheck:
    @property
    def name(self) -> str:
        return "sql_drop_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _kw_count(context.baseline_output, "drop"),
            _kw_count(output, "drop"),
        )


class SqlAlterCountCheck:
    @property
    def name(self) -> str:
        return "sql_alter_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _kw_count(context.baseline_output, "alter"),
            _kw_count(output, "alter"),
        )


class SqlCaseCountCheck:
    @property
    def name(self) -> str:
        return "sql_case_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _kw_count(context.baseline_output, "case"),
            _kw_count(output, "case"),
        )


class SqlDistinctCountCheck:
    @property
    def name(self) -> str:
        return "sql_distinct_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _kw_count(context.baseline_output, "distinct"),
            _kw_count(output, "distinct"),
        )


class SqlAggregateFuncCountCheck:
    @property
    def name(self) -> str:
        return "sql_agg_func_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _kw_count(context.baseline_output, "agg"),
            _kw_count(output, "agg"),
        )


class SqlStringFuncCountCheck:
    @property
    def name(self) -> str:
        return "sql_strfunc_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _kw_count(context.baseline_output, "strfunc"),
            _kw_count(output, "strfunc"),
        )


class SqlDateFuncCountCheck:
    @property
    def name(self) -> str:
        return "sql_datefunc_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _kw_count(context.baseline_output, "datefunc"),
            _kw_count(output, "datefunc"),
        )


class SqlSemicolonTermCountCheck:
    @property
    def name(self) -> str:
        return "sql_semicolon_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SEMI_RE.findall(_code(context.baseline_output))),
            len(_SEMI_RE.findall(_code(output))),
        )
