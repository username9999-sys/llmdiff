"""Tests for sql_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.sql_deep import (
    SqlAggregateFuncCountCheck,
    SqlAlterCountCheck,
    SqlCaseCountCheck,
    SqlCreateCountCheck,
    SqlCteCountCheck,
    SqlDateFuncCountCheck,
    SqlDeleteCountCheck,
    SqlDistinctCountCheck,
    SqlDropCountCheck,
    SqlFromCountCheck,
    SqlGroupByCountCheck,
    SqlHavingCountCheck,
    SqlInsertCountCheck,
    SqlLimitCountCheck,
    SqlOrderByCountCheck,
    SqlSemicolonTermCountCheck,
    SqlStringFuncCountCheck,
    SqlSubqueryCountCheck,
    SqlUnionCountCheck,
    SqlUpdateCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

SQL = (
    "WITH t AS (SELECT 1) "
    "SELECT DISTINCT a, count(*), upper(b) FROM t "
    "WHERE x IN (SELECT y FROM u) "
    "GROUP BY a HAVING count(*) > 1 "
    "UNION ALL SELECT 2 FROM v "
    "ORDER BY a LIMIT 10;\n"
    "INSERT INTO t VALUES (1);\n"
    "UPDATE t SET a = 2;\n"
    "DELETE FROM t WHERE a = 3;\n"
    "CREATE TABLE z (id INT);\n"
    "ALTER TABLE z ADD c TEXT;\n"
    "DROP TABLE z;\n"
    "SELECT CASE WHEN a THEN 1 ELSE 0 END, now() FROM t;\n"
)


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_select_clauses():
    ctx = _ctx(SQL, SQL)
    assert SqlFromCountCheck().score(SQL, ctx) == 1.0
    assert SqlGroupByCountCheck().score(SQL, ctx) == 1.0
    assert SqlOrderByCountCheck().score(SQL, ctx) == 1.0
    assert SqlHavingCountCheck().score(SQL, ctx) == 1.0
    assert SqlLimitCountCheck().score(SQL, ctx) == 1.0


def test_set_and_subquery():
    ctx = _ctx(SQL, SQL)
    assert SqlUnionCountCheck().score(SQL, ctx) == 1.0
    assert SqlSubqueryCountCheck().score(SQL, ctx) == 1.0
    assert SqlCteCountCheck().score(SQL, ctx) == 1.0
    assert SqlDistinctCountCheck().score(SQL, ctx) == 1.0
    assert SqlCaseCountCheck().score(SQL, ctx) == 1.0


def test_dml_ddl():
    ctx = _ctx(SQL, SQL)
    assert SqlInsertCountCheck().score(SQL, ctx) == 1.0
    assert SqlUpdateCountCheck().score(SQL, ctx) == 1.0
    assert SqlDeleteCountCheck().score(SQL, ctx) == 1.0
    assert SqlCreateCountCheck().score(SQL, ctx) == 1.0
    assert SqlAlterCountCheck().score(SQL, ctx) == 1.0
    assert SqlDropCountCheck().score(SQL, ctx) == 1.0


def test_functions_and_semicolon():
    ctx = _ctx(SQL, SQL)
    assert SqlAggregateFuncCountCheck().score(SQL, ctx) == 1.0
    assert SqlStringFuncCountCheck().score(SQL, ctx) == 1.0
    assert SqlDateFuncCountCheck().score(SQL, ctx) == 1.0
    assert SqlSemicolonTermCountCheck().score(SQL, ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": SQL,
         "candidate_output": SQL},
        {"id": "b", "prompt": "p", "baseline_output": SQL,
         "candidate_output": "SELECT 1"},
    ])
    result = evaluate(corpus, dimensions=("sql_from_count", "sql_groupby_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["sql_from_count"].losses == 1
    assert dims["sql_groupby_count"].losses == 1
