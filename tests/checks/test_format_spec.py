"""Tests for format_spec dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.format_spec import (
    CronLineCountCheck,
    DockerfileExposeCountCheck,
    DockerfileFromCountCheck,
    DockerfileRunCountCheck,
    EnvCommentCountCheck,
    EnvVarCountCheck,
    GitignorePatternCountCheck,
    IniKeyCountCheck,
    IniSectionCountCheck,
    MakefileRuleCountCheck,
    MakefileTargetCountCheck,
    NginxBlockCountCheck,
    NginxDirectiveCountCheck,
    SqlJoinCountCheck,
    SqlKeywordDensityCheck,
    SqlSelectCountCheck,
    SqlWhereCountCheck,
    SystemdUnitCountCheck,
    TomlKeyCountCheck,
    TomlTableCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

SQL = "SELECT a FROM t WHERE x = 1 JOIN u ON t.id = u.id"
DOCKER = "FROM python:3.12\nRUN pip install x\nEXPOSE 8000"
INI = "[section]\nkey = value\n"
TOML = "[package]\nname = \"x\"\n"
ENV = "# comment\nFOO=bar\nBAZ=qux\n"
MAKE = "build:\n\techo hi\ntest:\n\techo test\n"
NGINX = "server {\n    listen 80;\n    server_name x;\n}"
SYSTEMD = "[Unit]\n[Service]\n[Install]\n"
CRON = "# comment\n0 * * * * /bin/true\n"
GITIGNORE = "# comment\n*.pyc\n__pycache__/\n"


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_sql_counts():
    ctx = _ctx(SQL, SQL)
    assert SqlSelectCountCheck().score(SQL, ctx) == 1.0
    assert SqlJoinCountCheck().score(SQL, ctx) == 1.0
    assert SqlWhereCountCheck().score(SQL, ctx) == 1.0
    assert SqlKeywordDensityCheck().score(SQL, ctx) == 1.0
    nojoin = _ctx("SELECT a FROM t WHERE x = 1", "SELECT a FROM t WHERE x = 1")
    assert SqlJoinCountCheck().score(SQL, nojoin) < 1.0

def test_docker_counts():
    ctx = _ctx(DOCKER, DOCKER)
    assert DockerfileFromCountCheck().score(DOCKER, ctx) == 1.0
    assert DockerfileRunCountCheck().score(DOCKER, ctx) == 1.0
    assert DockerfileExposeCountCheck().score(DOCKER, ctx) == 1.0


def test_ini_toml_counts():
    ictx = _ctx(INI, INI)
    assert IniSectionCountCheck().score(INI, ictx) == 1.0
    assert IniKeyCountCheck().score(INI, ictx) == 1.0
    tctx = _ctx(TOML, TOML)
    assert TomlTableCountCheck().score(TOML, tctx) == 1.0
    assert TomlKeyCountCheck().score(TOML, tctx) == 1.0


def test_env_counts():
    ctx = _ctx(ENV, ENV)
    assert EnvVarCountCheck().score(ENV, ctx) == 1.0
    assert EnvCommentCountCheck().score(ENV, ctx) == 1.0
    assert EnvVarCountCheck().score("FOO=bar\n", ctx) < 1.0


def test_makefile_counts():
    ctx = _ctx(MAKE, MAKE)
    assert MakefileTargetCountCheck().score(MAKE, ctx) == 1.0
    assert MakefileRuleCountCheck().score(MAKE, ctx) == 1.0


def test_nginx_systemd_counts():
    nctx = _ctx(NGINX, NGINX)
    assert NginxBlockCountCheck().score(NGINX, nctx) == 1.0
    assert NginxDirectiveCountCheck().score(NGINX, nctx) == 1.0
    sctx = _ctx(SYSTEMD, SYSTEMD)
    assert SystemdUnitCountCheck().score(SYSTEMD, sctx) == 1.0


def test_cron_gitignore_counts():
    cctx = _ctx(CRON, CRON)
    assert CronLineCountCheck().score(CRON, cctx) == 1.0
    gctx = _ctx(GITIGNORE, GITIGNORE)
    assert GitignorePatternCountCheck().score(GITIGNORE, gctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": SQL,
         "candidate_output": SQL},
        {"id": "b", "prompt": "p", "baseline_output": SQL,
         "candidate_output": "SELECT 1"},
    ])
    result = evaluate(corpus, dimensions=("sql_select_count", "sql_where_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["sql_select_count"].losses == 0
    assert dims["sql_where_count"].losses == 1
