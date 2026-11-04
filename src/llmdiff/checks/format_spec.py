"""Format-specific dimensions: SQL, Dockerfile, INI, TOML, env, and lockfile shapes."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "SqlSelectCountCheck",
    "SqlJoinCountCheck",
    "SqlWhereCountCheck",
    "SqlKeywordDensityCheck",
    "DockerfileFromCountCheck",
    "DockerfileRunCountCheck",
    "DockerfileExposeCountCheck",
    "IniSectionCountCheck",
    "IniKeyCountCheck",
    "TomlTableCountCheck",
    "TomlKeyCountCheck",
    "EnvVarCountCheck",
    "EnvCommentCountCheck",
    "MakefileTargetCountCheck",
    "MakefileRuleCountCheck",
    "NginxBlockCountCheck",
    "NginxDirectiveCountCheck",
    "SystemdUnitCountCheck",
    "CronLineCountCheck",
    "GitignorePatternCountCheck",
]

_MAX = 200_000
_FENCE_RE = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)

_SQL_SELECT = re.compile(r"\bselect\b", re.IGNORECASE)
_SQL_JOIN = re.compile(r"\b(?:(?:inner|left|right|full|cross)\s+)?join\b", re.IGNORECASE)
_SQL_WHERE = re.compile(r"\bwhere\b", re.IGNORECASE)
_SQL_KEYWORDS = re.compile(
    r"\b(?:select|from|where|join|group by|order by|having|limit|"
    r"insert|update|delete|create|alter|drop)\b",
    re.IGNORECASE,
)
_DOCKER_FROM = re.compile(r"^FROM\s+\S+", re.MULTILINE | re.IGNORECASE)
_DOCKER_RUN = re.compile(r"^RUN\s+", re.MULTILINE | re.IGNORECASE)
_DOCKER_EXPOSE = re.compile(r"^EXPOSE\s+", re.MULTILINE | re.IGNORECASE)
_INI_SECTION = re.compile(r"^\s*\[[^\]]+\]\s*$", re.MULTILINE)
_INI_KEY = re.compile(r"^\s*[\w.-]+\s*=", re.MULTILINE)
_TOML_TABLE = re.compile(r"^\s*\[[^\]]+\]\s*$", re.MULTILINE)
_TOML_KEY = re.compile(r"^\s*[\w.-]+\s*=", re.MULTILINE)
_ENV_VAR = re.compile(r"^[A-Z_][A-Z0-9_]*=", re.MULTILINE)
_ENV_COMMENT = re.compile(r"^\s*#", re.MULTILINE)
_MAKE_TARGET = re.compile(r"^[A-Za-z_][\w.-]*\s*:", re.MULTILINE)
_MAKE_RULE = re.compile(r"^\t", re.MULTILINE)
_NGINX_BLOCK = re.compile(r"\b(?:server|location|http|events|upstream)\s*\{", re.IGNORECASE)
_NGINX_DIRECTIVE = re.compile(
    r"^\s*(?:listen|server_name|root|proxy_pass|return|rewrite|include)\s+",
    re.MULTILINE | re.IGNORECASE,
)
_SYSTEMD_UNIT = re.compile(r"^\s*\[(?:Unit|Service|Install)\]\s*$", re.MULTILINE)


def _s(text: str) -> str:
    return text[:_MAX]


def _code(text: str) -> str:
    blocks = [m.group(1) for m in _FENCE_RE.finditer(_s(text))]
    return "\n".join(blocks) if blocks else _s(text)


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


class SqlSelectCountCheck:
    @property
    def name(self) -> str:
        return "sql_select_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SQL_SELECT.findall(_code(context.baseline_output))),
            len(_SQL_SELECT.findall(_code(output))),
        )


class SqlJoinCountCheck:
    @property
    def name(self) -> str:
        return "sql_join_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SQL_JOIN.findall(_code(context.baseline_output))),
            len(_SQL_JOIN.findall(_code(output))),
        )


class SqlWhereCountCheck:
    @property
    def name(self) -> str:
        return "sql_where_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SQL_WHERE.findall(_code(context.baseline_output))),
            len(_SQL_WHERE.findall(_code(output))),
        )


class SqlKeywordDensityCheck:
    @property
    def name(self) -> str:
        return "sql_kw_density"

    def score(self, output: str, context: CheckContext) -> float:
        def density(text: str) -> float:
            code = _code(text)
            words = len(re.findall(r"\w+", code))
            return len(_SQL_KEYWORDS.findall(code)) / words if words else 0.0

        return _ratio_match(density(context.baseline_output), density(output))


class DockerfileFromCountCheck:
    @property
    def name(self) -> str:
        return "docker_from_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_DOCKER_FROM.findall(_code(context.baseline_output))),
            len(_DOCKER_FROM.findall(_code(output))),
        )


class DockerfileRunCountCheck:
    @property
    def name(self) -> str:
        return "docker_run_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_DOCKER_RUN.findall(_code(context.baseline_output))),
            len(_DOCKER_RUN.findall(_code(output))),
        )


class DockerfileExposeCountCheck:
    @property
    def name(self) -> str:
        return "docker_expose_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_DOCKER_EXPOSE.findall(_code(context.baseline_output))),
            len(_DOCKER_EXPOSE.findall(_code(output))),
        )


class IniSectionCountCheck:
    @property
    def name(self) -> str:
        return "ini_section_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_INI_SECTION.findall(_s(context.baseline_output))),
            len(_INI_SECTION.findall(_s(output))),
        )


class IniKeyCountCheck:
    @property
    def name(self) -> str:
        return "ini_key_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_INI_KEY.findall(_s(context.baseline_output))),
            len(_INI_KEY.findall(_s(output))),
        )


class TomlTableCountCheck:
    @property
    def name(self) -> str:
        return "toml_table_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_TOML_TABLE.findall(_s(context.baseline_output))),
            len(_TOML_TABLE.findall(_s(output))),
        )


class TomlKeyCountCheck:
    @property
    def name(self) -> str:
        return "toml_key_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_TOML_KEY.findall(_s(context.baseline_output))),
            len(_TOML_KEY.findall(_s(output))),
        )


class EnvVarCountCheck:
    @property
    def name(self) -> str:
        return "env_var_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ENV_VAR.findall(_s(context.baseline_output))),
            len(_ENV_VAR.findall(_s(output))),
        )


class EnvCommentCountCheck:
    @property
    def name(self) -> str:
        return "env_comment_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ENV_COMMENT.findall(_s(context.baseline_output))),
            len(_ENV_COMMENT.findall(_s(output))),
        )


class MakefileTargetCountCheck:
    @property
    def name(self) -> str:
        return "make_target_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_MAKE_TARGET.findall(_s(context.baseline_output))),
            len(_MAKE_TARGET.findall(_s(output))),
        )


class MakefileRuleCountCheck:
    @property
    def name(self) -> str:
        return "make_rule_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_MAKE_RULE.findall(_s(context.baseline_output))),
            len(_MAKE_RULE.findall(_s(output))),
        )


class NginxBlockCountCheck:
    @property
    def name(self) -> str:
        return "nginx_block_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_NGINX_BLOCK.findall(_s(context.baseline_output))),
            len(_NGINX_BLOCK.findall(_s(output))),
        )


class NginxDirectiveCountCheck:
    @property
    def name(self) -> str:
        return "nginx_directive_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_NGINX_DIRECTIVE.findall(_s(context.baseline_output))),
            len(_NGINX_DIRECTIVE.findall(_s(output))),
        )


class SystemdUnitCountCheck:
    @property
    def name(self) -> str:
        return "systemd_unit_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SYSTEMD_UNIT.findall(_s(context.baseline_output))),
            len(_SYSTEMD_UNIT.findall(_s(output))),
        )


class CronLineCountCheck:
    @property
    def name(self) -> str:
        return "cron_line_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(
                1
                for ln in _s(text).splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            )

        return _count_match(count(context.baseline_output), count(output))


class GitignorePatternCountCheck:
    @property
    def name(self) -> str:
        return "gitignore_pattern_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(
                1
                for ln in _s(text).splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            )

        return _count_match(count(context.baseline_output), count(output))
