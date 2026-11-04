"""Deep Docker dimensions: Dockerfile instructions, compose services, and healthchecks."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "DockerFromStageCountCheck",
    "DockerRunCountDeepCheck",
    "DockerCopyCountCheck",
    "DockerAddCountCheck",
    "DockerWorkdirCountCheck",
    "DockerEnvCountCheck",
    "DockerArgCountCheck",
    "DockerCmdCountCheck",
    "DockerEntrypointCountCheck",
    "DockerHealthcheckCountCheck",
    "DockerUserCountCheck",
    "DockerVolumeCountCheck",
    "DockerLabelCountCheck",
    "DockerOnbuildCountCheck",
    "DockerShellCountCheck",
    "ComposeServiceCountCheck",
    "ComposeImageCountCheck",
    "ComposePortCountCheck",
    "ComposeVolumeCountCheck",
    "ComposeEnvCountCheck",
]

_MAX = 200_000
_FENCE_RE = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)

_DOCKER_INSTR = {
    "from_stage": re.compile(r"^FROM\s+\S+", re.MULTILINE | re.IGNORECASE),
    "run": re.compile(r"^RUN\s+", re.MULTILINE | re.IGNORECASE),
    "copy": re.compile(r"^COPY\s+", re.MULTILINE | re.IGNORECASE),
    "add": re.compile(r"^ADD\s+", re.MULTILINE | re.IGNORECASE),
    "workdir": re.compile(r"^WORKDIR\s+", re.MULTILINE | re.IGNORECASE),
    "env": re.compile(r"^ENV\s+", re.MULTILINE | re.IGNORECASE),
    "arg": re.compile(r"^ARG\s+", re.MULTILINE | re.IGNORECASE),
    "cmd": re.compile(r"^CMD\s+", re.MULTILINE | re.IGNORECASE),
    "entrypoint": re.compile(r"^ENTRYPOINT\s+", re.MULTILINE | re.IGNORECASE),
    "healthcheck": re.compile(r"^HEALTHCHECK\s+", re.MULTILINE | re.IGNORECASE),
    "user": re.compile(r"^USER\s+", re.MULTILINE | re.IGNORECASE),
    "volume": re.compile(r"^VOLUME\s+", re.MULTILINE | re.IGNORECASE),
    "label": re.compile(r"^LABEL\s+", re.MULTILINE | re.IGNORECASE),
    "onbuild": re.compile(r"^ONBUILD\s+", re.MULTILINE | re.IGNORECASE),
    "shell": re.compile(r"^SHELL\s+", re.MULTILINE | re.IGNORECASE),
}
_COMPOSE_SERVICE_RE = re.compile(r"^  [\w.-]+:\s*$", re.MULTILINE)
_COMPOSE_IMAGE_RE = re.compile(r"^\s+image:\s*\S+", re.MULTILINE)
_COMPOSE_PORT_RE = re.compile(r'^\s+-\s+"?\d+:\d+"?', re.MULTILINE)
_COMPOSE_VOLUME_RE = re.compile(r"^\s+-\s+\S+:\S+", re.MULTILINE)
_COMPOSE_ENV_RE = re.compile(r"^\s+-\s+[A-Z_][A-Z0-9_]*=", re.MULTILINE)


def _s(text: str) -> str:
    return text[:_MAX]


def _code(text: str) -> str:
    blocks = [m.group(1) for m in _FENCE_RE.finditer(_s(text))]
    return "\n".join(blocks) if blocks else _s(text)


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _instr_count(text: str, key: str) -> int:
    return len(_DOCKER_INSTR[key].findall(_code(text)))


class DockerFromStageCountCheck:
    @property
    def name(self) -> str:
        return "docker_from_stage_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _instr_count(context.baseline_output, "from_stage"),
            _instr_count(output, "from_stage"),
        )


class DockerRunCountDeepCheck:
    @property
    def name(self) -> str:
        return "docker_run_count_deep"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _instr_count(context.baseline_output, "run"),
            _instr_count(output, "run"),
        )


class DockerCopyCountCheck:
    @property
    def name(self) -> str:
        return "docker_copy_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _instr_count(context.baseline_output, "copy"),
            _instr_count(output, "copy"),
        )


class DockerAddCountCheck:
    @property
    def name(self) -> str:
        return "docker_add_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _instr_count(context.baseline_output, "add"),
            _instr_count(output, "add"),
        )


class DockerWorkdirCountCheck:
    @property
    def name(self) -> str:
        return "docker_workdir_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _instr_count(context.baseline_output, "workdir"),
            _instr_count(output, "workdir"),
        )


class DockerEnvCountCheck:
    @property
    def name(self) -> str:
        return "docker_env_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _instr_count(context.baseline_output, "env"),
            _instr_count(output, "env"),
        )


class DockerArgCountCheck:
    @property
    def name(self) -> str:
        return "docker_arg_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _instr_count(context.baseline_output, "arg"),
            _instr_count(output, "arg"),
        )


class DockerCmdCountCheck:
    @property
    def name(self) -> str:
        return "docker_cmd_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _instr_count(context.baseline_output, "cmd"),
            _instr_count(output, "cmd"),
        )


class DockerEntrypointCountCheck:
    @property
    def name(self) -> str:
        return "docker_entrypoint_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _instr_count(context.baseline_output, "entrypoint"),
            _instr_count(output, "entrypoint"),
        )


class DockerHealthcheckCountCheck:
    @property
    def name(self) -> str:
        return "docker_healthcheck_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _instr_count(context.baseline_output, "healthcheck"),
            _instr_count(output, "healthcheck"),
        )


class DockerUserCountCheck:
    @property
    def name(self) -> str:
        return "docker_user_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _instr_count(context.baseline_output, "user"),
            _instr_count(output, "user"),
        )


class DockerVolumeCountCheck:
    @property
    def name(self) -> str:
        return "docker_volume_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _instr_count(context.baseline_output, "volume"),
            _instr_count(output, "volume"),
        )


class DockerLabelCountCheck:
    @property
    def name(self) -> str:
        return "docker_label_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _instr_count(context.baseline_output, "label"),
            _instr_count(output, "label"),
        )


class DockerOnbuildCountCheck:
    @property
    def name(self) -> str:
        return "docker_onbuild_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _instr_count(context.baseline_output, "onbuild"),
            _instr_count(output, "onbuild"),
        )


class DockerShellCountCheck:
    @property
    def name(self) -> str:
        return "docker_shell_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _instr_count(context.baseline_output, "shell"),
            _instr_count(output, "shell"),
        )


class ComposeServiceCountCheck:
    @property
    def name(self) -> str:
        return "compose_service_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_COMPOSE_SERVICE_RE.findall(_s(context.baseline_output))),
            len(_COMPOSE_SERVICE_RE.findall(_s(output))),
        )


class ComposeImageCountCheck:
    @property
    def name(self) -> str:
        return "compose_image_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_COMPOSE_IMAGE_RE.findall(_s(context.baseline_output))),
            len(_COMPOSE_IMAGE_RE.findall(_s(output))),
        )


class ComposePortCountCheck:
    @property
    def name(self) -> str:
        return "compose_port_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_COMPOSE_PORT_RE.findall(_s(context.baseline_output))),
            len(_COMPOSE_PORT_RE.findall(_s(output))),
        )


class ComposeVolumeCountCheck:
    @property
    def name(self) -> str:
        return "compose_volume_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_COMPOSE_VOLUME_RE.findall(_s(context.baseline_output))),
            len(_COMPOSE_VOLUME_RE.findall(_s(output))),
        )


class ComposeEnvCountCheck:
    @property
    def name(self) -> str:
        return "compose_env_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_COMPOSE_ENV_RE.findall(_s(context.baseline_output))),
            len(_COMPOSE_ENV_RE.findall(_s(output))),
        )
