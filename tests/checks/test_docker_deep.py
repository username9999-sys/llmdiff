"""Tests for docker_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.docker_deep import (
    ComposeEnvCountCheck,
    ComposeImageCountCheck,
    ComposePortCountCheck,
    ComposeServiceCountCheck,
    ComposeVolumeCountCheck,
    DockerArgCountCheck,
    DockerCmdCountCheck,
    DockerCopyCountCheck,
    DockerEntrypointCountCheck,
    DockerEnvCountCheck,
    DockerFromStageCountCheck,
    DockerHealthcheckCountCheck,
    DockerLabelCountCheck,
    DockerRunCountDeepCheck,
    DockerUserCountCheck,
    DockerVolumeCountCheck,
    DockerWorkdirCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

DOCKERFILE = (
    "FROM python:3.12 AS base\n"
    "ARG VERSION=1\n"
    "ENV APP_ENV=prod\n"
    "WORKDIR /app\n"
    "COPY . .\n"
    "RUN pip install -r req.txt\n"
    "RUN apt-get update\n"
    "LABEL maintainer=x\n"
    "USER app\n"
    "VOLUME /data\n"
    "EXPOSE 8000\n"
    "HEALTHCHECK CMD curl -f http://localhost || exit 1\n"
    "ENTRYPOINT [\"python\"]\n"
    "CMD [\"app.py\"]\n"
    "FROM base AS final\n"
)
COMPOSE = (
    "services:\n"
    "  web:\n"
    "    image: nginx\n"
    "    ports:\n"
    '      - "80:80"\n'
    '      - "443:443"\n'
    "    volumes:\n"
    "      - ./html:/usr/share/nginx/html\n"
    "    environment:\n"
    "      - NGINX_HOST=example.com\n"
    "  db:\n"
    "    image: postgres\n"
)


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_dockerfile_instructions():
    ctx = _ctx(DOCKERFILE, DOCKERFILE)
    assert DockerFromStageCountCheck().score(DOCKERFILE, ctx) == 1.0
    assert DockerRunCountDeepCheck().score(DOCKERFILE, ctx) == 1.0
    assert DockerCopyCountCheck().score(DOCKERFILE, ctx) == 1.0
    assert DockerWorkdirCountCheck().score(DOCKERFILE, ctx) == 1.0
    assert DockerEnvCountCheck().score(DOCKERFILE, ctx) == 1.0
    assert DockerArgCountCheck().score(DOCKERFILE, ctx) == 1.0
    assert DockerCmdCountCheck().score(DOCKERFILE, ctx) == 1.0
    assert DockerEntrypointCountCheck().score(DOCKERFILE, ctx) == 1.0
    assert DockerHealthcheckCountCheck().score(DOCKERFILE, ctx) == 1.0
    assert DockerUserCountCheck().score(DOCKERFILE, ctx) == 1.0
    assert DockerVolumeCountCheck().score(DOCKERFILE, ctx) == 1.0
    assert DockerLabelCountCheck().score(DOCKERFILE, ctx) == 1.0
    assert DockerRunCountDeepCheck().score("FROM x\n", ctx) < 1.0


def test_compose():
    ctx = _ctx(COMPOSE, COMPOSE)
    assert ComposeServiceCountCheck().score(COMPOSE, ctx) == 1.0
    assert ComposeImageCountCheck().score(COMPOSE, ctx) == 1.0
    assert ComposePortCountCheck().score(COMPOSE, ctx) == 1.0
    assert ComposeVolumeCountCheck().score(COMPOSE, ctx) == 1.0
    assert ComposeEnvCountCheck().score(COMPOSE, ctx) == 1.0
    assert ComposeServiceCountCheck().score("services:\n", ctx) < 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": DOCKERFILE,
         "candidate_output": DOCKERFILE},
        {"id": "b", "prompt": "p", "baseline_output": DOCKERFILE,
         "candidate_output": "FROM scratch\n"},
    ])
    result = evaluate(corpus, dimensions=("docker_run_count_deep", "docker_from_stage_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["docker_run_count_deep"].losses == 1
    assert dims["docker_from_stage_count"].losses == 1
