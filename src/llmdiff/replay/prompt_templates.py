"""Prompt templating for replay.

Prompts in a corpus may contain placeholders filled from each case's metadata
(``{var}`` for :meth:`str.format`, ``{{ var }}`` for Jinja2, ``$var`` for
:class:`string.Template`). Jinja2 is used when installed; otherwise we fall
back to ``str.format`` and then :class:`string.Template`. :class:`TemplateProvider`
wraps a provider so the template is rendered per case before the call.
"""

from __future__ import annotations

import string
from collections.abc import Mapping
from typing import Any

from llmdiff.replay.provider import Provider, ProviderResponse

__all__ = ["render_template", "TemplateProvider", "JINJA2_AVAILABLE"]

try:  # optional dependency
    import jinja2 as _jinja2  # type: ignore[import-not-found]

    JINJA2_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    _jinja2 = None  # type: ignore[assignment]
    JINJA2_AVAILABLE = False


def render_template(template: str, variables: Mapping[str, Any]) -> str:
    """Render ``template`` with ``variables``.

    Prefers Jinja2 (``{{ name }}`` syntax) when available; falls back to
    ``str.format`` (``{name}``), then :class:`string.Template` (``$name``).
    Missing keys raise :class:`KeyError` for format/Template and
    :class:`jinja2.UndefinedError` semantics for Jinja2 (undefined renders as
    empty string unless strict).
    """
    has_jinja = "{{" in template and "}}" in template
    has_format = "{" in template and "}" in template and not has_jinja
    if has_jinja and JINJA2_AVAILABLE:
        env = _jinja2.Environment(autoescape=False, keep_trailing_newline=True)
        return env.from_string(template).render(**dict(variables))
    if has_format:
        return template.format(**dict(variables))
    if "$" in template:
        subs = {k: str(v) for k, v in variables.items()}
        return string.Template(template).safe_substitute(**subs)
    return template
    # Looks like Jinja syntax but Jinja2 is missing: emulate the common case.
    out = template
    for key, value in variables.items():
        out = out.replace("{{ " + key + " }}", str(value)).replace("{{" + key + "}}", str(value))
    return out


class TemplateProvider:
    """Wrap a provider, rendering a prompt template per call.

    The template is applied to the incoming prompt: variables come from the
    ``template_vars`` passed as call params (merged over the provider-level
    defaults). The rendered text is what the wrapped provider sees.
    """

    def __init__(
        self,
        provider: Provider,
        template: str,
        *,
        default_vars: Mapping[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.template = template
        self.default_vars = dict(default_vars or {})

    @property
    def name(self) -> str:
        return f"template({self.provider.name})"

    def render(self, prompt: str, extra_vars: Mapping[str, Any] | None = None) -> str:
        variables = dict(self.default_vars)
        if extra_vars:
            variables.update(extra_vars)
        variables.setdefault("prompt", prompt)
        return render_template(self.template, variables)

    def complete(self, prompt: str, **params: Any) -> ProviderResponse:
        template_vars = params.pop("template_vars", None)
        rendered = self.render(prompt, template_vars)
        return self.provider.complete(rendered, **params)
