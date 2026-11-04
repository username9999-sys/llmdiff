"""Deep multi-language code dimensions: language-specific constructs and idioms."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "PythonFstringCountCheck",
    "PythonListcompCountCheck",
    "PythonDecoratorCountCheck",
    "PythonTypeHintCountCheck",
    "PythonAsyncDefCountCheck",
    "JsConstCountCheck",
    "JsArrowFuncCountCheck",
    "JsTemplateLiteralCountCheck",
    "JsDestructuringCountCheck",
    "JsSpreadCountCheck",
    "GoGoroutineCountCheck",
    "GoChannelCountCheck",
    "GoDeferCountCheck",
    "GoErrorCheckCountCheck",
    "RustMatchArmCountCheck",
    "RustResultUnwrapCountCheck",
    "RustLifetimeCountCheck",
    "JavaAnnotationCountCheck",
    "JavaStreamCountCheck",
    "CSharpAsyncCountCheck",
]

_MAX = 200_000
_FENCE_RE = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)

_PY_FSTRING = re.compile(r"f[\"'][^\"']*[{\[]")
_PY_LISTCOMP = re.compile(r"\[[^\[\]]+\bfor\b[^\[\]]+\bin\b[^\[\]]+\]")
_PY_DECOR = re.compile(r"^\s*@\w+", re.MULTILINE)
_PY_TYPEHINT = re.compile(
    r"\)\s*->\s*\w+|:\s*(?:int|str|bool|float|list|dict|set|tuple|None|Any)\b"
)
_PY_ASYNC_DEF = re.compile(r"^\s*async\s+def\s+\w+", re.MULTILINE)
_JS_CONST = re.compile(r"\bconst\s+\w+\s*=")
_JS_ARROW = re.compile(r"=>")
_JS_TMPL = re.compile(r"`[^`]*\$\{[^}]+\}[^`]*`")
_JS_DESTRUCT = re.compile(
    r"(?:const|let|var)\s*\{[^}]+\}\s*=|(?:const|let|var)\s*\[[^\]]+\]\s*="
)
_JS_SPREAD = re.compile(r"\.\.\.\w+")
_GO_GO = re.compile(r"\bgo\s+\w+\s*\(")
_GO_CHAN = re.compile(r"\bchan\b|<-\s*\w+|\w+\s*<-")
_GO_DEFER = re.compile(r"^\s*defer\s+", re.MULTILINE)
_GO_ERR = re.compile(r"\berr\s*!=\s*nil\b|\berr\s*==\s*nil\b")
_RUST_MATCH_ARM = re.compile(r"^\s*\w+\s*(?:\([^)]*\)|\{[^}]*\})?\s*=>", re.MULTILINE)
_RUST_UNWRAP = re.compile(r"\.(?:unwrap|expect|unwrap_or)\s*\(")
_RUST_LIFETIME = re.compile(r"'\w+")
_JAVA_ANNO = re.compile(r"^\s*@\w+", re.MULTILINE)
_JAVA_STREAM = re.compile(r"\.\s*(?:stream|map|filter|collect|forEach)\s*\(")
_CSHARP_ASYNC = re.compile(r"\b(?:async|await)\b")


def _s(text: str) -> str:
    return text[:_MAX]


def _code(text: str) -> str:
    blocks = [m.group(1) for m in _FENCE_RE.finditer(_s(text))]
    return "\n".join(blocks) if blocks else _s(text)


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class PythonFstringCountCheck:
    @property
    def name(self) -> str:
        return "py_fstring_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PY_FSTRING.findall(_code(context.baseline_output))),
            len(_PY_FSTRING.findall(_code(output))),
        )


class PythonListcompCountCheck:
    @property
    def name(self) -> str:
        return "py_listcomp_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PY_LISTCOMP.findall(_code(context.baseline_output))),
            len(_PY_LISTCOMP.findall(_code(output))),
        )


class PythonDecoratorCountCheck:
    @property
    def name(self) -> str:
        return "py_decorator_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PY_DECOR.findall(_code(context.baseline_output))),
            len(_PY_DECOR.findall(_code(output))),
        )


class PythonTypeHintCountCheck:
    @property
    def name(self) -> str:
        return "py_typehint_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PY_TYPEHINT.findall(_code(context.baseline_output))),
            len(_PY_TYPEHINT.findall(_code(output))),
        )


class PythonAsyncDefCountCheck:
    @property
    def name(self) -> str:
        return "py_async_def_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PY_ASYNC_DEF.findall(_code(context.baseline_output))),
            len(_PY_ASYNC_DEF.findall(_code(output))),
        )


class JsConstCountCheck:
    @property
    def name(self) -> str:
        return "js_const_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_JS_CONST.findall(_code(context.baseline_output))),
            len(_JS_CONST.findall(_code(output))),
        )


class JsArrowFuncCountCheck:
    @property
    def name(self) -> str:
        return "js_arrow_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_JS_ARROW.findall(_code(context.baseline_output))),
            len(_JS_ARROW.findall(_code(output))),
        )


class JsTemplateLiteralCountCheck:
    @property
    def name(self) -> str:
        return "js_tmpl_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_JS_TMPL.findall(_code(context.baseline_output))),
            len(_JS_TMPL.findall(_code(output))),
        )


class JsDestructuringCountCheck:
    @property
    def name(self) -> str:
        return "js_destructure_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_JS_DESTRUCT.findall(_code(context.baseline_output))),
            len(_JS_DESTRUCT.findall(_code(output))),
        )


class JsSpreadCountCheck:
    @property
    def name(self) -> str:
        return "js_spread_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_JS_SPREAD.findall(_code(context.baseline_output))),
            len(_JS_SPREAD.findall(_code(output))),
        )


class GoGoroutineCountCheck:
    @property
    def name(self) -> str:
        return "go_goroutine_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_GO_GO.findall(_code(context.baseline_output))),
            len(_GO_GO.findall(_code(output))),
        )


class GoChannelCountCheck:
    @property
    def name(self) -> str:
        return "go_chan_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_GO_CHAN.findall(_code(context.baseline_output))),
            len(_GO_CHAN.findall(_code(output))),
        )


class GoDeferCountCheck:
    @property
    def name(self) -> str:
        return "go_defer_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_GO_DEFER.findall(_code(context.baseline_output))),
            len(_GO_DEFER.findall(_code(output))),
        )


class GoErrorCheckCountCheck:
    @property
    def name(self) -> str:
        return "go_errcheck_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_GO_ERR.findall(_code(context.baseline_output))),
            len(_GO_ERR.findall(_code(output))),
        )


class RustMatchArmCountCheck:
    @property
    def name(self) -> str:
        return "rust_match_arm_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_RUST_MATCH_ARM.findall(_code(context.baseline_output))),
            len(_RUST_MATCH_ARM.findall(_code(output))),
        )


class RustResultUnwrapCountCheck:
    @property
    def name(self) -> str:
        return "rust_unwrap_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_RUST_UNWRAP.findall(_code(context.baseline_output))),
            len(_RUST_UNWRAP.findall(_code(output))),
        )


class RustLifetimeCountCheck:
    @property
    def name(self) -> str:
        return "rust_lifetime_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_RUST_LIFETIME.findall(_code(context.baseline_output))),
            len(_RUST_LIFETIME.findall(_code(output))),
        )


class JavaAnnotationCountCheck:
    @property
    def name(self) -> str:
        return "java_anno_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_JAVA_ANNO.findall(_code(context.baseline_output))),
            len(_JAVA_ANNO.findall(_code(output))),
        )


class JavaStreamCountCheck:
    @property
    def name(self) -> str:
        return "java_stream_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_JAVA_STREAM.findall(_code(context.baseline_output))),
            len(_JAVA_STREAM.findall(_code(output))),
        )


class CSharpAsyncCountCheck:
    @property
    def name(self) -> str:
        return "csharp_async_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_CSHARP_ASYNC.findall(_code(context.baseline_output))),
            len(_CSHARP_ASYNC.findall(_code(output))),
        )
