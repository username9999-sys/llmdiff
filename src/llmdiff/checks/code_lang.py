"""Multi-language code dimensions: detect and compare language-specific constructs."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "PythonImportStyleCheck",
    "JsImportStyleCheck",
    "GoImportStyleCheck",
    "RustUseStyleCheck",
    "JavaPackageCheck",
    "CIncludeStyleCheck",
    "RubyRequireCheck",
    "PhpUseCheck",
    "CSharpUsingCheck",
    "KotlinImportCheck",
    "SwiftImportCheck",
    "LangDetectMatchCheck",
    "ShebangMatchCheck",
    "SemicolonDensityCheck",
    "BraceDensityCheck",
    "ParenDensityCheck",
    "ArrowFuncCountCheck",
    "LambdaCountCheck",
    "ClassKeywordCountCheck",
    "InterfaceKeywordCountCheck",
]

_MAX = 200_000
_FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

_PY_IMPORT = re.compile(r"^\s*(?:from\s+[\w.]+\s+import|import\s+[\w.]+)", re.MULTILINE)
_JS_IMPORT = re.compile(r"^\s*import\s+.*?\s+from\s+['\"]", re.MULTILINE)
_GO_IMPORT = re.compile(r'^\s*(?:import\s+"\w|"\w[\w./]*")', re.MULTILINE)
_RUST_USE = re.compile(r"^\s*use\s+[\w:]+", re.MULTILINE)
_JAVA_PKG = re.compile(r"^\s*package\s+[\w.]+\s*;", re.MULTILINE)
_C_INCLUDE = re.compile(r'^\s*#include\s*[<"][^>"]+[>"]', re.MULTILINE)
_RUBY_REQ = re.compile(r'^\s*require\s+["\']', re.MULTILINE)
_PHP_USE = re.compile(r"^\s*use\s+[\w\\]+\s*;", re.MULTILINE)
_CSHARP_USING = re.compile(r"^\s*using\s+[\w.]+\s*;", re.MULTILINE)
_KOTLIN_IMPORT = re.compile(r"^\s*import\s+[\w.]+", re.MULTILINE)
_SWIFT_IMPORT = re.compile(r"^\s*import\s+(?:class|struct|enum|protocol\s+)?[\w.]+", re.MULTILINE)
_SHEBANG = re.compile(r"^#!.*$", re.MULTILINE)
_ARROW = re.compile(r"=>|->")
_LAMBDA = re.compile(r"\blambda\b|\bfunction\s*\(|=>")
_CLASS_KW = re.compile(
    r"^\s*(?:public\s+|private\s+|abstract\s+|final\s+)*class\s+\w+",
    re.MULTILINE,
)
_IFACE_KW = re.compile(r"^\s*(?:public\s+)?interface\s+\w+", re.MULTILINE)

_LANG_HINTS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"\bdef\s+\w+\s*\(|:\s*$|\bprint\s*\(", re.MULTILINE),
    "javascript": re.compile(r"\b(?:const|let|var)\s+\w+\s*=|=>|console\.log", re.MULTILINE),
    "go": re.compile(r"\bfunc\s+\w+\s*\(|package\s+\w+", re.MULTILINE),
    "rust": re.compile(r"\bfn\s+\w+\s*\(|let\s+mut\s+", re.MULTILINE),
    "java": re.compile(r"\bpublic\s+(?:static\s+)?(?:class|void|int|String)\b", re.MULTILINE),
    "c": re.compile(r"#include\s*<|\bint\s+main\s*\("),
    "cpp": re.compile(r"#include\s*<|std::|\bcout\b"),
    "ruby": re.compile(r"\bdef\s+\w+.*\bend\b|puts\s+"),
    "php": re.compile(r"<\?php|\$\w+\s*="),
    "csharp": re.compile(r"\busing\s+System|Console\.Write"),
    "kotlin": re.compile(r"\bfun\s+\w+\s*\(|val\s+\w+\s*="),
    "swift": re.compile(r"\bfunc\s+\w+\s*\(|let\s+\w+\s*:"),
}


def _s(text: str) -> str:
    return text[:_MAX]


def _code(text: str) -> str:
    blocks = [m.group(2) for m in _FENCE_RE.finditer(_s(text))]
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


class PythonImportStyleCheck:
    @property
    def name(self) -> str:
        return "py_imports"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PY_IMPORT.findall(_code(context.baseline_output))),
            len(_PY_IMPORT.findall(_code(output))),
        )


class JsImportStyleCheck:
    @property
    def name(self) -> str:
        return "js_imports"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_JS_IMPORT.findall(_code(context.baseline_output))),
            len(_JS_IMPORT.findall(_code(output))),
        )


class GoImportStyleCheck:
    @property
    def name(self) -> str:
        return "go_imports"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_GO_IMPORT.findall(_code(context.baseline_output))),
            len(_GO_IMPORT.findall(_code(output))),
        )


class RustUseStyleCheck:
    @property
    def name(self) -> str:
        return "rust_use"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_RUST_USE.findall(_code(context.baseline_output))),
            len(_RUST_USE.findall(_code(output))),
        )


class JavaPackageCheck:
    @property
    def name(self) -> str:
        return "java_package"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_JAVA_PKG.findall(_code(context.baseline_output))),
            len(_JAVA_PKG.findall(_code(output))),
        )


class CIncludeStyleCheck:
    @property
    def name(self) -> str:
        return "c_includes"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_C_INCLUDE.findall(_code(context.baseline_output))),
            len(_C_INCLUDE.findall(_code(output))),
        )


class RubyRequireCheck:
    @property
    def name(self) -> str:
        return "ruby_require"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_RUBY_REQ.findall(_code(context.baseline_output))),
            len(_RUBY_REQ.findall(_code(output))),
        )


class PhpUseCheck:
    @property
    def name(self) -> str:
        return "php_use"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PHP_USE.findall(_code(context.baseline_output))),
            len(_PHP_USE.findall(_code(output))),
        )


class CSharpUsingCheck:
    @property
    def name(self) -> str:
        return "csharp_using"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_CSHARP_USING.findall(_code(context.baseline_output))),
            len(_CSHARP_USING.findall(_code(output))),
        )


class KotlinImportCheck:
    @property
    def name(self) -> str:
        return "kotlin_imports"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_KOTLIN_IMPORT.findall(_code(context.baseline_output))),
            len(_KOTLIN_IMPORT.findall(_code(output))),
        )


class SwiftImportCheck:
    @property
    def name(self) -> str:
        return "swift_imports"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SWIFT_IMPORT.findall(_code(context.baseline_output))),
            len(_SWIFT_IMPORT.findall(_code(output))),
        )


class LangDetectMatchCheck:
    """1.0 when the detected dominant language matches the baseline's."""

    @property
    def name(self) -> str:
        return "lang_detect"

    def score(self, output: str, context: CheckContext) -> float:
        def detect(text: str) -> str:
            code = _code(text)
            best, best_n = "unknown", 0
            for lang, pat in _LANG_HINTS.items():
                n = len(pat.findall(code))
                if n > best_n:
                    best, best_n = lang, n
            return best

        base, cand = detect(context.baseline_output), detect(output)
        return 1.0 if base == cand else 0.0


class ShebangMatchCheck:
    """1.0 when the shebang line matches the baseline's."""

    @property
    def name(self) -> str:
        return "shebang"

    def score(self, output: str, context: CheckContext) -> float:
        def first_shebang(text: str) -> str:
            m = _SHEBANG.search(_s(text))
            return m.group(0) if m else ""

        base, cand = first_shebang(context.baseline_output), first_shebang(output)
        if not base and not cand:
            return 1.0
        return 1.0 if base == cand else 0.0


class SemicolonDensityCheck:
    @property
    def name(self) -> str:
        return "semicolon_density"

    def score(self, output: str, context: CheckContext) -> float:
        def density(text: str) -> float:
            code = _code(text)
            lines = [ln for ln in code.splitlines() if ln.strip()]
            if not lines:
                return 0.0
            return sum(1 for ln in lines if ln.rstrip().endswith(";")) / len(lines)

        return _ratio_match(density(context.baseline_output), density(output))


class BraceDensityCheck:
    @property
    def name(self) -> str:
        return "brace_density"

    def score(self, output: str, context: CheckContext) -> float:
        def density(text: str) -> float:
            code = _code(text)
            return code.count("{") / max(1, len(code))

        return _ratio_match(density(context.baseline_output), density(output))


class ParenDensityCheck:
    @property
    def name(self) -> str:
        return "paren_density"

    def score(self, output: str, context: CheckContext) -> float:
        def density(text: str) -> float:
            code = _code(text)
            return code.count("(") / max(1, len(code))

        return _ratio_match(density(context.baseline_output), density(output))


class ArrowFuncCountCheck:
    """Count of arrow-function tokens (=> or ->)."""

    @property
    def name(self) -> str:
        return "arrow_func_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ARROW.findall(_code(context.baseline_output))),
            len(_ARROW.findall(_code(output))),
        )


class LambdaCountCheck:
    @property
    def name(self) -> str:
        return "lambda_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_LAMBDA.findall(_code(context.baseline_output))),
            len(_LAMBDA.findall(_code(output))),
        )


class ClassKeywordCountCheck:
    @property
    def name(self) -> str:
        return "class_kw_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_CLASS_KW.findall(_code(context.baseline_output))),
            len(_CLASS_KW.findall(_code(output))),
        )


class InterfaceKeywordCountCheck:
    @property
    def name(self) -> str:
        return "interface_kw_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_IFACE_KW.findall(_code(context.baseline_output))),
            len(_IFACE_KW.findall(_code(output))),
        )
