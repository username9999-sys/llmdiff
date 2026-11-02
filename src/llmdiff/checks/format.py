"""Format/structure dimensions for markdown, HTML, YAML, CSV, XML, LaTeX."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from llmdiff.core.checks import CheckContext

__all__ = [
    "MarkdownHeadingsCheck",
    "MarkdownListsCheck",
    "MarkdownTablesCheck",
    "LinkCountCheck",
    "LatexBalanceCheck",
    "XmlValidCheck",
    "HtmlTagBalanceCheck",
    "YamlValidCheck",
    "CsvShapeCheck",
]

_MAX = 500_000
_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_LIST_RE = re.compile(r"^\s*[-*+]\s+\S", re.MULTILINE)
_TABLE_RE = re.compile(r"^\|.*\|\s*$", re.MULTILINE)
_URL_RE = re.compile(r"https?://[^\s)\]]+")
_TAG_RE = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)[^>]*?(/?)>")
_VOID_TAGS = {
    "br", "hr", "img", "input", "meta", "link", "area",
    "base", "col", "embed", "source", "track", "wbr",
}


class MarkdownHeadingsCheck:
    """1.0 when heading count matches the baseline, else proportional."""

    @property
    def name(self) -> str:
        return "md_headings"

    def score(self, output: str, context: CheckContext) -> float:
        base = len(_HEADING_RE.findall(context.baseline_output[:_MAX]))
        cand = len(_HEADING_RE.findall(output[:_MAX]))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class MarkdownListsCheck:
    """1.0 when list-item count matches the baseline, else proportional."""

    @property
    def name(self) -> str:
        return "md_lists"

    def score(self, output: str, context: CheckContext) -> float:
        base = len(_LIST_RE.findall(context.baseline_output[:_MAX]))
        cand = len(_LIST_RE.findall(output[:_MAX]))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class MarkdownTablesCheck:
    """1.0 when the output has a markdown table iff the baseline does."""

    @property
    def name(self) -> str:
        return "md_tables"

    def score(self, output: str, context: CheckContext) -> float:
        base = bool(_TABLE_RE.search(context.baseline_output[:_MAX]))
        cand = bool(_TABLE_RE.search(output[:_MAX]))
        return 1.0 if base == cand else 0.0


class LinkCountCheck:
    """1.0 when URL count matches the baseline, else proportional."""

    @property
    def name(self) -> str:
        return "links"

    def score(self, output: str, context: CheckContext) -> float:
        base = len(_URL_RE.findall(context.baseline_output[:_MAX]))
        cand = len(_URL_RE.findall(output[:_MAX]))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class LatexBalanceCheck:
    """1.0 when $...$ delimiters are balanced (even count of unescaped $)."""

    @property
    def name(self) -> str:
        return "latex_balance"

    def score(self, output: str, context: CheckContext) -> float:
        sample = output[:_MAX]
        count = 0
        prev = ""
        for ch in sample:
            if ch == "$" and prev != "\\":
                count += 1
            prev = ch
        return 1.0 if count % 2 == 0 else 0.0


class XmlValidCheck:
    """1.0 when the output parses as XML, else 0.0."""

    @property
    def name(self) -> str:
        return "xml_valid"

    def score(self, output: str, context: CheckContext) -> float:
        sample = output.strip()[:_MAX]
        if not sample.startswith("<"):
            return 0.0
        try:
            ET.fromstring(sample)
        except ET.ParseError:
            return 0.0
        return 1.0


class HtmlTagBalanceCheck:
    """Proportion of HTML tags that are properly closed (void tags excluded)."""

    @property
    def name(self) -> str:
        return "html_balance"

    def score(self, output: str, context: CheckContext) -> float:
        sample = output[:_MAX]
        stack: list[str] = []
        total = 0
        for m in _TAG_RE.finditer(sample):
            closing, name, self_closing = m.group(1), m.group(2).lower(), m.group(3)
            if name in _VOID_TAGS or self_closing:
                continue
            total += 1
            if closing:
                if stack and stack[-1] == name:
                    stack.pop()
            else:
                stack.append(name)
        if total == 0:
            return 1.0
        matched = total - len(stack)
        return max(0.0, matched / total)


class YamlValidCheck:
    """1.0 when the output parses as YAML (pyyaml if available), else 0.0."""

    @property
    def name(self) -> str:
        return "yaml_valid"

    def score(self, output: str, context: CheckContext) -> float:
        sample = output.strip()[:_MAX]
        if not sample:
            return 1.0
        try:
            import yaml

            yaml.safe_load(sample)
            return 1.0
        except Exception:
            return 0.0


class CsvShapeCheck:
    """1.0 when CSV column count matches the baseline header, else proportional."""

    @property
    def name(self) -> str:
        return "csv_shape"

    def score(self, output: str, context: CheckContext) -> float:
        import csv
        import io

        def cols(text: str) -> int:
            sample = text[:_MAX]
            try:
                row = next(csv.reader(io.StringIO(sample)))
            except (csv.Error, StopIteration):
                return 0
            return len(row)

        base = cols(context.baseline_output)
        cand = cols(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)
