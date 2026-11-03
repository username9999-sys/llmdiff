"""Built-in comparison dimensions."""

from __future__ import annotations

from llmdiff.checks.bleu import BleuCheck
from llmdiff.checks.code import (
    BraceBalanceCheck,
    CodeBlockCountCheck,
    CodeHasDefinitionsCheck,
    FencedCodeCheck,
    PythonSyntaxCheck,
)
from llmdiff.checks.composite import WeightedCompositeCheck
from llmdiff.checks.cost import CostCheck, LatencyCheck, TokenCountCheck
from llmdiff.checks.custom_regex import RegexForbiddenCheck, RegexMustMatchCheck
from llmdiff.checks.diffview import DiffHunkCountCheck, EditDistanceCheck, LineChangeRatioCheck
from llmdiff.checks.entropy import EntropyCheck, RepetitionCheck
from llmdiff.checks.format import (
    CsvShapeCheck,
    HtmlTagBalanceCheck,
    LatexBalanceCheck,
    LinkCountCheck,
    MarkdownHeadingsCheck,
    MarkdownListsCheck,
    MarkdownTablesCheck,
    XmlValidCheck,
    YamlValidCheck,
)
from llmdiff.checks.grounding import CitationSupportCheck, EntityConsistencyCheck
from llmdiff.checks.instructions import (
    FormatFollowCheck,
    JsonModeCheck,
    LengthInstructionCheck,
    NoApologyCheck,
)
from llmdiff.checks.judge import JudgeCheck
from llmdiff.checks.lists import (
    ListItemCountCheck,
    ListItemOverlapCheck,
    ListOrderCheck,
    TableShapeCheck,
)
from llmdiff.checks.ngram import (
    NgramF1Check,
    NgramPrecisionCheck,
    NgramRecallCheck,
    TokenNgramCheck,
)
from llmdiff.checks.numeric import MeanAbsoluteErrorCheck, NumberDriftCheck, NumberPreservedCheck
from llmdiff.checks.qa import (
    AnswersPromptCheck,
    CitationCheck,
    LanguageMatchCheck,
    NumberConsistencyCheck,
)
from llmdiff.checks.readability import (
    FleschKincaidCheck,
    SentenceStatsCheck,
    SyllableCountCheck,
)
from llmdiff.checks.rouge import Rouge1Check, Rouge2Check, RougeLCheck
from llmdiff.checks.safety import PiiCheck, PromptLeakCheck, SecretCheck, ToxicityCheck
from llmdiff.checks.semantic import SemanticSimilarityCheck
from llmdiff.checks.sentiment import SentimentCheck, ToneMatchCheck
from llmdiff.checks.structure import JsonValidCheck, StructureCheck
from llmdiff.checks.text import ContentCheck, ExactMatchCheck, LengthCheck, SimilarityCheck
from llmdiff.checks.topics import KeywordCoverageCheck, TopicDriftCheck
from llmdiff.checks.unicode import (
    ControlCharCheck,
    MojibakeCheck,
    RtlMarkCheck,
    UnicodeScriptCheck,
)
from llmdiff.core.checks import Check

__all__ = [
    "builtin_checks",
    "ExactMatchCheck",
    "SimilarityCheck",
    "LengthCheck",
    "ContentCheck",
    "JsonValidCheck",
    "StructureCheck",
    "TokenNgramCheck",
    "NgramPrecisionCheck",
    "NgramRecallCheck",
    "NgramF1Check",
    "EntropyCheck",
    "RepetitionCheck",
    "PythonSyntaxCheck",
    "BraceBalanceCheck",
    "CodeBlockCountCheck",
    "CodeHasDefinitionsCheck",
    "FencedCodeCheck",
    "MarkdownHeadingsCheck",
    "MarkdownListsCheck",
    "MarkdownTablesCheck",
    "LinkCountCheck",
    "LatexBalanceCheck",
    "XmlValidCheck",
    "HtmlTagBalanceCheck",
    "YamlValidCheck",
    "CsvShapeCheck",
    "PiiCheck",
    "SecretCheck",
    "ToxicityCheck",
    "PromptLeakCheck",
    "AnswersPromptCheck",
    "NumberConsistencyCheck",
    "CitationCheck",
    "LanguageMatchCheck",
    "NumberPreservedCheck",
    "NumberDriftCheck",
    "MeanAbsoluteErrorCheck",
    "ListItemCountCheck",
    "ListItemOverlapCheck",
    "ListOrderCheck",
    "TableShapeCheck",
    "EditDistanceCheck",
    "DiffHunkCountCheck",
    "LineChangeRatioCheck",
    "LatencyCheck",
    "TokenCountCheck",
    "CostCheck",
    "WeightedCompositeCheck",
    "UnicodeScriptCheck",
    "MojibakeCheck",
    "ControlCharCheck",
    "RtlMarkCheck",
    "RegexMustMatchCheck",
    "RegexForbiddenCheck",
    "SemanticSimilarityCheck",
    "JudgeCheck",
    "Rouge1Check",
    "Rouge2Check",
    "RougeLCheck",
    "BleuCheck",
    "FleschKincaidCheck",
    "SyllableCountCheck",
    "SentenceStatsCheck",
    "FormatFollowCheck",
    "JsonModeCheck",
    "LengthInstructionCheck",
    "NoApologyCheck",
    "KeywordCoverageCheck",
    "TopicDriftCheck",
    "SentimentCheck",
    "ToneMatchCheck",
    "EntityConsistencyCheck",
    "CitationSupportCheck",
]


def builtin_checks() -> tuple[Check, ...]:
    """Return one instance of every built-in comparison dimension."""
    return (
        ExactMatchCheck(),
        SimilarityCheck(),
        LengthCheck(),
        ContentCheck(),
        JsonValidCheck(),
        StructureCheck(),
        TokenNgramCheck(),
        NgramPrecisionCheck(),
        NgramRecallCheck(),
        NgramF1Check(),
        EntropyCheck(),
        RepetitionCheck(),
        PythonSyntaxCheck(),
        BraceBalanceCheck(),
        CodeBlockCountCheck(),
        CodeHasDefinitionsCheck(),
        FencedCodeCheck(),
        MarkdownHeadingsCheck(),
        MarkdownListsCheck(),
        MarkdownTablesCheck(),
        LinkCountCheck(),
        LatexBalanceCheck(),
        XmlValidCheck(),
        HtmlTagBalanceCheck(),
        YamlValidCheck(),
        CsvShapeCheck(),
        PiiCheck(),
        SecretCheck(),
        ToxicityCheck(),
        PromptLeakCheck(),
        AnswersPromptCheck(),
        NumberConsistencyCheck(),
        CitationCheck(),
        LanguageMatchCheck(),
        NumberPreservedCheck(),
        NumberDriftCheck(),
        MeanAbsoluteErrorCheck(),
        ListItemCountCheck(),
        ListItemOverlapCheck(),
        ListOrderCheck(),
        TableShapeCheck(),
        EditDistanceCheck(),
        DiffHunkCountCheck(),
        LineChangeRatioCheck(),
        LatencyCheck(),
        TokenCountCheck(),
        CostCheck(),
        UnicodeScriptCheck(),
        MojibakeCheck(),
        ControlCharCheck(),
        RtlMarkCheck(),
        RegexMustMatchCheck(),
        RegexForbiddenCheck(),
        SemanticSimilarityCheck(),
        JudgeCheck(),
        Rouge1Check(),
        Rouge2Check(),
        RougeLCheck(),
        BleuCheck(),
        FleschKincaidCheck(),
        SyllableCountCheck(),
        SentenceStatsCheck(),
        FormatFollowCheck(),
        JsonModeCheck(),
        LengthInstructionCheck(),
        NoApologyCheck(),
        KeywordCoverageCheck(),
        TopicDriftCheck(),
        SentimentCheck(),
        ToneMatchCheck(),
        EntityConsistencyCheck(),
        CitationSupportCheck(),
    )
