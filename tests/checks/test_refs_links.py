"""Tests for refs_links dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.refs_links import (
    AbsolutePathCountCheck,
    ArxivIdCountCheck,
    Base64UrlCountCheck,
    CommitHashCountCheck,
    DoiCountCheck,
    EmailRefCountCheck,
    FilePathCountCheck,
    HexColorCountCheck,
    HttpVsHttpsRatioCheck,
    IsbnCountCheck,
    IssueRefCountCheck,
    MacAddressCountCheck,
    MarkdownRefLinkCountCheck,
    PrRefCountCheck,
    RelativePathCountCheck,
    SemverCountCheck,
    ShortLinkCountCheck,
    UrlCountCheck,
    UrlDomainSetJaccardCheck,
    UuidCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_url_count():
    ctx = _ctx("see https://a.com and https://b.com", "see https://a.com and https://b.com")
    assert UrlCountCheck().score("https://a.com https://b.com", ctx) == 1.0
    assert UrlCountCheck().score("https://a.com", ctx) < 1.0


def test_url_domain_jaccard():
    ctx = _ctx("https://a.com/x", "https://a.com/x")
    assert UrlDomainSetJaccardCheck().score("https://a.com/y", ctx) == 1.0
    assert UrlDomainSetJaccardCheck().score("https://b.com/y", ctx) < 1.0


def test_https_ratio():
    ctx = _ctx("https://a.com", "https://a.com")
    assert HttpVsHttpsRatioCheck().score("https://a.com", ctx) == 1.0
    assert HttpVsHttpsRatioCheck().score("http://a.com", ctx) < 1.0


def test_email_count():
    ctx = _ctx("a@b.com", "a@b.com")
    assert EmailRefCountCheck().score("a@b.com c@d.com", ctx) < 1.0
    assert EmailRefCountCheck().score("a@b.com", ctx) == 1.0


def test_filepath_count():
    ctx = _ctx("src/main.py", "src/main.py")
    assert FilePathCountCheck().score("src/main.py", ctx) == 1.0
    assert FilePathCountCheck().score("no path", ctx) < 1.0


def test_abs_rel_path_count():
    ctx = _ctx("/etc/hosts", "/etc/hosts")
    assert AbsolutePathCountCheck().score("/etc/hosts", ctx) == 1.0
    rctx = _ctx("src/main.py", "src/main.py")
    assert RelativePathCountCheck().score("src/main.py", rctx) == 1.0


def test_issue_pr_ref_count():
    ctx = _ctx("fixes #123", "fixes #123")
    assert IssueRefCountCheck().score("fixes #123 and #124", ctx) < 1.0
    assert IssueRefCountCheck().score("fixes #123", ctx) == 1.0
    pctx = _ctx("PR #42", "PR #42")
    assert PrRefCountCheck().score("PR #42", pctx) == 1.0


def test_commit_hash_count():
    ctx = _ctx("abc1234", "abc1234")
    assert CommitHashCountCheck().score("abc1234", ctx) == 1.0


def test_doi_arxiv_count():
    ctx = _ctx("doi 10.1000/xyz123", "doi 10.1000/xyz123")
    assert DoiCountCheck().score("10.1000/xyz123", ctx) == 1.0
    actx = _ctx("arXiv:2301.00001", "arXiv:2301.00001")
    assert ArxivIdCountCheck().score("arXiv:2301.00001", actx) == 1.0


def test_isbn_count():
    ctx = _ctx("978-0-13-468599-1", "978-0-13-468599-1")
    assert IsbnCountCheck().score("978-0-13-468599-1", ctx) == 1.0


def test_uuid_count():
    u = "123e4567-e89b-12d3-a456-426614174000"
    ctx = _ctx(u, u)
    assert UuidCountCheck().score(u, ctx) == 1.0
    assert UuidCountCheck().score("not-a-uuid", ctx) < 1.0


def test_semver_count():
    ctx = _ctx("1.2.3", "1.2.3")
    assert SemverCountCheck().score("1.2.3", ctx) == 1.0
    assert SemverCountCheck().score("1.2", ctx) < 1.0


def test_hexcolor_count():
    ctx = _ctx("#FF00AA", "#FF00AA")
    assert HexColorCountCheck().score("#FF00AA", ctx) == 1.0
    assert HexColorCountCheck().score("#FF00AA #00FF00", ctx) < 1.0


def test_mac_count():
    ctx = _ctx("aa:bb:cc:dd:ee:ff", "aa:bb:cc:dd:ee:ff")
    assert MacAddressCountCheck().score("aa:bb:cc:dd:ee:ff", ctx) == 1.0


def test_b64url_count():
    url = "https://a.com/" + "A" * 30
    ctx = _ctx(url, url)
    assert Base64UrlCountCheck().score(url, ctx) == 1.0


def test_md_reflink_count():
    ctx = _ctx("[a][b]", "[a][b]")
    assert MarkdownRefLinkCountCheck().score("[a][b]", ctx) == 1.0


def test_shortlink_count():
    ctx = _ctx("https://bit.ly/abc", "https://bit.ly/abc")
    assert ShortLinkCountCheck().score("https://bit.ly/abc", ctx) == 1.0
    assert ShortLinkCountCheck().score("https://a.com/abc", ctx) < 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "see https://a.com and #1",
         "candidate_output": "see https://a.com and #1"},
        {"id": "b", "prompt": "p", "baseline_output": "see https://a.com and #1",
         "candidate_output": "no refs here"},
    ])
    result = evaluate(corpus, dimensions=("url_count", "issue_ref_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["url_count"].losses == 1
    assert dims["issue_ref_count"].losses == 1
