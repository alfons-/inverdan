"""Tests de la capa de revisión LLM (signals/llm_reviewer.py). Todo con mocks; no
toca ni Alpaca ni Anthropic. El veredicto llega por el tool `submit_verdict`."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from types import SimpleNamespace
from unittest.mock import MagicMock

from inverdan.signals.llm_reviewer import LLMReviewer, LLMReview
from inverdan.signals.signal_types import Signal


def _signal(symbol="AAPL", action="BUY"):
    return Signal(symbol=symbol, action=action, confidence=0.7, price=200.0, reasoning="RSI bajo")


def _settings(fail_open=True, use_web_search=True):
    s = MagicMock()
    s.llm_review.model = "claude-opus-4-8"
    s.llm_review.fail_open = fail_open
    s.llm_review.timeout = 20.0
    s.llm_review.news_lookback_hours = 24
    s.llm_review.max_headlines = 10
    s.llm_review.api_key = "test"
    s.llm_review.use_web_search = use_web_search
    s.llm_review.web_search_max_uses = 3
    return s


def _news(headlines=("AAPL sube",), raise_err=False):
    nc = MagicMock()
    if raise_err:
        nc.get_news.side_effect = RuntimeError("news down")
    else:
        items = [SimpleNamespace(headline=h, created_at="2026-06-20") for h in headlines]
        nc.get_news.return_value = SimpleNamespace(data={"news": items})
    return nc


def _verdict_block(approve, confidence, reason):
    return SimpleNamespace(type="tool_use", name="submit_verdict",
                           input={"approve": approve, "confidence": confidence, "reason": reason})


def _anthropic(verdict=None, raise_err=False, no_verdict=False):
    """verdict = (approve, confidence, reason) | None."""
    c = MagicMock()
    if raise_err:
        c.messages.create.side_effect = RuntimeError("api down")
    elif no_verdict:
        c.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="no opino")], stop_reason="end_turn")
    else:
        a, conf, rsn = verdict
        c.messages.create.return_value = SimpleNamespace(
            content=[_verdict_block(a, conf, rsn)], stop_reason="tool_use")
    return c


class TestLLMReviewer:
    def test_approve(self):
        r = LLMReviewer(_settings(), news_client=_news(),
                        anthropic_client=_anthropic((True, 0.9, "sin riesgo")))
        ok, reason = r.review(_signal())
        assert ok is True and reason == "sin riesgo"

    def test_veto(self):
        r = LLMReviewer(_settings(), news_client=_news(("AAPL presenta resultados mañana",)),
                        anthropic_client=_anthropic((False, 0.8, "earnings inminentes")))
        ok, reason = r.review(_signal())
        assert ok is False and "earnings" in reason

    def test_fail_open_on_api_error(self):
        r = LLMReviewer(_settings(fail_open=True), news_client=_news(), anthropic_client=_anthropic(raise_err=True))
        ok, reason = r.review(_signal())
        assert ok is True and "fail-open" in reason

    def test_fail_closed_on_api_error(self):
        r = LLMReviewer(_settings(fail_open=False), news_client=_news(), anthropic_client=_anthropic(raise_err=True))
        ok, _ = r.review(_signal())
        assert ok is False

    def test_fail_open_on_news_error(self):
        r = LLMReviewer(_settings(fail_open=True), news_client=_news(raise_err=True),
                        anthropic_client=_anthropic((True, 1.0, "x")))
        ok, reason = r.review(_signal())
        assert ok is True and "fail-open" in reason

    def test_no_verdict_respects_fail_mode(self):
        r = LLMReviewer(_settings(fail_open=False), news_client=_news(), anthropic_client=_anthropic(no_verdict=True))
        ok, _ = r.review(_signal())
        assert ok is False

    def test_web_search_tool_included_when_enabled(self):
        r = LLMReviewer(_settings(use_web_search=True), news_client=_news(),
                        anthropic_client=_anthropic((True, 0.9, "ok")))
        names = [t.get("type") or t.get("name") for t in r._tools()]
        assert any("web_search" in str(n) for n in names) and any("submit_verdict" == n for n in names)

    def test_web_search_tool_absent_when_disabled(self):
        r = LLMReviewer(_settings(use_web_search=False), news_client=_news(),
                        anthropic_client=_anthropic((True, 0.9, "ok")))
        names = [t.get("type") or t.get("name") for t in r._tools()]
        assert not any("web_search" in str(n) for n in names)


def _earnings_client(has_soon=None, raise_err=False, no_verdict=False):
    c = MagicMock()
    if raise_err:
        c.messages.create.side_effect = RuntimeError("api down")
    elif no_verdict:
        c.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="?")], stop_reason="end_turn")
    else:
        block = SimpleNamespace(type="tool_use", name="submit_earnings_check",
                                input={"has_earnings_soon": has_soon, "detail": "MSFT reporta el 29-jul"})
        c.messages.create.return_value = SimpleNamespace(content=[block], stop_reason="tool_use")
    return c


class TestCheckEarnings:
    def test_detects_earnings(self):
        r = LLMReviewer(_settings(), news_client=_news(), anthropic_client=_earnings_client(True))
        soon, detail = r.check_earnings("MSFT", 2)
        assert soon is True and "29-jul" in detail

    def test_no_earnings(self):
        r = LLMReviewer(_settings(), news_client=_news(), anthropic_client=_earnings_client(False))
        soon, _ = r.check_earnings("GLD", 2)
        assert soon is False

    def test_api_error_returns_none(self):
        # None = "no se pudo determinar" → el guardia NO actúa sobre incertidumbre
        r = LLMReviewer(_settings(), news_client=_news(), anthropic_client=_earnings_client(raise_err=True))
        soon, _ = r.check_earnings("MSFT", 2)
        assert soon is None

    def test_no_verdict_returns_none(self):
        r = LLMReviewer(_settings(), news_client=_news(), anthropic_client=_earnings_client(no_verdict=True))
        soon, _ = r.check_earnings("MSFT", 2)
        assert soon is None
