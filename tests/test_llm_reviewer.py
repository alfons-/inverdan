"""Tests de la capa de revisión LLM (signals/llm_reviewer.py). Todo con mocks; no
toca ni Alpaca ni Anthropic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from types import SimpleNamespace
from unittest.mock import MagicMock

from inverdan.signals.llm_reviewer import LLMReviewer, LLMReview
from inverdan.signals.signal_types import Signal


def _signal(symbol="AAPL", action="BUY"):
    return Signal(symbol=symbol, action=action, confidence=0.7, price=200.0, reasoning="RSI bajo")


def _settings(fail_open=True):
    s = MagicMock()
    s.llm_review.model = "claude-opus-4-8"
    s.llm_review.fail_open = fail_open
    s.llm_review.timeout = 8.0
    s.llm_review.news_lookback_hours = 24
    s.llm_review.max_headlines = 10
    s.llm_review.api_key = "test"
    return s


def _news(headlines=("AAPL sube",), raise_err=False):
    nc = MagicMock()
    if raise_err:
        nc.get_news.side_effect = RuntimeError("news down")
    else:
        items = [SimpleNamespace(headline=h, created_at="2026-06-20") for h in headlines]
        nc.get_news.return_value = SimpleNamespace(data={"news": items})
    return nc


def _anthropic(parsed=None, raise_err=False):
    c = MagicMock()
    if raise_err:
        c.messages.parse.side_effect = RuntimeError("api down")
    else:
        c.messages.parse.return_value = SimpleNamespace(parsed_output=parsed)
    return c


class TestLLMReviewer:
    def test_approve(self):
        r = LLMReviewer(_settings(), news_client=_news(),
                        anthropic_client=_anthropic(LLMReview(approve=True, confidence=0.9, reason="sin riesgo")))
        ok, reason = r.review(_signal())
        assert ok is True and reason == "sin riesgo"

    def test_veto(self):
        r = LLMReviewer(_settings(), news_client=_news(("AAPL presenta resultados mañana",)),
                        anthropic_client=_anthropic(LLMReview(approve=False, confidence=0.8, reason="earnings inminentes")))
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
                        anthropic_client=_anthropic(LLMReview(approve=True, confidence=1.0, reason="x")))
        ok, reason = r.review(_signal())
        assert ok is True and "fail-open" in reason

    def test_unparseable_response_respects_fail_mode(self):
        r = LLMReviewer(_settings(fail_open=False), news_client=_news(), anthropic_client=_anthropic(parsed=None))
        ok, _ = r.review(_signal())
        assert ok is False
