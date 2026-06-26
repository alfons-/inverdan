"""Tests del formato de avisos Pushover: abrir/cerrar · largo/corto · P&L."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock
from inverdan.utils.pushover import PushoverNotifier
from inverdan.events.bus import OrderFilledEvent


def _notifier():
    n = PushoverNotifier("tok", "usr", MagicMock())
    sent = []
    n._send = lambda title, message, priority=0: sent.append((title, message, priority))
    return n, sent


def _fill(**kw):
    base = dict(symbol="AAPL", side="buy", shares=10, fill_price=200.0, order_id="x",
                stop_price=195.0, take_profit_price=210.0)
    base.update(kw)
    return OrderFilledEvent(**base)


class TestPushoverFills:
    def test_open_long(self):
        n, sent = _notifier()
        n._on_order_filled(_fill(side="buy", position_side="long", is_close=False))
        title, msg, _ = sent[0]
        assert "Abre largo" in title and "AAPL" in title

    def test_open_short(self):
        n, sent = _notifier()
        n._on_order_filled(_fill(symbol="NVDA", side="sell", position_side="short", is_close=False))
        title, _, _ = sent[0]
        assert "Abre corto" in title and "NVDA" in title

    def test_close_long_profit(self):
        n, sent = _notifier()
        n._on_order_filled(_fill(side="sell", is_close=True, position_side="long",
                                 pnl=98.5, close_reason="take_profit", fill_price=210.0))
        title, msg, _ = sent[0]
        assert "Cierra largo" in title
        assert "+98.50" in msg and "take-profit" in msg

    def test_close_short_loss(self):
        n, sent = _notifier()
        n._on_order_filled(_fill(symbol="NVDA", side="buy", is_close=True, position_side="short",
                                 pnl=-45.2, close_reason="stop_loss", fill_price=205.0))
        title, msg, _ = sent[0]
        assert "Cierra corto" in title
        assert "-45.20" in msg and "stop-loss" in msg
