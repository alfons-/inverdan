"""Dashboard en tiempo real con Rich."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Optional

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..config.settings import Settings
from ..dashboard.state import DashboardState
from ..execution.portfolio import PortfolioSnapshot
from ..utils.market_hours import is_market_open, now_et

console = Console()

_ACTION_COLORS = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}
_PNL_COLOR = lambda v: "green" if v >= 0 else "red"


class DashboardRenderer:
    """Renderiza el dashboard en el terminal usando Rich Live."""

    def __init__(self, settings: Settings, state: DashboardState):
        self._cfg = settings
        self._state = state
        self._running = threading.Event()
        self._running.set()

    def run(self) -> None:
        """Bucle principal del dashboard. Llamar en el hilo principal."""
        with Live(
            self._build_layout(),
            console=console,
            refresh_per_second=int(1 / self._cfg.dashboard.refresh_rate),
            screen=True,
        ) as live:
            while self._running.is_set():
                try:
                    live.update(self._build_layout())
                    time.sleep(self._cfg.dashboard.refresh_rate)
                except KeyboardInterrupt:
                    break
                except Exception:
                    time.sleep(1)

    def _build_layout(self) -> Layout:
        snap = self._state.get_snapshot()
        portfolio = snap["portfolio"]
        signals = snap["signals"]
        logs = snap["logs"]
        prices = snap["prices"]

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        layout["left"].split_column(
            Layout(name="portfolio", ratio=1),
            Layout(name="positions", ratio=2),
        )
        layout["right"].split_column(
            Layout(name="signals", ratio=2),
            Layout(name="logs", ratio=1),
        )

        # Header
        market_status = "[green]ABIERTO[/green]" if is_market_open() else "[red]CERRADO[/red]"
        et_time = now_et().strftime("%Y-%m-%d %H:%M:%S ET")
        auto = "[green]AUTO-TRADE ON[/green]" if snap["auto_trade"] else "[yellow]AUTO-TRADE OFF[/yellow]"
        header_text = Text.from_markup(
            f"  [bold cyan]INVERDAN[/bold cyan] - Monitor de Mercados   "
            f"Mercado: {market_status}   {et_time}   {auto}"
        )
        layout["header"].update(Panel(header_text, style="bold"))

        # Portfolio
        layout["portfolio"].update(self._render_portfolio(portfolio))
        # Posiciones
        layout["positions"].update(self._render_positions(portfolio, prices))
        # Señales
        layout["signals"].update(self._render_signals(signals))
        # Logs
        layout["logs"].update(self._render_logs(logs))

        # Footer
        uptime = datetime.utcnow() - snap["started_at"]
        footer = Text.from_markup(
            f"  [dim]Uptime: {str(uptime).split('.')[0]}   "
            f"Símbolos: {', '.join(self._cfg.symbols)}   "
            f"Ctrl+C para salir[/dim]"
        )
        layout["footer"].update(Panel(footer))

        return layout

    def _render_portfolio(self, snap: Optional[PortfolioSnapshot]) -> Panel:
        if not snap:
            return Panel("[dim]Cargando...[/dim]", title="Portfolio")

        t = Table(show_header=False, box=None, padding=(0, 1))
        t.add_column("Clave", style="dim")
        t.add_column("Valor")

        pnl_color = _PNL_COLOR(snap.daily_pnl)
        upnl_color = _PNL_COLOR(snap.total_unrealized_pnl)
        win_rate = snap.wins_today / max(snap.trades_today, 1) * 100

        t.add_row("Equity:", f"[bold white]${snap.equity:,.2f}[/bold white]")
        t.add_row("Compra disponible:", f"${snap.buying_power:,.2f}")
        t.add_row("PnL día:", f"[{pnl_color}]{snap.daily_pnl:+.2f}[/{pnl_color}]")
        t.add_row("PnL no realizado:", f"[{upnl_color}]{snap.total_unrealized_pnl:+.2f}[/{upnl_color}]")
        t.add_row("Operaciones hoy:", f"{snap.trades_today} ({snap.wins_today}W/{snap.losses_today}L, {win_rate:.0f}%)")

        return Panel(t, title="[bold]Portfolio[/bold]", border_style="blue")

    def _render_positions(self, snap: Optional[PortfolioSnapshot], prices: dict) -> Panel:
        t = Table(box=box.SIMPLE_HEAD, show_edge=False)
        t.add_column("Símbolo", style="bold")
        t.add_column("Lado")
        t.add_column("Cant.")
        t.add_column("Entrada", justify="right")
        t.add_column("Actual", justify="right")
        t.add_column("PnL $", justify="right")
        t.add_column("PnL %", justify="right")
        t.add_column("SL", justify="right", style="dim")
        t.add_column("TP", justify="right", style="dim")

        if snap and snap.positions:
            for pos in snap.positions:
                current = prices.get(pos.symbol, pos.current_price)
                pnl = (current - pos.entry_price) * pos.qty * (1 if pos.side == "long" else -1)
                pnl_pct = pnl / (pos.entry_price * pos.qty) * 100
                c = _PNL_COLOR(pnl)
                side_color = "green" if pos.side == "long" else "red"

                t.add_row(
                    pos.symbol,
                    f"[{side_color}]{pos.side.upper()}[/{side_color}]",
                    str(pos.qty),
                    f"${pos.entry_price:.2f}",
                    f"${current:.2f}",
                    f"[{c}]{pnl:+.2f}[/{c}]",
                    f"[{c}]{pnl_pct:+.1f}%[/{c}]",
                    f"${pos.stop_loss:.2f}" if pos.stop_loss else "-",
                    f"${pos.take_profit:.2f}" if pos.take_profit else "-",
                )
        else:
            t.add_row("[dim]Sin posiciones abiertas[/dim]", *[""] * 8)

        return Panel(t, title="[bold]Posiciones Abiertas[/bold]", border_style="cyan")

    def _render_signals(self, signals: list) -> Panel:
        t = Table(box=box.SIMPLE_HEAD, show_edge=False)
        t.add_column("Hora", style="dim", width=8)
        t.add_column("Símbolo", style="bold", width=6)
        t.add_column("Señal", width=5)
        t.add_column("Conf.", justify="right", width=6)
        t.add_column("Precio", justify="right", width=8)
        t.add_column("Razón")

        for s in signals[:self._cfg.dashboard.signal_history_count]:
            action_color = _ACTION_COLORS.get(s.action, "white")
            t.add_row(
                s.timestamp.strftime("%H:%M:%S"),
                s.symbol,
                f"[{action_color}][bold]{s.action}[/bold][/{action_color}]",
                f"{s.confidence:.0%}",
                f"${s.price:.2f}",
                Text(s.reasoning[:50], overflow="ellipsis"),
            )

        if not signals:
            t.add_row("[dim]Esperando señales...[/dim]", *[""] * 5)

        return Panel(t, title="[bold]Señales en Tiempo Real[/bold]", border_style="green")

    def _render_logs(self, logs: list) -> Panel:
        lines = "\n".join(f"[dim]{line}[/dim]" for line in logs[:15])
        return Panel(
            Text.from_markup(lines or "[dim]Sin logs recientes[/dim]"),
            title="[bold]Sistema[/bold]",
            border_style="dim",
        )

    def stop(self) -> None:
        self._running.clear()
