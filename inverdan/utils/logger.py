"""Logging estructurado con rotación de ficheros."""
from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

_loggers: dict = {}
_LOG_DIR: Path = Path("logs")


def setup_logger(log_dir: Path = _LOG_DIR) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("inverdan")
    root.setLevel(logging.DEBUG)
    if root.handlers:
        return

    fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    # Consola
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Fichero rotativo
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "system.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"inverdan.{name}")


class TradeLogger:
    """Logger dedicado para el audit trail de operaciones."""

    def __init__(self, log_dir: Path = _LOG_DIR):
        log_dir.mkdir(parents=True, exist_ok=True)
        self._trade_log = log_dir / "trades.log"
        self._signal_log = log_dir / "signals.log"

    def log_signal(self, data: dict) -> None:
        data["_ts"] = datetime.utcnow().isoformat()
        with open(self._signal_log, "a") as f:
            f.write(json.dumps(data) + "\n")

    def log_trade(self, data: dict) -> None:
        data["_ts"] = datetime.utcnow().isoformat()
        with open(self._trade_log, "a") as f:
            f.write(json.dumps(data) + "\n")
