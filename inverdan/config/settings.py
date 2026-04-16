"""Gestión de configuración con validación Pydantic."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import yaml
from alpaca.data.enums import DataFeed
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent.parent


class AlpacaSettings(BaseModel):
    paper_trading: bool = True
    data_feed: DataFeed = DataFeed.IEX
    api_key: str = Field(default_factory=lambda: os.environ.get("ALPACA_API_KEY", ""))
    api_secret: str = Field(default_factory=lambda: os.environ.get("ALPACA_API_SECRET", ""))

    @field_validator("api_key", "api_secret")
    @classmethod
    def must_not_be_empty(cls, v: str, info) -> str:
        if not v:
            raise ValueError(
                f"{info.field_name} no puede estar vacío. "
                "Configura ALPACA_API_KEY y ALPACA_API_SECRET en el fichero .env"
            )
        return v


class IndicatorSettings(BaseModel):
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    stoch_k: int = 14
    stoch_d: int = 3
    ema_fast: int = 9
    ema_slow: int = 21
    sma_200: int = 200


class MLSettings(BaseModel):
    model_type: str = "random_forest"
    confidence_threshold: float = Field(0.65, ge=0.5, le=1.0)
    feature_lookback: int = 20
    model_path: str = "models/"
    min_bars_required: int = 50


class RiskSettings(BaseModel):
    max_position_pct: float = Field(0.05, gt=0, le=0.2)
    max_total_exposure: float = Field(0.80, gt=0, le=1.0)
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 4.0
    max_daily_loss_pct: float = Field(0.05, gt=0, le=1.0)
    max_consecutive_losses: int = 5
    max_orders_per_minute: int = 3
    min_stock_price: float = 5.0
    min_daily_volume: int = 500_000


class DashboardSettings(BaseModel):
    refresh_rate: float = 1.0
    max_log_lines: int = 50
    signal_history_count: int = 20


class TrainingSettings(BaseModel):
    lookback_days: int = 365
    forward_return_periods: int = 5
    buy_threshold: float = 0.005
    sell_threshold: float = -0.005
    test_split: float = 0.2
    n_estimators: int = 200
    max_depth: int = 10
    random_state: int = 42


class Settings(BaseModel):
    alpaca: AlpacaSettings
    symbols: List[str] = ["AAPL", "TSLA", "NVDA", "MSFT"]
    timeframe: str = "1Min"
    bar_buffer_size: int = 500
    indicators: IndicatorSettings = IndicatorSettings()
    ml: MLSettings = MLSettings()
    risk: RiskSettings = RiskSettings()
    dashboard: DashboardSettings = DashboardSettings()
    training: TrainingSettings = TrainingSettings()

    @property
    def root_path(self) -> Path:
        return _ROOT

    @property
    def models_path(self) -> Path:
        p = _ROOT / self.ml.model_path
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def logs_path(self) -> Path:
        p = _ROOT / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_path(self) -> Path:
        p = _ROOT / "data" / "historical"
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_settings(config_path: Optional[str] = None) -> Settings:
    path = Path(config_path) if config_path else _ROOT / "config.yaml"
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f)
    else:
        raw = {}

    # Alpaca credentials come from .env, merge into raw config
    alpaca_raw = raw.get("alpaca", {})
    alpaca_raw.setdefault("api_key", os.environ.get("ALPACA_API_KEY", ""))
    alpaca_raw.setdefault("api_secret", os.environ.get("ALPACA_API_SECRET", ""))
    raw["alpaca"] = alpaca_raw

    return Settings(**raw)
