"""Gestión de configuración con validación Pydantic."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import yaml
from alpaca.data.enums import DataFeed
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_ROOT / ".env")


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
    # Suelo mínimo del stop como % del precio. El ATR de 1 min es diminuto y
    # dejaba los stops al 1%, demasiado finos: el 81% de las salidas eran stops
    # disparados por ruido intradía antes de que la tesis funcionara.
    min_stop_pct: float = Field(0.025, gt=0, le=0.2)
    # Trailing stop (%) que coloca el protector: asegura ganancias dejando correr
    # al ganador (el stop sigue al precio), en vez del stop estático que se
    # quedaba clavado en la entrada sin proteger el beneficio.
    trailing_stop_pct: float = Field(3.0, gt=0, le=20)
    max_daily_loss_pct: float = Field(0.05, gt=0, le=1.0)
    max_consecutive_losses: int = 5
    max_orders_per_minute: int = 3
    min_stock_price: float = 5.0
    min_daily_volume: int = 500_000
    # Filtro de tendencia mayor: veta operar contra la tendencia del timeframe
    # indicado (por defecto SMA50 en velas diarias). Evita shortear en tendencia
    # alcista mayor / comprar en bajista, sin depender del SMA200 intradía (~3,3 h).
    trend_timeframe: str = "1Day"
    trend_sma_period: int = Field(50, gt=1)
    # Banda neutra alrededor de la media de tendencia: si el precio está dentro de
    # ±trend_buffer_pct del SMA, no se opera (tendencia poco clara), para evitar el
    # whipsaw de dirección en valores laterales (p. ej. NVDA oscilando sobre su SMA).
    trend_buffer_pct: float = Field(0.01, ge=0, le=0.1)


class PushoverSettings(BaseModel):
    enabled: bool = False
    api_token: str = Field(default_factory=lambda: os.environ.get("PUSHOVER_API_TOKEN", ""))
    user_key: str = Field(default_factory=lambda: os.environ.get("PUSHOVER_USER_KEY", ""))
    min_signal_confidence: float = Field(0.0, ge=0.0, le=1.0)
    # Dispositivos destino, separados por coma (la cuenta de Pushover es compartida
    # y sin esto las notificaciones llegan a TODOS sus dispositivos). Vacío = todos.
    device: str = ""


class LLMReviewSettings(BaseModel):
    """Capa opcional: Claude revisa titulares de noticias y puede VETAR (nunca crear)
    operaciones antes de enviarlas. Desactivada por defecto; requiere ANTHROPIC_API_KEY."""
    enabled: bool = False
    model: str = "claude-opus-4-8"
    api_key: str = Field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    fail_open: bool = True          # si el LLM falla/tarda → operar igual (no bloquear el bot)
    timeout: float = Field(8.0, gt=0)
    news_lookback_hours: int = Field(24, gt=0)
    max_headlines: int = Field(10, gt=0, le=50)


class DashboardSettings(BaseModel):
    refresh_rate: float = 1.0
    max_log_lines: int = 50
    signal_history_count: int = 20


class TrainingSettings(BaseModel):
    lookback_days: int = 365
    forward_return_periods: int = 5
    buy_threshold: float = 0.005
    sell_threshold: float = -0.005
    # Etiquetado por cuantiles: fracción de cola por clase. 0.33 → tercio
    # inferior=SELL, tercio superior=BUY, resto=HOLD (clases equilibradas).
    label_quantile: float = Field(0.33, gt=0.0, le=0.5)
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
    pushover: PushoverSettings = PushoverSettings()
    llm_review: LLMReviewSettings = LLMReviewSettings()

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
