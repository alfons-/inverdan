"""Cliente de datos históricos de Alpaca."""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from ..config.settings import Settings
from ..utils.logger import get_logger

logger = get_logger("data.historical")

_TIMEFRAME_MAP = {
    "1Min": TimeFrame(1, TimeFrameUnit.Minute),
    "5Min": TimeFrame(5, TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
    "1Day": TimeFrame(1, TimeFrameUnit.Day),
}


class HistoricalDataClient:
    def __init__(self, settings: Settings):
        self._cfg = settings
        self._client = StockHistoricalDataClient(
            api_key=settings.alpaca.api_key,
            secret_key=settings.alpaca.api_secret,
        )

    def fetch_bars(
        self,
        symbol: str,
        days: int = 365,
        timeframe: Optional[str] = None,
        cache: bool = True,
    ) -> pd.DataFrame:
        tf = timeframe or self._cfg.timeframe
        cache_path = self._cfg.data_path / f"{symbol}_{tf}_{days}d.parquet"

        if cache and cache_path.exists():
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            if age_hours < 1:
                logger.debug(f"Usando caché para {symbol}")
                return pd.read_parquet(cache_path)

        end = datetime.utcnow()
        start = end - timedelta(days=days)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_TIMEFRAME_MAP.get(tf, TimeFrame(1, TimeFrameUnit.Minute)),
            start=start,
            end=end,
            feed=self._cfg.alpaca.data_feed,
        )

        logger.info(f"Descargando datos históricos para {symbol} ({days} días, {tf})")
        bars = self._client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            logger.warning(f"No hay datos para {symbol}")
            return pd.DataFrame()

        # Normalizar índice
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[symbol] if symbol in df.index.get_level_values(0) else df.droplevel(0)

        df.index = pd.to_datetime(df.index, utc=True)
        df = df[["open", "high", "low", "close", "volume"]].sort_index()

        if cache:
            df.to_parquet(cache_path)
            logger.debug(f"Caché guardada: {cache_path}")

        return df

    def fetch_multiple(self, symbols: List[str], days: int = 365) -> dict[str, pd.DataFrame]:
        result = {}
        for sym in symbols:
            try:
                result[sym] = self.fetch_bars(sym, days=days)
                time.sleep(0.3)  # Respetar rate limits
            except Exception as e:
                logger.error(f"Error descargando {sym}: {e}")
        return result
