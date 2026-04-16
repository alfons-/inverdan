#!/usr/bin/env python3
"""
Script de entrenamiento de modelos ML para INVERDAN.
=======================================================
Uso:
    python train.py                            # Entrena todos los símbolos del config
    python train.py --symbols AAPL TSLA        # Solo estos símbolos
    python train.py --days 730                 # 2 años de datos
    python train.py --force                    # Reentrenar aunque ya existan modelos
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inverdan.config.settings import load_settings
from inverdan.ml.trainer import train_symbol
from inverdan.utils.logger import setup_logger, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos ML - INVERDAN")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbols", nargs="+", help="Símbolos a entrenar (por defecto: todos del config)")
    parser.add_argument("--days", type=int, help="Días de datos históricos")
    parser.add_argument("--force", action="store_true", help="Reentrenar aunque ya exista el modelo")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_settings(args.config)
    setup_logger(settings.logs_path)
    logger = get_logger("train")

    symbols = args.symbols or settings.symbols
    if args.days:
        settings.training.lookback_days = args.days

    logger.info(f"Entrenando modelos para: {symbols}")
    logger.info(f"Datos históricos: {settings.training.lookback_days} días")
    logger.info(f"Split test: {settings.training.test_split:.0%}")
    logger.info(f"Threshold BUY: +{settings.training.buy_threshold:.1%} | SELL: {settings.training.sell_threshold:.1%}")

    results = {}
    for symbol in symbols:
        model_file = settings.models_path / f"rf_{symbol}.joblib"
        if model_file.exists() and not args.force:
            logger.info(f"Modelo {symbol} ya existe. Usa --force para reentrenar.")
            results[symbol] = "skipped"
            continue

        success = train_symbol(symbol, settings, settings.models_path)
        results[symbol] = "ok" if success else "error"

    # Resumen
    print("\n" + "=" * 50)
    print("RESUMEN DE ENTRENAMIENTO")
    print("=" * 50)
    for sym, status in results.items():
        icon = "✓" if status == "ok" else ("⚡" if status == "skipped" else "✗")
        print(f"  {icon} {sym}: {status}")
    print("=" * 50)

    ok = sum(1 for v in results.values() if v == "ok")
    print(f"\nModelos entrenados: {ok}/{len(symbols)}")
    print(f"Guardados en: {settings.models_path}")
    print("\nAhora puedes ejecutar: python main.py --auto-trade")


if __name__ == "__main__":
    main()
