#!/bin/bash
# ===================================================================
# INVERDAN - Arranque completo
#   1. Entrena los modelos ML que falten (rf_<SYMBOL>.joblib)
#   2. Lanza el bot en segundo plano con --auto-trade
#   3. Lanza el dashboard web en http://localhost:5050
# Uso:
#   ./start_full.sh              # entrenamiento + bot + dashboard
#   ./start_full.sh --skip-train # solo arranca bot y dashboard
#   ./start_full.sh --force      # reentrena aunque ya existan modelos
# ===================================================================
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

PORT="${PORT:-5050}"
PY="${PY:-python3}"

SKIP_TRAIN=0
FORCE=""
for arg in "$@"; do
  case "$arg" in
    --skip-train) SKIP_TRAIN=1 ;;
    --force)      FORCE="--force" ;;
  esac
done

mkdir -p logs models data

# 1. Entrenamiento ----------------------------------------------------
if [ "$SKIP_TRAIN" -eq 0 ]; then
  echo ""
  echo ">>> Entrenando modelos ML..."
  $PY train.py $FORCE 2>&1 | tee -a logs/train.log
fi

# 2. Bot en segundo plano ---------------------------------------------
if [ -f bot.pid ] && kill -0 "$(cat bot.pid)" 2>/dev/null; then
  echo ">>> Bot ya corriendo (PID $(cat bot.pid)). Saltando arranque."
else
  echo ""
  echo ">>> Arrancando bot con --auto-trade..."
  nohup $PY main.py --no-dashboard --auto-trade \
        > logs/bot_stdout.log 2>&1 &
  echo $! > bot.pid
  sleep 1
  echo "    Bot PID: $(cat bot.pid)"
fi

# 3. Dashboard --------------------------------------------------------
echo ""
echo ">>> Dashboard → http://localhost:${PORT}"
echo ""
$PY web/app.py --port "$PORT" --host 0.0.0.0
