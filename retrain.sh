#!/bin/bash
# ===================================================================
# INVERDAN - Reentrenamiento automático de modelos ML
# ===================================================================
# Lo ejecuta el LaunchAgent com.inverdan.retrain (sábados por la mañana,
# con el mercado cerrado). Pasos:
#   1. Reentrena los modelos de todos los símbolos del config (train.py --force),
#      usando datos históricos frescos de Alpaca.
#   2. Si el entrenamiento va bien, reinicia el bot VÍA launchd para que cargue
#      los modelos nuevos (el bot solo lee los modelos al arrancar).
#
# Si el entrenamiento falla, el bot se queda con los modelos anteriores intactos.
# Uso manual:  ./retrain.sh
# ===================================================================
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

PY="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
LOG="logs/retrain.log"
UID_NUM="$(id -u)"
mkdir -p logs

{
    echo ""
    echo "===== Reentrenamiento $(date '+%Y-%m-%d %H:%M:%S') ====="
} >> "$LOG"

# 1. Reentrenar todos los modelos del config
if "$PY" train.py --force >> "$LOG" 2>&1; then
    echo "[OK] Entrenamiento completado a las $(date '+%H:%M:%S')" >> "$LOG"

    # 2. Reiniciar el bot gestionado por launchd para que cargue los modelos nuevos.
    #    'kickstart -k' mata la instancia actual y la relanza limpiamente.
    if launchctl kickstart -k "gui/${UID_NUM}/com.inverdan.bot" >> "$LOG" 2>&1; then
        echo "[OK] Bot reiniciado vía launchd; modelos nuevos en uso." >> "$LOG"
    else
        echo "[WARN] No se pudo reiniciar el bot (¿agente no cargado?)." >> "$LOG"
    fi
else
    echo "[ERROR] Falló el entrenamiento; el bot conserva los modelos previos." >> "$LOG"
fi
