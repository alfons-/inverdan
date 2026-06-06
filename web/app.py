#!/usr/bin/env python3
"""
INVERDAN Web Dashboard
Servidor Flask que expone la API REST para el dashboard de monitorización.

Uso:
    python3 web/app.py
    python3 web/app.py --port 8080
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import psutil
import yaml
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

# ── Rutas base ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

STATE_FILE   = ROOT / "state.json"
CONFIG_FILE  = ROOT / "config.yaml"
SIGNALS_LOG  = ROOT / "logs" / "signals.log"
TRADES_LOG   = ROOT / "logs" / "trades.log"
SYSTEM_LOG   = ROOT / "logs" / "system.log"
PID_FILE     = ROOT / "bot.pid"
_MAIN_PY = (ROOT / "main.py").resolve()


def _cmdline_is_dashboard_bot(cmdline: list[str] | None) -> bool:
    """True si el proceso es main.py --no-dashboard de esta instalación (ruta canónica)."""
    if not cmdline:
        return False
    if "--no-dashboard" not in cmdline:
        return False
    for arg in cmdline:
        try:
            if Path(arg).resolve() == _MAIN_PY:
                return True
        except (OSError, ValueError):
            continue
    return False


app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Logger dedicado del dashboard web (errores que el front-end no puede ver)
_web_log_dir = ROOT / "logs"
_web_log_dir.mkdir(exist_ok=True)
_web_logger = logging.getLogger("inverdan.web")
if not _web_logger.handlers:
    _h = logging.FileHandler(_web_log_dir / "web.log")
    _h.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _web_logger.addHandler(_h)
    _web_logger.setLevel(logging.INFO)


# ── Manejadores globales de error ────────────────────────────────────────────
# Garantizan que cualquier endpoint /api/* devuelva JSON aunque algo reviente.
@app.errorhandler(Exception)
def _handle_any_exception(e):
    if isinstance(e, HTTPException):
        # Errores HTTP estándar (404, 405, 400 mal-formed JSON, etc.)
        if request.path.startswith("/api/"):
            return jsonify({
                "ok": False,
                "error": e.description,
                "code": e.code,
            }), e.code
        return e

    tb = traceback.format_exc()
    _web_logger.error(f"Excepción no controlada en {request.method} {request.path}\n{tb}")
    if request.path.startswith("/api/"):
        return jsonify({
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }), 500
    # Para rutas no-API dejamos que Flask muestre la página por defecto
    return ("Internal Server Error", 500)

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_state() -> dict:
    """Lee el estado del bot desde state.json."""
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def read_jsonl(path: Path, limit: int = 100) -> list:
    """Lee las últimas N líneas de un fichero JSON Lines."""
    if not path.exists():
        return []
    lines = []
    try:
        with open(path) as f:
            raw = f.readlines()
        for line in reversed(raw[-limit * 2:]):
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                    if len(lines) >= limit:
                        break
                except Exception:
                    continue
    except Exception:
        pass
    return lines


def read_log_tail(path: Path, lines: int = 100) -> list:
    """Lee las últimas N líneas de texto de un log."""
    if not path.exists():
        return []
    try:
        with open(path) as f:
            all_lines = f.readlines()
        return [l.rstrip() for l in all_lines[-lines:]]
    except Exception:
        return []


def get_bot_pid() -> int | None:
    """Retorna el PID del bot si está corriendo."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            if not psutil.pid_exists(pid):
                PID_FILE.unlink(missing_ok=True)
            else:
                p = psutil.Process(pid)
                if p.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                    PID_FILE.unlink(missing_ok=True)
                else:
                    try:
                        args = p.cmdline()
                    except Exception:
                        args = []
                    if _cmdline_is_dashboard_bot(args):
                        return pid
                    # PID reutilizado por otro programa o bot.pid obsoleto
                    PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass
    # Buscar por línea de comandos (no usar subcadena "inverdan": la ruta puede no contenerla)
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline_list = proc.info["cmdline"] or []
            if _cmdline_is_dashboard_bot(cmdline_list):
                return proc.info["pid"]
        except Exception:
            continue
    return None


def get_bot_status() -> dict:
    pid = get_bot_pid()
    if not pid:
        return {"running": False, "pid": None, "cpu": 0, "memory_mb": 0, "uptime": None}
    try:
        proc = psutil.Process(pid)
        with proc.oneshot():
            # interval=0 evita bloqueo y reduce fallos esporádicos en macOS
            cpu = proc.cpu_percent(interval=0)
            mem = proc.memory_info().rss / 1024 / 1024
            create_time = proc.create_time()
            uptime_s = int(time.time() - create_time)
            uptime = str(timedelta(seconds=uptime_s))
        return {"running": True, "pid": pid, "cpu": round(cpu, 1),
                "memory_mb": round(mem, 1), "uptime": uptime}
    except Exception:
        # get_bot_pid() ya validó el proceso; aquí solo fallaron métricas (permisos, race, etc.)
        return {"running": True, "pid": pid, "cpu": 0, "memory_mb": 0, "uptime": None}


def get_system_metrics() -> dict:
    return {
        "cpu_pct": psutil.cpu_percent(interval=0.1),
        "ram_pct": psutil.virtual_memory().percent,
        "ram_used_gb": round(psutil.virtual_memory().used / 1e9, 1),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "disk_pct": psutil.disk_usage("/").percent,
        "disk_free_gb": round(psutil.disk_usage("/").free / 1e9, 1),
        "platform": platform.system(),
        "hostname": platform.node(),
        "python": platform.python_version(),
    }


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    resp = render_template("index.html")
    from flask import make_response
    r = make_response(resp)
    r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    return r


@app.route("/api/status")
def api_status():
    state = read_state()
    bot = get_bot_status()
    system = get_system_metrics()

    from inverdan.utils.market_hours import is_market_open, now_et
    try:
        market_open = is_market_open()
        market_time = now_et().strftime("%Y-%m-%d %H:%M:%S ET")
    except Exception:
        market_open = False
        market_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    return jsonify({
        "bot": bot,
        "system": system,
        "market": {"open": market_open, "time": market_time},
        "portfolio": state.get("portfolio", {}),
        "risk": state.get("risk", {}),
        "auto_trade": state.get("auto_trade", False),
        "server_time": datetime.utcnow().isoformat(),
    })


@app.route("/api/signals")
def api_signals():
    limit = int(request.args.get("limit", 50))
    signals = read_jsonl(SIGNALS_LOG, limit)
    return jsonify({"signals": signals, "count": len(signals)})


def _pnl_pct(trade: dict) -> float | None:
    """
    Rentabilidad % de una operación cerrada, a partir de los campos ya
    registrados (pnl, precio de salida, cantidad y acción de cierre).

    El coste (entrada × cantidad) se deduce del pnl:
      cierre SELL (cerraba un largo):  entrada = salida − pnl/qty
      cierre BUY  (cerraba un corto):  entrada = salida + pnl/qty
    """
    pnl = trade.get("pnl")
    qty = trade.get("qty")
    price = trade.get("price")
    action = trade.get("action")
    if pnl is None or not qty or not price:
        return None
    try:
        per_share = pnl / qty
        entry = (price - per_share) if action == "SELL" else (price + per_share)
        cost = abs(entry * qty)
        return round(pnl / cost * 100, 2) if cost else None
    except (TypeError, ZeroDivisionError):
        return None


@app.route("/api/trades")
def api_trades():
    limit = int(request.args.get("limit", 100))
    trades = read_jsonl(TRADES_LOG, limit)

    # Enriquecer cada operación con su rentabilidad %
    for t in trades:
        t["pnl_pct"] = _pnl_pct(t)

    # Estadísticas solo sobre operaciones CERRADAS (cierres reales SL/TP, que
    # llevan "close_reason"). Las aperturas registran "pnl": null o 0.0 y no
    # representan un resultado; incluirlas distorsiona el win-rate y rompe la
    # comparación None > int. Win-rate estándar = ganadoras / (ganadoras+perdedoras).
    closed = [t for t in trades if t.get("close_reason")]
    wins   = sum(1 for t in closed if (t.get("pnl") or 0) > 0)
    losses = sum(1 for t in closed if (t.get("pnl") or 0) < 0)
    decided = wins + losses
    total_pnl = sum((t.get("pnl") or 0) for t in closed)
    win_rate = wins / decided * 100 if decided else 0

    return jsonify({
        "trades": trades,
        "stats": {
            "total": len(closed),
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
        }
    })


@app.route("/api/logs")
def api_logs():
    lines = int(request.args.get("lines", 100))
    log_type = request.args.get("type", "system")

    path_map = {
        "system": SYSTEM_LOG,
        "signals": SIGNALS_LOG,
        "trades": TRADES_LOG,
    }
    path = path_map.get(log_type, SYSTEM_LOG)
    entries = read_log_tail(path, lines)
    return jsonify({"logs": entries, "type": log_type})


@app.route("/api/positions")
def api_positions():
    state = read_state()
    return jsonify({"positions": state.get("positions", [])})


@app.route("/api/config", methods=["GET"])
def api_config_get():
    try:
        with open(CONFIG_FILE) as f:
            cfg = yaml.safe_load(f)
        # No exponer las API keys
        if "alpaca" in cfg:
            cfg["alpaca"].pop("api_key", None)
            cfg["alpaca"].pop("api_secret", None)
        return jsonify({"config": cfg, "ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/config", methods=["POST"])
def api_config_post():
    """Guarda configuración. Solo campos seguros (no credenciales)."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"ok": False, "error": "No data"}), 400

        with open(CONFIG_FILE) as f:
            cfg = yaml.safe_load(f) or {}

        # Campos permitidos (nunca tocar credenciales)
        allowed_top = {"symbols", "timeframe", "bar_buffer_size", "indicators",
                       "ml", "risk", "dashboard", "training"}

        for key in allowed_top:
            if key in data:
                cfg[key] = data[key]

        # paper_trading sí se puede cambiar desde UI
        if "alpaca" in data and "paper_trading" in data["alpaca"]:
            cfg.setdefault("alpaca", {})["paper_trading"] = data["alpaca"]["paper_trading"]

        with open(CONFIG_FILE, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

        return jsonify({"ok": True, "message": "Configuración guardada. Reinicia el bot para aplicar."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def _spawn_bot(auto_trade: bool) -> dict:
    """Lanza main.py --no-dashboard como subproceso y devuelve el resultado."""
    from dotenv import dotenv_values
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / "bot_stdout.log"

    cmd = [sys.executable, str(ROOT / "main.py"), "--no-dashboard"]
    if auto_trade:
        cmd.append("--auto-trade")

    # Construir entorno explícito: hereda el actual y fuerza los valores del .env,
    # evitando que variables vacías o incorrectas del proceso Flask se propaguen.
    env = os.environ.copy()
    env.update(dotenv_values(ROOT / ".env"))

    log_fh = open(log_path, "a")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        log_fh.close()
        raise

    # NO escribir bot.pid aquí — main.py lo escribe él mismo al arrancar.
    # Si Flask escribiera el PID antes de que main.py lo lea, main.py vería su
    # propio PID en bot.pid y abortaría pensando que ya hay una instancia activa.
    _web_logger.info(f"Bot lanzado: PID={proc.pid} cmd={cmd}")
    return {"ok": True, "message": f"Bot iniciado (PID {proc.pid})"}


@app.route("/api/control", methods=["POST"])
def api_control():
    """Controla el bot: start, stop, restart, toggle_auto_trade, emergency_stop."""
    try:
        data = request.get_json(silent=True) or {}
    except Exception:
        data = {}
    action = data.get("action", "")

    try:
        pid = get_bot_pid()
    except Exception as e:
        _web_logger.error(f"get_bot_pid falló: {e}\n{traceback.format_exc()}")
        # Ante fallo de inspección de procesos asumimos que no hay bot activo
        pid = None

    if action == "start":
        if pid:
            return jsonify({"ok": False, "message": "El bot ya está corriendo."})
        try:
            return jsonify(_spawn_bot(bool(data.get("auto_trade"))))
        except Exception as e:
            _web_logger.error(f"start falló: {e}\n{traceback.format_exc()}")
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500

    elif action == "stop":
        if not pid:
            return jsonify({"ok": False, "message": "El bot no está corriendo."})
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            if psutil.pid_exists(pid):
                os.kill(pid, signal.SIGKILL)
            PID_FILE.unlink(missing_ok=True)
            return jsonify({"ok": True, "message": f"Bot detenido (PID {pid})"})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    elif action == "restart":
        auto_trade = bool(data.get("auto_trade"))

        def _do_restart(old_pid):
            if old_pid:
                try:
                    os.kill(old_pid, signal.SIGTERM)
                    time.sleep(5)
                    if psutil.pid_exists(old_pid):
                        os.kill(old_pid, signal.SIGKILL)
                except Exception:
                    pass
            PID_FILE.unlink(missing_ok=True)
            # Esperar a que Alpaca libere la conexión WebSocket
            time.sleep(30)
            # No lanzar si ya hay un bot corriendo (evita conflictos con threads paralelos)
            if get_bot_pid():
                _web_logger.info("restart (bg): ya hay un bot corriendo, omitiendo spawn.")
                return
            try:
                _spawn_bot(auto_trade)
            except Exception as e:
                _web_logger.error(f"restart (bg) falló: {e}\n{traceback.format_exc()}")

        threading.Thread(target=_do_restart, args=(pid,), daemon=True, name="bot-restart").start()
        return jsonify({"ok": True, "message": "Reiniciando bot en ~30s (Alpaca necesita liberar la conexión WebSocket)…"})

    elif action == "emergency_stop":
        """Cierra posiciones + detiene el bot."""
        state = read_state()
        state["emergency_stop"] = True
        state["emergency_stop_at"] = datetime.utcnow().isoformat()
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(3)
                if psutil.pid_exists(pid):
                    os.kill(pid, signal.SIGKILL)
                PID_FILE.unlink(missing_ok=True)
            except Exception:
                pass
        return jsonify({"ok": True, "message": "PARADA DE EMERGENCIA ejecutada. Revisa posiciones en Alpaca."})

    elif action == "toggle_auto_trade":
        state = read_state()
        current = state.get("auto_trade", False)
        state["auto_trade"] = not current
        state["auto_trade_changed_at"] = datetime.utcnow().isoformat()
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
        mode = "ACTIVADO" if state["auto_trade"] else "DESACTIVADO"
        return jsonify({"ok": True, "auto_trade": state["auto_trade"],
                        "message": f"Auto-trade {mode}"})

    return jsonify({"ok": False, "error": f"Acción desconocida: {action}"}), 400


@app.route("/api/pnl_history")
def api_pnl_history():
    """Historial de PnL para el gráfico."""
    trades = read_jsonl(TRADES_LOG, 500)
    history = []
    cumulative = 0.0
    for t in reversed(trades):
        pnl = t.get("pnl") or 0   # aperturas tienen pnl=null
        cumulative += pnl
        history.append({
            "ts": t.get("_ts", ""),
            "symbol": t.get("symbol", ""),
            "pnl": round(pnl, 2),
            "cumulative": round(cumulative, 2),
        })
    return jsonify({"history": history})


@app.route("/api/market")
def api_market():
    """Último snapshot de indicadores por símbolo."""
    state = read_state()
    return jsonify({"market": state.get("market", {})})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Crear directorio logs si no existe
    (ROOT / "logs").mkdir(exist_ok=True)
    if not STATE_FILE.exists():
        STATE_FILE.write_text("{}")

    print(f"\n  INVERDAN Dashboard → http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)
