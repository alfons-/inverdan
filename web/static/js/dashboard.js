/* ═══════════════════════════════════════════════════════
   INVERDAN Dashboard — JavaScript
   ═══════════════════════════════════════════════════════ */

'use strict';

// ── Estado global ──────────────────────────────────────
const state = {
  section: 'overview',
  signalFilter: 'all',
  logType: 'system',
  config: null,
  lastStatus: null,
  pnlChart: null,
  autoTrade: false,
};

const REFRESH_MS = 2000;   // Polling general
const LOG_MS     = 5000;   // Polling de logs

// ── Utilidades ─────────────────────────────────────────
const $ = id => document.getElementById(id);
const fmt = {
  usd: v => v == null ? '—' : (v >= 0 ? '+$' : '-$') + Math.abs(v).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2}),
  usdPlain: v => v == null ? '—' : '$' + Number(v).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2}),
  pct: v => v == null ? '—' : (v >= 0 ? '+' : '') + Number(v).toFixed(2) + '%',
  pctPlain: v => v == null ? '—' : Number(v).toFixed(1) + '%',
  ts: s => s ? new Date(s).toLocaleTimeString('es-ES') : '—',
  tsDate: s => s ? new Date(s).toLocaleString('es-ES', {hour:'2-digit', minute:'2-digit', second:'2-digit', day:'2-digit', month:'2-digit'}) : '—',
};

function colorClass(v) {
  if (v > 0) return 'text-green';
  if (v < 0) return 'text-red';
  return 'text-muted';
}
function pnlHtml(v) {
  return `<span class="${colorClass(v)}">${fmt.usd(v)}</span>`;
}

// ── TOAST ───────────────────────────────────────────────
function toast(msg, type = 'info', duration = 3500) {
  const icons = { success: '✓', error: '✗', info: 'ℹ', warning: '⚠' };
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${icons[type] || ''}</span><span class="toast-msg">${msg}</span>`;
  $('toastContainer').appendChild(el);
  setTimeout(() => el.remove(), duration);
}

// ── MODAL ───────────────────────────────────────────────
let _modalCallback = null;
function confirm(title, body, onConfirm, btnClass = 'btn-danger') {
  $('modalTitle').textContent = title;
  $('modalBody').textContent = body;
  const btn = $('modalConfirmBtn');
  btn.className = `btn ${btnClass}`;
  _modalCallback = onConfirm;
  btn.onclick = () => { closeModal(); onConfirm(); };
  $('modalOverlay').classList.add('open');
}
function closeModal() {
  $('modalOverlay').classList.remove('open');
  _modalCallback = null;
}

// ── CLOCK ───────────────────────────────────────────────
function startClock() {
  const update = () => {
    $('clock').textContent = new Date().toLocaleTimeString('es-ES', {
      hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
  };
  update();
  setInterval(update, 1000);
}

// ── NAVIGATION ─────────────────────────────────────────
function navigate(section) {
  document.querySelectorAll('.nav-item').forEach(el => {
    el.classList.toggle('active', el.dataset.section === section);
  });
  document.querySelectorAll('.section').forEach(el => {
    el.classList.toggle('active', el.id === `section-${section}`);
  });
  const titles = {
    overview: 'Resumen', positions: 'Posiciones', signals: 'Señales',
    trades: 'Operaciones', server: 'Servidor', config: 'Configuración', logs: 'Logs',
    manual: 'Manual de Usuario'
  };
  $('pageTitle').textContent = titles[section] || section;
  state.section = section;

  if (section === 'signals') fetchSignals();
  if (section === 'trades')  fetchTrades();
  if (section === 'config')  loadConfig();
  if (section === 'logs')    loadLogs('system');
  if (section === 'server')  fetchStatus();
}

document.querySelectorAll('.nav-item, .card-link').forEach(el => {
  if (el.dataset.section) {
    el.addEventListener('click', e => {
      e.preventDefault();
      navigate(el.dataset.section);
      if (window.innerWidth < 900) $('sidebar').classList.remove('open');
    });
  }
});

$('sidebarToggle').addEventListener('click', () => {
  $('sidebar').classList.toggle('open');
});

// ── PNL CHART ────────────────────────────────────────────
function initPnlChart() {
  const ctx = document.getElementById('pnlChart').getContext('2d');
  state.pnlChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'PnL Acumulado',
        data: [],
        borderColor: '#00d4aa',
        backgroundColor: 'rgba(0,212,170,0.07)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#151820',
          borderColor: '#252a38',
          borderWidth: 1,
          titleColor: '#7b8299',
          bodyColor: '#e8eaf0',
          callbacks: {
            label: ctx => ' PnL: $' + ctx.parsed.y.toFixed(2)
          }
        }
      },
      scales: {
        x: {
          display: false,
        },
        y: {
          grid: { color: '#1a1e28' },
          ticks: {
            color: '#7b8299',
            font: { size: 11 },
            callback: v => '$' + v.toFixed(0)
          },
          border: { display: false }
        }
      }
    }
  });
}

async function fetchPnlChart() {
  try {
    const r = await fetch('/api/pnl_history');
    const data = await r.json();
    const h = data.history || [];
    if (!h.length) return;
    const chart = state.pnlChart;
    chart.data.labels = h.map(x => fmt.ts(x.ts));
    chart.data.datasets[0].data = h.map(x => x.cumulative);
    const total = h[h.length - 1]?.cumulative ?? 0;
    chart.data.datasets[0].borderColor = total >= 0 ? '#00c97a' : '#f04b5a';
    chart.data.datasets[0].backgroundColor = total >= 0
      ? 'rgba(0,201,122,0.07)' : 'rgba(240,75,90,0.07)';
    chart.update('none');
    const badge = $('pnlChartTotal');
    badge.textContent = fmt.usd(total);
    badge.className = 'card-badge ' + (total >= 0 ? 'text-green' : 'text-red');
  } catch {}
}

// ── STATUS FETCH ─────────────────────────────────────────
async function fetchStatus() {
  const dot = $('refreshDot');
  dot.classList.add('active');
  try {
    const r = await fetch('/api/status');
    const data = await r.json();
    state.lastStatus = data;
    updateOverview(data);
    updateBotStatus(data.bot);
    updateMarketBadge(data.market);
    updateSystemMetrics(data.system);
  } catch (e) {
    // silencioso
  } finally {
    dot.classList.remove('active');
  }
}

function updateOverview(data) {
  const p = data.portfolio || {};
  const r = data.risk || {};

  // KPIs
  $('kpiEquity').textContent = fmt.usdPlain(p.equity);
  $('kpiEquitySub').textContent = 'Valor total del portfolio';

  const dpnl = p.daily_pnl ?? 0;
  $('kpiDailyPnl').innerHTML = `<span class="${colorClass(dpnl)}">${fmt.usd(dpnl)}</span>`;
  $('kpiDailyPnlSub').textContent = 'Resultado de hoy';

  const upnl = p.total_unrealized_pnl ?? 0;
  $('kpiUnrealizedPnl').innerHTML = `<span class="${colorClass(upnl)}">${fmt.usd(upnl)}</span>`;
  $('kpiUnrealizedSub').textContent = 'En posiciones abiertas';

  $('kpiBuyingPower').textContent = fmt.usdPlain(p.buying_power);

  const trades = p.trades_today ?? 0;
  const wins   = p.wins_today ?? 0;
  const losses = p.losses_today ?? 0;
  const winRate = trades > 0 ? (wins / trades * 100).toFixed(0) : 0;
  $('kpiTrades').textContent = trades;
  $('kpiWinRate').innerHTML = `<span class="text-green">${wins}W</span> / <span class="text-red">${losses}L</span> — ${winRate}% win rate`;

  const at = data.auto_trade;
  state.autoTrade = at;
  $('kpiAutoTrade').innerHTML = at
    ? '<span class="badge badge-green">ACTIVO</span>'
    : '<span class="badge badge-muted">INACTIVO</span>';

  // Sync controls
  const cb = $('ctrlAutoTrade');
  if (cb) cb.checked = at;
  const cbPaper = $('ctrlPaperMode');
  if (cbPaper && data.config) cbPaper.checked = data.config.paper_trading !== false;
}

function updateBotStatus(bot) {
  if (!bot) return;
  $('botRunning').innerHTML = bot.running
    ? '<span class="badge badge-green">CORRIENDO</span>'
    : '<span class="badge badge-red">DETENIDO</span>';
  $('ctrlBotStatus') && ($('ctrlBotStatus').innerHTML = $('botRunning').innerHTML);
  $('botPid').textContent = bot.pid ?? '—';
  $('botCpu').textContent = bot.running ? bot.cpu + '%' : '—';
  $('botMem').textContent = bot.running ? bot.memory_mb + ' MB' : '—';
  $('botUptime').textContent = bot.uptime ?? '—';
  if ($('botCircuit') && state.lastStatus?.risk) {
    const cb = state.lastStatus.risk.circuit_open;
    $('botCircuit').innerHTML = cb
      ? '<span class="badge badge-red">ABIERTO</span>'
      : '<span class="badge badge-green">OK</span>';
  }
}

function updateMarketBadge(market) {
  const dot  = document.querySelector('.market-dot');
  const text = $('marketStatus');
  if (!market) return;
  if (market.open) {
    dot.classList.add('open');
    dot.classList.remove('closed');
    text.textContent = 'Mercado abierto';
  } else {
    dot.classList.add('closed');
    dot.classList.remove('open');
    text.textContent = 'Mercado cerrado';
  }
}

function updateSystemMetrics(sys) {
  if (!sys) return;
  const setBar = (barId, valId, pct, label) => {
    const bar = $(barId);
    if (bar) {
      bar.style.width = pct + '%';
      bar.style.background = pct > 85 ? 'var(--red)' : pct > 65 ? 'var(--yellow)' : 'var(--accent)';
    }
    if ($(valId)) $(valId).textContent = label;
  };
  setBar('cpuBar',  'cpuVal',  sys.cpu_pct,  `${sys.cpu_pct}%`);
  setBar('ramBar',  'ramVal',  sys.ram_pct,  `${sys.ram_used_gb} / ${sys.ram_total_gb} GB (${sys.ram_pct}%)`);
  setBar('diskBar', 'diskVal', sys.disk_pct, `${sys.disk_free_gb} GB libres (${sys.disk_pct}%)`);

  const info = $('systemInfo');
  if (info) info.innerHTML = `
    <b>Host:</b> ${sys.hostname}<br>
    <b>OS:</b> ${sys.platform}<br>
    <b>Python:</b> ${sys.python}
  `;
}

// ── POSITIONS ────────────────────────────────────────────
async function fetchPositions() {
  try {
    const r = await fetch('/api/positions');
    const data = await r.json();
    renderPositions(data.positions || []);
  } catch {}
}

function renderPositions(positions) {
  $('positionsCount').textContent = positions.length;
  const body = $('positionsBody');
  if (!positions.length) {
    body.innerHTML = '<tr><td colspan="9" class="empty">Sin posiciones abiertas</td></tr>';
    return;
  }
  body.innerHTML = positions.map(p => {
    const pnl = p.unrealized_pnl ?? 0;
    const pct = p.unrealized_pnl_pct ?? 0;
    const sideClass = p.side === 'long' ? 'badge-green' : 'badge-red';
    return `<tr>
      <td><b>${p.symbol}</b></td>
      <td><span class="badge ${sideClass}">${p.side.toUpperCase()}</span></td>
      <td>${p.qty}</td>
      <td>${fmt.usdPlain(p.entry_price)}</td>
      <td>${fmt.usdPlain(p.current_price)}</td>
      <td>${pnlHtml(pnl)}</td>
      <td><span class="${colorClass(pct)}">${fmt.pct(pct * 100)}</span></td>
      <td class="text-muted">${p.stop_loss ? fmt.usdPlain(p.stop_loss) : '—'}</td>
      <td class="text-muted">${p.take_profit ? fmt.usdPlain(p.take_profit) : '—'}</td>
    </tr>`;
  }).join('');
}

// ── SIGNALS ─────────────────────────────────────────────
async function fetchSignals() {
  try {
    const r = await fetch('/api/signals?limit=100');
    const data = await r.json();
    renderSignals(data.signals || []);
  } catch {}
}

function renderSignals(signals) {
  const filter = state.signalFilter;
  const search = ($('signalSearch')?.value || '').toUpperCase();
  const filtered = signals.filter(s => {
    if (filter !== 'all' && s.action !== filter) return false;
    if (search && !(s.symbol || '').includes(search)) return false;
    return true;
  });
  const body = $('signalsBody');
  if (!filtered.length) {
    body.innerHTML = '<tr><td colspan="9" class="empty">Sin señales</td></tr>';
    return;
  }
  body.innerHTML = filtered.map(s => {
    const cls = s.action === 'BUY' ? 'badge-green' : s.action === 'SELL' ? 'badge-red' : 'badge-yellow';
    const mlCls = s.ml_signal === 'BUY' ? 'text-green' : s.ml_signal === 'SELL' ? 'text-red' : 'text-muted';
    const rsi = s.ind_rsi ? Number(s.ind_rsi).toFixed(1) : '—';
    return `<tr>
      <td class="text-muted">${fmt.tsDate(s._ts || s.timestamp)}</td>
      <td><b>${s.symbol || '—'}</b></td>
      <td><span class="badge ${cls}">${s.action}</span></td>
      <td>${s.confidence ? (s.confidence * 100).toFixed(0) + '%' : '—'}</td>
      <td>${fmt.usdPlain(s.price)}</td>
      <td class="${mlCls}">${s.ml_signal || '—'} ${s.ml_confidence ? '(' + (s.ml_confidence*100).toFixed(0)+'%)' : ''}</td>
      <td class="${s.rule_signal === 'BUY' ? 'text-green' : s.rule_signal === 'SELL' ? 'text-red' : 'text-muted'}">${s.rule_signal || '—'}</td>
      <td class="text-muted">${rsi}</td>
      <td class="text-muted" title="${s.reasoning || ''}">${(s.reasoning || '').substring(0, 50)}${s.reasoning?.length > 50 ? '…' : ''}</td>
    </tr>`;
  }).join('');

  // Señales preview en overview
  const previewBody = $('signalsPreviewBody');
  if (previewBody) {
    const top5 = signals.slice(0, 5);
    if (!top5.length) {
      previewBody.innerHTML = '<tr><td colspan="6" class="empty">Sin señales recientes</td></tr>';
    } else {
      previewBody.innerHTML = top5.map(s => {
        const cls = s.action === 'BUY' ? 'badge-green' : s.action === 'SELL' ? 'badge-red' : 'badge-yellow';
        return `<tr>
          <td class="text-muted">${fmt.ts(s._ts || s.timestamp)}</td>
          <td><b>${s.symbol || '—'}</b></td>
          <td><span class="badge ${cls}">${s.action}</span></td>
          <td>${s.confidence ? (s.confidence*100).toFixed(0)+'%' : '—'}</td>
          <td>${fmt.usdPlain(s.price)}</td>
          <td class="text-muted">${(s.reasoning || '').substring(0, 40)}${s.reasoning?.length > 40 ? '…' : ''}</td>
        </tr>`;
      }).join('');
    }
  }
}

// Filter buttons wiring
document.querySelectorAll('.filter-btn[data-filter]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn[data-filter]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.signalFilter = btn.dataset.filter;
    fetchSignals();
  });
});
$('signalSearch')?.addEventListener('input', () => fetchSignals());

// ── TRADES ──────────────────────────────────────────────
async function fetchTrades() {
  try {
    const r = await fetch('/api/trades?limit=100');
    const data = await r.json();
    renderTrades(data.trades || [], data.stats || {});
  } catch {}
}

function renderTrades(trades, stats) {
  $('statTotal').textContent = stats.total ?? 0;
  $('statWinLoss').innerHTML = `<span class="text-green">${stats.wins ?? 0}W</span> / <span class="text-red">${stats.losses ?? 0}L</span>`;
  $('statWinRate').innerHTML = `<span class="${colorClass(stats.win_rate - 50)}">${fmt.pctPlain(stats.win_rate)}</span>`;
  $('statPnl').innerHTML = pnlHtml(stats.total_pnl);

  const body = $('tradesBody');
  if (!trades.length) {
    body.innerHTML = '<tr><td colspan="9" class="empty">Sin operaciones registradas</td></tr>';
    return;
  }
  body.innerHTML = trades.map(t => {
    const cls = t.action === 'BUY' ? 'badge-green' : 'badge-red';
    const pnl = t.pnl ?? null;
    return `<tr>
      <td class="text-muted">${fmt.tsDate(t._ts)}</td>
      <td><b>${t.symbol || '—'}</b></td>
      <td><span class="badge ${cls}">${t.action || '—'}</span></td>
      <td>${t.qty || '—'}</td>
      <td>${fmt.usdPlain(t.price)}</td>
      <td class="text-muted">${t.stop_loss ? fmt.usdPlain(t.stop_loss) : '—'}</td>
      <td class="text-muted">${t.take_profit ? fmt.usdPlain(t.take_profit) : '—'}</td>
      <td>${pnl != null ? pnlHtml(pnl) : '—'}</td>
      <td class="text-muted" style="font-size:11px">${(t.order_id || '').substring(0, 12)}…</td>
    </tr>`;
  }).join('');
}

// ── LOGS ────────────────────────────────────────────────
let _currentLogType = 'system';

function loadLogs(type, btn) {
  _currentLogType = type;
  document.querySelectorAll('.filter-btn[data-log]').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  else {
    const b = document.querySelector(`.filter-btn[data-log="${type}"]`);
    if (b) b.classList.add('active');
  }
  fetchLogs(type);
}

async function fetchLogs(type) {
  try {
    const r = await fetch(`/api/logs?type=${type}&lines=150`);
    const data = await r.json();
    renderLogs(data.logs || []);
  } catch {}
}

function renderLogs(lines) {
  const el = $('logTerminal');
  if (!lines.length) {
    el.innerHTML = '<span class="log-empty">Sin logs disponibles</span>';
    return;
  }
  el.innerHTML = lines.map(line => {
    let cls = 'log-line';
    if (/error|ERROR|CRITICAL/i.test(line)) cls += ' error';
    else if (/warn|WARNING/i.test(line)) cls += ' warn';
    else if (/INFO|iniciado|cargado|conectado/i.test(line)) cls += ' info';
    return `<div class="${cls}">${escHtml(line)}</div>`;
  }).join('');
  el.scrollTop = el.scrollHeight;
}

async function refreshLogs() {
  const el = $('botLogTerminal') || $('logTerminal');
  try {
    const r = await fetch('/api/logs?type=system&lines=80');
    const data = await r.json();
    if ($('botLogTerminal')) renderBotLogs(data.logs || []);
    else renderLogs(data.logs || []);
  } catch {}
}

function renderBotLogs(lines) {
  const el = $('botLogTerminal');
  if (!lines.length) { el.innerHTML = '<span class="log-empty">Sin logs disponibles</span>'; return; }
  el.innerHTML = lines.map(l => {
    let cls = 'log-line';
    if (/error|ERROR/i.test(l)) cls += ' error';
    else if (/WARN/i.test(l)) cls += ' warn';
    return `<div class="${cls}">${escHtml(l)}</div>`;
  }).join('');
  el.scrollTop = el.scrollHeight;
}

function refreshCurrentLogs() { fetchLogs(_currentLogType); }

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── CONFIG ──────────────────────────────────────────────
async function loadConfig() {
  try {
    const r = await fetch('/api/config');
    const data = await r.json();
    if (data.ok) {
      state.config = data.config;
      renderConfig(data.config);
    }
  } catch {}
}

function renderConfig(cfg) {
  const grid = $('configGrid');
  if (!cfg || !grid) return;

  const sections = [
    {
      key: 'alpaca', title: 'Alpaca / Broker',
      fields: [
        { label: 'Paper Trading (simulado)', key: 'paper_trading', type: 'checkbox' },
        { label: 'Feed de datos', key: 'data_feed', type: 'select', options: ['iex', 'sip'] },
      ]
    },
    {
      key: 'risk', title: 'Gestión de Riesgo',
      fields: [
        { label: 'Max. posición (%)', key: 'max_position_pct', type: 'number', step: 0.01 },
        { label: 'Max. exposición total (%)', key: 'max_total_exposure', type: 'number', step: 0.01 },
        { label: 'Stop-loss ATR mult.', key: 'stop_loss_atr_multiplier', type: 'number', step: 0.1 },
        { label: 'Take-profit ATR mult.', key: 'take_profit_atr_multiplier', type: 'number', step: 0.1 },
        { label: 'Pérdida diaria máx. (%)', key: 'max_daily_loss_pct', type: 'number', step: 0.01 },
        { label: 'Pérdidas consecutivas máx.', key: 'max_consecutive_losses', type: 'number' },
        { label: 'Órdenes por minuto máx.', key: 'max_orders_per_minute', type: 'number' },
        { label: 'Precio mínimo ($)', key: 'min_stock_price', type: 'number', step: 0.5 },
      ]
    },
    {
      key: 'ml', title: 'Machine Learning',
      fields: [
        { label: 'Tipo de modelo', key: 'model_type', type: 'select', options: ['random_forest', 'ensemble'] },
        { label: 'Umbral confianza', key: 'confidence_threshold', type: 'number', step: 0.01 },
        { label: 'Lookback features', key: 'feature_lookback', type: 'number' },
        { label: 'Barras mínimas', key: 'min_bars_required', type: 'number' },
      ]
    },
    {
      key: 'indicators', title: 'Indicadores Técnicos',
      fields: [
        { label: 'RSI período', key: 'rsi_period', type: 'number' },
        { label: 'MACD rápido', key: 'macd_fast', type: 'number' },
        { label: 'MACD lento', key: 'macd_slow', type: 'number' },
        { label: 'BB período', key: 'bb_period', type: 'number' },
        { label: 'ATR período', key: 'atr_period', type: 'number' },
        { label: 'EMA rápida', key: 'ema_fast', type: 'number' },
        { label: 'EMA lenta', key: 'ema_slow', type: 'number' },
      ]
    },
  ];

  grid.innerHTML = sections.map(sec => {
    const src = sec.key === 'alpaca' ? cfg : (cfg[sec.key] || {});
    const fields = sec.fields.map(f => {
      const val = src[f.key] ?? '';
      let input;
      if (f.type === 'checkbox') {
        input = `<input type="checkbox" data-section="${sec.key}" data-key="${f.key}" ${val ? 'checked' : ''} />`;
      } else if (f.type === 'select') {
        const opts = f.options.map(o => `<option value="${o}" ${o === val ? 'selected' : ''}>${o}</option>`).join('');
        input = `<select data-section="${sec.key}" data-key="${f.key}">${opts}</select>`;
      } else {
        input = `<input type="number" step="${f.step || 1}" value="${val}"
                   data-section="${sec.key}" data-key="${f.key}" />`;
      }
      return `<div class="config-field"><label>${f.label}</label>${input}</div>`;
    }).join('');
    return `<div class="config-section">
      <div class="config-section-title">${sec.title}</div>
      ${fields}
    </div>`;
  }).join('');
}

async function saveConfig() {
  if (!state.config) return;
  const updated = JSON.parse(JSON.stringify(state.config));
  document.querySelectorAll('[data-section][data-key]').forEach(el => {
    const sec = el.dataset.section;
    const key = el.dataset.key;
    let val = el.type === 'checkbox' ? el.checked : (el.tagName === 'SELECT' ? el.value : Number(el.value));
    if (sec === 'alpaca') {
      updated[key] = val;
    } else {
      if (!updated[sec]) updated[sec] = {};
      updated[sec][key] = val;
    }
  });
  try {
    const r = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updated),
    });
    const data = await r.json();
    if (data.ok) toast(data.message, 'success');
    else toast('Error: ' + data.error, 'error');
  } catch (e) {
    toast('Error de conexión', 'error');
  }
}

// ── BOT CONTROLS ─────────────────────────────────────────
async function controlBot(action) {
  const labels = { start: 'Iniciar', stop: 'Detener', restart: 'Reiniciar' };
  const types  = { start: 'btn-success', stop: 'btn-danger', restart: 'btn-warning' };
  const msgs   = {
    start: '¿Iniciar el bot de trading?',
    stop: '¿Detener el bot? Las órdenes abiertas en Alpaca permanecerán activas.',
    restart: '¿Reiniciar el bot? El sistema se detendrá y volverá a arrancar.',
  };

  confirm(labels[action] + ' Bot', msgs[action], async () => {
    try {
      const r = await fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, auto_trade: state.autoTrade }),
      });
      const data = await r.json();
      toast(data.message || data.error, data.ok ? 'success' : 'error');
      setTimeout(fetchStatus, 1500);
    } catch {
      toast('Error de conexión con el servidor', 'error');
    }
  }, types[action]);
}

async function toggleAutoTrade() {
  try {
    const r = await fetch('/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'toggle_auto_trade' }),
    });
    const data = await r.json();
    toast(data.message, data.ok ? 'info' : 'error');
    state.autoTrade = data.auto_trade;
    setTimeout(fetchStatus, 500);
  } catch {
    toast('Error de conexión', 'error');
  }
}

async function togglePaperMode() {
  const el = $('ctrlPaperMode');
  if (!el) return;
  if (!el.checked) {
    confirm(
      '⚠ ACTIVAR TRADING REAL',
      'Vas a desactivar el Paper Trading. Las operaciones se ejecutarán con DINERO REAL. ¿Estás seguro?',
      async () => {
        await updatePaperMode(false);
      },
      'btn-danger'
    );
    // Revertir el checkbox hasta confirmar
    el.checked = true;
  } else {
    await updatePaperMode(true);
  }
}

async function updatePaperMode(paperMode) {
  try {
    const r = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ alpaca: { paper_trading: paperMode } }),
    });
    const data = await r.json();
    toast((paperMode ? 'Paper Trading activado' : '⚠ Paper Trading DESACTIVADO — usando dinero real'), paperMode ? 'info' : 'warning');
    const el = $('ctrlPaperMode');
    if (el) el.checked = paperMode;
  } catch {
    toast('Error actualizando configuración', 'error');
  }
}

function emergencyStop() {
  confirm(
    '⛔ PARADA DE EMERGENCIA',
    'Se detendrá el bot inmediatamente. Las órdenes bracket en Alpaca permanecerán activas (stop-loss y take-profit siguen funcionando server-side). Deberás gestionar manualmente las posiciones abiertas desde Alpaca.',
    async () => {
      try {
        const r = await fetch('/api/control', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'emergency_stop' }),
        });
        const data = await r.json();
        toast(data.message, data.ok ? 'warning' : 'error');
        setTimeout(fetchStatus, 1500);
      } catch {
        toast('Error de conexión', 'error');
      }
    }
  );
}

// ── POLLING LOOP ─────────────────────────────────────────
async function refreshAll() {
  await fetchStatus();
  await fetchPositions();
  if (state.section === 'signals') await fetchSignals();
  if (state.section === 'trades')  await fetchTrades();
  if (state.section === 'server')  await refreshLogs();
}

// ── INIT ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  startClock();
  initPnlChart();

  // Carga inicial
  fetchStatus();
  fetchPositions();
  fetchSignals();  // Para el preview del overview
  fetchPnlChart();

  // Polling
  setInterval(refreshAll, REFRESH_MS);
  setInterval(fetchPnlChart, 10_000);
  setInterval(refreshCurrentLogs, LOG_MS);
});
