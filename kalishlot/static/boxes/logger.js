// Shared text log, docked as a slide-in panel. One instance for the whole
// page (not per box): any box can append a line via logEntry(), any box's
// header button opens the panel via openLogger(). New entries always render
// at the top. Persisted and broadcast server-side (GET/POST /api/log,
// WS /ws/log) so the log survives a reload and stays in sync across viewers
// — this module only ever inserts an entry into the DOM when it arrives over
// the websocket, so a locally-triggered log and one from another viewer are
// rendered by the exact same path.

const RETRY_START_MS = 2000;
const RETRY_MAX_MS = 15000;

let listEl = null;
let panelEl = null;

function addEntryToDom(entry, atTop) {
  const row = document.createElement('div');
  row.className = 'log-entry';
  row.innerHTML = `<span class="log-time">${entry.time}</span><span class="log-text"></span>`;
  row.querySelector('.log-text').textContent = entry.text;
  if (atTop) listEl.insertBefore(row, listEl.firstChild);
  else listEl.appendChild(row);
}

let retryMs = RETRY_START_MS;

function connect() {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${protocol}://${location.host}/ws/log`);
  socket.onopen = () => { retryMs = RETRY_START_MS; };
  socket.onmessage = (message) => {
    const event = JSON.parse(message.data);
    if (event.type === 'entry') addEntryToDom(event.entry, true);
  };
  socket.onclose = () => {
    setTimeout(connect, retryMs);
    retryMs = Math.min(retryMs * 2, RETRY_MAX_MS);
  };
}

export function initLogger() {
  panelEl = document.createElement('div');
  panelEl.className = 'log-panel';
  panelEl.innerHTML = `
    <div class="box-header">
      <span class="box-title">log</span>
      <button class="box-close" title="close log">✕</button>
    </div>
    <div class="log-entries"></div>`;
  document.body.appendChild(panelEl);
  listEl = panelEl.querySelector('.log-entries');
  panelEl.querySelector('.box-close').onclick = () => panelEl.classList.remove('open');

  fetch('/api/log')
    .then((response) => response.json())
    .then((entries) => { for (const entry of entries) addEntryToDom(entry, false); })
    .catch(() => {});

  connect();
}

export function openLogger() {
  panelEl.classList.add('open');
}

export async function logEntry(text) {
  await fetch('/api/log', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
}
