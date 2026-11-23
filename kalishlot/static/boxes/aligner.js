// Aligner box: describe an alignment as signals with goals plus the controls it
// may move, then run it automatically (adapters/aligner.py, alignment/recipe.py).
// One level for now; "+ alignment" for a nested inner level comes later.
// The setup lives on the server (it persists); every edit is sent as set_setup.
// Number inputs commit on Enter or when they lose focus (lab convention: Enter).
// Returns a cleanup function that closes the socket.

import { connectDeviceStream } from './stream.js';

const fmt = (v, digits = 4) => (v === null || v === undefined || !isFinite(v))
  ? '—' : parseFloat(Number(v).toPrecision(digits)).toString();
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const quantityOf = (signal) => (signal || '').split('/').pop();
const deviceOf = (signal) => (signal || '').split('/').slice(0, -1).join('/');
const coordOf = (signal) => ({ com_x: 'x', com_y: 'y' }[quantityOf(signal)] ?? null);

export function createAlignerBox(device, container, sendCommand) {
  container.innerHTML = `
    <div class="toolbar">
      <button class="al-start primary" title="run the alignment">Start</button>
      <button class="al-pause" title="hold before the next measurement (axes stand still)">Pause</button>
      <button class="al-stop" title="end after the current measurement">Stop</button>
      <label class="field" title="where the axes go when stopped">then
        <select class="al-onabort">
          <option value="best">return to best</option>
          <option value="start">return to start</option>
          <option value="stay">stay</option>
        </select></label>
      <button class="al-dry" title="measure once and show each row's miss; nothing moves">Dry run</button>
      <button class="al-refresh" title="re-read the open cameras, markers and motion boxes">↻</button>
      <span class="lamp al-state">idle</span>
    </div>
    <div class="toolbar al-fields">
      <label class="field" title="wait after a move before frames count">delay
        <input class="al-f" data-f="delay_s" type="number" step="any" min="0"> <span class="unit">s</span></label>
      <label class="field" title="centre-of-mass results per camera per measurement (median)">samples
        <input class="al-f" data-f="samples" type="number" step="1" min="1"></label>
      <label class="field" title="measurement budget of one run">max evals
        <input class="al-f" data-f="max_evaluations" type="number" step="1" min="1"></label>
    </div>
    <div class="al-levels"></div>
    <div class="toolbar">
      <button class="al-add-level" disabled title="a nested inner alignment: coming">+ alignment (nested: coming)</button>
    </div>
    <div class="al-report"></div>
    <div class="al-chart"></div>
    <pre class="al-log"></pre>
    <div class="status-line al-status"></div>`;

  const $ = (sel) => container.querySelector(sel);
  const status = $('.al-status');
  const fail = (error) => { status.textContent = error.message; };
  const send = (name, args) => sendCommand(device.device_id, name, args)
    .then((r) => { status.textContent = ''; return r; }).catch((e) => { fail(e); throw e; });

  let setup = structuredClone(device.setup);
  let catalog = device.catalog;
  let state = device.state;
  let history = device.history ?? [];
  let lastRows = device.result?.rows ?? null;
  const logLines = [];

  const pushSetup = () => send('set_setup', { setup }).catch(() => {});

  // ------------------------------------------------------------ toolbar
  $('.al-start').onclick = () => send('start').catch(() => {});
  $('.al-pause').onclick = () => send(state === 'paused' ? 'resume' : 'pause').catch(() => {});
  $('.al-stop').onclick = () => send('stop', { on_abort: $('.al-onabort').value }).catch(() => {});
  $('.al-onabort').onchange = () => { setup.fields.on_abort = $('.al-onabort').value; pushSetup(); };
  $('.al-dry').onclick = () => {
    status.textContent = 'measuring…';
    send('dry_run').then((r) => { lastRows = r.rows; showReport('dry run'); }).catch(() => {});
  };
  $('.al-refresh').onclick = () => send('refresh').then((r) => { catalog = r.catalog; renderLevels(); }).catch(() => {});

  for (const input of container.querySelectorAll('.al-f')) {
    const commit = () => {
      const v = parseFloat(input.value);
      if (isFinite(v) && v !== setup.fields[input.dataset.f]) { setup.fields[input.dataset.f] = v; pushSetup(); }
    };
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') commit(); });
    input.addEventListener('blur', commit);
  }

  function showFields() {
    for (const input of container.querySelectorAll('.al-f')) input.value = setup.fields[input.dataset.f];
    $('.al-onabort').value = setup.fields.on_abort ?? 'best';
  }

  function showState() {
    const lamp = $('.al-state');
    lamp.textContent = state;
    lamp.className = 'lamp al-state' + (state === 'running' ? ' lit' : state === 'idle' ? '' : ' warn');
    const idle = state === 'idle';
    $('.al-start').disabled = !idle;
    $('.al-dry').disabled = !idle;
    $('.al-pause').disabled = idle || state === 'stopping';
    $('.al-pause').textContent = state === 'paused' ? 'Resume' : 'Pause';
    $('.al-stop').disabled = idle;
    container.querySelector('.al-levels').classList.toggle('is-disabled', !idle);
    for (const input of container.querySelectorAll('.al-f')) input.disabled = !idle;
  }

  // ------------------------------------------------------------- levels
  function signalOptions(selected) {
    const sigs = catalog?.signals ?? [];
    const known = sigs.some((s) => s.id === selected);
    return `<option value="">signal…</option>` +
      (selected && !known ? `<option value="${esc(selected)}" selected>${esc(selected)} (not open)</option>` : '') +
      sigs.map((s) => `<option value="${esc(s.id)}" ${s.id === selected ? 'selected' : ''}>${esc(s.label)} [${esc(s.unit)}]</option>`).join('');
  }

  function valueEditor(row) {
    const v = row.value ?? {};
    const goal = row.goal;
    if (goal === 'max' || goal === 'min') {
      return `<input class="al-num" type="number" step="any" placeholder="good enough (optional)"
                value="${v.kind === 'number' ? esc(v.value) : ''}">`;
    }
    const coord = coordOf(row.signal);
    const kinds = [['number', 'number'], ...(coord ? [['marker', 'marker']] : []),
                   ...(goal === 'goto' ? [['hold', 'hold current']] : [])];
    let html = `<select class="al-kind">${kinds.map(([k, label]) =>
      `<option value="${k}" ${v.kind === k ? 'selected' : ''}>${label}</option>`).join('')}</select>`;
    if ((v.kind ?? 'number') === 'number') {
      html += ` <input class="al-num" type="number" step="any" value="${v.value ?? ''}">`;
    } else if (v.kind === 'marker') {
      const markers = catalog?.markers?.[deviceOf(row.signal)] ?? [];
      html += ` <select class="al-marker"><option value="">marker…</option>${markers.map((m) =>
        `<option value="${esc(m.id)}" ${m.id === v.marker || m.label === v.marker ? 'selected' : ''}>${esc(m.label)}${m.visible === false ? ' (hidden)' : ''}.${coord} = ${fmt(m[coord], 6)}</option>`).join('')}</select>`;
    }
    return html;
  }

  function renderLevels() {
    const box = container.querySelector('.al-levels');
    box.innerHTML = '';
    setup.levels.forEach((level, li) => {
      const fs = document.createElement('fieldset');
      fs.className = 'al-level';
      const goals = catalog?.goals ?? {};
      fs.innerHTML = `
        <legend>alignment ${li + 1}</legend>
        <table class="al-rows"><thead><tr><th title="untick to keep a row but leave it out of the alignment">on</th><th>signal</th><th>goal</th><th>value</th><th>tolerance</th><th></th></tr></thead><tbody></tbody></table>
        <button class="al-add-row">+ signal</button>
        <div class="al-controls-title">controls</div>
        <div class="al-controls"></div>`;
      const tbody = fs.querySelector('tbody');
      level.rows.forEach((row, ri) => {
        const tr = document.createElement('tr');
        const enabled = row.enabled !== false;
        tr.classList.toggle('al-off', !enabled);
        tr.innerHTML = `
          <td><input class="al-on" type="checkbox" ${enabled ? 'checked' : ''}
                title="${enabled ? 'untick: keep this row but leave it out of the alignment' : 'tick: take this row into the alignment again'}"></td>
          <td><select class="al-signal">${signalOptions(row.signal)}</select></td>
          <td><select class="al-goal">${Object.entries(goals).map(([k, label]) =>
            `<option value="${k}" ${row.goal === k ? 'selected' : ''}>${esc(label)}</option>`).join('')}</select></td>
          <td class="al-value">${valueEditor(row)}</td>
          <td>${row.goal === 'goto' ? `<input class="al-tol" type="number" step="any" min="0" value="${row.tolerance ?? ''}">` : ''}</td>
          <td><button class="al-del" title="remove this row">✕</button></td>`;
        const changed = () => { pushSetup(); renderLevels(); };
        tr.querySelector('.al-signal').onchange = (e) => {
          row.signal = e.target.value;
          if (row.value?.kind === 'marker' && !coordOf(row.signal)) row.value = { kind: 'number' };
          changed();
        };
        tr.querySelector('.al-goal').onchange = (e) => {
          row.goal = e.target.value;
          if (row.goal !== 'goto' && row.value?.kind === 'hold') row.value = { kind: 'number' };
          if (row.goal === 'goto' && row.tolerance === undefined) row.tolerance = 1;
          changed();
        };
        tr.querySelector('.al-del').onclick = () => { level.rows.splice(ri, 1); changed(); };
        tr.querySelector('.al-on').onchange = (e) => { row.enabled = e.target.checked; changed(); };
        const kind = tr.querySelector('.al-kind');
        if (kind) kind.onchange = () => { row.value = { kind: kind.value }; changed(); };
        const marker = tr.querySelector('.al-marker');
        if (marker) marker.onchange = () => {
          row.value = { kind: 'marker', device_id: deviceOf(row.signal), marker: marker.value, coord: coordOf(row.signal) };
          pushSetup();
        };
        const commitNum = (input, apply) => {
          const go = () => { const v = input.value === '' ? null : parseFloat(input.value); apply(v); pushSetup(); };
          input.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); });
          input.addEventListener('change', go);
        };
        const num = tr.querySelector('.al-num');
        if (num) commitNum(num, (v) => {
          const optimize = row.goal === 'max' || row.goal === 'min';
          row.value = v !== null && isFinite(v) ? { kind: 'number', value: v } : optimize ? {} : { kind: 'number' };
        });
        const tol = tr.querySelector('.al-tol');
        if (tol) commitNum(tol, (v) => { row.tolerance = v; });
        tbody.appendChild(tr);
      });
      fs.querySelector('.al-add-row').onclick = () => {
        level.rows.push({ enabled: true, signal: '', goal: 'goto', value: { kind: 'number' }, tolerance: 1 });
        pushSetup(); renderLevels();
      };

      const controls = fs.querySelector('.al-controls');
      const available = catalog?.controls ?? [];
      if (!available.length) controls.innerHTML = '<span class="unit">no motion-axes box open</span>';
      const chosen = Object.fromEntries((level.controls ?? []).map((c) => [c.axis, c]));
      for (const c of available) {
        const pick = chosen[c.axis];
        const div = document.createElement('div');
        div.className = 'al-control';
        div.innerHTML = `
          <label><input type="checkbox" ${pick ? 'checked' : ''}> ${esc(c.axis)}</label>
          <span class="unit">${fmt(c.position)} ${esc(c.units)}${c.frozen ? ' · frozen' : ''}${c.owner ? ` · in use by ${esc(c.owner)}` : ''}</span>
          <label class="field" title="first probe step (blank: the axis max step, ${fmt(c.max_step)})">step
            <input type="number" step="any" min="0" placeholder="${fmt(c.max_step)}" value="${pick?.step ?? ''}"></label>`;
        const [check, step] = div.querySelectorAll('input');
        check.onchange = () => {
          level.controls = (level.controls ?? []).filter((x) => x.axis !== c.axis);
          if (check.checked) level.controls.push({ axis: c.axis, step: parseFloat(step.value) || null });
          pushSetup();
        };
        const commitStep = () => {
          const entry = (level.controls ?? []).find((x) => x.axis === c.axis);
          if (entry) { entry.step = parseFloat(step.value) || null; pushSetup(); }
        };
        step.addEventListener('keydown', (e) => { if (e.key === 'Enter') commitStep(); });
        step.addEventListener('change', commitStep);
        controls.appendChild(div);
      }
      box.appendChild(fs);
    });
  }

  // ------------------------------------------------------------- report
  const labelOf = (id) => (catalog?.signals ?? []).find((s) => s.id === id)?.label ?? id;

  function showReport(title) {
    const el = container.querySelector('.al-report');
    if (!lastRows?.length) { el.innerHTML = ''; return; }
    const goals = catalog?.goals ?? {};
    el.innerHTML = `<div class="al-controls-title">${esc(title)}</div>
      <table class="al-rows al-readout"><thead><tr><th>signal</th><th>goal</th><th>target</th><th>value</th><th>miss</th><th></th></tr></thead>
      <tbody>${lastRows.map((r) => `<tr>
        <td>${esc(labelOf(r.signal))}</td><td>${esc(goals[r.goal] ?? r.goal)}</td>
        <td>${fmt(r.target, 6)}</td>
        <td>${r.value === null ? `<span class="unit">${esc(r.reason)}</span>` : fmt(r.value, 6)}</td>
        <td>${r.miss === null ? '' : (r.miss >= 0 ? '+' : '') + fmt(r.miss, 3)}</td>
        <td>${r.ok === null ? '' : r.ok ? '<span class="lamp lit">ok</span>' : '<span class="lamp warn">off</span>'}</td>
      </tr>`).join('')}</tbody></table>`;
  }

  // -------------------------------------------------------------- chart
  // One series: per measurement, the worst go-to miss in units of its tolerance
  // (log scale; at or below the dashed line = within tolerance). A level that
  // optimizes instead plots the optimized signal (linear).
  function chartValue(ev) {
    const goto = ev.rows.filter((r) => r.goal === 'goto');
    if (goto.length) {
      const tols = Object.fromEntries((setup.levels[0]?.rows ?? []).map((r) => [r.signal, r.tolerance]));
      if (goto.some((r) => r.miss === null)) return null;
      return Math.max(...goto.map((r) => Math.abs(r.miss) / (tols[r.signal] || 1)));
    }
    const opt = ev.rows.find((r) => r.goal === 'max' || r.goal === 'min');
    return opt?.value ?? null;
  }

  function renderChart() {
    const el = container.querySelector('.al-chart');
    const pts = history.map((ev) => ({ i: ev.index, v: chartValue(ev) })).filter((p) => p.v !== null && isFinite(p.v));
    if (pts.length < 1) { el.innerHTML = ''; return; }
    const isLog = history[0].rows.some((r) => r.goal === 'goto');
    const W = 460, H = 150, L = 44, R = 10, T = 22, B = 24;
    const xs = pts.map((p) => p.i);
    const x0 = Math.min(...xs), x1 = Math.max(x0 + 1, ...xs);
    const tx = (i) => L + (i - x0) / (x1 - x0) * (W - L - R);
    let ty, ticks;
    if (isLog) {
      const vals = pts.map((p) => Math.max(p.v, 1e-3));
      const lo = Math.floor(Math.log10(Math.min(0.5, ...vals))), hi = Math.ceil(Math.log10(Math.max(2, ...vals)));
      ty = (v) => T + (hi - Math.log10(Math.max(v, 1e-3))) / (hi - lo) * (H - T - B);
      ticks = []; for (let e = lo; e <= hi; e++) ticks.push(10 ** e);
    } else {
      const lo = Math.min(...pts.map((p) => p.v)), hi = Math.max(...pts.map((p) => p.v));
      const pad = (hi - lo) * 0.1 || Math.abs(hi) * 0.1 || 1;
      ty = (v) => T + (hi + pad - v) / (hi - lo + 2 * pad) * (H - T - B);
      ticks = [lo, (lo + hi) / 2, hi];
    }
    const path = pts.map((p, k) => `${k ? 'L' : 'M'}${tx(p.i).toFixed(1)},${ty(p.v).toFixed(1)}`).join('');
    el.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" class="al-svg" role="img" aria-label="alignment progress per measurement">
        <text x="${L}" y="13" class="al-ct">${isLog ? 'worst miss / tolerance per measurement' : 'optimized signal per measurement'}</text>
        ${ticks.map((t) => `<line x1="${L}" x2="${W - R}" y1="${ty(t)}" y2="${ty(t)}" class="al-grid"/>
          <text x="${L - 6}" y="${ty(t) + 3}" class="al-tick" text-anchor="end">${fmt(t, 3)}</text>`).join('')}
        ${isLog ? `<line x1="${L}" x2="${W - R}" y1="${ty(1)}" y2="${ty(1)}" class="al-tolline"/>` : ''}
        <text x="${W - R}" y="${H - 6}" class="al-tick" text-anchor="end">measurement</text>
        <path d="${path}" class="al-line"/>
        ${pts.map((p) => `<circle cx="${tx(p.i)}" cy="${ty(p.v)}" r="4" class="al-dot"/>`).join('')}
        <g class="al-hover" visibility="hidden"><line class="al-cross" y1="${T}" y2="${H - B}"/>
          <rect class="al-tip-bg" rx="4" height="18"/><text class="al-tip"></text></g>
        <rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent" class="al-hit"/>
      </svg>`;
    const svg = el.querySelector('svg'), hover = el.querySelector('.al-hover');
    el.querySelector('.al-hit').onmousemove = (e) => {
      const box = svg.getBoundingClientRect();
      const mx = (e.clientX - box.left) / box.width * W;
      const p = pts.reduce((a, b) => Math.abs(tx(b.i) - mx) < Math.abs(tx(a.i) - mx) ? b : a);
      const ev = history.find((h) => h.index === p.i);
      const txt = `#${p.i}  ${fmt(p.v, 3)}  [${Object.values(ev?.positions ?? {}).map((v) => fmt(v, 5)).join(' ')}]`;
      const X = tx(p.i);
      hover.setAttribute('visibility', 'visible');
      hover.querySelector('.al-cross').setAttribute('x1', X); hover.querySelector('.al-cross').setAttribute('x2', X);
      const t = hover.querySelector('.al-tip'); t.textContent = txt;
      const w = txt.length * 6.2 + 10, left = Math.min(Math.max(X - w / 2, 0), W - w);
      t.setAttribute('x', left + 5); t.setAttribute('y', T + 13);
      const bg = hover.querySelector('.al-tip-bg'); bg.setAttribute('x', left); bg.setAttribute('y', T);
      bg.setAttribute('width', w);
    };
    el.querySelector('.al-hit').onmouseleave = () => hover.setAttribute('visibility', 'hidden');
  }

  function log(line) {
    logLines.push(line);
    if (logLines.length > 200) logLines.shift();
    const pre = container.querySelector('.al-log');
    pre.textContent = logLines.join('\n');
    pre.scrollTop = pre.scrollHeight;
  }

  function logEvaluation(ev) {
    const pos = Object.values(ev.positions).map((v) => fmt(v, 6)).join(' ');
    const misses = ev.rows.map((r) => r.miss !== null ? (r.miss >= 0 ? '+' : '') + fmt(r.miss, 3)
      : r.value !== null ? fmt(r.value, 4) : 'invalid').join(' ');
    log(`#${String(ev.index).padStart(3)} [${pos}]  ${misses}`);
  }

  // ------------------------------------------------------------- stream
  function showAll() {
    showFields(); showState(); renderLevels();
    showReport(state === 'idle' ? 'last result' : 'latest measurement');
    renderChart();
  }
  history.forEach(logEvaluation);
  showAll();

  const stream = connectDeviceStream({
    deviceId: device.device_id,
    status,
    onEvent(event) {
      if (event.type === 'status') { state = event.state; showState(); if (state === 'running' && !history.length) logLines.length = 0; }
      else if (event.type === 'evaluation') {
        if (event.index === 1) { history = []; logLines.length = 0; }
        history.push(event); logEvaluation(event);
        lastRows = event.rows; showReport('latest measurement'); renderChart();
      } else if (event.type === 'log') log(event.line);
      else if (event.type === 'result') {
        lastRows = event.rows; showReport('last result');
        const s = event.summary ?? {};
        log(event.error ? `ended: ${event.error}`
          : `${s.target_reached ? 'TARGET REACHED' : 'target not reached'}${s.aborted ? ' (stopped)' : ''}: ${s.evaluations} measurements, ${fmt(s.seconds, 3)} s`);
      } else if (event.type === 'setup') {
        // our own edit echoed back: re-rendering would steal the focus mid-typing
        if (JSON.stringify(event.setup) !== JSON.stringify(setup)) {
          setup = structuredClone(event.setup); showFields(); renderLevels();
        }
      }
      else if (event.type === 'catalog') { catalog = event.catalog; renderLevels(); }
      else if (event.type === 'dry_run') { lastRows = event.rows; showReport('dry run'); }
      else if (event.type === 'error') status.textContent = `error: ${event.message}`;
    },
    onReattach(describe) {
      setup = structuredClone(describe.setup); catalog = describe.catalog; state = describe.state;
      history = describe.history ?? []; lastRows = describe.result?.rows ?? lastRows;
      logLines.length = 0; history.forEach(logEvaluation); showAll();
    },
  });

  return function cleanup() {
    stream.close();
  };
}
