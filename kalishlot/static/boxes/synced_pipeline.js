// Synced video + scope pipeline box: the parameters of a mode-video run as
// widgets, instead of pico_scope/run_config_local.py, and the run itself.
//
// Not a device: the adapter (adapters/synced_pipeline.py) is virtual, and this
// box is greyed out unless the dashboard has a camera and a PicoScope (open or
// lent to this very run). Parameters come from the adapter's own list
// (describe().params), so a parameter added there appears here with no change
// to this file.
//
// A widget shows the config file's value until it is edited; an edited one is
// marked and has a "reset" that goes back to the file. A "from box" row ticked
// takes the camera/scope box's value; unticked, the capture works the value out
// itself, and the parameters that feed that logic appear under the row.
// Inputs commit on Enter or focus loss (lab convention).
// Returns a cleanup function.

import { connectDeviceStream } from './stream.js';

const POLL_MS = 1500;
// what the capture does by itself for a row that is not taken from the box
const AUTO_RULE = {
  exposure: 'derived from the frame rate: the whole period less a 1% gap',
  gain: 'trimmed by the light-level check, towards the target peak',
  frame_rate: 'the requested rate, lowered if the camera cannot reach it',
  roi: 'the mode is located first, then a strip around it is chosen',
  scope_range: 'auto-ranged from a short probe of the signal',
};
const LOG_LINES = 400;

export function createSyncedPipelineBox(device, container, sendCommand) {
  container.innerHTML = `
    <div class="sp-gate status-line"></div>
    <fieldset class="sp-body">
      <div class="sp-camera" hidden>
        <label class="sp-row"><span>camera</span><select class="sp-camera-select"></select></label>
      </div>
      <div class="sp-folder">
        <label class="sp-row"><span>save folder</span>
          <input type="text" class="sp-folder-input" spellcheck="false"
                 placeholder="full path, e.g. D:\\measurements\\2026-10-05">
          <button class="sp-browse" title="open the folder window on the lab PC">browse…</button>
        </label>
        <div class="sp-folder-note status-line"></div>
      </div>
      <div class="sp-adopt"></div>
      <div class="sp-main"></div>
      <details class="sp-advanced"><summary>advanced capture</summary><div></div></details>
      <details class="sp-sync"><summary>sync and plot</summary><div></div></details>
      <div class="sp-presets toolbar">
        <span>preset</span>
        <select class="sp-preset-select"></select>
        <button class="sp-preset-load">load</button>
        <button class="sp-preset-save" title="save the current parameters under a name">save as…</button>
        <button class="sp-preset-delete">delete</button>
      </div>
    </fieldset>
    <div class="toolbar sp-run">
      <button class="sp-start">run pipeline ▶</button>
      <button class="sp-stop" hidden>stop ■</button>
      <span class="sp-run-state status-line"></span>
    </div>
    <pre class="sp-log"></pre>
    <div class="sp-message status-line"></div>`;

  const $ = (selector) => container.querySelector(selector);
  const gate = $('.sp-gate');
  const body = $('.sp-body');
  const message = $('.sp-message');
  const fail = (error) => { message.textContent = error.message; };
  const send = (name, args) => sendCommand(device.device_id, name, args)
    .then((result) => { message.textContent = ''; return result; });

  let state = device;       // the adapter's latest describe()
  let boxes = {};           // device_id -> describe() of the other devices
  let checkedFolder = null; // the path the note under the folder box is about
  const widgets = {};       // param key -> { row, read(), write(value), reset }
  const adoptRows = {};     // adopt key -> { box, note }

  // ---------------------------------------------------------------- widgets
  const valueOf = (key) => (key in state.edited ? state.edited[key] : state.config[key]);

  function makeInput(param) {
    let input;
    if (param.kind === 'bool') {
      input = document.createElement('input');
      input.type = 'checkbox';
    } else if (param.kind === 'choice') {
      input = document.createElement('select');
      for (const choice of param.choices) {
        const option = document.createElement('option');
        option.value = String(choice);
        option.textContent = choice === '' ? '(none)' : String(choice);
        input.appendChild(option);
      }
    } else {
      input = document.createElement('input');
      input.type = ['float', 'int', 'optfloat'].includes(param.kind) ? 'number' : 'text';
      if (input.type === 'number') input.step = 'any';
      input.spellcheck = false;
    }
    return input;
  }

  function display(param, value) {
    if (param.kind === 'intlist') return Array.isArray(value) ? value.join(', ') : '';
    return value === null || value === undefined ? '' : value;
  }

  function buildParam(param, parent) {
    const row = document.createElement('label');
    row.className = 'sp-row';
    const name = document.createElement('span');
    name.textContent = param.label;
    const input = makeInput(param);
    const unit = document.createElement('span');
    unit.className = 'unit';
    unit.textContent = param.unit ?? '';
    const reset = document.createElement('button');
    reset.className = 'sp-reset';
    reset.textContent = '↺';
    reset.title = 'back to the config file\'s value';
    row.append(name, input, unit, reset);
    parent.appendChild(row);

    const commit = () => {
      const value = param.kind === 'bool' ? input.checked : input.value;
      if (String(value) === input.dataset.committed) return;
      send('set_param', { key: param.key, value })
        .catch((error) => { fail(error); show(); });
    };
    if (param.kind === 'bool' || param.kind === 'choice') input.onchange = commit;
    else {
      input.addEventListener('keydown', (event) => { if (event.key === 'Enter') commit(); });
      input.addEventListener('blur', commit);
    }
    reset.onclick = (event) => {
      event.preventDefault();
      send('reset_param', { key: param.key }).catch(fail);
    };

    const widget = {
      row, param,
      write(value, edited) {
        const shown = display(param, value);
        if (param.kind === 'bool') input.checked = Boolean(shown);
        else if (document.activeElement !== input) input.value = String(shown);
        input.dataset.committed = String(param.kind === 'bool' ? Boolean(shown) : shown);
        row.classList.toggle('is-edited', edited);
        reset.hidden = !edited;
      },
    };
    widgets[param.key] = widget;
    return widget;
  }

  function build() {
    const groups = { main: $('.sp-main'), advanced: $('.sp-advanced div'),
                     sync: $('.sp-sync div') };
    const folder = state.params.find((p) => p.kind === 'folder');
    const inputs = {};
    for (const param of state.params) {
      if (param.kind === 'folder') continue;
      const parent = groups[param.group] ?? groups.main;
      buildParam(param, parent);
      inputs[param.key] = widgets[param.key];
    }
    // the folder gets its own row with browse, above the rest
    const folderInput = $('.sp-folder-input');
    const commitFolder = () => {
      if (folderInput.value === folderInput.dataset.committed) return;
      send('set_param', { key: folder.key, value: folderInput.value })
        .catch((error) => { fail(error); show(); });
    };
    folderInput.addEventListener('keydown', (event) => { if (event.key === 'Enter') commitFolder(); });
    folderInput.addEventListener('blur', commitFolder);
    $('.sp-browse').onclick = async (event) => {
      event.preventDefault();
      const button = event.target;
      button.disabled = true;
      $('.sp-folder-note').textContent = 'choose the folder in the window on the lab PC…';
      try { await send('pick_folder', {}); } catch (error) { fail(error); }
      button.disabled = false;
      show();
    };
    widgets[folder.key] = {
      param: folder,
      write(value) {
        if (document.activeElement !== folderInput) folderInput.value = value ?? '';
        folderInput.dataset.committed = folderInput.value;
      },
    };

    // "from box" rows, each followed by the parameters that feed the capture's
    // own logic while it is unticked
    const adoptDiv = $('.sp-adopt');
    for (const row of state.adopt_rows) {
      const wrap = document.createElement('div');
      wrap.className = 'sp-adopt-row';
      const label = document.createElement('label');
      label.className = 'sp-row';
      const check = document.createElement('input');
      check.type = 'checkbox';
      check.onchange = () => send('set_adopt', { key: row.key, value: check.checked })
        .catch((error) => { fail(error); show(); });
      const name = document.createElement('span');
      name.textContent = `${row.label}: from the box`;
      const note = document.createElement('span');
      note.className = 'sp-box-value readout';
      label.append(check, name, note);
      const auto = document.createElement('div');
      auto.className = 'sp-auto';
      const rule = document.createElement('div');
      rule.className = 'status-line';
      rule.textContent = `automatic: ${AUTO_RULE[row.key] ?? ''}`;
      auto.appendChild(rule);
      wrap.append(label, auto);
      adoptDiv.appendChild(wrap);
      adoptRows[row.key] = { check, note, auto, wrap };
    }
    // move each auto_of parameter under its row
    for (const param of state.params) {
      const target = param.auto_of && adoptRows[param.auto_of];
      if (target) target.auto.appendChild(widgets[param.key].row);
    }
    $('.sp-camera-select').onchange = (event) =>
      send('choose_camera', { device_id: event.target.value }).catch(fail);

    $('.sp-preset-load').onclick = () => {
      const name = $('.sp-preset-select').value;
      if (name) send('preset_load', { name }).then(refreshSoon).catch(fail);
    };
    $('.sp-preset-save').onclick = () => {
      const name = prompt('Save the current parameters as preset:');
      if (name) send('preset_save', { name }).then(refreshSoon).catch(fail);
    };
    $('.sp-preset-delete').onclick = () => {
      const name = $('.sp-preset-select').value;
      if (name && confirm(`Delete preset "${name}"?`)) {
        send('preset_delete', { name }).then(refreshSoon).catch(fail);
      }
    };
    $('.sp-start').onclick = () => send('start', {}).catch(fail);
    $('.sp-stop').onclick = () => send('stop', {}).catch(fail);
  }

  // ------------------------------------------------------------ box values
  function cameraBox() {
    return boxes[state.camera_id] ?? null;
  }

  function settingOf(describe, name) {
    const found = (describe?.settings ?? []).find((s) => s.name === name);
    return found?.value ?? null;
  }

  function boxValueText(key) {
    const camera = cameraBox();
    const scope = Object.values(boxes).find((d) => d.type === 'picoscope');
    if (key === 'exposure') {
      const v = settingOf(camera, 'exposure');
      return v ? `${Math.round(v)} µs` : '';
    }
    if (key === 'gain') {
      const v = settingOf(camera, 'gain');
      return v === null ? '' : `${Number(v).toFixed(1)} dB`;
    }
    if (key === 'frame_rate') {
      const v = settingOf(camera, 'framerate');
      return v ? `${Number(v).toFixed(1)} Hz` : 'this camera box has no frame rate';
    }
    if (key === 'roi') {
      const r = camera?.roi;
      return r ? `${r.width}×${r.height} at (${r.x}, ${r.y})` : 'whole sensor';
    }
    if (key === 'scope_range') {
      const channel = valueOf('capture.SCOPE_CHANNEL');
      const c = scope?.channels?.[channel];
      return c?.enabled ? `±${c.range_v} V ${c.coupling} (channel ${channel})`
        : `channel ${channel} is off in the PicoScope box`;
    }
    return '';
  }

  // ------------------------------------------------------------------ show
  function show() {
    const camera = state.dependencies.cameras;
    const missing = [];
    if (!camera.length) missing.push('a camera');
    if (!state.dependencies.scopes.length) missing.push('a PicoScope');
    gate.textContent = missing.length
      ? `open ${missing.join(' and ')} on the dashboard to use this box` : '';
    body.disabled = !state.ready;
    body.classList.toggle('is-disabled', !state.ready);

    const select = $('.sp-camera-select');
    $('.sp-camera').hidden = camera.length < 2;
    if (select.dataset.key !== camera.map((c) => c.device_id).join('|')) {
      select.dataset.key = camera.map((c) => c.device_id).join('|');
      select.innerHTML = '';
      for (const c of camera) {
        const option = document.createElement('option');
        option.value = c.device_id;
        option.textContent = c.device_id + (c.state === 'lent' ? ' (in use)' : '');
        select.appendChild(option);
      }
    }
    if (state.camera_id) select.value = state.camera_id;

    for (const [key, widget] of Object.entries(widgets)) {
      widget.write(valueOf(key), key in state.edited);
    }
    const folderNote = $('.sp-folder-note');
    const folder = valueOf('capture.OUTPUT_ROOT');
    $('.sp-folder-input').classList.toggle('is-edited', 'capture.OUTPUT_ROOT' in state.edited);
    if (folder) {
      // only when the path changed: a command counts as user activity on the
      // server, and this runs on every poll
      if (folder !== checkedFolder) {
        checkedFolder = folder;
        send('check_folder', { path: folder }).then((r) => {
          folderNote.textContent = (r.ok ? '✓ ' : '✗ ') + r.message;
          folderNote.classList.toggle('sp-bad', !r.ok);
        }).catch(() => { checkedFolder = null; });
      }
    } else {
      checkedFolder = null;
      folderNote.textContent = 'choose where the capture is saved';
      folderNote.classList.add('sp-bad');
    }

    const frameRateInBox = settingOf(cameraBox(), 'framerate') !== null;
    for (const [key, row] of Object.entries(adoptRows)) {
      row.check.checked = state.adopt[key];
      row.note.textContent = boxValueText(key);
      const forced = key === 'frame_rate' && !frameRateInBox;
      row.check.disabled = forced;
      // unticked: the capture decides, from the parameters shown here
      row.auto.hidden = state.adopt[key] && !forced;
    }

    const presets = $('.sp-preset-select');
    if (presets.dataset.key !== state.presets.join('|')) {
      presets.dataset.key = state.presets.join('|');
      presets.innerHTML = '';
      for (const name of state.presets) {
        const option = document.createElement('option');
        option.textContent = name;
        presets.appendChild(option);
      }
    }
    showRun();
  }

  function showRun() {
    const run = state.run;
    $('.sp-start').disabled = run.running || !state.ready;
    $('.sp-stop').hidden = !run.running;
    let text = '';
    if (run.running) text = `running${run.step ? ` — ${run.step.replace('mode_video_', '')}` : ''}`;
    else if (run.stopped) text = 'stopped';
    else if (run.error) text = `failed: ${run.error}`;
    else if (run.ended) text = 'finished';
    if (!run.running && run.session && !run.error) text += ` — ${run.session}`;
    const stateLine = $('.sp-run-state');
    stateLine.textContent = text;
    stateLine.classList.toggle('sp-bad', Boolean(run.error));
  }

  // -------------------------------------------------------------- the log
  const logEl = $('.sp-log');
  function appendLog(lines) {
    const stick = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 8;
    logEl.textContent += lines.join('\n') + '\n';
    const all = logEl.textContent.split('\n');
    if (all.length > LOG_LINES) logEl.textContent = all.slice(-LOG_LINES).join('\n');
    if (stick) logEl.scrollTop = logEl.scrollHeight;
  }

  // ----------------------------------------------------- polling + events
  let timer = null;
  let closed = false;
  async function refresh() {
    try {
      const response = await fetch('/api/devices');
      if (!response.ok) throw new Error(response.statusText);
      const all = await response.json();
      const mine = all.find((d) => d.device_id === device.device_id);
      if (mine) {
        const wasRunning = state.run?.running;
        state = mine;
        if (!wasRunning && mine.run.running) { logEl.textContent = ''; }
      }
      boxes = Object.fromEntries(all.map((d) => [d.device_id, d]));
      show();
    } catch { /* the stream's own status line reports a lost server */ }
    if (!closed) timer = setTimeout(refresh, POLL_MS);
  }
  const refreshSoon = () => { clearTimeout(timer); refresh(); };

  build();
  show();
  appendLog(state.log ?? []);
  refresh();

  const stream = connectDeviceStream({
    deviceId: device.device_id,
    status: $('.sp-message'),
    onEvent(event) {
      if (event.type === 'log') appendLog([event.line]);
      else if (event.type === 'run') { state.run = event.run; showRun(); }
      else if (event.type === 'param_applied') { state.edited[event.key] = event.value; show(); }
      else if (event.type === 'param_reset') { delete state.edited[event.key]; show(); }
      else if (event.type === 'adopt_applied') { state.adopt[event.key] = event.value; show(); }
    },
    onReattach(describe) { state = describe; show(); },
  });

  return function cleanup() {
    closed = true;
    clearTimeout(timer);
    stream.close();
  };
}
