// Motorized axes box ("jog box"): one row per axis with position readout,
// jog -/+ by a step, go-to, freeze toggle and state lamps; an E-STOP for the
// whole box. Control-only device: the WebSocket carries JSON 'axis' events.
// Every move goes through the server's SafeAxis: a refused move (out of range,
// frozen, E-stop, unknown position) comes back as an error in the status line
// and nothing moves. Inputs commit on Enter (lab convention).
// Returns a cleanup function that closes the socket.

import { connectDeviceStream } from './stream.js';

const fmt = (v, digits = 4) => (v === null || v === undefined || !isFinite(v))
  ? '—' : parseFloat(Number(v).toPrecision(digits)).toString();

export function createMotionAxesBox(device, container, sendCommand) {
  container.innerHTML = `
    <div class="toolbar">
      <button class="motion-estop" title="stop every axis now and hold (moves refused until reset)">E-STOP</button>
      <button class="motion-reset" title="allow moves again after an E-stop">reset E-stop</button>
      <button class="motion-refresh" title="re-read all axes">refresh</button>
    </div>
    <div class="motion-axes"></div>
    <div class="status-line motion-status"></div>`;

  const list = container.querySelector('.motion-axes');
  const status = container.querySelector('.motion-status');
  const fail = (error) => { status.textContent = error.message; };
  const send = (name, args) =>
    sendCommand(device.device_id, name, args).then(() => { status.textContent = ''; }).catch(fail);

  container.querySelector('.motion-estop').onclick = () => send('estop', { who: 'jog box' });
  container.querySelector('.motion-reset').onclick = () => send('reset_estop', { who: 'jog box' });
  container.querySelector('.motion-refresh').onclick = () => send('refresh');

  const rows = {}; // axis name -> { show(state) }

  function makeRow(state) {
    const name = state.name;
    const row = document.createElement('fieldset');
    row.className = 'motion-axis';
    row.innerHTML = `
      <legend class="motion-name"></legend>
      <div class="motion-readout"><span class="motion-pos"></span> <span class="unit"></span>
        <span class="motion-lamps"></span></div>
      <div class="motion-controls">
        <button class="motion-minus" title="move down by the step">−</button>
        <input class="motion-step" type="number" step="any" title="jog step">
        <button class="motion-plus" title="move up by the step">+</button>
        <input class="motion-goto" type="number" step="any" placeholder="go to" title="absolute target, Enter to move">
        <label class="motion-freeze" title="freeze: refuse every move of this axis"><input type="checkbox"> freeze</label>
      </div>
      <div class="motion-declare" hidden>
        position unknown: <input type="number" step="any" placeholder="declare">
      </div>
      <div class="motion-range"></div>`;
    row.querySelector('.motion-name').textContent = name;
    const step = row.querySelector('.motion-step');
    step.value = state.limits.max_step;
    const jog = (sign) => {
      const delta = sign * parseFloat(step.value);
      if (isFinite(delta) && delta !== 0) send('move_by', { axis: name, delta, who: 'jog box' });
    };
    row.querySelector('.motion-minus').onclick = () => jog(-1);
    row.querySelector('.motion-plus').onclick = () => jog(+1);
    const goto = row.querySelector('.motion-goto');
    goto.addEventListener('keydown', (e) => {
      const target = parseFloat(goto.value);
      if (e.key === 'Enter' && isFinite(target)) send('move_to', { axis: name, target, who: 'jog box' });
    });
    const freeze = row.querySelector('.motion-freeze input');
    freeze.onchange = () => send('freeze', { axis: name, frozen: freeze.checked, who: 'jog box' });
    const declare = row.querySelector('.motion-declare input');
    declare.addEventListener('keydown', (e) => {
      const value = parseFloat(declare.value);
      if (e.key === 'Enter' && isFinite(value)) send('set_known_position', { axis: name, value, who: 'jog box' });
    });

    function show(s) {
      row.querySelector('.motion-pos').textContent = fmt(s.position);
      row.querySelector('.unit').textContent = s.units;
      const lamps = [];
      if (s.moving) lamps.push('<span class="lamp lit">moving</span>');
      if (s.frozen) lamps.push(`<span class="lamp warn">frozen${s.frozen_in_config ? ' (config)' : ''}</span>`);
      if (s.estop) lamps.push('<span class="lamp alert">E-stop</span>');
      if (!s.closed_loop) lamps.push('<span class="lamp">open loop</span>');
      row.querySelector('.motion-lamps').innerHTML = lamps.join(' ');
      freeze.checked = s.frozen;
      freeze.disabled = s.frozen_in_config;
      row.querySelector('.motion-declare').hidden = s.position_known;
      const L = s.limits;
      row.querySelector('.motion-range').textContent =
        `range ${fmt(L.min)}…${fmt(L.max)} ${s.units} · max step ${fmt(L.max_step)} · ` +
        `slew ${fmt(L.max_slew)} ${s.units}/s · settle ${fmt(L.vibration_settle_s)} s` +
        (s.commanded !== null && s.closed_loop ? ` · commanded ${fmt(s.commanded)}` : '');
    }
    list.appendChild(row);
    return { show };
  }

  for (const state of device.axes ?? []) {
    rows[state.name] = makeRow(state);
    rows[state.name].show(state);
  }

  const stream = connectDeviceStream({
    deviceId: device.device_id,
    status,
    onEvent(event) {
      if (event.type === 'axis' && rows[event.state.name]) {
        rows[event.state.name].show(event.state);
      } else if (event.type === 'error') {
        status.textContent = `error: ${event.message}`;
      }
    },
    onReattach(describe) {
      for (const state of describe.axes ?? []) rows[state.name]?.show(state);
    },
  });

  return function cleanup() {
    stream.close();
  };
}
