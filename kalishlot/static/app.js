// Canvas logic: the grid of device boxes, the "+ add device" flow, and the
// lifecycle of each box. Device-specific UI lives in boxes/*.js — this file
// only maps device types to their box renderers.

import { createCameraBox } from './boxes/camera.js';
import { createPicoScopeBox } from './boxes/picoscope.js';
import { createRigolDGBox } from './boxes/rigol_dg.js';
import { createSyncedPipelineBox } from './boxes/synced_pipeline.js';
import { initLogger, openLogger } from './boxes/logger.js';
import { initIdle } from './boxes/idle.js';

const BOX_RENDERERS = {
  dummy_camera: createCameraBox,
  basler_camera: createCameraBox,
  ximea_camera: createCameraBox,
  rigol_dg: createRigolDGBox,
  picoscope: createPicoScopeBox,
  synced_pipeline: createSyncedPipelineBox,
};

// ?box=<device_id> turns this page into a satellite window showing just that
// one box, so an instrument can sit on another screen - see openSatellite().
const SATELLITE_ID = new URLSearchParams(location.search).get('box');
document.body.classList.toggle('satellite', SATELLITE_ID !== null);

const grid = SATELLITE_ID !== null ? null : GridStack.init({
  cellHeight: 90,
  margin: 8,
  float: true,
  handle: '.box-header',
});

initLogger();

// ------------------------------------------------------------ theme toggle
// Ultra-dark is applied in index.html before first paint; this only flips it.
const themeToggle = document.getElementById('theme-toggle');
function syncThemeToggle() {
  themeToggle.classList.toggle('lit', document.documentElement.dataset.theme === 'ultra-dark');
}
themeToggle.onclick = () => {
  const root = document.documentElement;
  if (root.dataset.theme === 'ultra-dark') delete root.dataset.theme;
  else root.dataset.theme = 'ultra-dark';
  try { localStorage.setItem('kalishlot-theme', root.dataset.theme ?? ''); } catch { /* not persisted */ }
  syncThemeToggle();
};
syncThemeToggle();

const openBoxes = new Map(); // device_id -> { element, cleanup }

// ------------------------------------------------------------- API helpers
async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail; } catch { /* keep statusText */ }
    throw new Error(detail);
  }
  return response.json();
}

export function sendCommand(deviceId, name, args = {}) {
  return api(`/api/devices/${encodeURIComponent(deviceId)}/command`, {
    method: 'POST',
    body: JSON.stringify({ name, args }),
  });
}

// ------------------------------------------------------------------- modal
const backdrop = document.getElementById('modal-backdrop');
const modalTitle = document.getElementById('modal-title');
const modalChoices = document.getElementById('modal-choices');
document.getElementById('modal-cancel').onclick = () => closeModal();
backdrop.onclick = (event) => { if (event.target === backdrop) closeModal(); };

let modalResolve = null;
function closeModal(value = null) {
  backdrop.hidden = true;
  if (modalResolve) { modalResolve(value); modalResolve = null; }
}

// Show a list of choices; resolves with the chosen item's value or null.
function askChoice(title, choices) {
  modalTitle.textContent = title;
  modalChoices.innerHTML = '';
  for (const choice of choices) {
    const button = document.createElement('button');
    button.className = 'choice';
    button.textContent = choice.label;
    button.onclick = () => closeModal(choice.value);
    modalChoices.appendChild(button);
  }
  if (choices.length === 0) {
    const note = document.createElement('p');
    note.textContent = 'nothing available';
    modalChoices.appendChild(note);
  }
  backdrop.hidden = false;
  return new Promise((resolve) => { modalResolve = resolve; });
}

// -------------------------------------------------------------- + add flow
document.getElementById('add-device').onclick = async () => {
  try {
    const types = await api('/api/device-types');
    const type = await askChoice('Choose device type',
      types.map((t) => ({ label: t.display_name, value: t.type })));
    if (!type) return;

    const available = await api(`/api/device-types/${type}/available`);
    const address = await askChoice('Choose device',
      available.map((d) => ({ label: d.label, value: d.address })));
    if (!address) return;

    const device = await api('/api/devices', {
      method: 'POST',
      body: JSON.stringify({ type, address }),
    });
    if (openBoxes.has(device.device_id)) {
      alert('this device already has a box on the canvas');
      return;
    }
    addBox(device);
  } catch (error) {
    alert(`could not add device:\n${error.message}`);
  }
};

// ---------------------------------------------------------------- box life
function addBox(device) {
  const element = document.createElement('div');
  element.className = 'grid-stack-item';
  element.innerHTML = `
    <div class="grid-stack-item-content">
      <div class="box-header">
        <span class="box-title"></span>
        <button class="box-log" title="open log">log</button>
        <button class="box-window" title="open in its own window (to move it to another screen)">⧉</button>
        <button class="box-close" title="close device and remove box">✕</button>
      </div>
      <div class="box-body"></div>
    </div>`;
  element.querySelector('.box-title').textContent = device.label;
  element.querySelector('.box-log').onclick = () => openLogger();
  element.querySelector('.box-window').onclick = () => openSatellite(device, element);

  document.querySelector('.grid-stack').appendChild(element);
  grid.makeWidget(element, { w: 5, h: 6 });

  const body = element.querySelector('.box-body');
  const renderer = BOX_RENDERERS[device.type];
  const cleanup = renderer
    ? renderer(device, body, sendCommand)
    : (() => { body.textContent = `no renderer for device type ${device.type}`; return () => {}; })();

  openBoxes.set(device.device_id, { element, cleanup });

  element.querySelector('.box-close').onclick = async () => {
    removeBox(device.device_id);
    try {
      await api(`/api/devices/${encodeURIComponent(device.device_id)}`, { method: 'DELETE' });
    } catch (error) {
      console.warn('closing device failed:', error);
    }
  };
}

// ------------------------------------------------------- satellite windows
// A box cannot leave the browser window it is drawn in, so "pop out" opens a
// second window of this same page that draws only that box. Nothing is handed
// over: the device belongs to the server, and the satellite attaches to it as
// one more viewer, exactly as a second browser tab would. The box in the grid
// stays live too. The window's name is per device, so pressing the button
// again focuses the open one instead of making another.
function openSatellite(device, element) {
  const { width, height } = element.getBoundingClientRect();
  const name = `kalishlot-${device.device_id.replace(/\W/g, '_')}`;
  const url = `/?box=${encodeURIComponent(device.device_id)}`;
  const features = `popup=yes,width=${Math.round(width)},height=${Math.round(height)}`;
  const opened = window.open(url, name, features);
  if (!opened) {
    alert('the browser blocked the new window - allow pop-ups for this page');
    return;
  }
  opened.focus();
}

// The satellite page: one box, filling the window.
function showSatelliteNote(text) {
  const holder = document.querySelector('.grid-stack');
  holder.className = 'satellite-note';
  holder.textContent = text;
}

async function runSatellite(deviceId) {
  document.title = 'Kalishlot';
  const holder = document.querySelector('.grid-stack');
  const say = (text) => { holder.textContent = text; holder.className = 'satellite-note'; };
  let device;
  try {
    const open = await api('/api/devices');
    device = open.find((d) => d.device_id === deviceId);
    if (!device) {     // lent to a script: the box waits for it, as in the grid
      const loan = (await api('/api/loans')).find((l) => l.device_id === deviceId);
      if (loan) device = { ...loan.describe, device_id: deviceId };
    }
  } catch (error) {
    say(`server unreachable: ${error.message}`);
    return;
  }
  if (!device) {
    say(`${deviceId} is not open - open it in the dashboard first`);
    return;
  }
  document.title = `${device.label} - Kalishlot`;
  holder.className = 'satellite-box';
  holder.innerHTML = `
    <div class="grid-stack-item-content">
      <div class="box-header">
        <span class="box-title"></span>
        <button class="box-log" title="open log">log</button>
        <button class="box-close" title="close this window (the device stays open)">✕</button>
      </div>
      <div class="box-body"></div>
    </div>`;
  holder.querySelector('.box-title').textContent = device.label;
  holder.querySelector('.box-log').onclick = () => openLogger();
  holder.querySelector('.box-close').onclick = () => window.close();
  const body = holder.querySelector('.box-body');
  const renderer = BOX_RENDERERS[device.type];
  if (renderer) renderer(device, body, sendCommand);
  else body.textContent = `no renderer for device type ${device.type}`;
}

function removeBox(deviceId) {
  const box = openBoxes.get(deviceId);
  if (!box) return;
  openBoxes.delete(deviceId);
  try { box.cleanup(); } catch { /* box already dead */ }
  grid.removeWidget(box.element);
}

// -------------------------------------------- re-attach on load / status
async function reattachOpenDevices() {
  const status = document.getElementById('server-status');
  try {
    const open = await api('/api/devices');
    for (const device of open) addBox(device);
    // devices lent to a script get their box too, built from the state they
    // had when lent; it waits and re-attaches once they are returned
    const loans = await api('/api/loans');
    for (const loan of loans) addBox({ ...loan.describe, device_id: loan.device_id });
    status.textContent = open.length
      ? `re-attached to ${open.length} running device(s)` : '';
  } catch (error) {
    status.textContent = `server unreachable: ${error.message}`;
  }
}

if (SATELLITE_ID === null) reattachOpenDevices();
else runSatellite(SATELLITE_ID);

// the idle watchdog closes the devices server-side; drop their boxes too, so
// the canvas is not left full of boxes pointing at nothing
initIdle({
  onDisconnected(deviceIds) {
    if (SATELLITE_ID !== null) {
      if (deviceIds.includes(SATELLITE_ID)) {
        showSatelliteNote('idle timeout - the device was disconnected');
      }
      return;
    }
    for (const deviceId of deviceIds) removeBox(deviceId);
    document.getElementById('server-status').textContent =
      `idle timeout — disconnected ${deviceIds.length} device(s)`;
  },
});
