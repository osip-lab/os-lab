// Canvas logic: the grid of device boxes, the "+ add device" flow, and the
// lifecycle of each box. Device-specific UI lives in boxes/*.js — this file
// only maps device types to their box renderers.

import { createCameraBox } from './boxes/camera.js';
import { createRigolDGBox } from './boxes/rigol_dg.js';

const BOX_RENDERERS = {
  dummy_camera: createCameraBox,
  basler_camera: createCameraBox,
  rigol_dg: createRigolDGBox,
};

const grid = GridStack.init({
  cellHeight: 90,
  margin: 6,
  float: true,
  handle: '.box-header',
});

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
        <button class="box-close" title="close device and remove box">✕</button>
      </div>
      <div class="box-body"></div>
    </div>`;
  element.querySelector('.box-title').textContent = device.label;

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
    status.textContent = open.length
      ? `re-attached to ${open.length} running device(s)` : '';
  } catch (error) {
    status.textContent = `server unreachable: ${error.message}`;
  }
}

reattachOpenDevices();
