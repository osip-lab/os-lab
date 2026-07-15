// Rigol function generator box: one section per channel with output on/off,
// waveform selector, frequency (with unit choice), amplitude (Vpp) and
// offset (V). Control-only device: the WebSocket carries JSON events only.
// Inputs commit on Enter or focus loss (lab convention) and are overwritten
// by broadcast 'channel' events with the values the instrument accepted, so
// every viewer stays in sync (including changes made from the front panel
// after a 'refresh').
// Returns a cleanup function that closes the socket.

import { connectDeviceStream } from './stream.js';

const FREQ_UNITS = [
  { label: 'mHz', factor: 1e-3 },
  { label: 'Hz', factor: 1 },
  { label: 'kHz', factor: 1e3 },
  { label: 'MHz', factor: 1e6 },
];

const WAVEFORM_LABELS = {
  sine: 'Sine', square: 'Square', ramp: 'Ramp',
  pulse: 'Pulse', noise: 'Noise', dc: 'DC',
};

export function createRigolDGBox(device, container, sendCommand) {
  container.innerHTML = `
    <div class="rigol-channels"></div>
    <div class="toolbar">
      <button class="rigol-refresh" title="re-read the instrument (e.g. after front-panel changes)">refresh</button>
      <button class="rigol-local" title="give the front panel back to whoever stands at the instrument">release front panel</button>
      <span class="rigol-status status-line"></span>
    </div>`;

  const channelsDiv = container.querySelector('.rigol-channels');
  const status = container.querySelector('.rigol-status');
  const fail = (error) => { status.textContent = error.message; };
  const send = (name, args) =>
    sendCommand(device.device_id, name, args).then(() => { status.textContent = ''; }).catch(fail);

  container.querySelector('.rigol-refresh').onclick = () => send('refresh');
  container.querySelector('.rigol-local').onclick = () => send('local');

  // --------------------------------------------------- one channel section
  const sections = {}; // channel number -> { show(state) }

  function numberRow(labelText, unitContent) {
    const row = document.createElement('label');
    row.className = 'rigol-row';
    const span = document.createElement('span');
    span.textContent = labelText;
    const input = document.createElement('input');
    input.type = 'number';
    input.step = 'any';
    row.appendChild(span);
    row.appendChild(input);
    if (unitContent) row.appendChild(unitContent);
    return { row, input };
  }

  function makeSection(channel) {
    const box = document.createElement('fieldset');
    box.className = 'rigol-channel';
    const legend = document.createElement('legend');
    legend.className = 'rigol-legend';
    legend.textContent = `CH${channel}`;
    box.appendChild(legend);

    // output on/off — this switches a live output, keep it prominent
    const onRow = document.createElement('label');
    onRow.className = 'rigol-output';
    const onCheck = document.createElement('input');
    onCheck.type = 'checkbox';
    const onText = document.createElement('span');
    onText.className = 'output-state';
    onRow.appendChild(onCheck);
    onRow.appendChild(onText);
    box.appendChild(onRow);
    onCheck.onchange = () =>
      send('set_channel', { channel, name: 'output', value: onCheck.checked });

    // waveform
    const waveRow = document.createElement('label');
    waveRow.className = 'rigol-row';
    waveRow.appendChild(Object.assign(document.createElement('span'), { textContent: 'waveform' }));
    const waveSelect = document.createElement('select');
    for (const name of device.waveforms ?? Object.keys(WAVEFORM_LABELS)) {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = WAVEFORM_LABELS[name] ?? name;
      waveSelect.appendChild(option);
    }
    waveRow.appendChild(waveSelect);
    box.appendChild(waveRow);
    waveSelect.onchange = () =>
      send('set_channel', { channel, name: 'waveform', value: waveSelect.value });

    // frequency with unit choice; committed in Hz
    const unitSelect = document.createElement('select');
    for (const unit of FREQ_UNITS) {
      const option = document.createElement('option');
      option.value = unit.label;
      option.textContent = unit.label;
      unitSelect.appendChild(option);
    }
    unitSelect.value = 'Hz';
    const freq = numberRow('frequency', unitSelect);
    box.appendChild(freq.row);

    const ampl = numberRow('amplitude');
    ampl.row.insertBefore(Object.assign(document.createElement('span'),
      { textContent: 'Vpp', className: 'unit' }), null);
    box.appendChild(ampl.row);

    const offset = numberRow('offset');
    offset.row.insertBefore(Object.assign(document.createElement('span'),
      { textContent: 'V', className: 'unit' }), null);
    box.appendChild(offset.row);

    const factor = () => FREQ_UNITS.find((u) => u.label === unitSelect.value).factor;

    // commit on Enter or focus loss, only when the value actually changed
    function commitOn(input, makeCommand) {
      const commit = () => {
        const value = parseFloat(input.value);
        if (!isFinite(value) || input.value === input.dataset.committed) return;
        input.dataset.committed = input.value;
        send('set_channel', makeCommand(value));
      };
      input.addEventListener('keydown', (e) => { if (e.key === 'Enter') commit(); });
      input.addEventListener('blur', commit);
    }
    commitOn(freq.input, (value) => ({ channel, name: 'frequency', value: value * factor() }));
    commitOn(ampl.input, (value) => ({ channel, name: 'amplitude', value }));
    commitOn(offset.input, (value) => ({ channel, name: 'offset', value }));

    // re-express the frequency when the unit dropdown changes (no command)
    unitSelect.onchange = () => {
      const hertz = parseFloat(freq.input.dataset.hertz ?? 'NaN');
      if (isFinite(hertz)) showFrequency(hertz, unitSelect.value);
    };

    function showFrequency(hertz, unitLabel = null) {
      if (unitLabel === null) {
        // pick the largest unit that keeps the number >= 1
        unitLabel = 'mHz';
        for (const unit of FREQ_UNITS) {
          if (hertz / unit.factor >= 1) unitLabel = unit.label;
        }
        unitSelect.value = unitLabel;
      }
      const unit = FREQ_UNITS.find((u) => u.label === unitLabel);
      freq.input.value = parseFloat((hertz / unit.factor).toPrecision(10));
      freq.input.dataset.committed = freq.input.value;
      freq.input.dataset.hertz = hertz;
    }

    function show(state) {
      onCheck.checked = state.on;
      onText.textContent = state.on ? 'output ON' : 'output off';
      onText.classList.toggle('lit', state.on);
      waveSelect.value = state.waveform;
      showFrequency(state.frequency_hz);
      ampl.input.value = parseFloat(state.amplitude_vpp.toPrecision(6));
      ampl.input.dataset.committed = ampl.input.value;
      offset.input.value = parseFloat(state.offset_v.toPrecision(6));
      offset.input.dataset.committed = offset.input.value;
    }

    channelsDiv.appendChild(box);
    return { show };
  }

  (device.channels ?? []).forEach((state, index) => {
    const channel = index + 1;
    sections[channel] = makeSection(channel);
    sections[channel].show(state);
  });

  // ----------------------------------------------------------- the stream
  const stream = connectDeviceStream({
    deviceId: device.device_id,
    status,
    onEvent(event) {
      if (event.type === 'channel' && sections[event.channel]) {
        sections[event.channel].show(event.state);
      } else if (event.type === 'error') {
        status.textContent = `error: ${event.message}`;
      }
    },
    onReattach(describe) {
      (describe.channels ?? []).forEach((state, index) => {
        if (sections[index + 1]) sections[index + 1].show(state);
      });
    },
  });

  return function cleanup() {
    stream.close();
  };
}
