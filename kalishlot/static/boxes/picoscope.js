// PicoScope box: rolling chart-recorder view of the streamed channels.
// The server sends 'scope_data' events at ~20 Hz, each holding the visible
// window min/max-envelope-decimated to <= ~1000 points per channel; the
// whole chart is redrawn once per event with uPlot (canvas) — there is no
// per-sample work anywhere in the browser.
// Returns a cleanup function that closes the socket.

import { connectDeviceStream } from './stream.js';

const CHANNEL_ORDER = ['A', 'B', 'C', 'D'];
const CHANNEL_COLORS = { A: '#4a9eda', B: '#e05555', C: '#52c46a', D: '#d9a13c' };

function formatVolts(volts) {
  return volts < 1 ? `±${volts * 1000} mV` : `±${volts} V`;
}

function formatRate(hertz) {
  if (hertz >= 1e6) return `${hertz / 1e6} MS/s`;
  if (hertz >= 1e3) return `${hertz / 1e3} kS/s`;
  return `${hertz} S/s`;
}

export function createPicoScopeBox(device, container, sendCommand) {
  container.innerHTML = `
    <div class="scope-controls" style="display:flex; gap:6px; align-items:center; flex-wrap:wrap;">
      <button data-command="play">play</button>
      <button data-command="pause">pause</button>
      <label style="display:flex; gap:4px; align-items:center; font-size:12px;">
        window <select class="scope-window"></select></label>
      <label style="display:flex; gap:4px; align-items:center; font-size:12px;">
        rate <select class="scope-rate"></select></label>
      <span class="scope-status" style="font-size:12px; opacity:0.8;"></span>
    </div>
    <div class="scope-channels" style="display:flex; gap:10px; flex-wrap:wrap;"></div>
    <div class="scope-chart" style="flex:1; min-height:0; position:relative;"></div>`;

  const status = container.querySelector('.scope-status');
  const fail = (error) => { status.textContent = error.message; };
  const send = (name, args) => sendCommand(device.device_id, name, args)
    .then(() => { status.textContent = ''; })
    .catch(fail);

  // ------------------------------------------------------- play and pause
  const buttons = {
    play: container.querySelector('[data-command="play"]'),
    pause: container.querySelector('[data-command="pause"]'),
  };
  function setPlaying(playing) {
    buttons.play.disabled = playing;
    buttons.pause.disabled = !playing;
    status.textContent = playing ? '' : 'display frozen (still acquiring)';
  }
  setPlaying(device.playing ?? true);
  buttons.play.onclick = () => send('play').then(() => setPlaying(true));
  buttons.pause.onclick = () => send('pause').then(() => setPlaying(false));

  // ------------------------------------------- window and sample-rate selects
  const windowSelect = container.querySelector('.scope-window');
  for (const seconds of device.window_choices_s ?? [0.1, 1, 10, 60]) {
    const option = document.createElement('option');
    option.value = seconds;
    option.textContent = seconds < 1 ? `${seconds * 1000} ms` : `${seconds} s`;
    windowSelect.appendChild(option);
  }
  const rateSelect = container.querySelector('.scope-rate');
  for (const hertz of device.rate_choices_hz ?? [100, 1000, 10000, 100000]) {
    const option = document.createElement('option');
    option.value = hertz;
    option.textContent = formatRate(hertz);
    rateSelect.appendChild(option);
  }
  function selectClosest(select, value) {
    let best = null;
    for (const option of select.options) {
      if (best === null
          || Math.abs(option.value - value) < Math.abs(best.value - value)) {
        best = option;
      }
    }
    if (best !== null) select.value = best.value;
  }
  for (const setting of device.settings ?? []) {
    if (setting.name === 'window_s') selectClosest(windowSelect, setting.value);
    if (setting.name === 'sample_rate_hz') selectClosest(rateSelect, setting.value);
  }
  windowSelect.onchange = () => send('set_setting',
    { name: 'window_s', value: parseFloat(windowSelect.value) });
  rateSelect.onchange = () => send('set_setting',
    { name: 'sample_rate_hz', value: parseFloat(rateSelect.value) });

  // ------------------------------------------------------ channel controls
  let chart = null; // created below, after the channel controls
  const channelsDiv = container.querySelector('.scope-channels');
  const channelControls = {}; // name -> { enable, range, coupling }

  for (const name of CHANNEL_ORDER) {
    const state = (device.channels ?? {})[name]
      ?? { enabled: name === 'A', coupling: 'DC', range_v: 5 };
    const group = document.createElement('span');
    group.style.cssText = 'display:flex; gap:4px; align-items:center; font-size:12px;';

    const enable = document.createElement('input');
    enable.type = 'checkbox';
    const label = document.createElement('span');
    label.textContent = name;
    label.style.cssText =
      `color:${CHANNEL_COLORS[name]}; font-weight:bold; min-width:1em;`;

    const range = document.createElement('select');
    for (const volts of device.ranges_v ?? [5]) {
      const option = document.createElement('option');
      option.value = volts;
      option.textContent = formatVolts(volts);
      range.appendChild(option);
    }
    const coupling = document.createElement('select');
    for (const kind of ['DC', 'AC']) {
      const option = document.createElement('option');
      option.value = kind;
      option.textContent = kind;
      coupling.appendChild(option);
    }

    group.appendChild(enable);
    group.appendChild(label);
    group.appendChild(range);
    group.appendChild(coupling);
    channelsDiv.appendChild(group);

    enable.onchange = () =>
      send('set_channel', { channel: name, enabled: enable.checked });
    range.onchange = () =>
      send('set_channel', { channel: name, range_v: parseFloat(range.value) });
    coupling.onchange = () =>
      send('set_channel', { channel: name, coupling: coupling.value });

    channelControls[name] = { enable, range, coupling };
    showChannel(name, state);
  }

  function showChannel(name, state) {
    const controls = channelControls[name];
    controls.enable.checked = state.enabled;
    selectClosest(controls.range, state.range_v);
    controls.coupling.value = state.coupling;
    if (chart) chart.setSeries(CHANNEL_ORDER.indexOf(name) + 1,
      { show: state.enabled });
  }

  // ---------------------------------------------------------------- chart
  const chartDiv = container.querySelector('.scope-chart');
  const axisStyle = {
    stroke: '#b8b8c0',
    grid: { stroke: '#3a3a42' },
    ticks: { stroke: '#3a3a42' },
  };
  chart = new uPlot({
    width: 400,
    height: 200,
    scales: { x: { time: false } },
    series: [
      { label: 't (s)' },
      ...CHANNEL_ORDER.map((name) => ({
        label: name,
        stroke: CHANNEL_COLORS[name],
        width: 1,
        points: { show: false },
        show: (device.channels ?? {})[name]?.enabled ?? (name === 'A'),
        value: (u, v) => (v == null ? '-' : `${v.toFixed(4)} V`),
      })),
    ],
    axes: [axisStyle, { ...axisStyle, label: 'V' }],
    cursor: { drag: { x: false, y: false } },
  }, [[0], [null], [null], [null], [null]], chartDiv);

  // size the chart with the box (leave room for the legend row)
  const resizeObserver = new ResizeObserver(() => {
    const legend = chartDiv.querySelector('.u-legend');
    const height = Math.max(
      chartDiv.clientHeight - (legend ? legend.offsetHeight : 30) - 4, 60);
    chart.setSize({ width: Math.max(chartDiv.clientWidth, 120), height });
  });
  resizeObserver.observe(chartDiv);

  function showData(event) {
    const span = event.span_s;
    let longest = 0;
    for (const name of CHANNEL_ORDER) {
      longest = Math.max(longest, event.channels[name]?.length ?? 0);
    }
    if (!longest) return;
    const x = new Array(longest);
    for (let i = 0; i < longest; i++) {
      x[i] = longest > 1 ? -span + (span * i) / (longest - 1) : 0;
    }
    const data = [x];
    for (const name of CHANNEL_ORDER) {
      const values = event.channels[name];
      if (!values) data.push(new Array(longest).fill(null));
      else if (values.length === longest) data.push(values);
      else {
        // buffer still filling after a restart: align at the newest sample
        data.push(new Array(longest - values.length).fill(null).concat(values));
      }
    }
    chart.setData(data);
  }

  // ----------------------------------------------------------- the stream
  const stream = connectDeviceStream({
    deviceId: device.device_id,
    status,
    onEvent(event) {
      if (event.type === 'scope_data') showData(event);
      else if (event.type === 'status') setPlaying(event.playing);
      else if (event.type === 'channel') showChannel(event.channel, event.state);
      else if (event.type === 'setting_applied') {
        if (event.name === 'window_s') selectClosest(windowSelect, event.value);
        if (event.name === 'sample_rate_hz') selectClosest(rateSelect, event.value);
      } else if (event.type === 'error') {
        status.textContent = `error: ${event.message}`;
      }
    },
    onReattach(describe) {
      setPlaying(describe.playing ?? true);
      for (const [name, state] of Object.entries(describe.channels ?? {})) {
        showChannel(name, state);
      }
      for (const setting of describe.settings ?? []) {
        if (setting.name === 'window_s') selectClosest(windowSelect, setting.value);
        if (setting.name === 'sample_rate_hz') selectClosest(rateSelect, setting.value);
      }
    },
  });

  return function cleanup() {
    stream.close();
    resizeObserver.disconnect();
    chart.destroy();
    chart = null;
  };
}
