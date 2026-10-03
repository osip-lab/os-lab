// PicoScope box: rolling chart-recorder view of the streamed channels.
// The server sends 'scope_data' events at ~20 Hz, each holding the visible
// window min/max-envelope-decimated to <= ~1000 points per channel; the
// whole chart is redrawn once per event with uPlot (canvas) — there is no
// per-sample work anywhere in the browser.
// Analyses on the paused snapshot (sidebands NA, pairs df/FSR, ...) are NOT
// implemented here: they are extension modules driven by the analysis host
// (extensions/host.js + registry.js — see kalishlot/ADDING_ANALYSES.md).
// The time axis zooms with the mouse wheel over the chart (x only - the y
// axes belong to the fixed y-lim / positive only controls); double-click
// zooms back out. While paused, the zoomed span is fetched from the
// server's full-resolution snapshot (command view_region), so zooming shows
// real detail rather than a magnified envelope. The zoom is this viewer's
// alone.
// Returns a cleanup function that closes the socket.

import { connectDeviceStream } from './stream.js';
import { createAnalysisHost } from './extensions/host.js';
import { ANALYSIS_EXTENSIONS } from './extensions/registry.js';

const CHANNEL_ORDER = ['A', 'B', 'C', 'D'];
// Dimmed for the dark lab: same hues, about 60% of the old brightness.
const CHANNEL_COLORS = { A: '#2b5f85', B: '#8a3030', C: '#2f7a40', D: '#87631f' };

function formatVolts(volts) {
  return volts < 1 ? `±${volts * 1000} mV` : `±${volts} V`;
}

// uPlot's default tick labels go through Intl.NumberFormat, which stops at
// 3 decimals: on the ±50 mV range every tick then read "0.002". Instead,
// show as many decimals as the tick spacing needs.
function tickValues(u, splits, axisIndex, space, incr) {
  const decimals = Math.max(0, -Math.floor(Math.log10(incr) + 1e-9));
  return splits.map((v) => (v == null ? ''
    : (Math.abs(v) < incr / 2 ? 0 : v).toFixed(decimals)));
}

// widen the y axis to fit the longest tick label (the default 50 px clips
// labels like "-0.0012")
function tickAxisSize(u, values) {
  const longest = Math.max(0, ...(values ?? []).map((v) => String(v).length));
  return Math.max(50, Math.ceil(longest * 6.5) + 20);
}

function formatRate(hertz) {
  if (hertz >= 1e6) return `${hertz / 1e6} MS/s`;
  if (hertz >= 1e3) return `${hertz / 1e3} kS/s`;
  return `${hertz} S/s`;
}

export function createPicoScopeBox(device, container, sendCommand) {
  container.innerHTML = `
    <div class="toolbar scope-controls">
      <span class="transport">
        <button data-command="play">play</button>
        <button data-command="pause">pause</button>
      </span>
      <label class="field">window <select class="scope-window"></select></label>
      <label class="field">rate <select class="scope-rate"></select></label>
      <label class="field"><input type="checkbox" class="scope-fixed-y"> fixed y-lim</label>
      <label class="field scope-positive-field"><input type="checkbox" class="scope-positive"> positive only</label>
      <label class="field"
             title="freeze the view on each rising crossing of the trigger dot (drag it up or down), with the crossing at the middle; rolls live again after a whole window without one">trigger
        <select class="scope-trigger">
          <option value="">off</option>
          <option>A</option><option>B</option><option>C</option><option>D</option>
        </select></label>
      <span class="scope-trigger-state readout"></span>
      <span class="scope-status status-line"></span>
    </div>
    <div class="toolbar scope-channels"></div>
    <div class="toolbar scope-analysis">
      <label class="field">analysis
        <select class="an-mode">
          <option value="off">off</option>
        </select></label>
      <label class="field an-common" style="display:none;">ch
        <select class="an-channel"></select></label>
    </div>
    <div class="scope-chart"></div>`;

  const status = container.querySelector('.scope-status');
  const fail = (error) => { status.textContent = error.message; };
  const send = (name, args) => sendCommand(device.device_id, name, args)
    .then(() => { status.textContent = ''; })
    .catch(fail);

  let chart = null; // the uPlot instance, created after the channel controls
  let isPlaying = device.playing ?? true;
  // the x axis always spans the whole window, so right after a start the
  // data enters at 0 and slides left instead of the axis growing with it
  let windowSeconds = (device.settings ?? [])
    .find((setting) => setting.name === 'window_s')?.value ?? 10;
  // the x zoom (see "x zoom" below); up here because setPlaying reads it
  let xView = null;        // [t_min, t_max] while zoomed, else null
  let detailTimer = null;
  let detailRequest = 0;   // only the newest view_region reply is drawn

  // ------------------------------------------------------------- analyses
  // the channel all analyses fit on; the host shows it while a mode is
  // active and the extensions read it through box.channel()
  const anChannel = container.querySelector('.an-channel');
  for (const name of CHANNEL_ORDER) {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    anChannel.appendChild(option);
  }
  const analysisHost = createAnalysisHost({
    row: container.querySelector('.scope-analysis'),
    device,
    sendCommand,
    extensions: ANALYSIS_EXTENSIONS[device.type] ?? [],
    isPlaying: () => isPlaying,
    note: (text) => { status.textContent = text; },
    box: { channel: () => anChannel.value },
  });

  // ------------------------------------------------------- play and pause
  const buttons = {
    play: container.querySelector('[data-command="play"]'),
    pause: container.querySelector('[data-command="pause"]'),
  };
  function setPlaying(playing) {
    buttons.play.disabled = playing;
    buttons.pause.disabled = !playing;
    status.textContent = playing ? '' : 'data frozen (still acquiring)';
    isPlaying = playing;
    analysisHost.setPlaying(playing);
    if (!playing && xView) requestDetail(); // the snapshot just froze
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

  // ------------------------------------------------------------ y limits
  // Fixed: each channel's axis spans its configured range, ±range_v, or
  // [-range_v/100, range_v] with "positive only" (a sliver below zero so the
  // baseline stays visible). Off: autoscale to the data, and "positive only"
  // is greyed out and ignored. Remembered per browser, not on the server -
  // it changes the view only, never the acquisition.
  const fixedYBox = container.querySelector('.scope-fixed-y');
  const positiveBox = container.querySelector('.scope-positive');
  const positiveField = container.querySelector('.scope-positive-field');
  const yPrefKey = `kalishlot.scope.ylim.${device.device_id}`;
  try {
    const saved = JSON.parse(localStorage.getItem(yPrefKey) ?? '{}');
    fixedYBox.checked = !!saved.fixed;
    positiveBox.checked = !!saved.positive;
  } catch { /* no storage: start unfixed */ }
  function applyYLimitControls() {
    positiveBox.disabled = !fixedYBox.checked;
    positiveField.classList.toggle('is-disabled', !fixedYBox.checked);
    try {
      localStorage.setItem(yPrefKey, JSON.stringify(
        { fixed: fixedYBox.checked, positive: positiveBox.checked }));
    } catch { /* not remembered, still applied */ }
    if (chart) chart.setData(lastData); // re-ranges even while paused
  }
  fixedYBox.onchange = applyYLimitControls;
  positiveBox.onchange = applyYLimitControls;

  function yRange(name, dataMin, dataMax) {
    if (fixedYBox.checked) {
      const volts = parseFloat(channelControls[name].range.value);
      return positiveBox.checked ? [-volts / 100, volts] : [-volts, volts];
    }
    if (dataMin == null || dataMax == null) return [-1, 1];
    return uPlot.rangeNum(dataMin, dataMax, 0.1, true); // uPlot's own default
  }

  // -------------------------------------------------------------- trigger
  // The server does the triggering (adapters/picoscope.py, Trigger) - it sees
  // every sample, the box only the envelope. Here: the channel select, the
  // dot at the middle of the time axis at the trigger level, dragged up or
  // down to set it, and what the trigger is doing ('triggered' while a frame
  // is frozen, 'auto' while rolling and waiting for a crossing).
  const triggerSelect = container.querySelector('.scope-trigger');
  const triggerState = container.querySelector('.scope-trigger-state');
  let trigger = { channel: device.trigger?.channel ?? null,
                  level_v: device.trigger?.level_v ?? 0 };
  let triggerDrag = false;
  const TRIGGER_DOT_PX = 5;     // radius, CSS px
  const TRIGGER_GRAB_PX = 9;    // how near the pointer must be to grab it

  function showTrigger(settings) {
    trigger = { channel: settings.channel ?? null, level_v: settings.level_v ?? 0 };
    triggerSelect.value = trigger.channel ?? '';
    if (!trigger.channel) triggerState.textContent = '';
    if (chart) chart.redraw(false);
  }

  triggerSelect.value = trigger.channel ?? '';

  function sendTrigger() {
    send('set_trigger', { channel: trigger.channel, level_v: trigger.level_v });
  }

  triggerSelect.onchange = () => {
    const channel = triggerSelect.value || null;
    const scale = channel && chart ? chart.scales[channel] : null;
    // a level off the new channel's axis would leave the dot out of sight:
    // start it in the middle of that axis instead
    if (scale && scale.min != null
        && !(trigger.level_v > scale.min && trigger.level_v < scale.max)) {
      trigger.level_v = Number(((scale.min + scale.max) / 2).toPrecision(4));
    }
    trigger.channel = channel;
    sendTrigger();
  };

  // where the dot is, in CSS px of the plot area, or null when not drawn
  function triggerDotPos(u) {
    const name = trigger.channel;
    if (!name || !u.series[CHANNEL_ORDER.indexOf(name) + 1]?.show) return null;
    const scale = u.scales[name];
    if (scale.min == null) return null;
    const t = -windowSeconds / 2;
    if (t < u.scales.x.min || t > u.scales.x.max) return null; // zoomed away
    return { x: u.valToPos(t, 'x'), y: u.valToPos(trigger.level_v, name) };
  }

  function drawTrigger(u) {
    const dot = triggerDotPos(u);
    if (!dot) return;
    const ratio = devicePixelRatio;
    const x = u.bbox.left + dot.x * ratio;
    const y = u.bbox.top + Math.min(Math.max(dot.y, 0), u.bbox.height / ratio) * ratio;
    const ctx = u.ctx;
    ctx.save();
    ctx.strokeStyle = CHANNEL_COLORS[trigger.channel];
    ctx.globalAlpha = 0.5;      // the level across the plot, faint
    ctx.setLineDash([4 * ratio, 4 * ratio]);
    ctx.lineWidth = ratio;
    ctx.beginPath();
    ctx.moveTo(u.bbox.left, y);
    ctx.lineTo(u.bbox.left + u.bbox.width, y);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
    ctx.fillStyle = CHANNEL_COLORS[trigger.channel];
    ctx.strokeStyle = '#d8dce8';
    ctx.lineWidth = 1.5 * ratio;
    ctx.beginPath();
    ctx.arc(x, y, TRIGGER_DOT_PX * ratio, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  function nearTriggerDot(event) {
    const dot = triggerDotPos(chart);
    return dot && Math.hypot(event.offsetX - dot.x, event.offsetY - dot.y)
      <= TRIGGER_GRAB_PX;
  }

  // in the capture phase, so grabbing the dot never starts an analysis drag
  function attachTriggerDrag(u) {
    u.over.addEventListener('pointerdown', (event) => {
      if (event.button !== 0 || !nearTriggerDot(event)) return;
      event.stopImmediatePropagation();
      event.preventDefault();
      triggerDrag = true;
      u.over.setPointerCapture(event.pointerId);
    }, true);
    u.over.addEventListener('pointermove', (event) => {
      if (!triggerDrag) {
        if (!u.over.style.cursor || u.over.style.cursor === 'ns-resize') {
          u.over.style.cursor = nearTriggerDot(event) ? 'ns-resize' : '';
        }
        return;
      }
      event.stopImmediatePropagation();
      trigger.level_v = Number(
        u.posToVal(event.offsetY, trigger.channel).toPrecision(4));
      u.redraw(false);
    }, true);
    u.over.addEventListener('pointerup', (event) => {
      if (!triggerDrag) return;
      event.stopImmediatePropagation();
      triggerDrag = false;
      u.over.releasePointerCapture(event.pointerId);
      sendTrigger();
    }, true);
  }

  // ------------------------------------------------------ channel controls
  const channelsDiv = container.querySelector('.scope-channels');
  const channelControls = {}; // name -> { enable, range, coupling }

  for (const name of CHANNEL_ORDER) {
    const state = (device.channels ?? {})[name]
      ?? { enabled: name === 'A', coupling: 'DC', range_v: 5 };
    const group = document.createElement('span');
    group.className = 'field';

    const enable = document.createElement('input');
    enable.type = 'checkbox';
    const label = document.createElement('span');
    label.textContent = name;
    label.className = 'chan-name';
    label.style.color = CHANNEL_COLORS[name];

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
    if (chart) {
      buildChart(); // no-op unless the set of enabled channels changed
      if (fixedYBox.checked) chart.setData(lastData); // the range may have moved
    }
  }

  // ---------------------------------------------------------------- chart
  const chartDiv = container.querySelector('.scope-chart');
  const axisStyle = {
    stroke: '#9aa1b5',
    font: '11px Consolas, monospace',
    grid: { stroke: '#252a38' },
    ticks: { stroke: '#2e3342' },
  };
  // Every channel has its own y scale (keyed by its name) and its own axis in
  // its colour, so a ±50 mV trace and a ±5 V one each fill the height. Axes
  // alternate left, right, left, right in channel order; only enabled
  // channels get one, and only the first draws the grid. uPlot cannot add or
  // move axes on a live chart, so the chart is rebuilt whenever the set of
  // enabled channels changes (the last data is carried over).
  let lastData = [[0], [null], [null], [null], [null]];
  let builtFor = null; // the enabled channels the current chart was built for

  function chartSize() {
    const legend = chartDiv.querySelector('.u-legend');
    return {
      width: Math.max(chartDiv.clientWidth, 120),
      height: Math.max(
        chartDiv.clientHeight - (legend ? legend.offsetHeight : 30) - 4, 60),
    };
  }

  function buildChart() {
    const enabled = CHANNEL_ORDER.filter((name) => channelControls[name].enable.checked);
    if (!enabled.length) enabled.push('A');
    const key = enabled.join('');
    if (chart && key === builtFor) return;
    builtFor = key;
    const size = chart ? chartSize() : { width: 400, height: 200 };
    if (chart) chart.destroy();

    const scales = { x: { time: false, range: () => xView ?? [-windowSeconds, 0] } };
    for (const name of CHANNEL_ORDER) {
      scales[name] = { auto: true, range: (u, min, max) => yRange(name, min, max) };
    }
    chart = new uPlot({
      ...size,
      scales,
      series: [
        { label: 't (s)' },
        ...CHANNEL_ORDER.map((name) => ({
          label: name,
          scale: name,
          stroke: CHANNEL_COLORS[name],
          width: 1,
          points: { show: false },
          show: enabled.includes(name),
          value: (u, v) => (v == null ? '-' : `${v.toFixed(4)} V`),
        })),
      ],
      axes: [
        { ...axisStyle, values: tickValues },
        ...enabled.map((name, i) => ({
          ...axisStyle,
          scale: name,
          side: i % 2 ? 1 : 3, // 3 = left, 1 = right
          stroke: CHANNEL_COLORS[name],
          grid: { ...axisStyle.grid, show: i === 0 },
          label: `${name} (V)`,
          values: tickValues,
          size: tickAxisSize,
        })),
      ],
      cursor: { drag: { x: false, y: false } },
      hooks: { draw: [(u) => analysisHost.draw(u), drawTrigger] },
    }, lastData, chartDiv);
    attachTriggerDrag(chart); // before the analysis host's own pointer handlers
    chart.over.addEventListener('wheel', onWheel, { passive: false });
    chart.over.addEventListener('dblclick', resetZoom);
    chart.over.title = 'mouse wheel: zoom the time axis · double-click: zoom out';
    analysisHost.attachChart(chart);
  }
  buildChart();
  applyYLimitControls();

  // size the chart with the box (leave room for the legend row)
  const resizeObserver = new ResizeObserver(() => chart.setSize(chartSize()));
  resizeObserver.observe(chartDiv);

  // uPlot data from evenly spread points between tFirst and tLast
  function chartData(tFirst, tLast, channels) {
    let longest = 0;
    for (const name of CHANNEL_ORDER) {
      longest = Math.max(longest, channels[name]?.length ?? 0);
    }
    if (!longest) return null;
    const x = new Array(longest);
    for (let i = 0; i < longest; i++) {
      x[i] = longest > 1 ? tFirst + ((tLast - tFirst) * i) / (longest - 1) : tLast;
    }
    const data = [x];
    for (const name of CHANNEL_ORDER) {
      const values = channels[name];
      if (!values) data.push(new Array(longest).fill(null));
      else if (values.length === longest) data.push(values);
      else {
        // buffer still filling after a restart: align at the newest sample
        data.push(new Array(longest - values.length).fill(null).concat(values));
      }
    }
    return data;
  }

  let windowData = lastData; // the whole window as last streamed, unzoomed

  function showData(event) {
    if (isPlaying && trigger.channel) {
      triggerState.textContent = event.trigger_state === 'triggered'
        ? 'triggered' : 'auto';
    }
    if (event.window_s && event.window_s !== windowSeconds) {
      windowSeconds = event.window_s;
      xView = null; // a zoom into the old window means nothing in the new one
    }
    const data = chartData(-event.span_s, 0, event.channels);
    if (!data) return;
    windowData = data;
    if (!isPlaying && xView) {
      requestDetail(); // the frozen window, at the zoom's resolution
      return;
    }
    lastData = data;
    chart.setData(data);
  }

  // ------------------------------------------------------------- x zoom
  // span factor per pixel of wheel travel: a mouse notch (~100 px) zooms
  // ~1.35x, and a trackpad's many small deltas zoom smoothly
  const ZOOM_PER_PIXEL = 0.003;
  const MIN_SPAN_SAMPLES = 20;  // the deepest zoom, in samples

  function onWheel(event) {
    event.preventDefault(); // the wheel zooms here, it does not scroll the page
    const [min, max] = xView ?? [-windowSeconds, 0];
    const t = chart.posToVal(event.offsetX, 'x'); // stays under the cursor
    const pixels = event.deltaY * (event.deltaMode === 1 ? 33 : 1); // lines
    const factor = Math.exp(pixels * ZOOM_PER_PIXEL);
    const minSpan = MIN_SPAN_SAMPLES / parseFloat(rateSelect.value);
    const span = Math.min(Math.max((max - min) * factor, minSpan), windowSeconds);
    if (span >= windowSeconds) {
      resetZoom();
      return;
    }
    let lo = t - (t - min) * (span / (max - min));
    lo = Math.min(Math.max(lo, -windowSeconds), -span);
    setView([lo, lo + span]);
  }

  function resetZoom() {
    if (!xView) return;
    xView = null;
    clearTimeout(detailTimer);
    detailRequest++; // a reply still on its way is no longer wanted
    lastData = windowData;
    chart.setData(windowData);
  }

  function setView(view) {
    xView = view;
    chart.setScale('x', { min: view[0], max: view[1] });
    if (!isPlaying) {
      // once the wheel stops, not at every notch
      clearTimeout(detailTimer);
      detailTimer = setTimeout(requestDetail, 150);
    }
  }

  function requestDetail() {
    if (!xView) return;
    const request = ++detailRequest;
    const [tMin, tMax] = xView;
    sendCommand(device.device_id, 'view_region', { t_min: tMin, t_max: tMax })
      .then((reply) => {
        if (request !== detailRequest || isPlaying || !xView) return;
        const data = reply.t_first == null ? null
          : chartData(reply.t_first, reply.t_last, reply.channels);
        if (!data) return;
        lastData = data;
        chart.setData(data);
      })
      .catch(fail);
  }

  // ----------------------------------------------------------- the stream
  const stream = connectDeviceStream({
    deviceId: device.device_id,
    status,
    onEvent(event) {
      if (event.type === 'scope_data') showData(event);
      else if (event.type === 'status') setPlaying(event.playing);
      else if (analysisHost.onEvent(event)) { /* an analysis extension's */ }
      else if (event.type === 'channel') showChannel(event.channel, event.state);
      else if (event.type === 'trigger') showTrigger(event);
      else if (event.type === 'setting_applied') {
        if (event.name === 'window_s') selectClosest(windowSelect, event.value);
        if (event.name === 'sample_rate_hz') selectClosest(rateSelect, event.value);
      } else if (event.type === 'error') {
        status.textContent = `error: ${event.message}`;
      }
    },
    onReattach(describe) {
      setPlaying(describe.playing ?? true);
      analysisHost.onReattach(describe);
      if (describe.trigger) showTrigger(describe.trigger);
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
    clearTimeout(detailTimer);
    chart.destroy();
    chart = null;
  };
}
