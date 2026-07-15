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
    <div class="scope-analysis" style="display:flex; gap:6px; align-items:center; flex-wrap:wrap; font-size:12px;">
      <label style="display:flex; gap:4px; align-items:center;">analysis
        <select class="an-mode">
          <option value="off">off</option>
          <option value="sidebands">sidebands (NA)</option>
          <option value="pairs">pairs (df/FSR)</option>
        </select></label>
      <label class="an-common" style="display:none; gap:4px; align-items:center;">ch
        <select class="an-channel"></select></label>
      <span class="an-controls" style="display:none; gap:6px; align-items:center; flex-wrap:wrap;">
        <label style="display:flex; gap:4px; align-items:center;">f_sb
          <input type="number" class="an-fsb" min="0" style="width:60px;"> MHz</label>
        <button class="an-mark" data-mark="roi" title="drag a horizontal window over the region of interest">ROI</button>
        <button class="an-mark" data-mark="x0" title="click the 0th-order mode">0th</button>
        <button class="an-mark" data-mark="xsb" title="click one sideband of the 0th-order mode">sideband</button>
        <button class="an-mark" data-mark="x1" title="click the 1st-order mode">1st</button>
        <button class="an-fit" disabled>fit</button>
        <button class="an-clear" title="clear the marks and the fit overlay">✕</button>
      </span>
      <span class="an-pairs" style="display:none; gap:6px; align-items:center; flex-wrap:wrap;">
        <button class="an-mark" data-mark="proi" title="drag a horizontal window over one pair (both its peaks)">pair ROI</button>
        <button class="an-mark" data-mark="p1" title="click the pair's first peak (the fundamental mode)">peak 1</button>
        <button class="an-mark" data-mark="p2" title="click the pair's second peak (the higher-order mode)">peak 2</button>
        <button class="an-pair-fit" disabled>fit pair</button>
        <button class="an-pair-undo" title="remove the last fitted pair">undo</button>
        <button class="an-pair-clear" title="remove all fitted pairs">✕</button>
      </span>
      <button class="an-copy" disabled style="display:none;" title="copy the results (tab-separated)">copy</button>
      <span class="an-result"></span>
    </div>
    <div class="scope-chart" style="flex:1; min-height:0; position:relative;"></div>`;

  const status = container.querySelector('.scope-status');
  const fail = (error) => { status.textContent = error.message; };
  const send = (name, args) => sendCommand(device.device_id, name, args)
    .then(() => { status.textContent = ''; })
    .catch(fail);

  // ---------------------------------------- analysis state (used by the
  // chart draw hook, so it must exist before the chart is constructed)
  const MARK_COLORS = { x0: '#52c46a', xsb: '#9a9aa2', x1: '#e05555',
                        p1: '#52c46a', p2: '#e05555' };
  const CURVE_COLOR = '#f0a030';
  const PAIR_COLORS = ['#e05555', '#52c46a', '#4a9eda', '#c95fd0',
                       '#3ec8c8', '#cfcf52'];
  const REGION_MARKS = new Set(['roi', 'proi']); // marked by dragging
  const EMPTY_MARKS = () => ({ roi: null, x0: null, xsb: null, x1: null,
                               proi: null, p1: null, p2: null });
  let chart = null; // the uPlot instance, created after the channel controls
  let isPlaying = device.playing ?? true;
  let analysisMode = 'off'; // mirrors the mode select, for the draw hook
  let analysisMarks = EMPTY_MARKS();
  let analysisResult = device.analysis ?? null; // last 'analysis_result' event
  let pairsState = device.analysis_pairs ?? null; // last 'analysis_pairs' event
  let armedMark = null;      // one of the analysisMarks keys
  let roiDragStart = null;
  let regionDragKey = null;  // which region mark the current drag writes
  let analysisUiSync = null; // assigned once the analysis controls are wired

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
    if (playing) {
      // the marks and the fit overlays belong to the discarded snapshot
      analysisMarks = EMPTY_MARKS();
      analysisResult = null;
      pairsState = null;
      roiDragStart = null;
      regionDragKey = null;
      if (chart) chart.redraw();
    }
    if (analysisUiSync) analysisUiSync();
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
    hooks: { draw: [drawAnalysis] },
  }, [[0], [null], [null], [null], [null]], chartDiv);

  // --------------------------------------------- analysis overlay drawing
  // ROI shading, mark lines and the fitted curves are painted straight onto
  // the uPlot canvas after each redraw; a fit curve has its own dense time
  // axis, so it cannot live in the shared-x data arrays as a series. Only
  // the selected mode's overlay is drawn.
  function drawAnalysis(u) {
    if (analysisMode === 'off') return;
    const ctx = u.ctx;
    const { top, height } = u.bbox;
    ctx.save();
    ctx.beginPath();
    ctx.rect(u.bbox.left, top, u.bbox.width, height);
    ctx.clip();
    const xPx = (t) => u.valToPos(t, 'x', true);
    const shadeRegion = ([t0, t1]) => {
      ctx.fillStyle = 'rgba(217, 161, 60, 0.13)';
      ctx.fillRect(xPx(t0), top, xPx(t1) - xPx(t0), height);
    };
    const verticalLine = (t) => {
      ctx.beginPath();
      ctx.moveTo(xPx(t), top);
      ctx.lineTo(xPx(t), top + height);
      ctx.stroke();
    };
    const dashedMarks = (names) => {
      ctx.lineWidth = devicePixelRatio;
      ctx.setLineDash([6 * devicePixelRatio, 4 * devicePixelRatio]);
      for (const name of names) {
        if (analysisMarks[name] == null) continue;
        ctx.strokeStyle = MARK_COLORS[name];
        verticalLine(analysisMarks[name]);
      }
      ctx.setLineDash([]);
    };
    const polyline = (curve, color) => {
      if (!curve?.t?.length) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6 * devicePixelRatio;
      ctx.beginPath();
      ctx.moveTo(xPx(curve.t[0]), u.valToPos(curve.v[0], 'y', true));
      for (let i = 1; i < curve.t.length; i++) {
        ctx.lineTo(xPx(curve.t[i]), u.valToPos(curve.v[i], 'y', true));
      }
      ctx.stroke();
    };
    if (analysisMode === 'sidebands') {
      if (analysisMarks.roi) shadeRegion(analysisMarks.roi);
      dashedMarks(['x0', 'xsb', 'x1']);
      polyline(analysisResult?.curve, CURVE_COLOR);
    } else if (analysisMode === 'pairs') {
      if (analysisMarks.proi) shadeRegion(analysisMarks.proi);
      dashedMarks(['p1', 'p2']);
      (pairsState?.pairs ?? []).forEach((pair, index) => {
        const color = PAIR_COLORS[index % PAIR_COLORS.length];
        ctx.strokeStyle = color;
        ctx.lineWidth = devicePixelRatio;
        verticalLine(pair.x01);
        verticalLine(pair.x02);
        polyline(pair.curve, color);
      });
    }
    ctx.restore();
  }

  // ------------------------------------------------- analysis interaction
  const modeSelect = container.querySelector('.an-mode');
  const anCommon = container.querySelector('.an-common');
  const anControls = container.querySelector('.an-controls');
  const anPairs = container.querySelector('.an-pairs');
  const anChannel = container.querySelector('.an-channel');
  const fsbInput = container.querySelector('.an-fsb');
  const fitButton = container.querySelector('.an-fit');
  const pairFitButton = container.querySelector('.an-pair-fit');
  const pairUndoButton = container.querySelector('.an-pair-undo');
  const pairClearButton = container.querySelector('.an-pair-clear');
  const copyButton = container.querySelector('.an-copy');
  const resultSpan = container.querySelector('.an-result');
  const markButtons = {};
  for (const button of container.querySelectorAll('.an-mark')) {
    markButtons[button.dataset.mark] = button;
  }

  for (const name of CHANNEL_ORDER) {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    anChannel.appendChild(option);
  }
  fsbInput.value = device.sideband_freq_default_mhz ?? 25;

  function setMode(mode) {
    analysisMode = mode;
    modeSelect.value = mode;
    setArm(null);
    chart.redraw();
    updateAnalysisUi();
  }
  modeSelect.onchange = () => setMode(modeSelect.value);

  function setArm(which) {
    armedMark = which;
    for (const [name, button] of Object.entries(markButtons)) {
      button.style.outline =
        name === which ? `1px solid ${MARK_COLORS[name] ?? '#d9a13c'}` : '';
    }
    chart.over.style.cursor = which ? 'crosshair' : '';
  }
  for (const [name, button] of Object.entries(markButtons)) {
    button.onclick = () => setArm(armedMark === name ? null : name);
  }

  function updateAnalysisUi() {
    const active = analysisMode !== 'off';
    const ready = active && !isPlaying;
    anCommon.style.display = active ? 'flex' : 'none';
    anControls.style.display = analysisMode === 'sidebands' ? 'flex' : 'none';
    anPairs.style.display = analysisMode === 'pairs' ? 'flex' : 'none';
    copyButton.style.display = active ? '' : 'none';
    for (const button of Object.values(markButtons)) button.disabled = !ready;
    fitButton.disabled = !(ready && analysisMarks.roi
      && analysisMarks.x0 != null && analysisMarks.xsb != null
      && analysisMarks.x1 != null);
    pairFitButton.disabled = !(ready && analysisMarks.proi
      && analysisMarks.p1 != null && analysisMarks.p2 != null);
    const nPairs = pairsState?.pairs?.length ?? 0;
    pairUndoButton.disabled = !ready || !nPairs;
    pairClearButton.disabled = !ready || !nPairs;
    copyButton.disabled = analysisMode === 'sidebands'
      ? !analysisResult?.results
      : !(analysisMode === 'pairs' && pairsState?.results?.rows);
    if (active && isPlaying) resultSpan.textContent = 'pause the stream to analyze';
    else showAnalysisResult();
  }
  analysisUiSync = updateAnalysisUi;

  function showAnalysisResult() {
    if (analysisMode === 'pairs') {
      const nPairs = pairsState?.pairs?.length ?? 0;
      const r = pairsState?.results;
      const meanStd = (mean, std, digits) => std != null
        ? `${mean.toFixed(digits)} ± ${std.toFixed(digits)}`
        : mean.toFixed(digits);
      if (r?.rows) {
        const na = r.NA_mean != null ? meanStd(r.NA_mean, r.NA_std, 4)
          : `unavailable${r.na_error ? ` (${r.na_error})` : ''}`;
        resultSpan.textContent =
          `${nPairs} pairs | df/FSR = ${meanStd(r.df_over_fsr_mean, r.df_over_fsr_std, 4)}`
          + ` | df = ${(r.df_over_fsr_mean * r.fsr_mhz).toFixed(2)} MHz`
          + ` (FSR = ${r.fsr_mhz.toFixed(1)} MHz) | NA = ${na}`;
      } else if (r?.error) {
        resultSpan.textContent = r.error;
      } else if (nPairs === 1) {
        resultSpan.textContent =
          '1 pair fitted — fit at least 2 (the FSR is the spacing between pairs)';
      } else {
        resultSpan.textContent = '';
      }
      return;
    }
    if (analysisMode !== 'sidebands' || !analysisResult?.results) {
      resultSpan.textContent = '';
      return;
    }
    const r = analysisResult.results;
    const na = r.NA != null ? r.NA.toFixed(4)
      : `unavailable${r.na_error ? ` (${r.na_error})` : ''}`;
    resultSpan.textContent =
      `mode spacing = ${r.mode_spacing_MHz.toFixed(3)} MHz | `
      + `HWHM₀ = ${r.linewidth_0_HWHM_MHz.toFixed(3)} MHz | `
      + `HWHM₁ = ${r.linewidth_1_HWHM_MHz.toFixed(3)} MHz | NA = ${na}`;
  }

  // marking on the chart: regions (ROI / pair ROI) are horizontal drags, the
  // peak marks single clicks — same arm-a-button-then-act pattern as the
  // camera circles
  chart.over.addEventListener('pointerdown', (event) => {
    if (!armedMark || isPlaying) return;
    chart.over.setPointerCapture(event.pointerId);
    if (REGION_MARKS.has(armedMark)) {
      regionDragKey = armedMark;
      roiDragStart = chart.posToVal(event.offsetX, 'x');
    }
    event.preventDefault();
  });
  chart.over.addEventListener('pointermove', (event) => {
    if (roiDragStart == null) return;
    const t = chart.posToVal(event.offsetX, 'x');
    analysisMarks[regionDragKey] =
      [Math.min(roiDragStart, t), Math.max(roiDragStart, t)];
    chart.redraw();
  });
  chart.over.addEventListener('pointerup', (event) => {
    if (!armedMark) return;
    if (REGION_MARKS.has(armedMark)) {
      roiDragStart = null;
      regionDragKey = null;
      const region = analysisMarks[armedMark];
      if (!region || region[1] - region[0] <= 0) {
        analysisMarks[armedMark] = null; // a click without a drag is no region
      }
    } else {
      analysisMarks[armedMark] = chart.posToVal(event.offsetX, 'x');
    }
    setArm(null);
    chart.redraw();
    updateAnalysisUi();
  });

  fitButton.onclick = () => {
    resultSpan.textContent = 'fitting…';
    sendCommand(device.device_id, 'analyze_sidebands', {
      channel: anChannel.value,
      t_min: analysisMarks.roi[0],
      t_max: analysisMarks.roi[1],
      x0: analysisMarks.x0,
      x_sb: analysisMarks.xsb,
      x1: analysisMarks.x1,
      f_sb_mhz: parseFloat(fsbInput.value) || null,
    }).catch((error) => { resultSpan.textContent = error.message; });
  };

  container.querySelector('.an-clear').onclick = () => {
    analysisMarks.roi = null;
    analysisMarks.x0 = null;
    analysisMarks.xsb = null;
    analysisMarks.x1 = null;
    analysisResult = null;
    setArm(null);
    chart.redraw();
    updateAnalysisUi();
  };

  pairFitButton.onclick = () => {
    resultSpan.textContent = 'fitting…';
    sendCommand(device.device_id, 'fit_pair', {
      channel: anChannel.value,
      t_min: analysisMarks.proi[0],
      t_max: analysisMarks.proi[1],
      x1: analysisMarks.p1,
      x2: analysisMarks.p2,
    }).then(() => {
      // this pair is fitted; clear the marks, ready for the next one
      analysisMarks.proi = null;
      analysisMarks.p1 = null;
      analysisMarks.p2 = null;
      chart.redraw();
      updateAnalysisUi();
    }).catch((error) => { resultSpan.textContent = error.message; });
  };
  // the pairs live on the server (all viewers share them), so undo/clear are
  // commands; the broadcast 'analysis_pairs' event updates the overlay
  pairUndoButton.onclick = () => send('undo_pair');
  pairClearButton.onclick = () => {
    analysisMarks.proi = null;
    analysisMarks.p1 = null;
    analysisMarks.p2 = null;
    setArm(null);
    send('clear_pairs');
  };

  copyButton.onclick = async () => {
    let text;
    let copied;
    if (analysisMode === 'pairs') {
      const r = pairsState?.results;
      if (!r?.rows) return;
      text = `${r.n_pairs}\t${r.df_over_fsr_mean.toFixed(6)}\t`
        + `${r.df_over_fsr_std != null ? r.df_over_fsr_std.toFixed(6) : ''}\t`
        + `${r.NA_mean != null ? r.NA_mean.toFixed(4) : 'N/A'}\t`
        + `${r.NA_std != null ? r.NA_std.toFixed(4) : ''}`;
      copied = 'copied: n pairs, df/FSR (mean, std), NA (mean, std)';
    } else {
      const r = analysisResult?.results;
      if (!r) return;
      text = `${r.mode_spacing_MHz.toFixed(4)}\t`
        + `${r.linewidth_0_HWHM_MHz.toFixed(4)}\t`
        + `${r.linewidth_1_HWHM_MHz.toFixed(4)}\t`
        + `${r.NA != null ? r.NA.toFixed(4) : 'N/A'}`;
      copied = 'copied: mode spacing, HWHM₀, HWHM₁, NA';
    }
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // http from another computer: fall back to the legacy copy command
      const scratch = document.createElement('textarea');
      scratch.value = text;
      document.body.appendChild(scratch);
      scratch.select();
      document.execCommand('copy');
      scratch.remove();
    }
    status.textContent = copied;
  };

  // start in the mode whose result an earlier session left behind
  if (pairsState?.pairs?.length) setMode('pairs');
  else if (analysisResult) setMode('sidebands');
  else updateAnalysisUi();

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
      else if (event.type === 'analysis_result') {
        analysisResult = event;
        // another viewer may have run the fit; make it visible here too
        if (analysisMode === 'off') setMode('sidebands');
        chart.redraw();
        updateAnalysisUi();
      } else if (event.type === 'analysis_pairs') {
        pairsState = event;
        if (analysisMode === 'off' && event.pairs?.length) setMode('pairs');
        chart.redraw();
        updateAnalysisUi();
      } else if (event.type === 'channel') showChannel(event.channel, event.state);
      else if (event.type === 'setting_applied') {
        if (event.name === 'window_s') selectClosest(windowSelect, event.value);
        if (event.name === 'sample_rate_hz') selectClosest(rateSelect, event.value);
      } else if (event.type === 'error') {
        status.textContent = `error: ${event.message}`;
      }
    },
    onReattach(describe) {
      setPlaying(describe.playing ?? true);
      analysisResult = describe.analysis ?? null;
      pairsState = describe.analysis_pairs ?? null;
      if (analysisMode === 'off') {
        if (pairsState?.pairs?.length) setMode('pairs');
        else if (analysisResult) setMode('sidebands');
      }
      chart.redraw();
      updateAnalysisUi();
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
