// Camera box: live video over WebSocket, play / pause / single-frame,
// exposure & gain inputs (commit on Enter or focus loss), Gaussian fit with
// ellipse overlay + cross-section plots, an ROI, and two draggable circles:
//   marker ◯ — a persistent annotation (cyan), local to this viewer;
//   guess ◯  — the fit's initial guess (dashed green): center -> (x_0, y_0),
//              radius -> sigma; lives on the server so the fit can use it.
// Shared by every camera-like device type (dummy, Basler).
//
// The ROI dropdown has the three states it can be in: no ROI, editing the
// rectangle (drag it out, then move it or pull its handles), and applied —
// at which point the cropped frame IS the image and every coordinate in the
// box (fit, guess, marker, the rectangle itself) counts from its corner.
// Whether the crop happens in the camera or in the server is the adapter's
// business; nothing here depends on which.
//
// The 'intensity' checkbox puts a strip chart beside the image: the 99th
// percentile pixel value over the last 30 s, against a fixed 0..full-scale
// axis, for watching the light while a knob on the bench is turned, with a
// dashed line at its mean over the last 10 s — the steadier number to read
// off while turning. The server also sends the median and the maximum of
// every frame — see LEVELS_SERIES to put either back on the chart (each
// drawn series gets its own mean line). It measures and remembers that
// window; nothing is kept beyond it and nothing is saved.
//
// Returns a cleanup function that closes the socket.

import { connectDeviceStream } from './stream.js';
import { logEntry } from './logger.js';

const STRIP = 70;        // cross-section strip thickness, px
const GAP = 4;

const COLOR_DATA = 'rgb(70, 140, 220)';        // cross-section data
const COLOR_FIT_CURVE = 'rgb(255, 165, 40)';   // cross-section fit curve
const COLOR_ELLIPSE = 'rgba(255, 90, 90, 0.67)';
const COLOR_MARKER = 'rgba(0, 220, 220, 0.86)';
const COLOR_GUESS = 'rgba(110, 255, 110, 0.86)';
const COLOR_GRID = 'rgba(255, 255, 255, 0.28)';
const COLOR_ROI = 'rgba(255, 205, 70, 0.95)';
const GRID_SPACING_MM = 1;
const ROI_HANDLE_PX = 9;   // hit radius for the rectangle's handles, css px
const ROI_MIN_PX = 16;     // matches CameraAdapterBase.MIN_ROI_PX

const LEVELS_MIN_W = 240;  // width the image gives up for the strip chart
const LEVELS_AVERAGE_S = 10;  // span of the dashed mean line on that chart
// Which of the server's numbers to draw — median = the background, p99 = the
// beam, max = saturation. It sends all three every time —
// point = [seconds ago, median, p99, max] — so a line commented out here is
// the only thing standing between it and the chart: uncomment it to get the
// series back, no server change and no reconnect needed. `index` is its slot
// in that point, named rather than positional so commenting one out does not
// silently shift the others onto the wrong data.
const LEVELS_SERIES = [
  // { label: 'median', stroke: 'rgb(110, 170, 240)', index: 1 },
  { label: 'p99', stroke: 'rgb(255, 190, 60)', index: 2 },
  // { label: 'max', stroke: 'rgb(232, 96, 96)', index: 3 },
];

export function createCameraBox(device, container, sendCommand) {
  // fit coordinates are pixels of the frame the camera currently delivers —
  // the whole sensor, or the ROI once one is applied. The video stream may be
  // downsampled, so all drawing is scaled from sensor_shape, which the server
  // re-sends (with a 'roi' event) whenever the crop changes.
  let [sensorH, sensorW] = device.sensor_shape ?? device.frame_shape;
  let [fullH, fullW] = device.sensor_full ?? [sensorH, sensorW];
  let roi = device.roi ?? null;   // {x, y, width, height} in sensor px, or null
  const pixelMm = device.pixel_size_mm ?? 0;
  const levelsMax = device.levels_max ?? 4095; // raw-data full scale

  container.innerHTML = `
    <div class="toolbar cam-controls">
      <span class="transport">
        <button data-command="play">play</button>
        <button data-command="pause">pause</button>
      </span>
      <button data-command="snap">single frame</button>
      <span class="subgroup cam-settings"></span>
      <span class="subgroup">
        <label class="field">
          <input type="checkbox" class="cam-fit"> fit</label>
      </span>
      <span class="subgroup">
        <label class="field" title="1 mm grid, centered on the image">
          <input type="checkbox" class="cam-grid"> grid</label>
      </span>
      <span class="subgroup">
        <label class="field"
               title="strip chart beside the image: the 99th percentile pixel value over the last 30 s, on a fixed 0 to full-scale axis, with a dashed line at its 10 s mean — watch it while you change something on the bench">
          <input type="checkbox" class="cam-levels"> intensity</label>
      </span>
      <span class="subgroup">
        <label class="field"
               title="fit only frames at least this bright (counts above background); empty or 0 = fit every frame">
          trigger <input type="number" class="cam-trigger" min="0" placeholder="off"></label>
        <span class="cam-brightness readout"
              title="live beam brightness, counts above background: mean inside the guess circle's bounding square, or the 99th-percentile pixel when no guess circle is set"></span>
      </span>
      <span class="subgroup">
        <label class="field"
               title="crop the camera to a region: pick Edit ROI, drag the rectangle on the image (its handles resize it, its middle moves it), then Apply ROI">
          ROI <select class="cam-roi">
            <option value="none">No ROI</option>
            <option value="edit">Edit ROI</option>
            <option value="apply">Apply ROI</option>
          </select></label>
      </span>
      <span class="subgroup">
        <button class="cam-mark" title="drag from the circle center to its edge">marker ◯</button>
        <button class="cam-mark-clear" title="clear the marker circle">✕</button>
        <button class="cam-guess" title="drag the fit initial guess: center → (x₀, y₀), radius → σ">guess ◯</button>
        <button class="cam-guess-clear" title="clear the guess circle">✕</button>
      </span>
      <span class="subgroup">
        <button class="cam-copy-figure" title="copy the figure (image, overlays, cross-sections, title) to the clipboard as PNG">copy figure</button>
        <button class="cam-copy-fit" title="copy the fitted beam radii w_x, w_y (mm, tab-separated) to the clipboard">copy w_x w_y</button>
        <button class="cam-record-fit" title="append the current fit values to the log">record fit values</button>
      </span>
      <span class="cam-status status-line"></span>
    </div>
    <div class="cam-info"></div>
    <div class="cam-view"></div>`;

  const status = container.querySelector('.cam-status');
  const info = container.querySelector('.cam-info');
  const view = container.querySelector('.cam-view');

  // ------------------------------------------------------- play/pause/snap
  const buttons = {
    play: container.querySelector('[data-command="play"]'),
    pause: container.querySelector('[data-command="pause"]'),
    snap: container.querySelector('[data-command="snap"]'),
  };

  function setPlaying(playing) {
    buttons.play.disabled = playing;
    buttons.pause.disabled = !playing;
    status.textContent = playing ? 'streaming' : 'paused';
  }
  setPlaying(device.playing ?? true);

  buttons.play.onclick = () => sendCommand(device.device_id, 'play')
    .then(() => setPlaying(true)).catch((e) => alert(e.message));
  buttons.pause.onclick = () => sendCommand(device.device_id, 'pause')
    .then(() => setPlaying(false)).catch((e) => alert(e.message));
  buttons.snap.onclick = () => sendCommand(device.device_id, 'pause')
    .then(() => { setPlaying(false); return sendCommand(device.device_id, 'snap'); })
    .catch((e) => alert(e.message));

  // ------------------------------------------------------------- settings
  // Number inputs from the adapter's settings schema. Commit on Enter or
  // focus loss; the input then shows the value the hardware accepted
  // (delivered as a 'setting_applied' event).
  const settingsSpan = container.querySelector('.cam-settings');
  const settingInputs = {}; // name -> { input, decimals }
  for (const setting of device.settings ?? []) {
    const label = document.createElement('label');
    label.className = 'field';
    label.textContent = `${setting.label}`;
    const input = document.createElement('input');
    input.type = 'number';
    input.min = setting.min;
    input.max = setting.max;
    input.step = setting.decimals ? Math.pow(10, -setting.decimals) : 1;
    input.value = setting.value.toFixed(setting.decimals);
    input.dataset.committed = input.value;
    const unit = document.createElement('span');
    unit.textContent = setting.unit;
    unit.className = 'unit';
    label.appendChild(input);
    label.appendChild(unit);
    settingsSpan.appendChild(label);
    settingInputs[setting.name] = { input, decimals: setting.decimals };

    const commit = () => {
      const value = parseFloat(input.value);
      if (!isFinite(value) || input.value === input.dataset.committed) return;
      input.dataset.committed = input.value;
      sendCommand(device.device_id, 'set_setting', { name: setting.name, value })
        .catch((e) => { status.textContent = e.message; });
    };
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') commit(); });
    input.addEventListener('blur', commit);
  }

  function showAppliedSetting(name, value, max) {
    const entry = settingInputs[name];
    if (!entry) return;
    // a ceiling that moves with other settings (the XIMEA's frame rate
    // with its exposure) comes along with the value
    if (max != null) entry.input.max = max;
    entry.input.value = value.toFixed(entry.decimals);
    entry.input.dataset.committed = entry.input.value;
  }

  // ------------------------------------------- view: canvases and layout
  // Desktop-GUI arrangement: row cross-section strip on top, image below it,
  // column cross-section strip to the right. The strips keep their slots
  // also when the fit is off. Sizes are computed here (not with CSS) so the
  // strips stay exactly aligned with the image axes.
  function makeCanvas(background) {
    const canvas = document.createElement('canvas');
    canvas.style.position = 'absolute';
    if (background) canvas.style.background = background;
    view.appendChild(canvas);
    return canvas;
  }
  const hCanvas = makeCanvas('#14161d');       // row cut, above the image
  const vCanvas = makeCanvas('#14161d');       // column cut, right of image
  const videoCanvas = makeCanvas('#0e1015');
  const overlay = makeCanvas(null);            // ellipses + circles, on top
  overlay.style.touchAction = 'none';
  // the intensity strip chart lives to the right of the column strip; it is
  // a uPlot, so it gets a div rather than a canvas of ours
  const levelsDiv = document.createElement('div');
  levelsDiv.className = 'cam-levels-chart';
  levelsDiv.style.position = 'absolute';
  levelsDiv.hidden = true;
  view.appendChild(levelsDiv);

  function place(element, x, y, w, h) {
    element.style.left = `${x}px`;
    element.style.top = `${y}px`;
    element.style.width = `${w}px`;
    element.style.height = `${h}px`;
  }

  function layout() {
    // the strip chart is given a minimum width out of the image's share, and
    // then whatever else is left over — so widening the box grows the chart
    const showChart = levelsCheck.checked;
    const reserved = showChart ? LEVELS_MIN_W + GAP : 0;
    const scale = Math.max(Math.min(
      (view.clientWidth - STRIP - GAP - reserved) / sensorW,
      (view.clientHeight - STRIP - GAP) / sensorH), 0.01);
    const w = Math.round(sensorW * scale);
    const h = Math.round(sensorH * scale);
    place(hCanvas, 0, 0, w, STRIP);
    place(videoCanvas, 0, STRIP + GAP, w, h);
    place(overlay, 0, STRIP + GAP, w, h);
    place(vCanvas, w + GAP, STRIP + GAP, STRIP, h);
    hCanvas.width = w; hCanvas.height = STRIP;
    vCanvas.width = STRIP; vCanvas.height = h;
    overlay.width = w; overlay.height = h;
    levelsDiv.hidden = !showChart;
    if (showChart) {
      const x = w + GAP + STRIP + GAP;
      const chartWidth = Math.max(view.clientWidth - x, LEVELS_MIN_W);
      place(levelsDiv, x, STRIP + GAP, chartWidth, h);
      sizeLevelsChart(chartWidth, h);
    }
    redrawOverlay();
    drawStrips();
  }
  const resizeObserver = new ResizeObserver(layout);
  resizeObserver.observe(view);

  // --------------------------------------------------------- overlay state
  let fitParams = null;   // last successful fit, sensor px
  let fitCross = null;    // {step, row, col} pixel cuts through the center
  let fitReason = '';
  let marker = null;      // {x, y, r} sensor px
  let editRect = null;    // ROI being edited: {x, y, w, h} in current-frame px
  let guess = device.guess
    ? { x: device.guess.x_0, y: device.guess.y_0, r: device.guess.sigma } : null;

  const toCss = (v) => v * overlay.width / sensorW; // aspect is preserved

  function drawCross(ctx, cx, cy, size) {
    ctx.beginPath();
    ctx.moveTo(cx - size, cy); ctx.lineTo(cx + size, cy);
    ctx.moveTo(cx, cy - size); ctx.lineTo(cx, cy + size);
    ctx.stroke();
  }

  function drawCircle(ctx, circle, color, dashed) {
    ctx.strokeStyle = color;
    ctx.setLineDash(dashed ? [6, 4] : []);
    ctx.beginPath();
    ctx.arc(toCss(circle.x), toCss(circle.y), toCss(circle.r), 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);
    drawCross(ctx, toCss(circle.x), toCss(circle.y), 5);
  }

  function drawGrid(ctx) {
    // spacing in sensor px for GRID_SPACING_MM, centered on the SENSOR so an
    // intersection falls at its middle — a one-pixel offset from dropping the
    // fractional half-cell at either edge doesn't matter here. Anchoring it
    // to the sensor rather than to the image keeps the grid still under the
    // beam when the view is cropped to an ROI.
    if (!pixelMm) return;
    const stepPx = GRID_SPACING_MM / pixelMm;
    ctx.strokeStyle = COLOR_GRID;
    ctx.setLineDash([]);
    ctx.lineWidth = 1;
    const cx = fullW / 2 - (roi ? roi.x : 0);
    const cy = fullH / 2 - (roi ? roi.y : 0);
    ctx.beginPath();
    for (let x = cx; x >= 0; x -= stepPx) { ctx.moveTo(toCss(x), 0); ctx.lineTo(toCss(x), overlay.height); }
    for (let x = cx + stepPx; x <= sensorW; x += stepPx) { ctx.moveTo(toCss(x), 0); ctx.lineTo(toCss(x), overlay.height); }
    for (let y = cy; y >= 0; y -= stepPx) { ctx.moveTo(0, toCss(y)); ctx.lineTo(overlay.width, toCss(y)); }
    for (let y = cy + stepPx; y <= sensorH; y += stepPx) { ctx.moveTo(0, toCss(y)); ctx.lineTo(overlay.width, toCss(y)); }
    ctx.stroke();
  }

  function redrawOverlay() {
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.lineWidth = 1;
    if (gridCheck.checked) drawGrid(ctx);
    if (fitParams) {
      // thin translucent ellipses at 1 sigma and at the beam radius w = 2 sigma
      ctx.strokeStyle = COLOR_ELLIPSE;
      const cx = toCss(fitParams.x_0), cy = toCss(fitParams.y_0);
      for (const k of [1, 2]) {
        ctx.beginPath();
        ctx.ellipse(cx, cy, toCss(k * fitParams.s_x), toCss(k * fitParams.s_y),
          fitParams.angle, 0, 2 * Math.PI);
        ctx.stroke();
      }
      drawCross(ctx, cx, cy, 6);
    }
    if (marker) drawCircle(ctx, marker, COLOR_MARKER, false);
    if (guess) drawCircle(ctx, guess, COLOR_GUESS, true);
    if (editRect) drawEditRect(ctx);
  }

  // The rectangle being edited: everything outside it is dimmed, so what the
  // camera would deliver after Apply ROI is what stays bright.
  function drawEditRect(ctx) {
    const x = toCss(editRect.x), y = toCss(editRect.y);
    const w = toCss(editRect.w), h = toCss(editRect.h);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
    ctx.fillRect(0, 0, overlay.width, y);
    ctx.fillRect(0, y + h, overlay.width, overlay.height - y - h);
    ctx.fillRect(0, y, x, h);
    ctx.fillRect(x + w, y, overlay.width - x - w, h);
    ctx.strokeStyle = COLOR_ROI;
    ctx.setLineDash([5, 3]);
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    ctx.fillStyle = COLOR_ROI;
    for (const [hx, hy] of roiHandlePoints(x, y, w, h)) {
      ctx.fillRect(hx - 3, hy - 3, 6, 6);
    }
  }

  // the eight grab points, in the order of ROI_HANDLES
  function roiHandlePoints(x, y, w, h) {
    return [[x, y], [x + w / 2, y], [x + w, y],
            [x, y + h / 2], [x + w, y + h / 2],
            [x, y + h], [x + w / 2, y + h], [x + w, y + h]];
  }

  // ------------------------------------------------- cross-section strips
  function polyline(ctx, points, color) {
    if (!points.length) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
    ctx.stroke();
  }

  function drawStrips() {
    const hCtx = hCanvas.getContext('2d');
    const vCtx = vCanvas.getContext('2d');
    hCtx.clearRect(0, 0, hCanvas.width, hCanvas.height);
    vCtx.clearRect(0, 0, vCanvas.width, vCanvas.height);
    if (!fitParams || !fitCross) return;
    const p = fitParams;
    const step = fitCross.step;
    // data: the image row/column through the fit center
    polyline(hCtx, fitCross.row.map((value, i) =>
      [i * step * hCanvas.width / sensorW,
       hCanvas.height * (1 - value / levelsMax)]), COLOR_DATA);
    polyline(vCtx, fitCross.col.map((value, i) =>
      [vCanvas.width * value / levelsMax,
       i * step * vCanvas.height / sensorH]), COLOR_DATA);
    // analytic cuts of the fitted 2D Gaussian along y = y0 and x = x0
    const sin2 = Math.sin(p.angle) ** 2, cos2 = Math.cos(p.angle) ** 2;
    const a = cos2 / (2 * p.s_x ** 2) + sin2 / (2 * p.s_y ** 2);
    const c = sin2 / (2 * p.s_x ** 2) + cos2 / (2 * p.s_y ** 2);
    const n = 200;
    const hPoints = [], vPoints = [];
    for (let i = 0; i <= n; i++) {
      const x = i / n * sensorW;
      const hValue = p.offset + p.amplitude * Math.exp(-a * (x - p.x_0) ** 2);
      hPoints.push([x * hCanvas.width / sensorW,
                    hCanvas.height * (1 - hValue / levelsMax)]);
      const y = i / n * sensorH;
      const vValue = p.offset + p.amplitude * Math.exp(-c * (y - p.y_0) ** 2);
      vPoints.push([vCanvas.width * vValue / levelsMax,
                    y * vCanvas.height / sensorH]);
    }
    polyline(hCtx, hPoints, COLOR_FIT_CURVE);
    polyline(vCtx, vPoints, COLOR_FIT_CURVE);
  }

  // -------------------------------------------------------- the info line
  // The fitted beam radii are the number people read across the room, so they
  // get their own oversized span; everything else stays at the usual size.
  const fitReadout = document.createElement('span');
  fitReadout.className = 'cam-fit-readout';
  const infoAux = document.createElement('span');
  infoAux.className = 'cam-info-aux';
  info.append(fitReadout, infoAux);

  function updateInfo() {
    if (fitParams && !fitReason) {
      const p = fitParams;
      fitReadout.textContent = `w_x = ${(p.w_x * pixelMm).toFixed(3)} mm, `
        + `w_y = ${(p.w_y * pixelMm).toFixed(3)} mm`;
    } else {
      fitReadout.textContent = '';
    }

    const parts = [];
    if (fitReason) parts.push(fitReason);
    else if (fitParams) {
      const p = fitParams;
      parts.push(`x₀ = ${p.x_0.toFixed(1)} px, y₀ = ${p.y_0.toFixed(1)} px, `
        + `θ = ${p.angle >= 0 ? '+' : ''}${p.angle.toFixed(2)} rad `
        + `(fit ${p.time.toFixed(2)} s)`);
    }
    if (marker) {
      parts.push(`marker: (${marker.x.toFixed(0)}, ${marker.y.toFixed(0)}) px, `
        + `r = ${marker.r.toFixed(1)} px = ${(marker.r * pixelMm).toFixed(3)} mm`);
    }
    if (guess) {
      parts.push(`guess: (${guess.x.toFixed(0)}, ${guess.y.toFixed(0)}) px, `
        + `σ = ${guess.r.toFixed(1)} px`);
    }
    if (roi) {
      parts.push(`ROI: ${roi.width}×${roi.height} px at (${roi.x}, ${roi.y}) `
        + `of ${fullW}×${fullH}`);
    }
    if (editRect) {
      parts.push(`ROI edit: ${Math.round(editRect.w)}×${Math.round(editRect.h)}`
        + ` px at (${Math.round(editRect.x)}, ${Math.round(editRect.y)})`);
    }
    // the leading separator also keeps the copied-figure title readable, since
    // that is built from info.textContent (both spans concatenated)
    const aux = parts.join('   |   ');
    infoAux.textContent = fitReadout.textContent && aux ? `   |   ${aux}` : aux;
  }

  // ------------------------------------------------------------------ fit
  const fitCheck = container.querySelector('.cam-fit');
  fitCheck.checked = device.fitting ?? false;
  fitCheck.onchange = () => {
    sendCommand(device.device_id, fitCheck.checked ? 'fit_on' : 'fit_off')
      .catch((e) => { status.textContent = e.message; });
  };

  // grid is a local display preference only (like the marker circle) — not
  // sent to the server, not shared across viewers.
  const gridCheck = container.querySelector('.cam-grid');
  gridCheck.onchange = () => redrawOverlay();

  // ------------------------------------------------- intensity strip chart
  // Median / 99th percentile / maximum pixel value over the last 30 s, for
  // watching the light while something on the bench is changed. The server
  // measures and keeps the window (it costs CPU there, so like the fit it is
  // shared by every viewer rather than switched on per browser), and sends
  // the whole window each time — this side only draws what it is given.
  const levelsCheck = container.querySelector('.cam-levels');
  const levelsWindowS = device.levels_window_s ?? 30;
  let levelsChart = null;
  let levelsPoints = null;   // newest window received, kept across re-layouts

  function ensureLevelsChart() {
    if (levelsChart) return levelsChart;
    const axisStyle = {
      stroke: '#9aa1b5',
      font: '11px Consolas, monospace',
      grid: { stroke: '#252a38' },
      ticks: { stroke: '#2e3342' },
    };
    levelsChart = new uPlot({
      width: LEVELS_MIN_W,
      height: 160,
      // Both axes are pinned. x, so a trace does not stretch sideways while
      // the first 30 s are still filling up; y to the sensor's full range,
      // so a height on this chart means the same thing from one session to
      // the next and saturation is always the top of the box rather than
      // wherever the last few seconds happened to reach.
      scales: {
        x: { time: false, range: [-levelsWindowS, 0] },
        y: { range: [0, levelsMax] },
      },
      series: [
        { label: 't (s)' },
        ...LEVELS_SERIES.map((series) => ({
          label: series.label,
          stroke: series.stroke,
          width: 1,
          points: { show: false },
          value: (u, v) => (v == null ? '-' : v.toFixed(1)),
        })),
      ],
      axes: [axisStyle, { ...axisStyle, label: 'counts' }],
      cursor: { drag: { x: false, y: false } },
      hooks: { draw: [drawLevelsAverages] },
      // one empty column per series, so this follows LEVELS_SERIES too
    }, [[0], ...LEVELS_SERIES.map(() => [null])], levelsDiv);
    return levelsChart;
  }

  // Mean of the last LEVELS_AVERAGE_S of one series, or null when the window
  // holds nothing that recent. Averaged over what is actually there, so the
  // line is honest in the first seconds after the monitor is switched on.
  function levelsAverage(index) {
    let sum = 0;
    let count = 0;
    for (const point of levelsPoints ?? []) {
      if (point[0] >= -LEVELS_AVERAGE_S) {
        sum += point[index];
        count += 1;
      }
    }
    return count ? sum / count : null;
  }

  // One dashed line per drawn series, in that series' color, at its recent
  // mean — the number to read off while a knob is being turned, steadier
  // than the trace itself. Drawn in a uPlot hook so it survives every
  // redraw, in canvas pixels like everything else in there.
  function drawLevelsAverages(chart) {
    // a hook draws in canvas pixels, where uPlot's own 1px lines and 11px
    // fonts have already been multiplied by the device ratio — match it, or
    // this comes out hairline on a HiDPI screen
    const ratio = uPlot.pxRatio || window.devicePixelRatio || 1;
    const ctx = chart.ctx;
    ctx.save();
    ctx.lineWidth = ratio;
    ctx.font = `${11 * ratio}px Consolas, monospace`;
    ctx.textBaseline = 'bottom';
    const left = chart.bbox.left;
    const right = left + chart.bbox.width;
    LEVELS_SERIES.forEach((series, i) => {
      if (!chart.series[i + 1].show) return;
      const mean = levelsAverage(series.index);
      if (mean === null) return;
      const y = Math.round(chart.valToPos(mean, 'y', true)) + 0.5;
      ctx.strokeStyle = series.stroke;
      ctx.fillStyle = series.stroke;
      ctx.setLineDash([6 * ratio, 4 * ratio]);
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(right, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillText(mean.toFixed(1), left + 4 * ratio, y - 2 * ratio);
    });
    ctx.restore();
  }

  function sizeLevelsChart(width, height) {
    if (!levelsChart) return;
    const legend = levelsDiv.querySelector('.u-legend');
    levelsChart.setSize({
      width: Math.max(width, 120),
      height: Math.max(height - (legend ? legend.offsetHeight : 30) - 4, 60),
    });
  }

  function drawLevels() {
    if (!levelsCheck.checked || !levelsPoints) return;
    const chart = ensureLevelsChart();
    chart.setData([levelsPoints.map((point) => point[0]),
                   ...LEVELS_SERIES.map((series) =>
                     levelsPoints.map((point) => point[series.index]))]);
  }

  function showLevelsEnabled(enabled) {
    levelsCheck.checked = enabled;
    if (!enabled) levelsPoints = null;
    else ensureLevelsChart();   // before layout(), which sizes it
    layout();                   // the image gives up / takes back the width
    drawLevels();
  }

  levelsCheck.onchange = () => {
    showLevelsEnabled(levelsCheck.checked);
    sendCommand(device.device_id,
                levelsCheck.checked ? 'levels_on' : 'levels_off')
      .catch((e) => { status.textContent = e.message; });
  };

  // The crop changed: the image is a different frame now, so everything drawn
  // in the old frame's pixels is stale. The guess is the exception — the
  // server moves it with the crop and broadcasts it — and the video canvas is
  // cleared so a frame from before the change is not left stretched over the
  // new aspect ratio.
  function applyRoiState(newRoi, shape, full) {
    newRoi = newRoi ?? null;
    // a reattach re-sends the state unchanged: don't throw away the marker
    // and the last fit just because the socket reconnected
    const changed = JSON.stringify(newRoi) !== JSON.stringify(roi);
    roi = newRoi;
    if (shape) [sensorH, sensorW] = shape;
    if (full) [fullH, fullW] = full;
    if (changed || editRect) {
      editRect = null;
      roiDrag = null;
      marker = null;
      fitParams = null;
      fitCross = null;
      videoCanvas.getContext('2d').clearRect(0, 0, videoCanvas.width,
                                             videoCanvas.height);
    }
    showRoiState();
    layout();          // the frame's aspect ratio moved: re-place the canvases
    updateInfo();
  }

  function clearFitDisplay() {
    fitParams = null;
    fitCross = null;
    fitReason = '';
    lastBrightness = null;
    paintBrightness();
    redrawOverlay();
    drawStrips();
    updateInfo();
  }

  // --------------------------------------------- fit trigger (blinking beam)
  // Frames dimmer than the threshold are not fitted (the last fit result
  // stays on screen). The live readout blinks with the beam — watch it and
  // set the threshold between the dark and bright values.
  const triggerInput = container.querySelector('.cam-trigger');
  const brightnessSpan = container.querySelector('.cam-brightness');
  let fitThreshold = 0;
  let lastBrightness = null; // newest 'brightness' event value, or null

  function paintBrightness() {
    if (lastBrightness === null) {
      brightnessSpan.textContent = '';
      return;
    }
    brightnessSpan.textContent = lastBrightness.toFixed(0);
    const above = fitThreshold <= 0 || lastBrightness >= fitThreshold;
    brightnessSpan.style.color = above ? '#52c46a' : 'rgba(224, 85, 85, 0.9)';
  }

  function showThreshold(value) {
    fitThreshold = value;
    triggerInput.value = value > 0 ? value : '';
    triggerInput.dataset.committed = triggerInput.value;
    paintBrightness();
  }
  showThreshold(device.fit_threshold ?? 0);

  const commitTrigger = () => {
    if (triggerInput.value === triggerInput.dataset.committed) return;
    const value = triggerInput.value === '' ? 0 : parseFloat(triggerInput.value);
    if (!isFinite(value) || value < 0) return;
    triggerInput.dataset.committed = triggerInput.value;
    sendCommand(device.device_id, 'set_fit_threshold', { value })
      .catch((e) => { status.textContent = e.message; });
  };
  triggerInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') commitTrigger(); });
  triggerInput.addEventListener('blur', commitTrigger);

  // ------------------------------------------------------------------ ROI
  // Three states in one dropdown: no crop, editing the rectangle, cropped.
  // 'Apply ROI' sends the rectangle in current-frame pixels; the server
  // translates it onto the sensor (so applying twice zooms further in) and
  // answers with a 'roi' event carrying the size the hardware snapped to.
  const roiSelect = container.querySelector('.cam-roi');
  const ROI_HANDLES = ['nw', 'n', 'ne', 'w', 'e', 'sw', 's', 'se'];
  const ROI_CURSORS = {nw: 'nwse-resize', n: 'ns-resize', ne: 'nesw-resize',
                       w: 'ew-resize', e: 'ew-resize', sw: 'nesw-resize',
                       s: 'ns-resize', se: 'nwse-resize', move: 'move'};
  let roiDrag = null;     // {handle, start:{x,y}, rect0} while dragging

  function showRoiState() {
    roiSelect.value = editRect ? 'edit' : (roi ? 'apply' : 'none');
    overlay.style.cursor = editRect ? 'crosshair' : (armed ? 'crosshair' : '');
  }

  roiSelect.onchange = () => {
    const mode = roiSelect.value;
    if (mode === 'edit') {
      setArmed(null);   // the rectangle owns the pointer while editing
      // start from the middle half of what is on screen; a drag on empty
      // image replaces it outright
      if (!editRect) {
        editRect = {x: sensorW / 4, y: sensorH / 4,
                    w: sensorW / 2, h: sensorH / 2};
      }
      status.textContent = 'drag the rectangle, then choose Apply ROI';
    } else if (mode === 'apply') {
      if (!editRect) {
        status.textContent = 'nothing to apply — pick Edit ROI and drag a rectangle first';
        showRoiState();
        return;
      }
      sendCommand(device.device_id, 'set_roi',
        {x: Math.round(editRect.x), y: Math.round(editRect.y),
         width: Math.round(editRect.w), height: Math.round(editRect.h)})
        .catch((e) => { status.textContent = e.message; });
      // the 'roi' event does the rest (new frame shape, rectangle cleared)
      return;
    } else {
      editRect = null;
      if (roi) {
        sendCommand(device.device_id, 'clear_roi')
          .catch((e) => { status.textContent = e.message; });
        return;   // again, the 'roi' event finishes it
      }
    }
    redrawOverlay();
    updateInfo();
  };

  function roiHandleAt(event) {
    if (!editRect) return null;
    const rect = overlay.getBoundingClientRect();
    const px = (event.clientX - rect.left) * overlay.width / rect.width;
    const py = (event.clientY - rect.top) * overlay.height / rect.height;
    const points = roiHandlePoints(toCss(editRect.x), toCss(editRect.y),
                                   toCss(editRect.w), toCss(editRect.h));
    for (let i = 0; i < points.length; i++) {
      if (Math.abs(px - points[i][0]) <= ROI_HANDLE_PX
          && Math.abs(py - points[i][1]) <= ROI_HANDLE_PX) return ROI_HANDLES[i];
    }
    const inside = px > toCss(editRect.x) && px < toCss(editRect.x + editRect.w)
      && py > toCss(editRect.y) && py < toCss(editRect.y + editRect.h);
    return inside ? 'move' : null;
  }

  function resizeEditRect(handle, rect0, dx, dy) {
    // work in edges, so dragging a handle past the opposite one just flips
    // the rectangle instead of collapsing it
    let left = rect0.x, top = rect0.y;
    let right = rect0.x + rect0.w, bottom = rect0.y + rect0.h;
    if (handle === 'move') {
      left += dx; right += dx; top += dy; bottom += dy;
      const shiftX = Math.min(0, left) + Math.max(0, right - sensorW);
      const shiftY = Math.min(0, top) + Math.max(0, bottom - sensorH);
      left -= shiftX; right -= shiftX; top -= shiftY; bottom -= shiftY;
    } else {
      if (handle.includes('w')) left += dx;
      if (handle.includes('e')) right += dx;
      if (handle.includes('n')) top += dy;
      if (handle.includes('s')) bottom += dy;
    }
    const x0 = Math.max(0, Math.min(left, right));
    const x1 = Math.min(sensorW, Math.max(left, right));
    const y0 = Math.max(0, Math.min(top, bottom));
    const y1 = Math.min(sensorH, Math.max(top, bottom));
    return {x: x0, y: y0, w: Math.max(x1 - x0, ROI_MIN_PX),
            h: Math.max(y1 - y0, ROI_MIN_PX)};
  }

  // -------------------------------------------------------------- circles
  const markButton = container.querySelector('.cam-mark');
  const guessButton = container.querySelector('.cam-guess');
  let armed = null;      // 'marker' | 'guess' — next drag draws this circle
  let dragCenter = null;

  // the armed glow (button.armed in CSS) lights up in each circle's color
  markButton.style.setProperty('--mark', COLOR_MARKER);
  guessButton.style.setProperty('--mark', COLOR_GUESS);
  function setArmed(which) {
    if (which && editRect) {
      // the circles and the ROI rectangle cannot share the pointer: arming
      // one abandons the unapplied rectangle
      editRect = null;
      roiDrag = null;
      showRoiState();
      redrawOverlay();
      updateInfo();
    }
    armed = which;
    markButton.classList.toggle('armed', armed === 'marker');
    guessButton.classList.toggle('armed', armed === 'guess');
    overlay.style.cursor = armed ? 'crosshair' : '';
  }
  markButton.onclick = () => setArmed(armed === 'marker' ? null : 'marker');
  guessButton.onclick = () => setArmed(armed === 'guess' ? null : 'guess');
  container.querySelector('.cam-mark-clear').onclick = () => {
    marker = null;
    redrawOverlay();
    updateInfo();
  };
  container.querySelector('.cam-guess-clear').onclick = () => {
    guess = null; // the 'guess' broadcast event confirms for all viewers
    redrawOverlay();
    updateInfo();
    sendCommand(device.device_id, 'clear_guess')
      .catch((e) => { status.textContent = e.message; });
  };

  function toSensor(event) {
    const rect = overlay.getBoundingClientRect();
    return { x: (event.clientX - rect.left) / rect.width * sensorW,
             y: (event.clientY - rect.top) / rect.height * sensorH };
  }
  overlay.onpointerdown = (event) => {
    if (editRect) {
      // while the ROI rectangle is being edited it owns the pointer: a grab
      // on a handle resizes, inside moves, anywhere else draws a new one
      overlay.setPointerCapture(event.pointerId);
      const handle = roiHandleAt(event);
      const point = toSensor(event);
      if (handle) {
        roiDrag = {handle, start: point, rect0: {...editRect}};
      } else {
        editRect = {x: point.x, y: point.y, w: ROI_MIN_PX, h: ROI_MIN_PX};
        roiDrag = {handle: 'se', start: point, rect0: {...editRect}};
      }
      event.preventDefault();
      return;
    }
    if (!armed) return;
    overlay.setPointerCapture(event.pointerId);
    dragCenter = toSensor(event);
    event.preventDefault();
  };
  overlay.onpointermove = (event) => {
    if (editRect && !roiDrag) {
      const handle = roiHandleAt(event);
      overlay.style.cursor = handle ? ROI_CURSORS[handle] : 'crosshair';
      return;
    }
    if (roiDrag) {
      const point = toSensor(event);
      editRect = resizeEditRect(roiDrag.handle, roiDrag.rect0,
                                point.x - roiDrag.start.x,
                                point.y - roiDrag.start.y);
      redrawOverlay();
      updateInfo();
      return;
    }
    if (!dragCenter) return;
    const point = toSensor(event);
    const circle = { x: dragCenter.x, y: dragCenter.y,
                     r: Math.hypot(point.x - dragCenter.x, point.y - dragCenter.y) };
    if (armed === 'marker') marker = circle;
    else guess = circle;
    redrawOverlay();
    updateInfo();
  };
  overlay.onpointerup = () => {
    if (roiDrag) {
      roiDrag = null;
      return;
    }
    if (!dragCenter) return;
    const which = armed;
    dragCenter = null;
    setArmed(null);
    if (which === 'guess' && guess) {
      sendCommand(device.device_id, 'set_guess',
        { x_0: guess.x, y_0: guess.y, sigma: Math.max(guess.r, 1) })
        .catch((e) => { status.textContent = e.message; });
    }
  };

  // ------------------------------------------------------ clipboard export
  function composeFigure() {
    // one PNG laid out like the box: title + info line, row cross-section
    // strip, image with overlays, column strip — at on-screen resolution
    const width = overlay.width, height = overlay.height;
    const titleHeight = 40;
    const figure = document.createElement('canvas');
    figure.width = width + GAP + STRIP;
    figure.height = titleHeight + STRIP + GAP + height;
    const ctx = figure.getContext('2d');
    ctx.fillStyle = '#1a1d26';
    ctx.fillRect(0, 0, figure.width, figure.height);
    ctx.fillStyle = '#e6e8ef';
    ctx.font = 'bold 13px system-ui, sans-serif';
    ctx.fillText(`${device.label} — ${new Date().toLocaleString()}`, 4, 16);
    ctx.fillStyle = '#939aae';
    ctx.font = '11px Consolas, monospace';
    ctx.fillText(info.textContent, 4, 32);
    ctx.fillStyle = '#14161d';
    ctx.fillRect(0, titleHeight, width, STRIP);
    ctx.fillRect(width + GAP, titleHeight + STRIP + GAP, STRIP, height);
    ctx.drawImage(hCanvas, 0, titleHeight);
    ctx.drawImage(videoCanvas, 0, titleHeight + STRIP + GAP, width, height);
    ctx.drawImage(overlay, 0, titleHeight + STRIP + GAP);
    ctx.drawImage(vCanvas, width + GAP, titleHeight + STRIP + GAP);
    return figure;
  }

  function downloadBlob(blob) {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    link.download = `${device.device_id.replace(/[^\w-]+/g, '_')}_${stamp}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 5000);
  }

  container.querySelector('.cam-copy-figure').onclick = async () => {
    const blob = await new Promise((resolve) =>
      composeFigure().toBlob(resolve, 'image/png'));
    try {
      // image clipboard needs a secure context (localhost or https)
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
      status.textContent = 'figure copied to clipboard';
    } catch {
      downloadBlob(blob);
      status.textContent = 'clipboard unavailable here — saved as PNG file instead';
    }
  };

  container.querySelector('.cam-copy-fit').onclick = async () => {
    if (!fitParams) {
      status.textContent = 'no fit result to copy — enable the fit first';
      return;
    }
    const text = `${(fitParams.w_x * pixelMm).toFixed(4)}\t`
      + `${(fitParams.w_y * pixelMm).toFixed(4)}`;
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
    status.textContent = `copied: w_x, w_y = ${text.replace('\t', ', ')} mm`;
  };

  container.querySelector('.cam-record-fit').onclick = () => {
    if (!fitParams) {
      status.textContent = 'no fit result to record — enable the fit first';
      return;
    }
    const p = fitParams;
    const text = `${device.label}: x0=${p.x_0.toFixed(1)} px, y0=${p.y_0.toFixed(1)} px, `
      + `w_x=${(p.w_x * pixelMm).toFixed(3)} mm, w_y=${(p.w_y * pixelMm).toFixed(3)} mm, `
      + `theta=${p.angle.toFixed(2)} rad`;
    logEntry(text);
    status.textContent = 'recorded fit values to the log';
  };

  // ----------------------------------------------------------- the stream
  const stream = connectDeviceStream({
    deviceId: device.device_id,
    status,
    onEvent(event) {
      if (event.type === 'status') setPlaying(event.playing);
      else if (event.type === 'setting_applied') {
        showAppliedSetting(event.name, event.value, event.max);
      }
      else if (event.type === 'fit_status') {
        fitCheck.checked = event.enabled;
        if (!event.enabled) clearFitDisplay();
      } else if (event.type === 'fit') {
        if (event.success) {
          fitParams = event.params;
          fitCross = event.cross ?? null;
          fitReason = '';
        } else {
          fitParams = null;
          fitCross = null;
          fitReason = `fit: ${event.reason}`;
        }
        redrawOverlay();
        drawStrips();
        updateInfo();
      } else if (event.type === 'guess') {
        guess = event.guess
          ? { x: event.guess.x_0, y: event.guess.y_0, r: event.guess.sigma } : null;
        redrawOverlay();
        updateInfo();
      } else if (event.type === 'levels_status') {
        showLevelsEnabled(event.enabled);
      } else if (event.type === 'levels') {
        levelsPoints = event.points ?? [];
        drawLevels();
      } else if (event.type === 'roi') {
        applyRoiState(event.roi, event.sensor_shape, event.sensor_full);
        status.textContent = event.roi
          ? `ROI ${event.roi.width}×${event.roi.height} applied` : 'ROI cleared';
      } else if (event.type === 'brightness') {
        lastBrightness = event.value;
        paintBrightness();
      } else if (event.type === 'fit_threshold') {
        showThreshold(event.value);
      } else if (event.type === 'error') {
        status.textContent = `error: ${event.message}`;
      }
    },
    async onFrame(blob) {
      const bitmap = await createImageBitmap(blob);
      if (videoCanvas.width !== bitmap.width || videoCanvas.height !== bitmap.height) {
        videoCanvas.width = bitmap.width;
        videoCanvas.height = bitmap.height;
      }
      videoCanvas.getContext('2d').drawImage(bitmap, 0, 0);
      bitmap.close();
    },
    onReattach(describe) {
      setPlaying(describe.playing ?? true);
      // the crop may have changed while this viewer was offline
      applyRoiState(describe.roi, describe.sensor_shape, describe.sensor_full);
      fitCheck.checked = describe.fitting ?? false;
      if (!fitCheck.checked) clearFitDisplay();
      guess = describe.guess
        ? { x: describe.guess.x_0, y: describe.guess.y_0, r: describe.guess.sigma } : null;
      showThreshold(describe.fit_threshold ?? 0);
      levelsPoints = describe.levels_points ?? null;
      showLevelsEnabled(describe.levels ?? false);
      for (const setting of describe.settings ?? []) {
        showAppliedSetting(setting.name, setting.value);
      }
      redrawOverlay();
      updateInfo();
    },
  });

  showRoiState();
  levelsPoints = device.levels_points ?? null;
  showLevelsEnabled(device.levels ?? false);
  updateInfo();

  return function cleanup() {
    stream.close();
    resizeObserver.disconnect();
    if (levelsChart) levelsChart.destroy();
  };
}
