// Camera box: live video over WebSocket, play / pause / single-frame,
// exposure & gain inputs (commit on Enter or focus loss), Gaussian fit with
// ellipse overlay + cross-section plots, and two draggable circles:
//   marker ◯ — a persistent annotation (cyan), local to this viewer;
//   guess ◯  — the fit's initial guess (dashed green): center -> (x_0, y_0),
//              radius -> sigma; lives on the server so the fit can use it.
// Shared by every camera-like device type (dummy, Basler).
// Returns a cleanup function that closes the socket.

import { connectDeviceStream } from './stream.js';

const STRIP = 70;        // cross-section strip thickness, px
const GAP = 4;

const COLOR_DATA = 'rgb(70, 140, 220)';        // cross-section data
const COLOR_FIT_CURVE = 'rgb(255, 165, 40)';   // cross-section fit curve
const COLOR_ELLIPSE = 'rgba(255, 90, 90, 0.67)';
const COLOR_MARKER = 'rgba(0, 220, 220, 0.86)';
const COLOR_GUESS = 'rgba(110, 255, 110, 0.86)';
const COLOR_GRID = 'rgba(255, 255, 255, 0.28)';
const GRID_SPACING_MM = 1;

export function createCameraBox(device, container, sendCommand) {
  // fit coordinates are full-resolution sensor pixels; the video stream may
  // be downsampled, so all drawing is scaled from sensor_shape
  const [sensorH, sensorW] = device.sensor_shape ?? device.frame_shape;
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
        <label class="field" title="1 mm grid, centered on the image">
          <input type="checkbox" class="cam-grid"> grid</label>
        <label class="field"
               title="fit only frames at least this bright (counts above background); empty or 0 = fit every frame">
          trigger <input type="number" class="cam-trigger" min="0" placeholder="off"></label>
        <span class="cam-brightness readout"
              title="live beam brightness, counts above background: mean inside the guess circle's bounding square, or the 99th-percentile pixel when no guess circle is set"></span>
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

  function showAppliedSetting(name, value) {
    const entry = settingInputs[name];
    if (!entry) return;
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

  function place(canvas, x, y, w, h) {
    canvas.style.left = `${x}px`;
    canvas.style.top = `${y}px`;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
  }

  function layout() {
    const scale = Math.max(Math.min(
      (view.clientWidth - STRIP - GAP) / sensorW,
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
    // spacing in sensor px for GRID_SPACING_MM, centered on the image so an
    // intersection falls at (sensorW/2, sensorH/2) — a one-pixel offset from
    // dropping the fractional half-cell at either edge doesn't matter here.
    if (!pixelMm) return;
    const stepPx = GRID_SPACING_MM / pixelMm;
    ctx.strokeStyle = COLOR_GRID;
    ctx.setLineDash([]);
    ctx.lineWidth = 1;
    const cx = sensorW / 2, cy = sensorH / 2;
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
  function updateInfo() {
    const parts = [];
    if (fitReason) parts.push(fitReason);
    else if (fitParams) {
      const p = fitParams;
      parts.push(`x₀ = ${p.x_0.toFixed(1)} px, y₀ = ${p.y_0.toFixed(1)} px, `
        + `w_x = ${(p.w_x * pixelMm).toFixed(3)} mm, `
        + `w_y = ${(p.w_y * pixelMm).toFixed(3)} mm, `
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
    info.textContent = parts.join('   |   ');
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

  // -------------------------------------------------------------- circles
  const markButton = container.querySelector('.cam-mark');
  const guessButton = container.querySelector('.cam-guess');
  let armed = null;      // 'marker' | 'guess' — next drag draws this circle
  let dragCenter = null;

  // the armed glow (button.armed in CSS) lights up in each circle's color
  markButton.style.setProperty('--mark', COLOR_MARKER);
  guessButton.style.setProperty('--mark', COLOR_GUESS);
  function setArmed(which) {
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
    if (!armed) return;
    overlay.setPointerCapture(event.pointerId);
    dragCenter = toSensor(event);
    event.preventDefault();
  };
  overlay.onpointermove = (event) => {
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

  // ----------------------------------------------------------- the stream
  const stream = connectDeviceStream({
    deviceId: device.device_id,
    status,
    onEvent(event) {
      if (event.type === 'status') setPlaying(event.playing);
      else if (event.type === 'setting_applied') showAppliedSetting(event.name, event.value);
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
      fitCheck.checked = describe.fitting ?? false;
      if (!fitCheck.checked) clearFitDisplay();
      guess = describe.guess
        ? { x: describe.guess.x_0, y: describe.guess.y_0, r: describe.guess.sigma } : null;
      showThreshold(describe.fit_threshold ?? 0);
      for (const setting of describe.settings ?? []) {
        showAppliedSetting(setting.name, setting.value);
      }
      redrawOverlay();
      updateInfo();
    },
  });

  updateInfo();

  return function cleanup() {
    stream.close();
    resizeObserver.disconnect();
  };
}
