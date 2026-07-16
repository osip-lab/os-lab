// Analysis-extension host: the engine shared by every analysis mode, so a
// device box stays free of analysis code and a new analysis is one extension
// module + a registry line (see kalishlot/ADDING_ANALYSES.md).
//
// The host owns the mode dropdown (extensions are exclusive, per-viewer),
// one hidden .subgroup slot per extension, the arm-a-button-then-mark engine
// on the chart (regions are drags, points are clicks), the generic overlay
// pass (region shading + dashed mark lines, then the extension's draw), the
// shared result line and copy button, and the lifecycle: marks and results
// are cleared on play, events are routed to the owning extension, and
// reattach restores each extension from describe().
//
// The embedding box provides the toolbar row (with an .an-mode select and an
// optional .an-common element shown while a mode is active), tells the host
// about the chart, play/pause changes, events and reattaches, and calls
// host.draw(u) from the chart's draw hook.

export function createAnalysisHost({ row, device, sendCommand, extensions,
                                     isPlaying, note, box }) {
  const noop = () => {};
  if (!extensions?.length) {
    row.style.display = 'none';
    return { attachChart: noop, draw: noop, setPlaying: noop,
             onEvent: () => false, onReattach: noop };
  }

  const modeSelect = row.querySelector('.an-mode');
  const commonEl = row.querySelector('.an-common');

  let chart = null;     // attached after the box constructs it
  let activeId = 'off';
  let armedMark = null; // name of an armed mark of the ACTIVE extension
  let dragStart = null; // region drag in progress: the starting x value
  let dragKey = null;   // which region mark the current drag writes

  const redraw = () => { if (chart) chart.redraw(); };

  // ------------------------------------------------ extension instantiation
  const instances = []; // { def, marks, markButtons, slot, hooks }

  function makeExtensionApi(instance) {
    return {
      device,
      box,
      slot: instance.slot,
      marks: instance.marks,
      send: (name, args) => sendCommand(device.device_id, name, args),
      ready: () => activeId === instance.def.id && !isPlaying(),
      setResult: (text) => { resultSpan.textContent = text; },
      redraw,
      sync: () => syncUi(),
      // wire the .an-mark buttons the extension put in its slot
      wireMarks() {
        for (const button of instance.slot.querySelectorAll('.an-mark')) {
          const name = button.dataset.mark;
          instance.markButtons[name] = button;
          // the armed glow (button.armed in CSS) lights up in the mark color
          button.style.setProperty('--mark',
            instance.def.marks[name]?.color ?? '#d9a13c');
          button.onclick = () => setArm(armedMark === name ? null : name);
        }
      },
      clearMarks() {
        for (const name of Object.keys(instance.marks)) {
          instance.marks[name] = null;
        }
        setArm(null);
        redraw();
        syncUi();
      },
    };
  }

  for (const def of extensions) {
    const option = document.createElement('option');
    option.value = def.id;
    option.textContent = def.label;
    modeSelect.appendChild(option);

    const marks = {};
    for (const name of Object.keys(def.marks ?? {})) marks[name] = null;
    const slot = document.createElement('span');
    slot.className = 'subgroup';
    slot.style.display = 'none';
    row.appendChild(slot);

    const instance = { def, marks, markButtons: {}, slot, hooks: null };
    instance.hooks = def.create(makeExtensionApi(instance));
    instances.push(instance);
  }

  const copyButton = document.createElement('button');
  copyButton.className = 'an-copy';
  copyButton.textContent = 'copy';
  copyButton.title = 'copy the results (tab-separated)';
  copyButton.disabled = true;
  copyButton.style.display = 'none';
  row.appendChild(copyButton);

  const resultSpan = document.createElement('span');
  resultSpan.className = 'an-result readout';
  row.appendChild(resultSpan);

  const active = () =>
    instances.find((instance) => instance.def.id === activeId) ?? null;

  // ------------------------------------------------------- mode and arming
  function setActive(id) {
    activeId = id;
    modeSelect.value = id;
    setArm(null);
    redraw();
    syncUi();
  }
  modeSelect.onchange = () => setActive(modeSelect.value);

  function setArm(name) {
    armedMark = name;
    for (const instance of instances) {
      for (const [n, button] of Object.entries(instance.markButtons)) {
        button.classList.toggle('armed',
          instance.def.id === activeId && n === name);
      }
    }
    if (chart) chart.over.style.cursor = name ? 'crosshair' : '';
  }

  function syncUi() {
    const instance = active();
    if (commonEl) commonEl.style.display = instance ? 'flex' : 'none';
    for (const i of instances) {
      i.slot.style.display = i === instance ? 'flex' : 'none';
    }
    copyButton.style.display = instance ? '' : 'none';
    const ready = !!instance && !isPlaying();
    for (const i of instances) {
      for (const button of Object.values(i.markButtons)) {
        button.disabled = !ready;
      }
      i.hooks.sync?.();
    }
    if (!instance) {
      resultSpan.textContent = '';
      return;
    }
    copyButton.disabled = !instance.hooks.hasResult();
    resultSpan.textContent = isPlaying()
      ? 'pause the stream to analyze' : instance.hooks.resultText();
  }

  copyButton.onclick = async () => {
    const payload = active()?.hooks.copy();
    if (!payload) return;
    try {
      await navigator.clipboard.writeText(payload.text);
    } catch {
      // http from another computer: fall back to the legacy copy command
      const scratch = document.createElement('textarea');
      scratch.value = payload.text;
      document.body.appendChild(scratch);
      scratch.select();
      document.execCommand('copy');
      scratch.remove();
    }
    if (note) note(payload.note);
  };

  // ----------------------------------------- marking on the chart overlay
  // regions (kind 'region') are horizontal drags, points single clicks —
  // the arm-a-button-then-act pattern shared with the camera circles
  function attachChart(newChart) {
    chart = newChart;
    chart.over.addEventListener('pointerdown', (event) => {
      if (!armedMark || isPlaying()) return;
      const instance = active();
      if (!instance) return;
      chart.over.setPointerCapture(event.pointerId);
      if (instance.def.marks[armedMark].kind === 'region') {
        dragKey = armedMark;
        dragStart = chart.posToVal(event.offsetX, 'x');
      }
      event.preventDefault();
    });
    chart.over.addEventListener('pointermove', (event) => {
      if (dragStart == null) return;
      const t = chart.posToVal(event.offsetX, 'x');
      active().marks[dragKey] =
        [Math.min(dragStart, t), Math.max(dragStart, t)];
      redraw();
    });
    chart.over.addEventListener('pointerup', (event) => {
      if (!armedMark) return;
      const instance = active();
      if (!instance) return;
      if (instance.def.marks[armedMark].kind === 'region') {
        dragStart = null;
        dragKey = null;
        const region = instance.marks[armedMark];
        if (!region || region[1] - region[0] <= 0) {
          instance.marks[armedMark] = null; // a click without a drag is no region
        }
      } else {
        instance.marks[armedMark] = chart.posToVal(event.offsetX, 'x');
      }
      setArm(null);
      redraw();
      syncUi();
    });
  }

  // ------------------------------------------------------- overlay drawing
  // painted straight onto the uPlot canvas after each redraw; a fit curve
  // has its own dense time axis, so it cannot live in the shared-x data
  // arrays as a series. Only the active extension's overlay is drawn.
  function draw(u) {
    const instance = active();
    if (!instance) return;
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
    // generic pass: shade the region marks, dash the point marks
    for (const [name, def] of Object.entries(instance.def.marks)) {
      if (def.kind === 'region' && instance.marks[name]) {
        shadeRegion(instance.marks[name]);
      }
    }
    ctx.lineWidth = devicePixelRatio;
    ctx.setLineDash([6 * devicePixelRatio, 4 * devicePixelRatio]);
    for (const [name, def] of Object.entries(instance.def.marks)) {
      if (def.kind === 'region' || instance.marks[name] == null) continue;
      ctx.strokeStyle = def.color;
      verticalLine(instance.marks[name]);
    }
    ctx.setLineDash([]);
    instance.hooks.draw?.(u, { ctx, xPx, shadeRegion, verticalLine, polyline });
    ctx.restore();
  }

  // -------------------------------------------------------------- lifecycle
  function setPlaying(playing) {
    if (playing) {
      // the marks and the fit overlays belong to the discarded snapshot
      for (const instance of instances) {
        for (const name of Object.keys(instance.marks)) {
          instance.marks[name] = null;
        }
        instance.hooks.clear?.();
      }
      dragStart = null;
      dragKey = null;
      redraw();
    }
    syncUi();
  }

  function onEvent(event) {
    for (const instance of instances) {
      if (!instance.def.eventTypes.includes(event.type)) continue;
      const hasResult = instance.hooks.onEvent(event);
      // another viewer may have run the fit; make it visible here too
      if (activeId === 'off' && hasResult) setActive(instance.def.id);
      redraw();
      syncUi();
      return true;
    }
    return false;
  }

  function onReattach(describe) {
    // restore every extension; activate the one a result was left in
    let select = null;
    for (const instance of instances) {
      if (instance.hooks.restore?.(describe)) select = instance.def.id;
    }
    if (activeId === 'off' && select) {
      setActive(select);
    } else {
      redraw();
      syncUi();
    }
  }

  // start in the mode whose result an earlier session left behind
  onReattach(device);

  return { attachChart, draw, setPlaying, onEvent, onReattach };
}
