// Idle-shutdown warning. The server closes every device after a timeout with
// no user activity (see server.py's idle watchdog) so a dashboard forgotten at
// the end of the day does not leave the hardware running all night. This
// module shows the last-chance warning: a modal with a countdown and a beep
// loud enough to notice from the optics table, and a button that restarts the
// server's countdown.
//
//   initIdle({ onDisconnected(deviceIds) {...} });

const RETRY_START_MS = 2000;
const RETRY_MAX_MS = 15000;

const BEEP_PERIOD_MS = 1000;
const BEEP_HZ = 880;
const BEEP_S = 0.12;
const BEEP_GAIN = 0.18;

let overlayEl = null;
let countdownEl = null;
let noteEl = null;
let countdownTimer = null;
let beepTimer = null;
let deadline = 0;

// ------------------------------------------------------------------- sound
// One AudioContext for the page. Browsers start it suspended until the user
// has interacted with the page, so we also resume it on the first gesture —
// by the time a warning fires (an hour in) that has long since happened.
let audio = null;

function audioContext() {
  if (!audio) {
    const Ctor = window.AudioContext ?? window.webkitAudioContext;
    if (!Ctor) return null;
    audio = new Ctor();
  }
  if (audio.state === 'suspended') audio.resume().catch(() => {});
  return audio;
}

function beep() {
  const ctx = audioContext();
  if (!ctx) return;
  const now = ctx.currentTime;
  const oscillator = ctx.createOscillator();
  const gain = ctx.createGain();
  oscillator.type = 'sine';
  oscillator.frequency.value = BEEP_HZ;
  // ramp the envelope instead of switching it: a hard start/stop clicks
  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(BEEP_GAIN, now + 0.01);
  gain.gain.linearRampToValueAtTime(0, now + BEEP_S);
  oscillator.connect(gain).connect(ctx.destination);
  oscillator.start(now);
  oscillator.stop(now + BEEP_S + 0.02);
}

// ------------------------------------------------------------- the overlay
function buildOverlay() {
  overlayEl = document.createElement('div');
  overlayEl.className = 'idle-backdrop';
  overlayEl.hidden = true;
  overlayEl.innerHTML = `
    <div class="idle-modal">
      <h2>still there?</h2>
      <p class="idle-note"></p>
      <div class="idle-countdown"></div>
      <button class="idle-keep">keep devices running</button>
    </div>`;
  document.body.appendChild(overlayEl);
  countdownEl = overlayEl.querySelector('.idle-countdown');
  noteEl = overlayEl.querySelector('.idle-note');
  overlayEl.querySelector('.idle-keep').onclick = dismiss;
  document.addEventListener('keydown', (event) => {
    if (!overlayEl.hidden && (event.key === 'Escape' || event.key === 'Enter')) dismiss();
  });
}

function stopTimers() {
  clearInterval(countdownTimer);
  clearInterval(beepTimer);
  countdownTimer = null;
  beepTimer = null;
}

function hideOverlay() {
  stopTimers();
  overlayEl.hidden = true;
}

function showWarning(graceS, timeoutS) {
  deadline = Date.now() + graceS * 1000;
  const minutes = Math.round(timeoutS / 60);
  noteEl.textContent =
    `No activity for ${minutes >= 1 ? `${minutes} min` : `${Math.round(timeoutS)} s`}`
    + ' — all devices are about to be disconnected.';
  tick();
  overlayEl.hidden = false;
  overlayEl.querySelector('.idle-keep').focus();
  stopTimers();
  countdownTimer = setInterval(tick, 200);
  beep();
  beepTimer = setInterval(beep, BEEP_PERIOD_MS);
}

function tick() {
  const left = Math.max(0, (deadline - Date.now()) / 1000);
  countdownEl.textContent = `${left.toFixed(1)} s`;
}

// The server owns the countdown; this only tells it we are here. The warning
// is hidden on the resulting 'idle_clear' broadcast so every viewer's overlay
// goes away together — whichever of them clicked.
function dismiss() {
  hideOverlay();
  fetch('/api/idle/dismiss', { method: 'POST' }).catch(() => {});
}

// -------------------------------------------------------------- the socket
export function initIdle({ onDisconnected } = {}) {
  buildOverlay();
  const unlock = () => audioContext();
  document.addEventListener('pointerdown', unlock, { once: true });
  document.addEventListener('keydown', unlock, { once: true });

  let retryMs = RETRY_START_MS;
  (function connect() {
    const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${protocol}://${location.host}/ws/idle`);
    socket.onopen = () => {
      retryMs = RETRY_START_MS;
      // a warning raised while this viewer was away (page reload, dropped
      // socket) is still counting down on the server — join it
      fetch('/api/idle')
        .then((response) => response.json())
        .then((state) => {
          if (state.warning_active) showWarning(state.grace_left_s, state.timeout_s);
        })
        .catch(() => {});
    };
    socket.onmessage = (message) => {
      const event = JSON.parse(message.data);
      if (event.type === 'idle_warning') showWarning(event.grace_s, event.timeout_s);
      else if (event.type === 'idle_clear') hideOverlay();
      else if (event.type === 'idle_disconnected') {
        hideOverlay();
        onDisconnected?.(event.devices ?? []);
      }
    };
    socket.onclose = () => {
      // a dropped socket must not leave a stale warning beeping forever
      hideOverlay();
      setTimeout(connect, retryMs);
      retryMs = Math.min(retryMs * 2, RETRY_MAX_MS);
    };
  })();
}
