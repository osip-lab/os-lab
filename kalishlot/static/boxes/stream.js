// Shared WebSocket plumbing for device boxes: connects to the device's
// stream and RECONNECTS automatically (with backoff) when the socket drops
// for any reason other than the box being closed. A lab dashboard stays
// open for hours — a transient drop (network blip, server restart, machine
// sleep) must heal by itself, not leave a dead box.
// A device lent to a standalone script (close code 4005, see "loans" in
// server.py) is waited for the same way: the box says who has it, offers a
// "reconnect" button for a borrower that never gave it back, and re-attaches
// by itself once the device is returned.
//
//   const stream = connectDeviceStream({
//     deviceId, status,            // status: element for state text
//     onEvent(event) {...},        // JSON events
//     onFrame(blob) {...},         // binary messages (optional)
//     onReattach(describe) {...},  // fresh describe() after a reconnect
//   });
//   ... stream.close() in the box cleanup.

const RETRY_START_MS = 2000;
const RETRY_MAX_MS = 15000;
const LOAN_POLL_MS = 1000;

export function connectDeviceStream(options) {
  const { deviceId, status, onEvent, onFrame, onReattach } = options;
  let socket = null;
  let closedByUs = false;
  let retryMs = RETRY_START_MS;
  let retryTimer = null;
  let everConnected = false;
  let loanShown = undefined; // the borrower the status line names, if any

  async function fetchDescribe() {
    // returns the device's fresh describe(), null if the device is gone,
    // and throws when the server itself is unreachable
    const response = await fetch('/api/devices');
    if (!response.ok) throw new Error(response.statusText);
    const open = await response.json();
    return open.find((d) => d.device_id === deviceId) ?? null;
  }

  async function fetchLoan() {
    const response = await fetch('/api/loans');
    if (!response.ok) throw new Error(response.statusText);
    const loans = await response.json();
    return loans.find((loan) => loan.device_id === deviceId) ?? null;
  }

  function showLoan(loan) {
    loanShown = loan?.borrower ?? null;
    if (!status) return;
    status.textContent =
      `on loan to ${loan?.borrower ?? 'a script'} — reattaches when it is returned `;
    const button = document.createElement('button');
    button.textContent = 'reconnect';
    button.title = 'take the device back now (for a script that never returned it)';
    button.onclick = async () => {
      button.disabled = true;
      const response = await fetch(
        `/api/devices/${encodeURIComponent(deviceId)}/return`, { method: 'POST' })
        .catch(() => null);
      if (response?.ok) { waitForReturn(); return; }
      const detail = response ? (await response.json()).detail : 'server unreachable';
      status.textContent = `reconnect failed: ${detail} `;
      status.appendChild(button);
      button.disabled = false;
    };
    status.appendChild(button);
  }

  // poll until the device is back (reattach), still on loan (keep waiting)
  // or gone altogether (e.g. the idle watchdog closed everything meanwhile)
  async function waitForReturn() {
    if (closedByUs) return;
    let describe, loan;
    try {
      [describe, loan] = await Promise.all([fetchDescribe(), fetchLoan()]);
    } catch {
      retryTimer = setTimeout(waitForReturn, RETRY_MAX_MS);
      return;
    }
    if (describe) { loanShown = undefined; connect(); return; }
    if (loan) {
      if (loan.borrower !== loanShown) showLoan(loan);
      retryTimer = setTimeout(waitForReturn, LOAN_POLL_MS);
      return;
    }
    if (status) status.textContent =
      'device is no longer open on the server — close this box and re-add it';
  }

  function connect() {
    const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(
      `${protocol}://${location.host}/ws/devices/${encodeURIComponent(deviceId)}`);
    socket.onopen = async () => {
      retryMs = RETRY_START_MS;
      if (everConnected && onReattach) {
        // pick up state changes that happened while we were disconnected
        try {
          const describe = await fetchDescribe();
          if (describe) onReattach(describe);
        } catch { /* box may be stale until the next event */ }
      }
      if (everConnected && status) status.textContent = '';
      everConnected = true;
    };
    socket.onmessage = (message) => {
      if (typeof message.data === 'string') onEvent(JSON.parse(message.data));
      else if (onFrame) onFrame(message.data);
    };
    socket.onclose = (event) => {
      if (closedByUs) return;
      if (event.code === 4005) {
        // lent to a script (e.g. mode_video_capture.py): wait for it back
        showLoan(null);
        waitForReturn();
        return;
      }
      if (event.code === 4004) {
        // the device was closed on the server (e.g. by another viewer):
        // nothing to reconnect to
        if (status) status.textContent = 'device was closed on the server';
        return;
      }
      if (status) status.textContent = 'connection lost — reconnecting…';
      retryTimer = setTimeout(retry, retryMs);
      retryMs = Math.min(retryMs * 2, RETRY_MAX_MS);
    };
  }

  async function retry() {
    // before reconnecting, ask whether the device still exists — after a
    // server restart it won't, and retrying a nonexistent device forever
    // would only hammer the server with rejected handshakes
    let describe;
    try {
      describe = await fetchDescribe();
    } catch {
      // the server itself is unreachable: back off and try again later
      retryTimer = setTimeout(retry, retryMs);
      retryMs = Math.min(retryMs * 2, RETRY_MAX_MS);
      return;
    }
    if (describe === null) {
      if (status) status.textContent =
        'device is no longer open on the server — close this box and re-add it';
      return;
    }
    connect();
  }
  connect();

  return {
    close() {
      closedByUs = true;
      clearTimeout(retryTimer);
      if (socket) socket.close();
    },
  };
}
