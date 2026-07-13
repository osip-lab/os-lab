// Shared WebSocket plumbing for device boxes: connects to the device's
// stream and RECONNECTS automatically (with backoff) when the socket drops
// for any reason other than the box being closed. A lab dashboard stays
// open for hours — a transient drop (network blip, server restart, machine
// sleep) must heal by itself, not leave a dead box.
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

export function connectDeviceStream(options) {
  const { deviceId, status, onEvent, onFrame, onReattach } = options;
  let socket = null;
  let closedByUs = false;
  let retryMs = RETRY_START_MS;
  let retryTimer = null;
  let everConnected = false;

  async function fetchDescribe() {
    // returns the device's fresh describe(), null if the device is gone,
    // and throws when the server itself is unreachable
    const response = await fetch('/api/devices');
    if (!response.ok) throw new Error(response.statusText);
    const open = await response.json();
    return open.find((d) => d.device_id === deviceId) ?? null;
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
