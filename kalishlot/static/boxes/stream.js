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
    const response = await fetch('/api/devices');
    if (!response.ok) throw new Error(response.statusText);
    const open = await response.json();
    const device = open.find((d) => d.device_id === deviceId);
    if (!device) throw new Error('device is no longer open on the server');
    return device;
  }

  function connect() {
    const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(
      `${protocol}://${location.host}/ws/devices/${encodeURIComponent(deviceId)}`);
    socket.onopen = async () => {
      retryMs = RETRY_START_MS;
      if (everConnected && onReattach) {
        // pick up state changes that happened while we were disconnected
        try { onReattach(await fetchDescribe()); } catch { /* box may be stale */ }
      }
      if (everConnected && status) status.textContent = '';
      everConnected = true;
    };
    socket.onmessage = (message) => {
      if (typeof message.data === 'string') onEvent(JSON.parse(message.data));
      else if (onFrame) onFrame(message.data);
    };
    socket.onclose = () => {
      if (closedByUs) return;
      if (status) status.textContent = 'connection lost — reconnecting…';
      retryTimer = setTimeout(connect, retryMs);
      retryMs = Math.min(retryMs * 2, RETRY_MAX_MS);
    };
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
