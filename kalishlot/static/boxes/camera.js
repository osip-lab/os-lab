// Camera box: live video over WebSocket plus play / pause / single-frame
// controls. Shared by every camera-like device type (dummy, Basler).
// Returns a cleanup function that closes the socket.

export function createCameraBox(device, container, sendCommand) {
  container.innerHTML = `
    <div class="cam-controls" style="display:flex; gap:6px; align-items:center; flex-wrap:wrap;">
      <button data-command="play">play</button>
      <button data-command="pause">pause</button>
      <button data-command="snap">single frame</button>
      <span class="cam-status" style="font-size:12px; opacity:0.8;"></span>
    </div>
    <div class="cam-view" style="flex:1; min-height:0; display:flex;">
      <canvas class="cam-canvas"
              style="flex:1; min-width:0; object-fit:contain; background:#111;"></canvas>
    </div>`;

  const canvas = container.querySelector('.cam-canvas');
  const context = canvas.getContext('2d');
  const status = container.querySelector('.cam-status');
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

  // ----------------------------------------------------------- the stream
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(
    `${protocol}://${location.host}/ws/devices/${encodeURIComponent(device.device_id)}`);
  socket.binaryType = 'blob';
  let closedByUs = false;

  socket.onmessage = async (message) => {
    if (typeof message.data === 'string') {
      const event = JSON.parse(message.data);
      if (event.type === 'status') setPlaying(event.playing);
      // more event types (settings, fit results) handled in later phases
      return;
    }
    const bitmap = await createImageBitmap(message.data);
    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
    }
    context.drawImage(bitmap, 0, 0);
    bitmap.close();
  };
  socket.onclose = () => {
    if (!closedByUs) status.textContent = 'stream disconnected';
  };

  return function cleanup() {
    closedByUs = true;
    socket.close();
  };
}
