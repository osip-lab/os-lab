// The play/pause toggle shared by the streaming boxes: one button that names
// the action a click takes, lit as a lamp for the current state — green while
// live (click to pause), amber while frozen (click to play).

export function showPlayToggle(button, playing) {
  button.textContent = playing ? '❚❚ pause' : '▶ play';
  button.title = playing ? 'live — click to pause' : 'paused — click to play';
  button.classList.toggle('live', playing);
  button.classList.toggle('frozen', !playing);
}
