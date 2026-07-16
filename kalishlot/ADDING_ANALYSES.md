# Kalishlot — adding an analysis extension

Analyses (fits and derived quantities on a device's paused data) are NOT
written into the device modules. They are **extensions**: one small JS module
+ one small Python module + two registry lines. The PicoScope's two analyses
(sidebands → NA, pairs → df/FSR) are the reference implementations. Adding
an analysis never requires touching the device box or the adapter.

## Is it an extension? (decide this first)

A feature is an analysis extension if it consumes only the host's declared
services: a toolbar slot, marks on the chart, overlay drawing, the paused
full-resolution snapshot, event broadcast and reattach state. If it must
change the device's data/frame pipeline or the box layout, it is a mode of
the device module instead — the camera's live Gaussian fit is the canonical
example: it runs continuously inside the frame pipeline and owns part of the
camera layout, so it is (correctly) built into the camera module and
variants of it belong there, not here.

## How it works

```
box (picoscope.js)              extension host                extensions
  transport/channels/chart -->  static/boxes/extensions/ -->  scope_sidebands.js + .py
  one .scope-analysis row       host.js: dropdown, marks,     scope_pairs.js + .py
  + snapshot on pause           overlay, result/copy,         (your new analysis)
                                lifecycle, event routing
```

- **Marks are per-viewer**, collected client-side and sent in one command.
- **Fits run server-side** on the pause-time full-resolution snapshot, with
  the same math module the offline scripts use (`pico_scope/mode_analysis.py`
  — never duplicate math in JS), and results are **broadcast** so every
  viewer sees them; `describe()` carries the last result for reattach.
- Modes are exclusive per viewer (the dropdown), cleared on `play`.
- The NA interpolators (slow cavity-design simulation) are pre-warmed in a
  background thread the moment an analysis is constructed
  (`adapters/analyses/util.py: warm_na_interpolators`), so the first fit
  doesn't pay the build time; all access goes through the locked
  `na_interpolators()` — never call `get_na_interpolators()` directly from
  an analysis.

## Step 1 — server side: `kalishlot/adapters/analyses/<name>.py`

One class; the host is the device adapter, providing `snapshot_region(args)`
(guard-checked, decimated fit input from the paused snapshot) and
`emit(event)` (broadcast to all viewers):

```python
from .util import finite_or_none, na_interpolators, warm_na_interpolators
from pico_scope.mode_analysis import ...   # the shared math


class MyAnalysis:
    COMMANDS = ('my_fit',)          # command names this analysis owns

    def __init__(self, host):
        self.host = host
        self._last = None           # last result event, for reattach
        warm_na_interpolators()     # only if the analysis needs the NA map

    def describe_state(self):
        return {'analysis_my': self._last}   # merged into describe()

    def reset(self):                # called on play: snapshot is discarded
        self._last = None

    def command(self, name, args):  # return None if the command isn't yours
        if name != 'my_fit':
            return None
        x, y, t_min, t_max = self.host.snapshot_region(args)
        ...  # fit via mode_analysis; raise ValueError -> HTTP 400 in the box
        event = {'type': 'analysis_my', ...}
        self._last = event
        self.host.emit(event)
        return {'ok': True}
```

Register it in the adapter's `__init__` (`adapters/picoscope.py`):

```python
self.analyses = [SidebandsAnalysis(self), PairsAnalysis(self), MyAnalysis(self)]
```

That is all — command dispatch, describe() merging and play-reset iterate
over `self.analyses` generically.

## Step 2 — client side: `kalishlot/static/boxes/extensions/<name>.js`

One exported object. The host generates nothing you don't declare:

```js
export const myExtension = {
  id: 'my',
  label: 'my analysis (unit)',          // dropdown entry
  marks: {                              // the host runs the marking engine:
    roi: { kind: 'region' },            //   region = drag on the chart
    peak: { kind: 'point', color: '#52c46a' },  // point = one click,
  },                                    //   drawn as a dashed line
  eventTypes: ['analysis_my'],          // routed to onEvent below

  create(host) {
    let result = null;
    // your controls; .an-mark buttons are wired by host.wireMarks()
    host.slot.innerHTML = `
      <button class="an-mark" data-mark="roi" title="...">ROI</button>
      <button class="an-mark" data-mark="peak" title="...">peak</button>
      <button class="my-fit" disabled>fit</button>`;
    host.wireMarks();
    const fitButton = host.slot.querySelector('.my-fit');
    fitButton.onclick = () => {
      host.setResult('fitting…');
      host.send('my_fit', { channel: host.box.channel(),
                            t_min: host.marks.roi[0], t_max: host.marks.roi[1],
                            x: host.marks.peak })
        .catch((error) => host.setResult(error.message));
    };
    return {
      sync() {          // any state change: set your buttons' disabled state
        fitButton.disabled = !(host.ready() && host.marks.roi
          && host.marks.peak != null);
      },
      hasResult: () => !!result,        // enables the shared copy button
      resultText: () => result ? `... = ${result.value}` : '',
      copy: () => result ? { text: `${result.value}`, note: 'copied: ...' } : null,
      draw(u, helpers) {                // overlay beyond the generic marks
        helpers.polyline(result?.curve, '#f0a030');
      },
      onEvent(event) { result = event; return true; },  // true -> auto-select
      restore(describe) { result = describe.analysis_my ?? null; return !!result; },
      clear() { result = null; },       // on play
    };
  },
};
```

Host services available in `create(host)`: `device` (the describe dict),
`box` (box-specific extras — for the scope, `channel()`), `slot` (your
subgroup element), `marks` (live values: region `[t0, t1]`, point `t`, or
null), `send(name, args)`, `ready()` (this mode active and stream paused),
`setResult(text)`, `wireMarks()`, `clearMarks()`, `redraw()`, `sync()`.

Register it in `static/boxes/extensions/registry.js` (keep the imports
static — no `import()` — so the page render still proves every module
parses):

```js
export const ANALYSIS_EXTENSIONS = {
  picoscope: [sidebandsExtension, pairsExtension, myExtension],
};
```

## Step 3 — accumulating analyses (server-owned state)

If the analysis accumulates across several commands (the pairs mode: one
`fit_pair` per marked pair, plus `undo_pair`/`clear_pairs`), the SERVER
class owns the growing list and re-broadcasts the FULL state after every
change — viewers render whatever the last event holds; nothing is
client-local except unsent marks. See `scope_pairs.py` / `scope_pairs.js`.

## Step 4 — giving another device box analyses

The host is device-agnostic. A box embeds it with ~20 lines (see
`picoscope.js`): a `.toolbar` row containing an `.an-mode` select (an `off`
option) and optionally an `.an-common` element shown while a mode is active;
`createAnalysisHost({row, device, sendCommand, extensions, isPlaying, note,
box})`; then forward: `hooks: {draw: [(u) => host.draw(u)]}` on the uPlot,
`host.attachChart(chart)`, `host.setPlaying(...)` from its play/pause,
`host.onEvent(event)` in the stream handler (returns true if consumed) and
`host.onReattach(describe)`. Server side, give the adapter the same
`self.analyses` composition as `adapters/picoscope.py` (a pause-time
snapshot + `snapshot_region()` are prerequisites).

## Testing without hardware

`python kalishlot/smoke_test.py` covers the server and static page. For the
UI, render the box from a faked device descriptor in a scratch HTML harness
(stub `window.WebSocket`, import the box module, pass a describe-shaped
object with a stored result for your event type) and screenshot it with a
headless browser — reattach/restore and the toolbar render without a scope.
For the math, drive the server class directly with a fake host
(`snapshot_region` returning synthetic data, `emit` collecting events).
