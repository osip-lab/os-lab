// Sidebands (NA) analysis extension for the PicoScope box: mark a region,
// the 0th-order mode, one of its sidebands and the 1st-order mode on the
// paused snapshot; the server fits six Lorentzians (same math as the offline
// script) and broadcasts mode spacing, linewidths and NA to every viewer.

const CURVE_COLOR = '#f0a030';

export const sidebandsExtension = {
  id: 'sidebands',
  label: 'sidebands (NA)',
  marks: {
    roi: { kind: 'region' },
    x0: { kind: 'point', color: '#52c46a' },
    xsb: { kind: 'point', color: '#9a9aa2' },
    x1: { kind: 'point', color: '#e05555' },
  },
  eventTypes: ['analysis_result'],

  create(host) {
    let result = null; // last 'analysis_result' event

    host.slot.innerHTML = `
      <label class="field">f_sb
        <input type="number" class="an-fsb" min="0"> <span class="unit">MHz</span></label>
      <button class="an-mark" data-mark="roi" title="drag a horizontal window over the region of interest">ROI</button>
      <button class="an-mark" data-mark="x0" title="click the 0th-order mode">0th</button>
      <button class="an-mark" data-mark="xsb" title="click one sideband of the 0th-order mode">sideband</button>
      <button class="an-mark" data-mark="x1" title="click the 1st-order mode">1st</button>
      <button class="an-fit" disabled>fit</button>
      <button class="an-clear" title="clear the marks and the fit overlay">✕</button>`;
    host.wireMarks();
    const fsbInput = host.slot.querySelector('.an-fsb');
    const fitButton = host.slot.querySelector('.an-fit');
    fsbInput.value = host.device.sideband_freq_default_mhz ?? 25;

    fitButton.onclick = () => {
      host.setResult('fitting…');
      host.send('analyze_sidebands', {
        channel: host.box.channel(),
        t_min: host.marks.roi[0],
        t_max: host.marks.roi[1],
        x0: host.marks.x0,
        x_sb: host.marks.xsb,
        x1: host.marks.x1,
        f_sb_mhz: parseFloat(fsbInput.value) || null,
      }).catch((error) => host.setResult(error.message));
    };

    host.slot.querySelector('.an-clear').onclick = () => {
      result = null;
      host.clearMarks();
    };

    return {
      sync() {
        fitButton.disabled = !(host.ready() && host.marks.roi
          && host.marks.x0 != null && host.marks.xsb != null
          && host.marks.x1 != null);
      },
      hasResult: () => !!result?.results,
      resultText() {
        if (!result?.results) return '';
        const r = result.results;
        const na = r.NA != null ? r.NA.toFixed(4)
          : `unavailable${r.na_error ? ` (${r.na_error})` : ''}`;
        return `mode spacing = ${r.mode_spacing_MHz.toFixed(3)} MHz | `
          + `HWHM₀ = ${r.linewidth_0_HWHM_MHz.toFixed(3)} MHz | `
          + `HWHM₁ = ${r.linewidth_1_HWHM_MHz.toFixed(3)} MHz | NA = ${na}`;
      },
      copy() {
        const r = result?.results;
        if (!r) return null;
        return {
          text: `${r.mode_spacing_MHz.toFixed(4)}\t`
            + `${r.linewidth_0_HWHM_MHz.toFixed(4)}\t`
            + `${r.linewidth_1_HWHM_MHz.toFixed(4)}\t`
            + `${r.NA != null ? r.NA.toFixed(4) : 'N/A'}`,
          note: 'copied: mode spacing, HWHM₀, HWHM₁, NA',
        };
      },
      draw(u, helpers) {
        helpers.polyline(result?.curve, CURVE_COLOR);
      },
      onEvent(event) {
        result = event;
        return true; // a broadcast fit result always makes this mode visible
      },
      restore(describe) {
        result = describe.analysis ?? null;
        return !!result;
      },
      clear() {
        result = null;
      },
    };
  },
};
