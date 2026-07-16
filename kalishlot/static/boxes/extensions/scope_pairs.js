// Pairs (df/FSR) analysis extension for the PicoScope box: mark and fit one
// Lorentzian pair at a time on the paused snapshot; the fitted pairs
// accumulate on the SERVER (all viewers share them, so undo/clear are
// commands and every change re-broadcasts the full 'analysis_pairs' state)
// and with two or more pairs the df/FSR -> NA results appear.

const PAIR_COLORS = ['#e05555', '#52c46a', '#4a9eda', '#c95fd0',
                     '#3ec8c8', '#cfcf52'];

export const pairsExtension = {
  id: 'pairs',
  label: 'pairs (df/FSR)',
  marks: {
    proi: { kind: 'region' },
    p1: { kind: 'point', color: '#52c46a' },
    p2: { kind: 'point', color: '#e05555' },
  },
  eventTypes: ['analysis_pairs'],

  create(host) {
    let state = null; // last 'analysis_pairs' event

    host.slot.innerHTML = `
      <button class="an-mark" data-mark="proi" title="drag a horizontal window over one pair (both its peaks)">pair ROI</button>
      <button class="an-mark" data-mark="p1" title="click the pair's first peak (the fundamental mode)">peak 1</button>
      <button class="an-mark" data-mark="p2" title="click the pair's second peak (the higher-order mode)">peak 2</button>
      <button class="an-pair-fit" disabled>fit pair</button>
      <button class="an-pair-undo" title="remove the last fitted pair">undo</button>
      <button class="an-pair-clear" title="remove all fitted pairs">✕</button>`;
    host.wireMarks();
    const fitButton = host.slot.querySelector('.an-pair-fit');
    const undoButton = host.slot.querySelector('.an-pair-undo');
    const clearButton = host.slot.querySelector('.an-pair-clear');

    fitButton.onclick = () => {
      host.setResult('fitting…');
      host.send('fit_pair', {
        channel: host.box.channel(),
        t_min: host.marks.proi[0],
        t_max: host.marks.proi[1],
        x1: host.marks.p1,
        x2: host.marks.p2,
      }).then(() => {
        // this pair is fitted; clear the marks, ready for the next one
        host.clearMarks();
      }).catch((error) => host.setResult(error.message));
    };
    undoButton.onclick = () => {
      host.send('undo_pair').catch((error) => host.setResult(error.message));
    };
    clearButton.onclick = () => {
      host.clearMarks();
      host.send('clear_pairs').catch((error) => host.setResult(error.message));
    };

    return {
      sync() {
        fitButton.disabled = !(host.ready() && host.marks.proi
          && host.marks.p1 != null && host.marks.p2 != null);
        const nPairs = state?.pairs?.length ?? 0;
        undoButton.disabled = !host.ready() || !nPairs;
        clearButton.disabled = !host.ready() || !nPairs;
      },
      hasResult: () => !!state?.results?.rows,
      resultText() {
        const nPairs = state?.pairs?.length ?? 0;
        const r = state?.results;
        const meanStd = (mean, std, digits) => std != null
          ? `${mean.toFixed(digits)} ± ${std.toFixed(digits)}`
          : mean.toFixed(digits);
        if (r?.rows) {
          const na = r.NA_mean != null ? meanStd(r.NA_mean, r.NA_std, 4)
            : `unavailable${r.na_error ? ` (${r.na_error})` : ''}`;
          return `${nPairs} pairs | df/FSR = `
            + `${meanStd(r.df_over_fsr_mean, r.df_over_fsr_std, 4)}`
            + ` | df = ${(r.df_over_fsr_mean * r.fsr_mhz).toFixed(2)} MHz`
            + ` (FSR = ${r.fsr_mhz.toFixed(1)} MHz) | NA = ${na}`;
        }
        if (r?.error) return r.error;
        if (nPairs === 1) {
          return '1 pair fitted — fit at least 2 '
            + '(the FSR is the spacing between pairs)';
        }
        return '';
      },
      copy() {
        const r = state?.results;
        if (!r?.rows) return null;
        return {
          text: `${r.n_pairs}\t${r.df_over_fsr_mean.toFixed(6)}\t`
            + `${r.df_over_fsr_std != null ? r.df_over_fsr_std.toFixed(6) : ''}\t`
            + `${r.NA_mean != null ? r.NA_mean.toFixed(4) : 'N/A'}\t`
            + `${r.NA_std != null ? r.NA_std.toFixed(4) : ''}`,
          note: 'copied: n pairs, df/FSR (mean, std), NA (mean, std)',
        };
      },
      draw(u, helpers) {
        (state?.pairs ?? []).forEach((pair, index) => {
          const color = PAIR_COLORS[index % PAIR_COLORS.length];
          helpers.ctx.strokeStyle = color;
          helpers.ctx.lineWidth = devicePixelRatio;
          helpers.verticalLine(pair.x01);
          helpers.verticalLine(pair.x02);
          helpers.polyline(pair.curve, color);
        });
      },
      onEvent(event) {
        state = event;
        return !!event.pairs?.length;
      },
      restore(describe) {
        state = describe.analysis_pairs ?? null;
        return !!state?.pairs?.length;
      },
      clear() {
        state = null;
      },
    };
  },
};
