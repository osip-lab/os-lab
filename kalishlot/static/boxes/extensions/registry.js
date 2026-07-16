// Which analysis extensions each device type offers, in dropdown order.
// Static imports on purpose (no dynamic import()): the whole frontend stays
// one script graph, so rendering any box proves every module parses.

import { sidebandsExtension } from './scope_sidebands.js';
import { pairsExtension } from './scope_pairs.js';

export const ANALYSIS_EXTENSIONS = {
  picoscope: [sidebandsExtension, pairsExtension],
};
