// Keyboard control of the dashboard. GUI-wide and box-agnostic: a box takes
// part only by putting attributes on its own elements, so a new box type gets
// keyboard control by declaring keys, not by wiring handlers here.
//
//   data-key="F"            the element's hotkey (several: "P Space"). A
//                           button / checkbox / radio / <summary> is clicked,
//                           a text or number field is focused with its text
//                           selected, a <select> is focused and opened.
//   data-key-zone           a sub-focus inside a box (the marker list): its
//                           own data-key enters it, and the data-keys inside
//                           it work only while it is entered.
//   data-zone-row           a row of a zone: ↑ ↓ Home End move between rows;
//     data-row-toggle         Space clicks it,
//     data-row-rename         F2 / Enter focuses it for typing,
//     data-row-delete         Delete (pressed twice) clicks it.
//
// Keys are matched by physical position (event.code), so they work whatever
// the keyboard layout (Hebrew included). Only bare keys and Shift+key are
// used: Ctrl / Alt / Meta combinations are the browser's and pass through.
//
// Levels, each left with Esc: typing in a field -> a zone -> the focused box.
// Tab / Shift+Tab move between boxes. Keys on elements outside every box (the
// header) work whatever box is focused, and win over a box's own keys.

const BOX = '.grid-stack-item-content';
const MODALS = '#modal-backdrop:not([hidden]), .cam-modal-backdrop, '
  + '.sp-start-backdrop:not([hidden]), .idle-backdrop:not([hidden])';
const PICKABLE = 'button, input:not([type="hidden"]), select, textarea, summary, [data-key-zone]';
const PICK_LETTERS = 'ASDFGHJKLQWERTYUIOPZXCVBNM';
const TIP = '  ⌨ ';                 // separates a tooltip from its key hint
const DELETE_CONFIRM_MS = 2000;

let focusedBox = null;   // the BOX element keys go to
let zone = null;         // the entered data-key-zone element, inside focusedBox
let rowIndex = 0;        // current row of the zone
let deleteArmed = null;  // {row, until} after a first Delete
let pick = null;         // {badges: [{code, el, badge}], typed}
let helpEl = null;
const globalActions = []; // [{spec, label, run}] keys without an element

// ------------------------------------------------------------------ keys
const CODE_NAMES = {
  Space: 'Space', Minus: '-', Equal: '=', BracketLeft: '[', BracketRight: ']',
  Semicolon: ';', Quote: "'", Backquote: '`', Backslash: '\\', Comma: ',',
  Period: '.', Slash: '/',
};

function specOf(event) {
  if (event.ctrlKey || event.altKey || event.metaKey) return null;
  const code = event.code;
  let name;
  if (/^Key[A-Z]$/.test(code)) name = code.slice(3);
  else if (/^(Digit|Numpad)[0-9]$/.test(code)) name = code.slice(-1);
  else name = CODE_NAMES[code] ?? code;   // F2, Delete, ArrowUp, Escape, ...
  return (event.shiftKey ? 'Shift+' : '') + name;
}

const specsOf = (el) => (el.dataset.key ?? '').split(/\s+/).filter(Boolean);
const showSpec = (spec) => (spec === 'Shift+/' ? '?' : spec);
const showKeys = (el) => specsOf(el).map(showSpec).join(' / ');

// ------------------------------------------------------------ predicates
function isVisible(el) {
  if (el.getClientRects().length) return true;
  // a radio of a segmented control hides its circle; its label shows
  const label = el.closest('label');
  return !!label && label.getClientRects().length > 0;
}

function isTyping(el) {
  if (!el || !el.tagName) return false;
  if (el.isContentEditable || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT') return true;
  return el.tagName === 'INPUT'
    && !['checkbox', 'radio', 'button', 'submit', 'reset', 'range', 'color'].includes(el.type);
}

const modalOpen = () => [...document.querySelectorAll(MODALS)].some((m) => m.getClientRects().length);
const boxes = () => [...document.querySelectorAll(BOX)].filter((b) => b.getClientRects().length);
const zoneOf = (el, within) => {
  const z = el.parentElement?.closest('[data-key-zone]');
  return z && within.contains(z) ? z : null;
};

// ------------------------------------------------------------- box focus
function currentBox() {
  if (focusedBox && !focusedBox.isConnected) setBox(null);
  if (focusedBox) return focusedBox;
  const all = boxes();
  return all.length === 1 ? all[0] : null;   // a lone box (satellite) needs no Tab
}

function setBox(box, { scroll = true } = {}) {
  if (box === focusedBox) return;
  leaveZone();
  focusedBox?.classList.remove('kb-focus');
  focusedBox = box;
  if (!box) return;
  box.classList.add('kb-focus');
  if (scroll) box.scrollIntoView({ block: 'nearest', inline: 'nearest' });
}

function cycleBox(step) {
  const all = boxes().sort((a, b) => {
    const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
    return Math.abs(ra.top - rb.top) > 20 ? ra.top - rb.top : ra.left - rb.left;
  });
  if (!all.length) return;
  const at = all.indexOf(focusedBox);
  const next = at < 0 ? (step > 0 ? 0 : all.length - 1) : (at + step + all.length) % all.length;
  setBox(all[next]);
}

// ------------------------------------------------------------------ zones
const zoneRows = () => (zone ? [...zone.querySelectorAll('[data-zone-row]')] : []);

function enterZone(el) {
  if (!el.getClientRects().length) return;
  leaveZone();
  zone = el;
  zone.classList.add('kb-zone');
  rowIndex = 0;
  markRow();
}

function leaveZone() {
  if (!zone) return;
  zone.classList.remove('kb-zone');
  zone.querySelectorAll('.kb-row, .kb-armed').forEach((r) => r.classList.remove('kb-row', 'kb-armed'));
  zone = null;
  deleteArmed = null;
}

// rows are rebuilt by their box at will; this puts the mark back each time
function markRow() {
  if (!zone) return;
  if (!zone.isConnected || !zone.getClientRects().length) { leaveZone(); return; }
  const rows = zoneRows();
  rowIndex = Math.max(0, Math.min(rowIndex, rows.length - 1));
  rows.forEach((r, i) => {
    r.classList.toggle('kb-row', i === rowIndex);
    r.classList.toggle('kb-armed', !!deleteArmed && i === rowIndex
      && Date.now() < deleteArmed.until);
  });
  rows[rowIndex]?.scrollIntoView({ block: 'nearest' });
}

function zoneKey(spec) {
  const rows = zoneRows();
  const row = rows[rowIndex];
  const moveTo = (i) => { rowIndex = i; deleteArmed = null; markRow(); };
  switch (spec) {
    case 'ArrowDown': moveTo(Math.min(rowIndex + 1, rows.length - 1)); return true;
    case 'ArrowUp': moveTo(Math.max(rowIndex - 1, 0)); return true;
    case 'Home': moveTo(0); return true;
    case 'End': moveTo(rows.length - 1); return true;
    case 'Space': row?.querySelector('[data-row-toggle]')?.click(); return !!row;
    case 'F2': case 'Enter': {
      const field = row?.querySelector('[data-row-rename]');
      if (!field) return false;
      field.focus();
      field.select?.();
      return true;
    }
    case 'Delete': {
      if (!row) return false;
      if (deleteArmed?.row === rowIndex && Date.now() < deleteArmed.until) {
        deleteArmed = null;
        row.querySelector('[data-row-delete]')?.click();
      } else {
        deleteArmed = { row: rowIndex, until: Date.now() + DELETE_CONFIRM_MS };
        markRow();
        setTimeout(markRow, DELETE_CONFIRM_MS + 50);
      }
      return true;
    }
    default: return false;
  }
}

// --------------------------------------------------------------- dispatch
function matches(el, spec) {
  return specsOf(el).includes(spec) && isVisible(el);
}

function findTarget(spec, box) {
  // the header's keys first: they work whatever box is focused
  const outside = [...document.querySelectorAll('[data-key]')]
    .filter((el) => !el.closest(BOX) && matches(el, spec));
  if (outside.length) return outside[0];
  if (!box) return null;
  const inBox = [...box.querySelectorAll('[data-key]')].filter((el) => matches(el, spec));
  if (zone) {
    const own = inBox.find((el) => zoneOf(el, box) === zone);
    if (own) return own;
  }
  return inBox.find((el) => !zoneOf(el, box)) ?? null;
}

// a checkbox or radio is small or invisible (segmented); its label shows
const shownOf = (el) => ((el.type === 'checkbox' || el.type === 'radio' || !el.getClientRects().length)
  && el.closest('label')) || el;

function flash(el, cls = 'kb-flash') {
  const shown = shownOf(el);
  shown.classList.remove('kb-flash', 'kb-refused');
  void shown.offsetWidth;   // restart the animation
  shown.classList.add(cls);
  setTimeout(() => shown.classList.remove(cls), 400);
}

function activate(el) {
  if (el.disabled) { flash(el, 'kb-refused'); return; }
  const box = el.closest(BOX);
  if (box && box !== focusedBox) setBox(box);
  flash(el);
  if (el.hasAttribute('data-key-zone')) { enterZone(el); return; }
  if (el.tagName === 'SELECT') {
    el.focus();
    try { el.showPicker?.(); } catch { /* focused is enough: arrows change it */ }
    return;
  }
  if (isTyping(el)) { el.focus(); el.select?.(); return; }
  el.click();
}

function onKeyDown(event) {
  if (event.defaultPrevented || event.isComposing) return;
  const spec = specOf(event);
  if (!spec) return;
  if (modalOpen()) return;   // a dialog has the keyboard (its own Esc / Tab)

  if (helpEl) {
    if (spec === 'Escape' || spec === 'Shift+/') { closeHelp(); event.preventDefault(); }
    return;
  }
  if (pick) { pickKey(event, spec); return; }

  const target = event.target;
  if (isTyping(target)) {
    if (spec === 'Escape') {
      // the field's own Esc (e.g. undo a rename) has run already; then out
      target.blur();
      event.preventDefault();
    } else if (spec === 'Tab' || spec === 'Shift+Tab') {
      target.blur();   // fields commit on blur
      cycleBox(spec === 'Tab' ? 1 : -1);
      event.preventDefault();
    }
    return;
  }

  if (spec === 'Tab' || spec === 'Shift+Tab') {
    cycleBox(spec === 'Tab' ? 1 : -1);
    event.preventDefault();
    return;
  }
  if (spec === 'Escape') {
    if (zone) leaveZone();
    else if (document.querySelector('.log-panel.open')) document.querySelector('.log-panel .box-close')?.click();
    else setBox(null);
    event.preventDefault();
    return;
  }

  const global = globalActions.find((a) => a.spec === spec);
  if (global) { global.run(); event.preventDefault(); return; }

  const box = currentBox();
  if (zone && zoneKey(spec)) { event.preventDefault(); return; }
  if (spec === ';' && box) { startPick(box); event.preventDefault(); return; }

  const el = findTarget(spec, box);
  if (el) { activate(el); event.preventDefault(); }
}

// ----------------------------------------------------------- pick mode
// Every control of the focused box gets a letter badge; typing it activates
// that control. Reaches what has no hotkey of its own (rows built at run time,
// the boxes that have not been given keys yet).
function startPick(box) {
  const root = zone ?? box;
  const els = [...root.querySelectorAll(PICKABLE)].filter((el) => isVisible(el) && !el.disabled);
  if (!els.length) return;
  const n = els.length;
  const codes = n <= PICK_LETTERS.length ? [...PICK_LETTERS].slice(0, n)
    : els.map((_, i) => PICK_LETTERS[Math.floor(i / PICK_LETTERS.length) % PICK_LETTERS.length]
      + PICK_LETTERS[i % PICK_LETTERS.length]);
  pick = { typed: '', badges: [] };
  els.forEach((el, i) => {
    const rect = shownOf(el).getBoundingClientRect();
    const badge = document.createElement('div');
    badge.className = 'kb-badge';
    badge.textContent = codes[i];
    badge.style.left = `${Math.max(0, rect.left - 4)}px`;
    badge.style.top = `${Math.max(0, rect.top - 6)}px`;
    document.body.appendChild(badge);
    pick.badges.push({ code: codes[i], el, badge });
  });
}

function endPick() {
  if (!pick) return;
  pick.badges.forEach((b) => b.badge.remove());
  pick = null;
}

function pickKey(event, spec) {
  event.preventDefault();
  const letter = /^[A-Z]$/.test(spec) ? spec : null;
  if (!letter) { endPick(); return; }
  pick.typed += letter;
  const left = pick.badges.filter((b) => b.code.startsWith(pick.typed));
  if (!left.length) { endPick(); return; }
  if (left.length === 1 && left[0].code === pick.typed) {
    const el = left[0].el;
    endPick();
    activate(el);
    return;
  }
  for (const b of pick.badges) b.badge.hidden = !b.code.startsWith(pick.typed);
}

// ------------------------------------------------------------------ help
function describe(el) {
  const own = el.getAttribute('aria-label')
    || (el.tagName === 'INPUT' && (el.type === 'checkbox' || el.type === 'radio')
      ? el.closest('label')?.textContent : null)
    || (isTyping(el) ? el.closest('label')?.firstChild?.textContent : null)
    || el.textContent;
  const text = (own ?? '').replace(/\s+/g, ' ').trim();
  if (text && text.length < 40) return text;
  return (el.title.split(TIP)[0] || text).slice(0, 60);
}

function openHelp() {
  const box = currentBox();
  helpEl = document.createElement('div');
  helpEl.className = 'kb-help';
  const rows = [];
  const add = (keys, what) => rows.push([keys, what]);

  add('Tab / Shift+Tab', 'focus the next / previous box');
  add('Esc', 'leave: a field, then a list, then the box');
  add(';', 'pick any control of the focused box by its letter');
  for (const a of globalActions) add(showSpec(a.spec), a.label);
  for (const el of document.querySelectorAll('[data-key]')) {
    if (el.closest(BOX) || !el.getClientRects().length) continue;
    if (globalActions.some((a) => specsOf(el).includes(a.spec))) continue;
    add(showKeys(el), describe(el));
  }
  const groups = [['everywhere', rows.splice(0)]];
  if (box) {
    for (const el of box.querySelectorAll('[data-key]')) {
      if (zoneOf(el, box) || !isVisible(el)) continue;
      if (globalActions.some((a) => specsOf(el).includes(a.spec))) continue;
      add(showKeys(el), describe(el));
    }
    groups.push([`this box — ${box.querySelector('.box-title')?.textContent ?? ''}`, rows.splice(0)]);
    for (const z of box.querySelectorAll('[data-key-zone]')) {
      if (!z.getClientRects().length) continue;
      add('↑ ↓ Home End', 'move in the list');
      add('Space', 'show / hide');
      add('F2 / Enter', 'rename (Enter keeps, Esc undoes)');
      add('Delete ×2', 'delete');
      for (const el of z.querySelectorAll('[data-key]')) {
        if (zoneOf(el, box) === z) add(showKeys(el), describe(el));
      }
      groups.push([`after ${showKeys(z)} — ${z.getAttribute('aria-label') ?? 'list'}`, rows.splice(0)]);
    }
  } else {
    groups.push(['no box focused', [['Tab', 'focus one to see its keys']]]);
  }

  const title = document.createElement('h2');
  title.textContent = 'keyboard  (? or Esc closes)';
  helpEl.appendChild(title);
  for (const [name, items] of groups) {
    if (!items.length) continue;
    const h = document.createElement('h3');
    h.textContent = name;
    helpEl.appendChild(h);
    for (const [keys, what] of items) {
      const row = document.createElement('div');
      row.className = 'kb-help-row';
      const k = document.createElement('kbd');
      k.textContent = keys;
      const w = document.createElement('span');
      w.textContent = what;
      row.append(k, w);
      helpEl.appendChild(row);
    }
  }
  helpEl.onclick = closeHelp;
  document.body.appendChild(helpEl);
}

function closeHelp() {
  helpEl?.remove();
  helpEl = null;
}

// -------------------------------------------------------------- tooltips
// Each keyed element's tooltip gets its key appended (and the label around a
// checkbox, which is what the pointer is usually on). Boxes set and reset
// titles freely; the observer puts the hint back whenever it goes missing.
function hintFor(el) {
  const box = el.closest(BOX);
  const z = box ? zoneOf(el, box) : null;
  return z ? `${showKeys(z)}, then ${showKeys(el)}` : showKeys(el);
}

function withHint(target, hint) {
  const base = (target.getAttribute('title') ?? '').replace(/\s*⌨ [^]*$/, '');
  const wanted = base ? `${base}${TIP}${hint}` : `${TIP.trim()} ${hint}`;
  if (target.getAttribute('title') !== wanted) target.setAttribute('title', wanted);
}

function decorate(el) {
  if (!el.dataset.key) return;
  const hint = hintFor(el);
  withHint(el, hint);
  const label = el.closest('label');
  if (label && label !== el && label.querySelectorAll('[data-key]').length === 1) withHint(label, hint);
}

// (the observer already batches: one call per burst of changes)
function onMutations(records) {
  const batch = new Set();
  for (const r of records) {
    if (r.type === 'attributes') { if (r.target.dataset?.key) batch.add(r.target); continue; }
    for (const node of r.addedNodes) {
      if (node.nodeType !== 1) continue;
      if (node.dataset.key) batch.add(node);
      node.querySelectorAll('[data-key]').forEach((el) => batch.add(el));
    }
  }
  batch.forEach((el) => el.isConnected && decorate(el));
  if (zone && records.some((r) => r.type === 'childList' && zone.contains(r.target))) markRow();
}

// ------------------------------------------------------------------ setup
/** Open a zone row's name for typing, with the zone entered on that row and
 *  its box focused - so after Enter (or Esc) the arrows go on in the list. */
export function editZoneRow(row) {
  const box = row.closest(BOX);
  const rowZone = row.closest('[data-key-zone]');
  if (box) setBox(box, { scroll: false });
  if (rowZone && rowZone !== zone) enterZone(rowZone);
  if (zone === rowZone) {
    rowIndex = Math.max(0, zoneRows().indexOf(row));
    markRow();
  }
  const field = row.querySelector('[data-row-rename]');
  field?.focus();
  field?.select?.();
}

export function toggleHelp() {
  if (helpEl) closeHelp();
  else openHelp();
}

/** A key with no element to carry it (still listed by `?`). */
export function registerGlobalKey(spec, label, run) {
  globalActions.push({ spec, label, run });
}

export function initKeyboard() {
  document.addEventListener('keydown', onKeyDown);
  // a click in a box makes it the focused one, so mouse and keys agree
  document.addEventListener('pointerdown', (event) => {
    const box = event.target.closest?.(BOX);
    if (box) {
      if (box !== focusedBox) setBox(box, { scroll: false });
      else if (zone && !zone.contains(event.target)) leaveZone();
    }
    endPick();
  }, true);
  window.addEventListener('scroll', endPick, true);
  window.addEventListener('resize', endPick);
  registerGlobalKey('Shift+/', 'this help', toggleHelp);
  document.querySelectorAll('[data-key]').forEach(decorate);
  new MutationObserver(onMutations).observe(document.body, {
    subtree: true, childList: true, attributes: true, attributeFilter: ['title', 'data-key'],
  });
}
