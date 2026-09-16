/*
 * Minimal, dependency-free DOM + fetch driver for real behavioural tests of
 * auto_tune/ui/static/hpo.js.
 *
 * The HPO front-end is an IIFE that talks to `document`, `window`, `fetch`,
 * `setInterval`/`clearInterval`. This module supplies just enough of those to
 * execute the *real* script (no string assertions) and to control async
 * ordering deterministically:
 *
 *   - elements are built by parsing the real single_page.html template, so the
 *     ids/classes/defaults under test are the ones that ship;
 *   - `fetch` never performs I/O: every request is queued until the scenario
 *     resolves it, which is what makes out-of-order/stale replies testable;
 *   - timers never fire on their own, so polling cannot interleave.
 *
 * No npm packages, no jsdom.
 */

'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

class Element {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.parentNode = null;
    this.style = {};
    this.attributes = {};
    this.listeners = {};
    this.disabled = false;
    this.type = '';
    this.checked = false;
    this._text = '';
    this._className = '';
    this._classList = null;
    this.defaultValue = '';
    this.value = '';
    this.options = this.tagName === 'SELECT' ? [] : undefined;
  }

  get className() { return this._className; }
  set className(value) { this._className = value == null ? '' : String(value); this._classList = null; }

  _tokens() { return this._className.split(/\s+/).filter(Boolean); }

  get classList() {
    if (!this._classList) {
      const el = this;
      this._classList = {
        add: function () {
          const t = new Set(el._tokens());
          for (const n of arguments) t.add(n);
          el._className = Array.from(t).join(' ');
        },
        remove: function () {
          const t = new Set(el._tokens());
          for (const n of arguments) t.delete(n);
          el._className = Array.from(t).join(' ');
        },
        contains: function (n) { return el._tokens().indexOf(n) >= 0; },
        toggle: function (n, force) {
          const t = new Set(el._tokens());
          let on;
          if (force === true) on = true;
          else if (force === false) on = false;
          else on = !t.has(n);
          if (on) t.add(n); else t.delete(n);
          el._className = Array.from(t).join(' ');
          return on;
        },
      };
    }
    return this._classList;
  }

  get textContent() {
    const nested = this.children.map((c) => c.textContent).join('');
    return nested + this._text;
  }

  set textContent(value) {
    this.children = [];
    this._text = value == null ? '' : String(value);
  }

  get firstChild() { return this.children[0] || null; }
  get childNodes() { return this.children; }
  get firstElementChild() { return this.children[0] || null; }

  appendChild(child) {
    if (child.tagName === '#FRAGMENT') {
      // DocumentFragment semantics: appending moves its children into this node.
      for (const kid of child.children.slice()) this.appendChild(kid);
      return child;
    }
    if (child.parentNode) child.parentNode.removeChild(child);
    child.parentNode = this;
    this.children.push(child);
    if (this.tagName === 'SELECT' && child.tagName === 'OPTION') this.options.push(child);
    return child;
  }

  removeChild(child) {
    const i = this.children.indexOf(child);
    if (i >= 0) this.children.splice(i, 1);
    if (this.options) {
      const oi = this.options.indexOf(child);
      if (oi >= 0) this.options.splice(oi, 1);
    }
    child.parentNode = null;
    return child;
  }

  addEventListener(type, fn) {
    (this.listeners[type] = this.listeners[type] || []).push(fn);
  }

  dispatch(type, event) {
    const fns = this.listeners[type] || [];
    for (const fn of fns) fn(event || {});
  }

  setAttribute(key, value) { this.attributes[key] = String(value); }
  getAttribute(key) { return this.attributes[key] === undefined ? null : this.attributes[key]; }
  querySelectorAll() { return []; }
  querySelector() { return null; }
}

const VOID_TAGS = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img',
  'input', 'link', 'meta', 'param', 'source', 'track', 'wbr']);

function buildDom(html) {
  const registry = {};
  const all = [];        // every parsed element (class selectors must find them)
  const inOrder = [];    // every parsed element, in real document order
  const tagRe = /<(\/?)([a-zA-Z][a-zA-Z0-9]*)((?:"[^"]*"|[^>"])*)>/g;
  let match;
  let lastSelect = null;
  let cursor = 0;
  const stack = [];
  // Server-rendered text between tags is real content (e.g. the monitor cards),
  // so capture it onto the innermost open element.
  const flushText = (end) => {
    const raw = html.slice(cursor, end);
    const el = stack[stack.length - 1];
    if (!raw || !el || el.tagName === 'SCRIPT' || el.tagName === 'STYLE') return;
    const text = raw.replace(/\s+/g, ' ').trim();
    // Append to the element's own text instead of replacing it: setting
    // ``textContent`` would drop the element children parsed so far.
    if (text) el._text = el._text ? el._text + ' ' + text : text;
  };
  while ((match = tagRe.exec(html)) !== null) {
    flushText(match.index);
    cursor = tagRe.lastIndex;
    const closing = match[1] === '/';
    const tag = match[2];
    const attrs = match[3] || '';
    const tagLower = tag.toLowerCase();
    if (closing) {
      if (stack.length) stack.pop();
      continue;
    }

    if (tagLower === 'select') lastSelect = null;
    if (tagLower === 'option' && lastSelect) {
      const opt = new Element('option');
      const om = /value="([^"]*)"/.exec(attrs);
      opt.value = om ? om[1] : '';
      const tm = /<option[^>]*>([^<]*)/.exec(html.slice(match.index, match.index + 400));
      opt.textContent = tm ? tm[1] : '';
      lastSelect.appendChild(opt);
      all.push(opt);
      inOrder.push(opt);
      continue;
    }

    const el = new Element(tag);
    const clsMatch = /class="([^"]+)"/.exec(attrs);
    if (clsMatch) el.className = clsMatch[1];
    inOrder.push(el);
    // Every parsed element is addressable, exactly like a browser: a
    // class-selector must find class-only elements too (not just id-bearing ones).
    all.push(el);
    const idMatch = /id="([^"]+)"/.exec(attrs);
    if (idMatch) {
      el.id = idMatch[1];
      registry[el.id] = el;
    }
    const typeMatch = /type="([^"]+)"/.exec(attrs);
    if (typeMatch) el.type = typeMatch[1];
    // Bare boolean attributes must behave like a browser: `<button disabled>` is
    // disabled before any script touches it (the create gate depends on this).
    if (/(?:^|\s)disabled(?:\s|=|$)/.test(attrs)) el.disabled = true;
    const valMatch = /value="([^"]*)"/.exec(attrs);
    if (valMatch) { el.value = valMatch[1]; el.defaultValue = valMatch[1]; }
    for (const key of ['name', 'placeholder', 'min', 'max', 'step']) {
      const m = new RegExp(key + '="([^"]*)"').exec(attrs);
      if (m) el.setAttribute(key, m[1]);
    }
    if (tagLower === 'select') lastSelect = el;
    // Build a real tree: the parent is the innermost open element. Static
    // markup therefore supports the same parentNode/children walks the front
    // end uses (placeMonitor, collectButtons, scoping assertions).
    const parent = stack[stack.length - 1];
    if (parent) parent.appendChild(el);
    if (!VOID_TAGS.has(tagLower) && !attrs.endsWith('/')) stack.push(el);
  }
  flushText(html.length);

  const document = {
    _registry: registry,
    _all: all,
    _domReady: [],
    getElementById(id) { return Object.prototype.hasOwnProperty.call(registry, id) ? registry[id] : null; },
    createElement(tag) { const el = new Element(tag); all.push(el); return el; },
    createDocumentFragment() { return new Element('#fragment'); },
    querySelectorAll(selector) {
      const parts = String(selector).trim().split(/\s+/);
      const cls = parts[0].replace(/^\./, '');
      return all.filter((el) => el.classList.contains(cls));
    },
    querySelector(selector) {
      const found = document.querySelectorAll(selector);
      return found.length ? found[0] : null;
    },
    addEventListener(type, fn) {
      if (type === 'DOMContentLoaded') document._domReady.push(fn);
    },
    fireDomReady() {
      for (const fn of document._domReady) fn();
    },
  };
  return { document, registry, all, inOrder };
}

/**
 * Collect the page's script blocks in document order.
 *
 * ``src`` is kept for external scripts so the caller can load them exactly like
 * the browser would (``/static/x.js`` → ``ui/static/x.js``).
 */
function extractScripts(html) {
  const blocks = [];
  const re = /<script([^>]*)>([\s\S]*?)<\/script>/g;
  let match;
  while ((match = re.exec(html)) !== null) {
    const srcMatch = /src="([^"]+)"/.exec(match[1] || '');
    blocks.push({ src: srcMatch ? srcMatch[1] : null, code: match[2] });
  }
  return blocks;
}

function createHarness(options) {
  const options_ = options || {};
  const uiDir = options_.uiDir;
  // pageHtml: a *rendered* single_page.html. When given, the harness executes the
  // page's real scripts (inline + /static/*.js) in document order, so scenarios can
  // drive the production training-monitor code, not a reimplementation of it.
  const pageHtml = options_.pageHtml || null;
  const html = pageHtml
    ? fs.readFileSync(pageHtml, 'utf8')
    : fs.readFileSync(path.join(uiDir, 'templates', 'single_page.html'), 'utf8');
  const { document, registry, all, inOrder } = buildDom(html);

  const fetchQueue = [];
  const requestLog = [];
  const downloads = [];
  const alerts = [];
  const storage = {};
  const timers = [];
  const listeners = {};
  let timerSeq = 0;

  const sandbox = {
    document,
    fetch(url, opts) {
      return new Promise((resolve, reject) => {
        const entry = { url: String(url), opts: opts || {}, resolve, reject };
        requestLog.push(String(url));
        fetchQueue.push(entry);
        const signal = entry.opts.signal;
        if (signal && typeof signal.addEventListener === 'function') {
          // Honour AbortController like the browser: a superseded stream rejects.
          signal.addEventListener('abort', () => {
            const i = fetchQueue.indexOf(entry);
            if (i >= 0) fetchQueue.splice(i, 1);
            const err = new Error('aborted');
            err.name = 'AbortError';
            reject(err);
          });
        }
      });
    },
    setInterval(fn) { timerSeq += 1; const token = { id: timerSeq, fn, interval: true }; timers.push(token); return token; },
    // Synthetic like setInterval: the front-end only uses setTimeout for
    // fire-and-forget UI work (auto-hide, reload, batched log flush), so keeping
    // it from firing on its own makes scenarios deterministic. `flushTimeouts()`
    // fires them on demand, which is what a browser would eventually do.
    setTimeout(fn) { timerSeq += 1; const token = { id: timerSeq, fn, interval: false }; timers.push(token); return token; },
    clearInterval(token) {
      const i = timers.indexOf(token);
      if (i >= 0) timers.splice(i, 1);
    },
    clearTimeout(token) { const i = timers.indexOf(token); if (i >= 0) timers.splice(i, 1); },
    AbortController,
    TextDecoder,
    TextEncoder,
    alert(msg) { alerts.push(String(msg)); },
    addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
    location: {
      _href: '',
      get href() { return this._href; },
      set href(value) { this._href = String(value); downloads.push(String(value)); },
    },
    localStorage: {
      getItem(k) { return Object.prototype.hasOwnProperty.call(storage, k) ? storage[k] : null; },
      setItem(k, v) { storage[k] = String(v); },
      removeItem(k) { delete storage[k]; },
    },
    console,
    Promise,
    Date,
    Number,
    String,
    Boolean,
    Array,
    Object,
    JSON,
    Math,
    RegExp,
    Error,
    isFinite,
    parseFloat,
    parseInt,
  };
  // In a browser `window` *is* the global object, so `window.hpoX = fn` makes a
  // bare `hpoX()` call resolve. Mirror that faithfully.
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  if (pageHtml) {
    for (const block of extractScripts(html)) {
      const code = block.src
        ? fs.readFileSync(path.join(uiDir, 'static', path.basename(block.src)), 'utf8')
        : block.code;
      vm.runInContext(code, sandbox, { filename: block.src || 'single_page.html' });
    }
  } else {
    vm.runInContext(fs.readFileSync(path.join(uiDir, 'static', 'hpo.js'), 'utf8'),
      sandbox, { filename: 'hpo.js' });
  }

  const api = {
    document,
    window: sandbox,
    registry,
    all,
    order: inOrder,
    pending() { return fetchQueue.map((e) => e.url); },
    pendingCount() { return fetchQueue.length; },
    requestLog,
    downloads,
    alerts,
    timers,
    state: sandbox._hpoState,
    fireDomReady() { document.fireDomReady(); },

    /** Inspect the first queued request matching `pattern` (url + parsed body). */
    queued(pattern) {
      const matcher = typeof pattern === 'function' ? pattern : (url) => url.indexOf(pattern) >= 0;
      const entry = fetchQueue.find((e) => matcher(e.url));
      if (!entry) return null;
      let body = null;
      try { body = JSON.parse((entry.opts && entry.opts.body) || 'null'); } catch (err) { body = null; }
      return { url: entry.url, method: (entry.opts && entry.opts.method) || 'GET', body: body };
    },

    /** Resolve the first queued request whose url matches `pattern`. */
    respond(pattern, status, body) {
      const matcher = typeof pattern === 'function' ? pattern : (url) => url.indexOf(pattern) >= 0;
      const i = fetchQueue.findIndex((e) => matcher(e.url));
      if (i < 0) {
        throw new Error('no queued request matching ' + String(pattern) +
          '; pending=' + JSON.stringify(fetchQueue.map((e) => e.url)));
      }
      const entry = fetchQueue.splice(i, 1)[0];
      entry.resolve({
        ok: status >= 200 && status < 300,
        status,
        json: () => Promise.resolve(body === undefined ? {} : body),
      });
      return api.drain();
    },

    /**
     * Answer every pending request via `handler(url) -> {status, body}` until
     * the front-end stops issuing requests (i.e. the page is quiescent).
     */
    async settle(handler) {
      for (let guard = 0; guard < 80; guard++) {
        await api.drain();
        if (!fetchQueue.length) return;
        const entry = fetchQueue.shift();
        const res = (handler && handler(entry.url)) || { status: 200, body: {} };
        entry.resolve({
          ok: res.status >= 200 && res.status < 300,
          status: res.status,
          json: () => Promise.resolve(res.body === undefined ? {} : res.body),
        });
      }
      throw new Error('settle did not quiesce; pending=' +
        JSON.stringify(fetchQueue.map((e) => e.url)));
    },

    /**
     * Resolve the first queued request matching `pattern` with a fake SSE body.
     * `chunks` are strings (encoded) or Uint8Arrays, fed one per `reader.read()`.
     */
    respondStream(pattern, chunks) {
      const matcher = typeof pattern === 'function' ? pattern : (url) => url.indexOf(pattern) >= 0;
      const i = fetchQueue.findIndex((e) => matcher(e.url));
      if (i < 0) {
        throw new Error('no queued request matching ' + String(pattern) +
          '; pending=' + JSON.stringify(fetchQueue.map((e) => e.url)));
      }
      const entry = fetchQueue.splice(i, 1)[0];
      let next = 0;
      const reader = {
        read() {
          if (next >= chunks.length) return Promise.resolve({ done: true, value: undefined });
          const chunk = chunks[next++];
          return Promise.resolve({
            done: false,
            value: typeof chunk === 'string' ? new TextEncoder().encode(chunk) : chunk,
          });
        },
      };
      entry.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({}),
        body: { getReader: () => reader },
      });
      return api.drain();
    },

    /**
     * Resolve the first queued request matching `pattern` with an SSE body the
     * *test* drives: nothing is delivered until ``push(chunk)`` is called and the
     * stream never ends until ``end()``. This is what makes "an event arrives
     * after the subscription was superseded" deterministically testable — the
     * old reader stays open across a study switch instead of being exhausted by
     * the harness itself.
     */
    respondStreamHold(pattern) {
      const matcher = typeof pattern === 'function' ? pattern : (url) => url.indexOf(pattern) >= 0;
      const i = fetchQueue.findIndex((e) => matcher(e.url));
      if (i < 0) {
        throw new Error('no queued request matching ' + String(pattern) +
          '; pending=' + JSON.stringify(fetchQueue.map((e) => e.url)));
      }
      const entry = fetchQueue.splice(i, 1)[0];
      const buffered = [];
      const waiters = [];
      let closed = false;
      const deliver = () => {
        while (waiters.length && buffered.length) {
          waiters.shift()({ done: false, value: buffered.shift() });
        }
        if (!buffered.length && closed) {
          while (waiters.length) waiters.shift()({ done: true, value: undefined });
        }
      };
      const reader = {
        read() {
          if (buffered.length) return Promise.resolve({ done: false, value: buffered.shift() });
          if (closed) return Promise.resolve({ done: true, value: undefined });
          return new Promise((resolve) => { waiters.push(resolve); deliver(); });
        },
      };
      entry.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({}),
        body: { getReader: () => reader },
      });
      return {
        push(chunk) {
          buffered.push(typeof chunk === 'string' ? new TextEncoder().encode(chunk) : chunk);
          deliver();
          return api.drain();
        },
        end() { closed = true; deliver(); return api.drain(); },
      };
    },

    /** Reject the first queued request matching `pattern` (network failure). */
    reject(pattern) {
      const matcher = typeof pattern === 'function' ? pattern : (url) => url.indexOf(pattern) >= 0;
      const i = fetchQueue.findIndex((e) => matcher(e.url));
      if (i < 0) {
        throw new Error('no queued request matching ' + String(pattern) +
          '; pending=' + JSON.stringify(fetchQueue.map((e) => e.url)));
      }
      const entry = fetchQueue.splice(i, 1)[0];
      entry.reject(new TypeError('Failed to fetch'));
      return api.drain();
    },

    respondAll(status, body) {
      const entries = fetchQueue.splice(0, fetchQueue.length);
      for (const entry of entries) {
        entry.resolve({
          ok: status >= 200 && status < 300,
          status,
          json: () => Promise.resolve(body === undefined ? {} : body),
        });
      }
      return api.drain();
    },

    /**
     * Fire the pending ``setTimeout`` callbacks once — exactly what a browser
     * does a few milliseconds later. Intervals (status polling) deliberately
     * stay pending so a scenario keeps deciding when a poll round happens.
     */
    flushTimeouts() {
      const due = timers.filter((t) => !t.interval);
      for (const token of due) {
        const i = timers.indexOf(token);
        if (i >= 0) timers.splice(i, 1);
        token.fn();
      }
      return api.drain();
    },

    /**
     * Advance the fake clock by one polling period: every pending interval
     * callback fires once. Intervals stay scheduled exactly like in a browser,
     * so a callback that clears its own token really stops future ticks.
     */
    tickIntervals() {
      const due = timers.filter((t) => t.interval);
      for (const token of due) {
        if (timers.indexOf(token) < 0) continue;   // cleared by an earlier tick
        token.fn();
      }
      return api.drain();
    },

    /** How many interval timers are currently scheduled. */
    intervalCount() { return timers.filter((t) => t.interval).length; },

    /**
     * Answer every queued request except the run event streams (`/api/runs/`),
     * which the scenario holds open itself. Used by the polling scenarios: the
     * *implementation* decides which requests a polling round issues, the test
     * only answers them, so the assertions stay about automatic behaviour.
     */
    async settleExceptStream(handler) {
      for (let guard = 0; guard < 80; guard++) {
        await api.drain();
        const url = api.pending().find((u) => u.indexOf('/api/runs/') < 0);
        if (!url) return;
        const res = (handler && handler(url)) || { status: 200, body: {} };
        await api.respond(url, res.status, res.body);
      }
      throw new Error('settleExceptStream did not quiesce; pending=' +
        JSON.stringify(api.pending()));
    },

    drain() {
      // Flush the microtask queue a bounded number of times (promise chains in
      // hpo.js are at most a handful of links deep).
      let chain = Promise.resolve();
      for (let i = 0; i < 40; i++) chain = chain.then(() => undefined);
      return chain;
    },

    $(id) { return document.getElementById(id); },
    text(id) {
      const el = document.getElementById(id);
      return el ? String(el.textContent) : null;
    },
    visible(id) {
      const el = document.getElementById(id);
      if (!el) return false;
      return el.style.display !== 'none' && !el.classList.contains('hidden');
    },
    setInput(id, value) {
      const el = document.getElementById(id);
      if (!el) throw new Error('missing input ' + id);
      el.value = String(value);
    },
  };
  return api;
}

module.exports = { createHarness, Element };

// ── scenario runner ─────────────────────────────────────────────────

const ID_A = 'hpo_' + 'a'.repeat(32);
const ID_B = 'hpo_' + 'b'.repeat(32);
const ID_C = 'hpo_' + 'c'.repeat(32);

function defaultsPayload() {
  return {
    dataset: { registered: true, display_name: 'demo', reason_code: 'REGISTERED_SNAPSHOT',
      needs_user_confirmation: false,
      snapshot: { snapshot_id: 'd'.repeat(64), short_id: 'dddddddd', dataset_name: 'demo',
        source_root: 'E:/data/demo', created_at: '2026-09-14T00:00:00Z',
        train_count: 184, val_count: 46, background_count: 99, image_count: 230 } },
    model: { path: 'E:/project/yolov8n.pt', name: 'yolov8n.pt', source: 'project.model',
      available: true, reason_code: 'MODEL_BOUND' },
    search: { sampler: 'tpe', budget: 10, epochs: 30, seed: 42, timeout_seconds: 3600,
      evaluation_mode: 'comprehensive', batch: 16, imgsz: 640, device: '0' },
    formal: { epochs: 100, batch: 16, imgsz: 640, device: '0' },
    // 已探测设备：GPU 编号列表 + 权威默认（GPU 可用时必须是 GPU，绝不静默回退 CPU）
    devices: { gpus: [0], default: '0' },
    device_notice: null,
    config_warnings: [],
  };
}

function statusPayload(studyId, overrides) {
  const base = {
    study_id: studyId,
    execution_status: 'READY',
    execution_revision: 1,
    stop_reason: null,
    budget: 10,
    claimed_count: 0,
    terminal_count: 0,
    success_count: 0,
    failed_count: 0,
    cancelled_count: 0,
    interrupted_count: 0,
    running_count: 0,
    remaining_count: 10,
    evaluation_mode: 'comprehensive',
    objective: 'comprehensive_composite_best_epoch_v1',
    current_trial_number: null,
    control_active: false,
    can_stop: false,
    can_resume: false,
    error_code: null,
    next_action: null,
    sampler: 'tpe',
    seed: 42,
    trials: [],
    ranking: [],
    has_success: false,
    snapshot_id: 'd'.repeat(64),
    model_display: 'yolov8n.pt',
    created_at: '2026-09-14T00:00:00Z',
    revision: 1,
    study_epochs: 30,
    snapshot_short_id: 'dddddddd',
    batch: 16,
    imgsz: 640,
    device: 'cpu',
    timeout_seconds: 3600,
    search_space: { sampler: 'tpe', evaluation_mode: 'comprehensive',
      evaluation_mode_label: '全面（四指标）',
      objective: 'comprehensive_composite_best_epoch_v1',
      objective_label: '全面综合分数', direction: 'maximize',
      score: { objective: 'comprehensive_composite_best_epoch_v1',
        tie_break: 'earliest_best_epoch',
        weights: { 'metrics/mAP50(B)': 0.10, 'metrics/mAP50-95(B)': 0.50,
                   'metrics/precision(B)': 0.20, 'metrics/recall(B)': 0.20 } },
      search_space_version: 'v1', parameters: {}, fixed: { epochs: 30, batch: 16,
        imgsz: 640, device: 'cpu', seed: 42, snapshot: 'dddddddd', model: 'yolov8n.pt' } },
    approved_for_formal_training: false,
    best: null,
  };
  return Object.assign(base, overrides || {});
}

function bestPayload(studyId) {
  return {
    study_id: studyId,
    trial_id: studyId + '_t0001',
    trial_number: 1,
    trial_display_number: 2,
    value: 0.9,
    epoch: 5,
    evaluation_mode: 'comprehensive',
    objective: 'comprehensive_composite_best_epoch_v1',
    metrics: { 'metrics/mAP50(B)': 0.9, 'metrics/mAP50-95(B)': 0.9,
               'metrics/precision(B)': 0.9, 'metrics/recall(B)': 0.9 },
    search: { optimizer: 'SGD', lr0: 0.01, lrf: 0.01, momentum: 0.9,
      weight_decay: 0.0005, warmup_epochs: 0 },
    ranked_count: 2,
    is_tie: false,
    artifacts: { best_pt_available: true, last_pt_available: false },
  };
}

const EXPERIMENT_RUN_ID = 'manual:11111111-2222-3333-4444-555555555555';

/**
 * Five linked formal runs covering the identity matrix the third rework must
 * get right:
 *   train2  running,     runtime + index identity resolved
 *   train3  completed,   runtime + index identity resolved
 *   train9  unknown,     no runtime in the source metadata, index has nothing
 *   train10 completed,   legacy metadata (no runtime) but the index holds it
 *   train11 completed,   runtime present but the index row is ambiguous
 */
function formalRuns() {
  return [
    { train_name: 'train2', runtime_run_id: 'manual:uuid-2',
      runtime_identity_missing: false, history_run_id: 'manual:train2',
      experiment_run_id: 'manual:uuid-2', experiment_identity_reason: null,
      status: 'running', result_available: false, source_readable: true,
      metrics: { mAP50: 0.12, mAP50_95: 0.03 }, best_pt_available: false,
      source_trial_number: 1, training_config: { epochs: 2, batch: 1, imgsz: 96, device: '0' } },
    { train_name: 'train3', runtime_run_id: 'manual:uuid-3',
      runtime_identity_missing: false, history_run_id: 'manual:train3',
      experiment_run_id: 'manual:uuid-3', experiment_identity_reason: null,
      status: 'completed', result_available: true, source_readable: true,
      metrics: { mAP50: 0.71, mAP50_95: 0.44 }, best_pt_available: true,
      source_trial_number: 1, training_config: { epochs: 2, batch: 1, imgsz: 96, device: '0' } },
    { train_name: 'train9', runtime_run_id: null,
      runtime_identity_missing: true, history_run_id: null,
      experiment_run_id: null, experiment_identity_reason: 'EXPERIMENT_NOT_INDEXED',
      status: 'unknown', result_available: false, metrics: {}, source_readable: true,
      best_pt_available: false, source_trial_number: 1,
      training_config: { epochs: 2, batch: 1, imgsz: 96, device: 'cpu' } },
    { train_name: 'train10', runtime_run_id: null,
      runtime_identity_missing: true, history_run_id: 'manual:train10',
      experiment_run_id: EXPERIMENT_RUN_ID, experiment_identity_reason: null,
      status: 'completed', result_available: true, source_readable: true,
      metrics: { mAP50: 0.63, mAP50_95: 0.31 }, best_pt_available: true,
      source_trial_number: 1, training_config: { epochs: 2, batch: 1, imgsz: 96, device: 'cpu' } },
    { train_name: 'train11', runtime_run_id: 'manual:uuid-11',
      runtime_identity_missing: false, history_run_id: 'manual:train11',
      experiment_run_id: null, experiment_identity_reason: 'EXPERIMENT_AMBIGUOUS',
      status: 'completed', result_available: true, source_readable: true,
      metrics: { mAP50: 0.55, mAP50_95: 0.21 }, best_pt_available: false,
      source_trial_number: 1, training_config: { epochs: 2, batch: 1, imgsz: 96, device: 'cpu' } },
  ];
}

/** 正式训练“打开结果文件夹”场景的夹具：覆盖可打开与不可打开的每一种终态。 */
function folderRuns() {
  const base = {
    runtime_identity_missing: false, experiment_identity_reason: null,
    source_trial_number: 1, source_readable: true,
    training_config: { epochs: 2, batch: 1, imgsz: 96, device: '0' },
  };
  return [
    Object.assign({}, base, { train_name: 'train2', status: 'running',
      runtime_run_id: 'manual:uuid-2', history_run_id: 'manual:train2',
      experiment_run_id: 'manual:uuid-2', result_available: false, metrics: {},
      best_pt_available: false }),
    Object.assign({}, base, { train_name: 'train3', status: 'completed',
      runtime_run_id: 'manual:uuid-3', history_run_id: 'manual:train3',
      experiment_run_id: 'manual:uuid-3', result_available: true,
      metrics: { mAP50: 0.71, mAP50_95: 0.44 }, best_pt_available: true }),
    Object.assign({}, base, { train_name: 'train4', status: 'failed',
      runtime_run_id: 'manual:uuid-4', history_run_id: 'manual:train4',
      experiment_run_id: 'manual:uuid-4', result_available: false, metrics: {},
      best_pt_available: false }),
    // 已完成但没有 best.pt：结果目录仍然可以打开（只是没有可下载的权重）
    Object.assign({}, base, { train_name: 'train5', status: 'completed',
      runtime_run_id: 'manual:uuid-5', history_run_id: 'manual:train5',
      experiment_run_id: 'manual:uuid-5', result_available: true,
      metrics: { mAP50: 0.5, mAP50_95: 0.2 }, best_pt_available: false }),
    Object.assign({}, base, { train_name: 'train6', status: 'stopping',
      runtime_run_id: 'manual:uuid-6', history_run_id: 'manual:train6',
      experiment_run_id: 'manual:uuid-6', result_available: false, metrics: {},
      best_pt_available: false }),
    // 来源事实不合法（服务端受控扫描不会返回这类行；这里锁定前端不得放行）
    Object.assign({}, base, { train_name: 'train7', status: 'completed',
      runtime_run_id: 'manual:uuid-7', history_run_id: 'manual:train7',
      experiment_run_id: 'manual:uuid-7', result_available: true, metrics: {},
      best_pt_available: false, source_readable: false }),
  ];
}

function defaultHandler(url) {
  if (url.indexOf('/api/hpo/defaults') >= 0) return { status: 200, body: defaultsPayload() };
  if (url.indexOf('/api/hpo/snapshots') >= 0) {
    const snap = defaultsPayload().dataset.snapshot;
    return { status: 200, body: { snapshots: [Object.assign({ readable: true,
      selectable: true, reason_code: null }, snap)], count: 1 } };
  }
  if (url.indexOf('/api/hpo/local-models') >= 0) return { status: 200, body: { models: [], count: 0 } };
  if (url.indexOf('/formal-runs') >= 0) {
    return { status: 200, body: { runs: [], count: 0, warnings: [], truncated: false } };
  }
  if (url.indexOf('/api/hpo/studies?') >= 0) return { status: 200, body: { studies: [], count: 0 } };
  if (url.indexOf('/studies/' + ID_A) >= 0) return { status: 200, body: statusPayload(ID_A) };
  if (url.indexOf('/studies/' + ID_B) >= 0) return { status: 200, body: statusPayload(ID_B) };
  if (url.indexOf('/studies/' + ID_C) >= 0) return { status: 200, body: statusPayload(ID_C) };
  return { status: 200, body: {} };
}

async function bootstrap(api) {
  api.setInput('tuningModeSelect', 'hpo');
  api.fireDomReady();
  await api.settle(defaultHandler);
}

function collectButtons(node) {
  const found = [];
  const walk = (el) => {
    for (const child of el.children) {
      if (child.tagName === 'BUTTON') found.push(child);
      walk(child);
    }
  };
  walk(node);
  return found;
}

/** Select study A, answer its status with `overrides`, then answer formal-runs. */
async function selectStudyWithRuns(api, overrides, runs, warnings) {
  api.window.hpoSelectStudy(ID_A);
  await api.drain();
  await api.respond('/studies/' + ID_A, 200, statusPayload(ID_A, overrides));
  await api.drain();
  await api.respond('/formal-runs', 200, {
    study_id: ID_A, count: runs.length, truncated: false,
    warnings: warnings || [], runs: runs,
  });
  await api.settle(defaultHandler);
}

// ── training monitor helpers (page mode) ───────────────────────────

const RUN_A = 'manual:aaaaaaaa-1111-4111-8111-111111111111';

function sse(payload) {
  return 'data: ' + JSON.stringify(payload) + '\n\n';
}

/**
 * A terminal ``result`` exactly as ``finalize_training_run()`` emits it: the
 * epoch KPI is the *structured* ``{configured, completed, best}`` object (never
 * a scalar) and the metrics use the mapped report keys. Fixtures must mirror
 * the production shape, otherwise the tests pass while real training shows “—”.
 */
function finalizedResult(runName, epochs, metrics) {
  return {
    run_name: runName,
    params: { epochs: epochs },
    epochs: { configured: epochs, completed: epochs, best: epochs },
    metrics: metrics || { mAP50: 0.00037, mAP50_95: 0.00007,
                          precision: 0.00066, recall: 0.06452 },
    artifacts: { report_path: 'log/' + runName + '_report.json' },
  };
}

/** The four monitor cards plus the report link, read from the real DOM. */
function readMonitorCards(api) {
  const link = api.$('monitorReportLink');
  return {
    epochs: api.text('monitorEpochs'),
    map50: api.text('monitorMap50'),
    map5095: api.text('monitorMap5095'),
    pr: api.text('monitorPR'),
    reportVisible: !!(link && link.style.display !== 'none'),
    reportHref: link ? (link.href || null) : null,
  };
}

/** Startup endpoints of the rendered page (never a live stream by default). */
function pageHandler(url) {
  if (url.indexOf('/api/training/running') >= 0) {
    return { status: 200, body: { running: false } };
  }
  if (url.indexOf('/api/tuning/status') >= 0) return { status: 200, body: {} };
  return defaultHandler(url);
}

const SCENARIOS = {
  /* C1: A→B switch while A's status request is still in flight. */
  async switch_coalesce(api) {
    await bootstrap(api);
    api.window.hpoSelectStudy(ID_A);
    await api.drain();
    const pendingAfterA = api.pending().slice();
    // Switch to B before A answers: B's round must be queued, not dropped.
    api.window.hpoSelectStudy(ID_B);
    await api.drain();
    // Answer A first (stale generation): its 100% progress must not leak into B.
    await api.respond('/studies/' + ID_A, 200,
      statusPayload(ID_A, { execution_status: 'COMPLETED', terminal_count: 10,
                            budget: 10 }));
    const statusTextAfterStaleA = api.text('hpoStatusText');
    const percentAfterStaleA = api.text('hpoProgressPercent');
    // Let everything else settle; B's status must be asked for and rendered.
    await api.settle((url) => {
      if (url.indexOf('/studies/' + ID_B) >= 0) {
        return { status: 200, body: statusPayload(ID_B, {
          execution_status: 'RUNNING', control_active: true, can_stop: true,
          terminal_count: 4, budget: 10 }) };
      }
      return defaultHandler(url);
    });
    return {
      pendingAfterA: pendingAfterA,
      bStatusWasAsked: api.requestLog.some((u) => u.indexOf(ID_B) >= 0),
      statusTextAfterStaleA: statusTextAfterStaleA,
      statusTextAfterB: api.text('hpoStatusText'),
      percentAfterStaleA: percentAfterStaleA,
      percentAfterB: api.text('hpoProgressPercent'),
      studyId: api.state.studyId,
      stuckReading: /正在读取/.test(api.text('hpoStatusText') || ''),
    };
  },

  /* C1 (busy variant): B arrives while A's round is mid-flight and B must render. */
  async switch_while_round_busy(api) {
    await bootstrap(api);
    api.window.hpoSelectStudy(ID_A);
    await api.drain();
    // A's round is still in flight (status unanswered) when B is selected.
    api.window.hpoSelectStudy(ID_B);
    await api.drain();
    // Answer A: stale reply must be dropped AND B must still be fetched.
    await api.respond('/studies/' + ID_A, 200, statusPayload(ID_A));
    await api.settle((url) => {
      if (url.indexOf('/studies/' + ID_B) >= 0) {
        return { status: 200, body: statusPayload(ID_B, {
          execution_status: 'COMPLETED', success_count: 2, has_success: true }) };
      }
      return defaultHandler(url);
    });
    return {
      askedB: api.requestLog.some((u) => u.indexOf(ID_B) >= 0),
      statusText: api.text('hpoStatusText'),
      stuckReading: /正在读取/.test(api.text('hpoStatusText') || ''),
      studyId: api.state.studyId,
    };
  },

  /* C2: a stale train-best reply must not touch the newly selected study. */
  async formal_response_guard(api) {
    await bootstrap(api);
    api.window.hpoSelectStudy(ID_A);
    await api.settle((url) => {
      if (url.indexOf('/studies/' + ID_A) >= 0) {
        return { status: 200, body: statusPayload(ID_A, {
          execution_status: 'COMPLETED', success_count: 2, has_success: true,
          approved_for_formal_training: true, best: bestPayload(ID_A),
          ranking: [{ rank: 1, trial_number: 2, number: 1, value: 0.9, epoch: 5 }],
        }) };
      }
      return defaultHandler(url);
    });
    const btnEnabledForA = !api.$('hpoFormalBestBtn').disabled;
    api.window.hpoStartFormalTraining();
    await api.drain();
    const trainBestQueued = api.pending().some((u) => u.indexOf('/train-best') >= 0);
    // Switch to B before the submission answers: B has no best at all. Only the
    // B round is answered here; the train-best request deliberately stays queued.
    api.window.hpoSelectStudy(ID_B);
    await api.drain();
    await api.respond('/studies/' + ID_B, 200, statusPayload(ID_B));
    await api.respond('/formal-runs', 200,
      { runs: [], count: 0, warnings: [], truncated: false });
    await api.respond('/api/hpo/studies?', 200, { studies: [], count: 0 });
    const statusBeforeLateReply = api.text('hpoFormalStatus');
    const buttonBeforeLateReply = api.$('hpoFormalBestBtn').disabled;
    // Late 202 for A.
    await api.respond('/train-best', 202, {
      status: 'accepted', run_id: 'manual:uuid-a', train_name: 'train1',
      mode: 'formal', source: { study_id: ID_A, trial_id: ID_A + '_t0001',
        trial_number: 1, value: 0.9, epoch: 5 },
      training_config: { epochs: 2, batch: 1, imgsz: 96, device: '0' },
      differences: {}, links: {}, error_code: null, error: null, next_action: null,
    });
    await api.drain();
    return {
      btnEnabledForA: btnEnabledForA,
      trainBestQueued: trainBestQueued,
      statusBeforeLateReply: statusBeforeLateReply,
      buttonBeforeLateReply: buttonBeforeLateReply,
      formalStatusAfterLateReply: api.text('hpoFormalStatus'),
      buttonAfterLateReply: api.$('hpoFormalBestBtn').disabled,
      bestAfterSwitch: api.state.best,
      watchFormalAfterLateReply: api.state.watchFormal,
      studyId: api.state.studyId,
      trainBestRequests: api.requestLog.filter((u) => u.indexOf('/train-best') >= 0).length,
    };
  },

  /* C3: a successful poll must not erase a user-operation error. */
  async operation_error_survives_poll(api) {
    await bootstrap(api);
    api.window.hpoSelectStudy(ID_A);
    await api.settle(defaultHandler);

    api.window.hpoStartStudy();
    await api.drain();
    await api.respond('/start', 500, { error_code: 'HPO_EXECUTION_ERROR',
      error: '启动失败：执行器暂不可用。', next_action: '请稍后重试。' });
    const afterFailure = api.text('hpoStudyDetailError');

    // a normal poll round with a healthy status
    api.window.hpoRefreshRound(api.state.studyId, true);
    await api.settle(defaultHandler);
    const afterPoll = api.text('hpoStudyDetailError');

    // a new explicit operation clears only the operation half
    api.window.hpoStartStudy();
    await api.drain();
    await api.respond('/start', 202, { study_id: ID_A, started: true });
    await api.drain();
    const afterNewOperation = api.text('hpoStudyDetailError');
    return {
      afterFailure: afterFailure,
      afterPoll: afterPoll,
      afterNewOperation: afterNewOperation,
      operationError: api.state.operationError,
      serverDetailHidden: api.$('hpoStudyDetailError').classList.contains('hidden'),
    };
  },

  /* B3 + B5 + B6: first read of formal runs, identities, warnings, actions. */
  async formal_runs_first_read(api) {
    await bootstrap(api);
    api.window.hpoSelectStudy(ID_A);
    await api.drain();
    const watchFormalBefore = api.state.watchFormal;
    await api.respond('/studies/' + ID_A, 200, statusPayload(ID_A, {
      execution_status: 'COMPLETED', success_count: 1, has_success: true,
      approved_for_formal_training: true, best: bestPayload(ID_A),
    }));
    await api.drain();
    const askedFormalRuns = api.pending().some((u) => u.indexOf('/formal-runs') >= 0);
    await api.respond('/formal-runs', 200, {
      study_id: ID_A, count: 5, truncated: false,
      warnings: [{ code: 'FORMAL_SOURCE_UNREADABLE', train_name: 'train7' },
                 { code: 'FORMAL_RUNS_TRUNCATED', train_name: null }],
      runs: formalRuns(),
    });
    await api.settle(defaultHandler);
    const box = api.$('hpoFormalRunsList');
    const rows = [];
    for (const child of box.children) {
      const btns = collectButtons(child);
      rows.push({
        text: child.textContent,
        buttons: btns.map((b) => ({ label: b.textContent, disabled: b.disabled })),
      });
    }
    return {
      watchFormalBefore: watchFormalBefore,
      askedFormalRuns: askedFormalRuns,
      rows: rows,
      warningText: api.text('hpoFormalRunsWarning'),
      warningHidden: api.$('hpoFormalRunsWarning').classList.contains('hidden'),
      watchFormalAfter: api.state.watchFormal,
      timersRunning: api.timers.length,
      // 六参数表与搜索阶段模型下载默认折叠；关联结果/警告在 best 区之外始终可见
      bestDetailsOpen: !!api.$('hpoBestDetails').open,
      bestBodyHidden: api.$('hpoBestBody').classList.contains('hidden'),
      formalRunsInsideBestBody: (() => {
        let node = api.$('hpoFormalRuns');
        while (node) {
          if (node === api.$('hpoBestBody')) return true;
          node = node.parentNode;
        }
        return false;
      })(),
    };
  },

  /* “查看结果”必须传权威实验详情身份，而不是 JSON 历史 ID。 */
  async formal_runs_result_identity(api) {
    await bootstrap(api);
    const opened = [];
    api.window.showExperimentDetail = function (id) { opened.push(String(id)); };
    const runs = formalRuns();
    await selectStudyWithRuns(api, {
      execution_status: 'COMPLETED', success_count: 1, has_success: true,
      approved_for_formal_training: true, best: bestPayload(ID_A),
    }, runs);
    const box = api.$('hpoFormalRunsList');
    const rows = [];
    for (const child of box.children) {
      const result = collectButtons(child).find((b) => b.textContent === '查看结果');
      rows.push({
        trainName: String(runs[rows.length].train_name),
        disabled: result.disabled,
        title: result.title || null,
        clicked: (() => {
          if (result.disabled) return null;
          const before = opened.length;
          result.dispatch('click');
          return opened.length > before ? opened[opened.length - 1] : null;
        })(),
      });
    }
    return {
      opened: opened,
      rows: rows,
      historyIdsNeverOpened: opened.filter((id) => /^manual:train/.test(id)).length,
    };
  },

  /* 第三轮返修：已声明的 runtime 身份失配时不得改绑——监控只认 runtime 身份，
     结果只认实验索引身份，不可用时禁用入口并说明原因，绝不退回 JSON 历史 ID。 */
  async formal_runs_identity_guard(api) {
    await bootstrap(api);
    const monitored = [];
    const opened = [];
    api.window.locateTrainingRun = function (id) { monitored.push(String(id)); return true; };
    api.window.showExperimentDetail = function (id) { opened.push(String(id)); };
    const runs = formalRuns().concat([{
      // 声明了 runtime 身份但索引里没有它（另有一条同名同目录实验）：不得改绑
      train_name: 'train12',
      runtime_run_id: 'manual:11111111-1111-4111-8111-111111111111',
      runtime_identity_missing: false, history_run_id: 'manual:train12',
      experiment_run_id: null, experiment_identity_reason: 'EXPERIMENT_NOT_INDEXED',
      status: 'running', result_available: false, metrics: {},
      best_pt_available: false, source_trial_number: 1,
      training_config: { epochs: 2, batch: 1, imgsz: 96, device: 'cpu' },
    }, {
      // 终态：metadata 只有非法 runtime（manual:train13）且无活动控制器 →
      // 监控身份为 null，两个入口都必须禁用，历史 ID 只作展示
      train_name: 'train13', runtime_run_id: null, runtime_identity_missing: true,
      history_run_id: 'manual:train13', experiment_run_id: null,
      experiment_identity_reason: 'EXPERIMENT_NOT_INDEXED',
      status: 'completed', result_available: true,
      metrics: { mAP50: 0.5, mAP50_95: 0.3 }, best_pt_available: false,
      source_trial_number: 1,
      training_config: { epochs: 2, batch: 1, imgsz: 96, device: 'cpu' },
    }]);
    await selectStudyWithRuns(api, {
      execution_status: 'COMPLETED', success_count: 1, has_success: true,
      approved_for_formal_training: true, best: bestPayload(ID_A),
    }, runs);
    // 进入研究时会按权威投影自动订阅一次；之后只记录显式点击，两者的身份断言分开
    const autoSubscribed = monitored.slice();
    monitored.length = 0;
    const box = api.$('hpoFormalRunsList');
    const rows = [];
    for (let i = 0; i < box.children.length; i++) {
      const buttons = collectButtons(box.children[i]);
      const monitor = buttons.find((b) => b.textContent === '查看监控');
      const result = buttons.find((b) => b.textContent === '查看结果');
      const row = {
        trainName: String(runs[i].train_name),
        text: box.children[i].textContent,
        buttons: buttons.map((b) => ({ label: b.textContent, disabled: b.disabled })),
        monitorDisabled: monitor ? monitor.disabled : null,
        resultDisabled: result ? result.disabled : null,
        resultTitle: (result && result.title) || null,
        monitored: null,
        clicked: null,
      };
      if (monitor && !monitor.disabled) {
        const before = monitored.length;
        monitor.dispatch('click');
        if (monitored.length > before) row.monitored = monitored[monitored.length - 1];
      }
      if (result && !result.disabled) {
        const before = opened.length;
        result.dispatch('click');
        if (opened.length > before) row.clicked = opened[opened.length - 1];
      }
      rows.push(row);
    }
    return {
      rows: rows,
      autoSubscribed: autoSubscribed,
      monitored: monitored,
      opened: opened,
      historyIdsOpenedAsDetail: opened.filter((id) => /^manual:train/.test(id)).length,
      historyIdsOpenedAsMonitor: monitored.filter((id) => /^manual:train/.test(id)).length,
    };
  },

  /* 第五轮 Task 1：正式训练“打开结果文件夹”。
     身份只能来自当前页面的完整 studyId 与当前行的 train_name；请求期间不可重复
     提交、只影响当前记录；失败进既有操作错误区；迟到响应不改写当前研究。 */
  async formal_folder_open(api) {
    await bootstrap(api);
    const FOLDER = '打开结果文件夹';
    const runs = folderRuns();
    const studyStatus = {
      execution_status: 'COMPLETED', success_count: 2, has_success: true,
      approved_for_formal_training: true, best: bestPayload(ID_A),
    };
    await selectStudyWithRuns(api, studyStatus, runs);

    const box = () => api.$('hpoFormalRunsList');
    const folderBtn = (i) => collectButtons(box().children[i])
      .find((b) => b.textContent === FOLDER) || null;
    const rowLabels = (i) => collectButtons(box().children[i]).map((b) => b.textContent);
    const openCalls = () => api.requestLog.filter((u) => u.indexOf('/open-folder') >= 0);
    const rowText = (i) => box().children[i].textContent;

    const states = runs.map((run, i) => {
      const btn = folderBtn(i);
      return {
        trainName: run.train_name, status: run.status,
        present: !!btn, disabled: btn ? btn.disabled : null,
        labels: rowLabels(i),
      };
    });
    // 只有 completed + 来源事实合法 + best.pt 存在/缺失都无所谓 → 可打开
    const completedLabels = rowLabels(1);
    const completedBtnIndex = completedLabels.indexOf(FOLDER);
    const downloadIndex = completedLabels.indexOf('下载最终 best.pt');
    const resultsIndex = completedLabels.indexOf('查看结果');

    // 不符合规则的记录：即使绕过浏览器语义派发点击，也不得发出任何请求
    const refused = [];
    runs.forEach((run, i) => {
      const btn = folderBtn(i);
      if (btn && btn.disabled) { btn.dispatch('click'); refused.push(run.train_name); }
    });
    await api.drain();
    const refusedRequests = openCalls().length;

    // ── 成功路径：当前行 train3（completed，best.pt 存在）──
    const before = openCalls().length;
    const btn3 = folderBtn(1);
    if (btn3) btn3.dispatch('click');
    await api.drain();
    const firstRequest = {
      requests: openCalls().length - before,
      url: openCalls()[before] || null,
      body: (api.queued('/open-folder') || {}).body,
      pendingDisabled: btn3 ? btn3.disabled : null,
      // 一条记录的请求不得影响其他记录（train5 仍可点击）
      otherRowDisabled: folderBtn(3) ? folderBtn(3).disabled : null,
    };
    if (btn3) btn3.dispatch('click');       // pending 期间重复点击
    await api.drain();
    const afterRepeat = openCalls().length - before;

    if (api.pending().some((u) => u.indexOf('train3/open-folder') >= 0)) {
      await api.respond(ID_A + '/formal-runs/train3/open-folder', 200, {
        study_id: ID_A, train_name: 'train3', opened: true,
        error_code: null, error: null, next_action: null });
    }
    const afterSuccess = {
      disabled: btn3 ? btn3.disabled : null,
      hint: rowText(1),
      error: api.text('hpoStudyDetailError'),
      errorHidden: api.$('hpoStudyDetailError').classList.contains('hidden'),
    };

    // ── 失败路径：当前行 train5（completed，但服务端拒绝）──
    const failedBefore = openCalls().length;
    const btn5 = folderBtn(3);
    if (btn5) btn5.dispatch('click');
    await api.drain();
    const failureRequests = openCalls().length - failedBefore;
    if (api.pending().some((u) => u.indexOf('train5/open-folder') >= 0)) {
      await api.respond(ID_A + '/formal-runs/train5/open-folder', 409, {
        error_code: 'HPO_ARTIFACT_IDENTITY_MISMATCH',
        error: '该产物不属于当前研究或训练，未返回任何文件。',
        next_action: '请刷新界面后重新选择研究与试验。' });
    }
    const afterFailure = {
      disabled: btn5 ? btn5.disabled : null,
      errorHidden: api.$('hpoStudyDetailError').classList.contains('hidden'),
      errorText: api.text('hpoStudyDetailError'),
      errorInsideDetails: (() => {
        let node = api.$('hpoStudyDetailError');
        while (node) {
          if (node.tagName === 'DETAILS') return true;
          node = node.parentNode;
        }
        return false;
      })(),
      // 错误文案里绝不出现任何物理路径
      hasPath: /[A-Za-z]:[\\/]/.test(api.text('hpoStudyDetailError')),
      // 失败行自己不带成功提示，也不抹掉另一条记录已有的成功提示
      failedRowText: rowText(3),
      otherRowText: rowText(1),
    };

    // ── 迟到响应：切到研究 B 之后 A 的回复才到达 ──
    await selectStudyWithRuns(api, studyStatus, runs);
    if (folderBtn(1)) folderBtn(1).dispatch('click');
    await api.drain();
    const pendingForA = openCalls().length;
    api.window.hpoSelectStudy(ID_B);
    await api.drain();
    await api.respond('/studies/' + ID_B, 200, statusPayload(ID_B));
    await api.respond(ID_B + '/formal-runs', 200,
      { study_id: ID_B, count: 0, truncated: false, warnings: [], runs: [] });
    const errorBeforeStale = api.text('hpoStudyDetailError');
    await api.settle((url) => (url.indexOf('/open-folder') >= 0)
      ? { status: 200, body: { study_id: ID_A, train_name: 'train3', opened: true } }
      : defaultHandler(url));
    const afterStale = {
      studyId: api.state.studyId,
      errorBeforeStale: errorBeforeStale,
      errorAfterStale: api.text('hpoStudyDetailError'),
      listText: api.text('hpoFormalRunsList'),
    };

    return {
      states: states,
      completedOrder: { folder: completedBtnIndex, results: resultsIndex,
                        download: downloadIndex },
      refused: refused,
      refusedRequests: refusedRequests,
      firstRequest: firstRequest,
      afterRepeat: afterRepeat,
      afterSuccess: afterSuccess,
      failureRequests: failureRequests,
      afterFailure: afterFailure,
      pendingForA: pendingForA,
      afterStale: afterStale,
    };
  },

  /* 第五轮 Task 2：五项技术设置默认收进“高级技术选项”折叠区。
     默认值/设备探测/校验/提交字段不变，主摘要仍反映真实选择。 */
  async draft_collapsed_settings(api) {
    api.setInput('tuningModeSelect', 'hpo');
    api.fireDomReady();
    await api.drain();
    const beforeDefaults = {
      open: !!api.$('hpoDraftDetails').open,
      device: api.$('hpoDevice').value,
    };
    await api.settle(defaultHandler);
    const readCodings = () => ({
      open: !!api.$('hpoDraftDetails').open,
      sampler: api.$('hpoSampler').value,
      evaluationMode: api.$('hpoEvaluationMode').value,
      device: api.$('hpoDevice').value,
      imgsz: api.$('hpoImgsz').value,
      batch: api.$('hpoBatch').value,
      budget: api.$('hpoBudget').value,
      epochs: api.$('hpoEpochs').value,
      seed: api.$('hpoSeed').value,
      timeout: api.$('hpoTimeout').value,
      snapshot: api.$('hpoSnapshotSelect').value,
      model: api.$('hpoModelSelect').value,
      summary: api.$('hpoMainSummary').textContent,
      createDisabled: api.$('hpoCreateAndStartBtn').disabled,
    });
    const collapsed = readCodings();

    // 展开并逐项修改：主摘要必须立即反映真实选择
    api.$('hpoDraftDetails').open = true;
    await api.drain();
    const afterExpand = { summary: api.$('hpoMainSummary').textContent };
    api.setInput('hpoSampler', 'random');
    api.$('hpoSampler').dispatch('change');
    await api.drain();
    const afterSampler = api.$('hpoMainSummary').textContent;
    api.setInput('hpoImgsz', '320');
    api.$('hpoImgsz').dispatch('change');
    await api.drain();
    const afterImgsz = api.$('hpoMainSummary').textContent;
    api.setInput('hpoBatch', '8');
    api.$('hpoBatch').dispatch('change');
    await api.drain();
    const afterBatch = api.$('hpoMainSummary').textContent;
    api.setInput('hpoDevice', 'cpu');
    api.$('hpoDevice').dispatch('change');
    await api.drain();
    const afterDevice = api.$('hpoMainSummary').textContent;
    api.setInput('hpoEvaluationMode', 'quick');
    api.$('hpoEvaluationMode').dispatch('change');
    await api.drain();
    const afterMode = api.$('hpoMainSummary').textContent;

    // 创建请求仍携带完整配置
    api.window.hpoCreateAndStart();
    await api.drain();
    const createBody = (api.queued('/api/hpo/studies') || {}).body;
    await api.settle((url) => (url.indexOf('/start') >= 0)
      ? { status: 202, body: { started: true } } : defaultHandler(url));

    // 切换模式并重新进入 HPO：不得复制控件，也不得丢掉草稿值
    api.window.onTuningModeChange('full');
    api.window.onTuningModeChange('hpo');
    await api.settle(defaultHandler);
    const afterModeSwitch = {
      sampler: api.$('hpoSampler').value,
      evaluationMode: api.$('hpoEvaluationMode').value,
      device: api.$('hpoDevice').value,
      imgsz: api.$('hpoImgsz').value,
      batch: api.$('hpoBatch').value,
      // 每个控件在整页中仍然只有一个实例
      counts: ['hpoSampler', 'hpoEvaluationMode', 'hpoDevice', 'hpoImgsz',
               'hpoBatch', 'hpoBudget', 'hpoEpochs', 'hpoSeed', 'hpoTimeout',
               'hpoCreateConfirmDetail', 'hpoFieldError', 'hpoDraftDetails']
        .map((id) => api.all.filter((e) => e.id === id).length),
    };

    return {
      beforeDefaults: beforeDefaults,
      collapsed: collapsed,
      afterExpand: afterExpand,
      afterSampler: afterSampler,
      afterImgsz: afterImgsz,
      afterBatch: afterBatch,
      afterDevice: afterDevice,
      afterMode: afterMode,
      createBody: createBody,
      afterModeSwitch: afterModeSwitch,
    };
  },

  /* 第五轮 Task 2：折叠区内的字段校验失败时必须自动展开并在折叠区外报错，
     且不发送创建请求、不清除已选的数据快照与权重。 */
  async draft_hidden_field_error(api) {
    api.setInput('tuningModeSelect', 'hpo');
    api.fireDomReady();
    await api.settle(defaultHandler);
    const before = {
      open: !!api.$('hpoDraftDetails').open,
      snapshot: api.$('hpoSnapshotSelect').value,
      model: api.$('hpoModelSelect').value,
    };
    const read = () => ({
      open: !!api.$('hpoDraftDetails').open,
      errorHidden: api.$('hpoFieldError').classList.contains('hidden'),
      errorText: api.text('hpoFieldError'),
      errorInsideDetails: (() => {
        let node = api.$('hpoFieldError');
        while (node) {
          if (node.tagName === 'DETAILS') return true;
          node = node.parentNode;
        }
        return false;
      })(),
      snapshot: api.$('hpoSnapshotSelect').value,
      model: api.$('hpoModelSelect').value,
    });
    const createCalls = () =>
      api.requestLog.filter((u) => u === '/api/hpo/studies').length;

    // imgsz 不是 32 的整数倍（折叠区内字段）
    api.setInput('hpoImgsz', '100');
    api.window.hpoCreateAndStart();
    await api.drain();
    const badImgsz = read();
    const imgszCalls = createCalls();

    // Batch 越界
    api.setInput('hpoImgsz', '640');
    api.setInput('hpoBatch', '999');
    // 折叠区由用户收起后再次提交：必须再次自动展开
    api.$('hpoDraftDetails').open = false;
    api.window.hpoCreateAndStart();
    await api.drain();
    const badBatch = read();
    const batchCalls = createCalls();

    // 评价模式非法（取值不在允许集合内）
    api.setInput('hpoBatch', '16');
    api.$('hpoEvaluationMode').value = 'legacy_map50_95';
    api.$('hpoDraftDetails').open = false;
    api.window.hpoCreateAndStart();
    await api.drain();
    const badMode = read();
    const modeCalls = createCalls();

    // 未取得合法设备
    api.setInput('hpoEvaluationMode', 'comprehensive');
    api.$('hpoDevice').value = '';
    api.$('hpoDraftDetails').open = false;
    api.window.hpoCreateAndStart();
    await api.drain();
    const badDevice = read();
    const deviceCalls = createCalls();

    // 服务端字段错误同样必须展开折叠区（采样器不在前端校验范围内）
    api.setInput('hpoDevice', '0');
    api.$('hpoDraftDetails').open = false;
    api.window.hpoCreateAndStart();
    await api.drain();
    const queuedCreate = api.pending().some((u) => u === '/api/hpo/studies');
    await api.respond('/api/hpo/studies', 422, {
      error_code: 'INVALID_HPO_FIELD', field: 'sampler', reason_code: 'FIELD_VALUE',
      error: '取值不在允许的选项内，请从界面提供的选项中选择。',
      next_action: '请检查并修正输入后重试。' });
    const serverSampler = read();

    return {
      before: before,
      badImgsz: badImgsz, imgszCalls: imgszCalls,
      badBatch: badBatch, batchCalls: batchCalls,
      badMode: badMode, modeCalls: modeCalls,
      badDevice: badDevice, deviceCalls: deviceCalls,
      queuedCreate: queuedCreate,
      serverSampler: serverSampler,
    };
  },

  /* P1b：训练监控首屏（服务端渲染的“最近有效训练”投影）。 */
  async monitor_first_paint(api) {
    api.fireDomReady();
    await api.settle(pageHandler);
    return { cards: readMonitorCards(api) };
  },

  /* P1b：重连终态事件把结果投影到监控卡片；他 run / 重复 / 乱序事件不得污染。 */
  async monitor_terminal_metrics(api) {
    api.fireDomReady();
    await api.settle(pageHandler);
    const beforeSwitch = readMonitorCards(api);
    api.window.locateTrainingRun(RUN_A);
    await api.drain();
    await api.respondStream('/api/runs/', [
      sse({ event: 'training_log', run_id: RUN_A, event_seq: 1, message: 'epoch 1/2' }),
      sse({ status: 'completed', run_id: RUN_A, event_seq: 2, message: '训练完成',
            result: finalizedResult('train9', 2) }),
      // 另一条 run 的迟到事件：绝不能改变当前卡片
      sse({ status: 'completed', run_id: 'manual:bbbbbbbb-2222-4222-8222-222222222222',
            event_seq: 1, result: finalizedResult('train3', 9) }),
      // 重复/乱序 seq：不得重复处理，也不得回滚已投影的指标
      sse({ status: 'completed', run_id: RUN_A, event_seq: 2,
            result: finalizedResult('train9', 7) }),
      sse({ status: 'completed', run_id: RUN_A, event_seq: 1,
            result: finalizedResult('train9', 4) }),
    ]);
    await api.settle(pageHandler);
    return {
      beforeSwitch: beforeSwitch,
      afterTerminal: readMonitorCards(api),
      logText: api.text('monitorLog'),
    };
  },

  /* P1b：共用投影的取值规则（真实 0 / 缺失 / 报告链接）——使用真实最终化结构。 */
  async monitor_result_projection(api) {
    api.fireDomReady();
    await api.settle(pageHandler);
    const apply = api.window.TrainingMonitor.applyResultToMonitor;
    apply({ run_name: 'train1', params: { epochs: 0 },
            epochs: { configured: 0, completed: 0, best: 0 },
            metrics: { mAP50: 0, mAP50_95: 0, precision: 0, recall: 0 },
            artifacts: { report_path: 'log/train1_report.json' } });
    const zeros = readMonitorCards(api);
    apply({ run_name: 'train2', params: { epochs: 3 },
            epochs: { configured: 3, completed: 3, best: 2 },
            metrics: { mAP50: 0.5 }, artifacts: {} });
    const partial = readMonitorCards(api);
    apply({ run_name: 'train3', params: { epochs: 2 },
            epochs: { configured: 2, completed: 2, best: 2 },
            metrics: { mAP50: 0.12345, mAP50_95: 0.98765,
                       precision: 0.1234, recall: 0.9876 },
            artifacts: { report_path: 'log/train3_report.json' } });
    const rounded = readMonitorCards(api);
    return { zeros: zeros, partial: partial, rounded: rounded };
  },

  /* epochs 卡片的取值契约：结构化对象优先 configured，逐级回退，缺失诚实显示“—”。
     这正是“真实终态事件把 epochs 显示成 —”所对应的行为。 */
  async monitor_epochs_contract(api) {
    api.fireDomReady();
    await api.settle(pageHandler);
    const apply = api.window.TrainingMonitor.applyResultToMonitor;
    const read = () => api.text('monitorEpochs');
    const facts = {};
    // 1. 真实最终化结构：configured 优先（即使 completed/best/params 不同）
    apply({ run_name: 't1', params: { epochs: 9 },
            epochs: { configured: 2, completed: 5, best: 4 } });
    facts.configuredPreferred = read();
    // 2. configured 缺失 → completed
    apply({ run_name: 't2', params: { epochs: 9 },
            epochs: { completed: 5, best: 4 } });
    facts.completedFallback = read();
    // 3. configured/completed 都缺失 → params.epochs
    apply({ run_name: 't3', params: { epochs: 9 }, epochs: { best: 4 } });
    facts.paramsFallback = read();
    // 4. 真实 0 是有效值，不得当成缺失
    apply({ run_name: 't4', params: { epochs: 0 },
            epochs: { configured: 0, completed: 0, best: 0 } });
    facts.realZero = read();
    // 5. 整个 epochs 结构缺失 → 诚实显示“—”，绝不补造
    apply({ run_name: 't5', params: {}, metrics: {} });
    facts.missingStructure = read();
    // 6. 旧标量事件仍然兼容
    apply({ run_name: 't6', epochs: 3 });
    facts.legacyScalar = read();
    // 7. 候选值全部非法 → “—”，不回退成 0
    apply({ run_name: 't7', params: { epochs: 'none' },
            epochs: { configured: 'unknown', completed: null, best: 'x' } });
    facts.allIllegal = read();
    return facts;
  },

  /* P1b：普通训练原始流与重连流必须得到一致的 DOM 结果。 */
  async monitor_stream_parity(api) {
    api.fireDomReady();
    await api.settle(pageHandler);
    const result = finalizedResult('train5', 2,
      { mAP50: 0.1234, mAP50_95: 0.5678, precision: 0.891, recall: 0.234 });
    // A. 重连流（页面刷新 / 从关联结果定位）
    api.window.locateTrainingRun(RUN_A);
    await api.drain();
    await api.respondStream('/api/runs/', [
      sse({ status: 'completed', run_id: RUN_A, event_seq: 1, result: result }),
    ]);
    await api.settle(pageHandler);
    const reconnect = readMonitorCards(api);

    // B. 普通训练原始响应流（同一份 result）
    api.setInput('trainDataYaml', 'E:/data/data.yaml');
    api.setInput('trainModel', 'yolov8n.pt');
    api.setInput('trainEpochs', '2');
    api.setInput('trainImgsz', '96');
    api.setInput('trainBatch', '1');
    api.window.startTraining();
    await api.drain();
    await api.respondStream('/api/training/start', [
      sse({ status: 'done', run_id: 'manual:cccccccc-3333-4333-8333-333333333333',
            event_seq: 1, result: result }),
    ]);
    await api.settle(pageHandler);
    return { reconnect: reconnect, legacy: readMonitorCards(api) };
  },

  /* P1 返修：PAUSED 后的“检查并恢复”必须可见且可点击，pending 期间不可重复提交。 */
  async resume_button_state(api) {
    await bootstrap(api);
    // can_resume=false：入口隐藏，不留无意义的禁用按钮
    await selectStudyWithRuns(api, {
      execution_status: 'PAUSED', can_resume: false, terminal_count: 1 }, []);
    const notResumable = {
      visible: api.visible('hpoResumeBtn'),
      disabled: api.$('hpoResumeBtn').disabled,
    };
    // can_resume=true：可见且可用
    await selectStudyWithRuns(api, {
      execution_status: 'PAUSED', control_active: false, can_resume: true,
      terminal_count: 1, budget: 2 }, []);
    const resumable = {
      visible: api.visible('hpoResumeBtn'),
      disabled: api.$('hpoResumeBtn').disabled,
      label: api.$('hpoResumeBtn').textContent,
    };

    const resumeUrls = () => api.requestLog.filter((u) => u.indexOf('/resume') >= 0);
    api.window.hpoResumeStudy();
    await api.drain();
    const firstClick = {
      requests: resumeUrls().length,
      url: resumeUrls()[0] || null,
      // 提交后立即进入 pending：不能重复点击
      disabled: api.$('hpoResumeBtn').disabled,
      label: api.$('hpoResumeBtn').textContent,
      status: api.text('hpoStatusText'),
    };
    api.window.hpoResumeStudy();
    await api.drain();
    const afterSecondClick = resumeUrls().length;

    // 服务端接受后等待轮询收敛为 RUNNING：恢复入口消失，绝不伪造已恢复
    await api.settle((url) => {
      if (url.indexOf('/resume') >= 0) {
        return { status: 202, body: { status: 'accepted' } };
      }
      if (url.indexOf('/studies/' + ID_A) >= 0) {
        return { status: 200, body: statusPayload(ID_A, {
          execution_status: 'RUNNING', control_active: true, can_stop: true,
          can_resume: false }) };
      }
      return defaultHandler(url);
    });
    return {
      notResumable: notResumable,
      resumable: resumable,
      firstClick: firstClick,
      afterSecondClick: afterSecondClick,
      studyId: api.state.studyId,
      running: {
        visible: api.visible('hpoResumeBtn'),
        statusText: api.text('hpoStatusText'),
        stopVisible: api.visible('hpoStopBtn'),
      },
    };
  },

  /* P1 返修：恢复失败或结果未知时，按钮必须重新可操作且错误保持可见。 */
  async resume_failure_reenables(api) {
    await bootstrap(api);
    await selectStudyWithRuns(api, {
      execution_status: 'PAUSED', control_active: false, can_resume: true,
      terminal_count: 1 }, []);
    api.window.hpoResumeStudy();
    await api.drain();
    await api.settle((url) => {
      if (url.indexOf('/resume') >= 0) {
        return { status: 409, body: {
          error_code: 'HPO_EXECUTION_CONFLICT', error: '该研究当前不可恢复。',
          next_action: '请稍后重试。' } };
      }
      return defaultHandler(url);
    });
    const afterServerError = {
      disabled: api.$('hpoResumeBtn').disabled,
      visible: api.visible('hpoResumeBtn'),
      errorHidden: api.$('hpoStudyDetailError').classList.contains('hidden'),
      errorText: api.text('hpoStudyDetailError'),
    };

    // 网络结果未知：同样必须重新可操作，错误替换为未知结果提示
    api.window.hpoResumeStudy();
    await api.drain();
    await api.reject('/resume');
    const afterNetworkError = {
      disabled: api.$('hpoResumeBtn').disabled,
      errorText: api.text('hpoStudyDetailError'),
      resumeRequests: api.requestLog.filter((u) => u.indexOf('/resume') >= 0).length,
    };
    // 操作错误必须留在折叠区之外
    const inDetails = (() => {
      let node = api.$('hpoStudyDetailError');
      while (node) {
        if (node.tagName === 'DETAILS') return true;
        node = node.parentNode;
      }
      return false;
    })();
    return {
      afterServerError: afterServerError,
      afterNetworkError: afterNetworkError,
      errorInsideDetails: inDetails,
    };
  },

  /* 少选择：可靠绑定只留摘要；缺失/不合法才自动展开并说明原因。 */
  async binding_convergence(api) {
    await bootstrap(api);
    const reliable = {
      noticeHidden: api.$('hpoBindingNotice').classList.contains('hidden'),
      modelSelectValue: api.$('hpoModelSelect').value,
      modelSelectVisible: !api.$('hpoModelSelect').classList.contains('hidden'),
      snapshotSelectVisible: !api.$('hpoSnapshotSelect').classList.contains('hidden'),
      draftDetailsOpen: !!api.$('hpoDraftDetails').open,
      createConfirm: api.$('hpoCreateConfirm').textContent,
      datasetSummary: api.text('hpoDatasetSummary'),
      modelSummary: api.text('hpoModelSummary'),
    };
    // 绑定只能来自受控列表：选择即成为提交值，主摘要只显示模型名称
    api.setInput('hpoModelSelect', 'E:/models/other.pt');
    api.$('hpoModelSelect').dispatch('change');
    await api.drain();
    const afterSelect = {
      selectValue: api.$('hpoModelSelect').value,
      draftModel: api.state ? null : null,
      modelSummary: api.text('hpoModelSummary'),
      confirm: api.$('hpoCreateConfirm').textContent,
    };
    return {
      reliable: reliable,
      afterSelect: afterSelect,
      modelPathExists: api.all.filter((e) => e.id === 'hpoModelPath').length,
    };
  },

  /* 绑定缺失/不合法：自动展开必要选择并显示原因，且不自动换数据/权重。 */
  async binding_missing_is_explained(api) {
    const missing = defaultsPayload();
    missing.dataset = { registered: true, display_name: null, snapshot: null,
      reason_code: 'NO_MATCHING_SNAPSHOT', needs_user_confirmation: true };
    missing.model = { path: null, name: null, source: null, available: false,
      reason_code: 'MODEL_CONFIG_INVALID' };
    api.setInput('tuningModeSelect', 'hpo');
    api.fireDomReady();
    await api.settle((url) => {
      if (url.indexOf('/api/hpo/defaults') >= 0) return { status: 200, body: missing };
      if (url.indexOf('/api/hpo/snapshots') >= 0) {
        return { status: 200, body: { snapshots: [], count: 0 } };
      }
      return defaultHandler(url);
    });
    return {
      noticeHidden: api.$('hpoBindingNotice').classList.contains('hidden'),
      noticeText: api.$('hpoBindingNotice').textContent,
      modelValue: api.$('hpoModelSelect').value,
      snapshotValue: api.$('hpoSnapshotSelect').value,
      snapshotSelectVisible: !api.$('hpoSnapshotSelect').classList.contains('hidden'),
      modelSelectVisible: !api.$('hpoModelSelect').classList.contains('hidden'),
      mainSummary: api.$('hpoMainSummary').textContent,
    };
  },

  /* 公共控制区位置：DOM 顺序上必须排在 LLM/HPO 可变内容之前；无任务不显示
     停止/恢复/启动按钮。像素几何由 Codex 用真实浏览器验收。 */
  async control_bar_position(api) {
    await bootstrap(api);
    const common = api.$('tuningCommonControls');
    const idx = (el) => api.order.indexOf(el);
    const firstWithClass = (cls) =>
      api.order.filter((e) => e.classList.contains(cls))[0] || null;
    const llmCard = firstWithClass('llm-only');
    const order = {
      beforeModeSelect: idx(common) < idx(api.$('tuningModeSelect')),
      beforeLlmSuggestion: llmCard !== null && idx(common) < idx(llmCard),
      beforeSearchConfig: idx(common) < idx(api.$('hpoSearchConfig')),
      beforeBestArea: idx(common) < idx(api.$('hpoBestArea')),
      beforeDraft: idx(common) < idx(api.$('hpoCreateDraft')),
      beforeProgressCard: idx(common) < idx(api.$('hpoProgressCard')),
      beforeHpoMainSummary: idx(common) < idx(api.$('hpoMainSummary')),
    };
    const buttons = {
      start: api.visible('hpoStartBtn'),
      stop: api.visible('hpoStopBtn'),
      resume: api.visible('hpoResumeBtn'),
    };
    // READY 且无活动控制器：允许显式启动，因此“启动此任务”出现，停止/恢复不出现
    await selectStudyWithRuns(api, { execution_status: 'READY' }, []);
    const afterReady = {
      start: api.visible('hpoStartBtn'),
      stop: api.visible('hpoStopBtn'),
      resume: api.visible('hpoResumeBtn'),
    };
    await selectStudyWithRuns(api, {
      execution_status: 'RUNNING', control_active: true, can_stop: true }, []);
    const running = {
      start: api.visible('hpoStartBtn'),
      stop: api.visible('hpoStopBtn'),
      resume: api.visible('hpoResumeBtn'),
    };
    return { order: order, buttons: buttons, afterReady: afterReady, running: running };
  },

  /* 列表读取失败也不得丢掉已绑定/权威默认的数据与权重。 */
  async binding_survives_a_failed_listing(api) {
    api.setInput('tuningModeSelect', 'hpo');
    api.fireDomReady();
    await api.settle((url) => {
      if (url.indexOf('/api/hpo/defaults') >= 0) return { status: 200, body: defaultsPayload() };
      if (url.indexOf('/api/hpo/snapshots') >= 0) return { status: 500, body: { error: 'x' } };
      if (url.indexOf('/api/hpo/local-models') >= 0) return { status: 500, body: {} };
      return defaultHandler(url);
    });
    return {
      snapshotValue: api.$('hpoSnapshotSelect').value,
      snapshotHint: api.$('hpoSnapshotHint').textContent,
      modelSelectValue: api.$('hpoModelSelect').value,
      modelSummary: api.text('hpoModelSummary'),
      datasetSummary: api.text('hpoDatasetSummary'),
      noticeHidden: api.$('hpoBindingNotice').classList.contains('hidden'),
    };
  },

  /* 错误与 warnings 始终可见：即使折叠、即使轮询成功也不消失。 */
  async errors_survive_collapse_and_poll(api) {
    await bootstrap(api);
    await selectStudyWithRuns(api, { execution_status: 'READY' },
      [], [{ code: 'FORMAL_SOURCE_UNREADABLE', train_name: 'train7' }]);
    const warning = {
      hidden: api.$('hpoFormalRunsWarning').classList.contains('hidden'),
      text: api.$('hpoFormalRunsWarning').textContent,
      bestDetailsOpen: !!api.$('hpoBestDetails').open,
      formalRunsHidden: api.$('hpoFormalRuns').classList.contains('hidden'),
    };
    api.window.hpoStartStudy();
    await api.drain();
    await api.respond('/start', 500, { error_code: 'HPO_EXECUTION_ERROR',
      error: '启动失败：执行器暂不可用。', next_action: '请稍后重试。' });
    const afterFailure = api.text('hpoStudyDetailError');
    // 一次正常轮询不得抹掉用户操作错误
    api.window.hpoRefreshRound(ID_A, true);
    await api.settle(defaultHandler);
    return {
      warning: warning,
      afterFailure: afterFailure,
      detailErrorHidden: api.$('hpoStudyDetailError').classList.contains('hidden'),
      afterPoll: api.text('hpoStudyDetailError'),
    };
  },

  /* 第四轮 Task 2：评价模式进入创建请求；正式训练条件与搜索阶段锁定一致。 */
  async evaluation_mode_and_locked_formal(api) {
    await bootstrap(api);
    const modeSelect = api.$('hpoEvaluationMode');
    // 新建研究选择器只有全面/快速两项（legacy 只能被读取与展示）
    const selectorValues = modeSelect.options.map((o) => o.value);
    // 默认全面模式（服务端权威默认）
    const defaultMode = modeSelect.value;
    api.setInput('hpoEvaluationMode', 'quick');
    api.$('hpoEvaluationMode').dispatch('change');
    await api.drain();
    api.window.hpoCreateAndStart();
    await api.drain();
    const createRequest = api.queued('/api/hpo/studies');
    await api.settle((url) => (url.indexOf('/resume') >= 0 || url.indexOf('/start') >= 0)
      ? { status: 202, body: { started: true } } : defaultHandler(url));
    const quickBody = createRequest ? createRequest.body : null;

    // 正式训练：只提交 epochs 输入，其余三项取当前研究的权威执行条件
    await selectStudyWithRuns(api, {
      execution_status: 'COMPLETED', success_count: 2, has_success: true,
      approved_for_formal_training: true, best: bestPayload(ID_A),
      batch: 4, imgsz: 96, device: '0'}, []);
    api.setInput('hpoFormalEpochs', '7');
    api.window.hpoStartFormalTraining();
    await api.drain();
    const formalRequest = api.queued('/train-best');
    await api.respond('/train-best', 202, {
      status: 'accepted', run_id: RUN_A, train_name: 'train1', mode: 'formal',
      source: {}, training_config: { epochs: 7, batch: 4, imgsz: 96, device: '0' },
      differences: { epochs: { original: 30, requested: 7 } }, links: {},
      error_code: null, error: null, next_action: null });
    await api.settle(defaultHandler);
    return {
      selectorValues: selectorValues,
      defaultMode: defaultMode,
      quickBody: quickBody,
      formalBody: formalRequest ? formalRequest.body : null,
      formalStatus: api.text('hpoFormalStatus'),
    };
  },

  /* 第四轮 Task 3：进度只来自服务端事实；当前项缺失诚实显示“—”。 */
  async progress_panel(api) {
    await bootstrap(api);
    const read = () => ({
      status: api.text('hpoProgressStatus'),
      counts: api.text('hpoProgressCounts'),
      percent: api.text('hpoProgressPercent'),
      width: api.$('hpoProgressBar').style.width,
      current: api.text('hpoProgressCurrent'),
      tally: api.text('hpoProgressTally'),
      best: api.text('hpoProgressBest'),
      message: api.text('hpoProgressMessage'),
      ranking: api.text('hpoRanking'),
      rankingNote: api.text('hpoRankingNote'),
      visible: api.visible('hpoProgress'),
    });
    await selectStudyWithRuns(api, {
      execution_status: 'RUNNING', control_active: true, can_stop: true,
      budget: 10, terminal_count: 4, success_count: 3, failed_count: 1,
      cancelled_count: 0, interrupted_count: 0, running_count: 1,
      remaining_count: 6, current_trial_number: 5,
      evaluation_mode: 'comprehensive', best: bestPayload(ID_A),
      // 同分排名：诚实说明按编号优先，绝不声称精度更高
      ranking: [{ rank: 1, trial_number: 2, number: 1, value: 0.9, epoch: 5 },
                { rank: 2, trial_number: 1, number: 0, value: 0.9, epoch: 3 }]}, []);
    const running = read();
    await selectStudyWithRuns(api, {
      execution_status: 'PAUSED', can_resume: true, budget: 10,
      terminal_count: 4, success_count: 3, evaluation_mode: 'comprehensive'}, []);
    const paused = read();
    await selectStudyWithRuns(api, {
      execution_status: 'COMPLETED', budget: 10, terminal_count: 10,
      success_count: 9, remaining_count: 0,
      evaluation_mode: 'comprehensive', best: bestPayload(ID_A)}, []);
    const completed = read();
    await selectStudyWithRuns(api, {
      execution_status: 'BLOCKED', budget: 10, terminal_count: 4,
      failed_count: 4, current_trial_number: null,
      evaluation_mode: 'comprehensive'}, []);
    const blocked = read();
    return {
      running: running, paused: paused, completed: completed, blocked: blocked,
      trialsBodyCount: api.all.filter((e) => e.id === 'hpoTrialsBody').length,
      trialTableElements: api.all.filter((e) => e.id === 'hpoTrialsTable').length,
    };
  },

  /* 第四轮 Task 3：HPO 历史记录入口默认收起，展开后可翻页与选择。 */
  async history_toggle(api) {
    await bootstrap(api);
    const initial = {
      sectionVisible: api.visible('hpoHistorySection'),
      toggleVisible: api.visible('hpoHistoryToggle'),
      label: api.$('hpoHistoryToggle').textContent,
    };
    api.window.hpoToggleHistory();
    await api.settle((url) => {
      if (url.indexOf('/api/hpo/studies?') >= 0) {
        return { status: 200, body: { count: 3, offset: 0, limit: 20, studies: [
          { study_id: ID_A, readable: true, transient: false,
            created_at: '2026-09-15T10:00:00Z', sampler: 'tpe', budget: 10,
            execution_status: 'COMPLETED', execution_error_code: null,
            terminal_count: 6, success_count: 5, dataset_name: 'demo',
            evaluation_mode: 'comprehensive' },
          { study_id: ID_B, readable: true, transient: false,
            created_at: '2026-09-14T10:00:00Z', sampler: 'random', budget: 4,
            execution_status: 'COMPLETED', execution_error_code: null,
            terminal_count: 4, success_count: 2, dataset_name: 'legacy-ds',
            evaluation_mode: 'legacy_map50_95' },
          // 旧记录：无法解析数据集名称 → 诚实缺失文案，绝不显示短身份
          { study_id: ID_C, readable: true, transient: false,
            created_at: '2026-09-13T10:00:00Z', sampler: 'tpe', budget: 2,
            execution_status: 'COMPLETED', execution_error_code: null,
            terminal_count: 2, success_count: 1, dataset_name: null,
            evaluation_mode: 'legacy_map50_95' },
          // 损坏记录：可见但不可选，且绝不冒充“数据集不可用”或提供入口
          { study_id: 'hpo_' + 'e'.repeat(32), readable: false, transient: false,
            created_at: null, sampler: null, budget: null,
            execution_status: null, execution_error_code: 'HPO_CORRUPT_STUDY',
            terminal_count: null, success_count: null, dataset_name: null,
            evaluation_mode: null },
        ] } };
      }
      return defaultHandler(url);
    });
    const opened = {
      sectionVisible: api.visible('hpoHistorySection'),
      label: api.$('hpoHistoryToggle').textContent,
      rows: api.$('hpoHistoryList').children.map((c) => c.textContent),
    };
    // 可选择的“查看”按钮只属于可读记录：损坏记录行没有任何入口
    // （必须在选择之前读取：选择会重新渲染列表，children 是活动数组）
    const listRows = api.$('hpoHistoryList').children.slice();
    const selectableRows = listRows
      .map((child, idx) => ({ idx: idx, buttons: collectButtons(child).length }))
      .filter((r) => r.buttons > 0)
      .map((r) => r.idx);
    const lastRowHasButton = collectButtons(listRows[listRows.length - 1]).length;
    const target = collectButtons(api.$('hpoHistoryList'))[0];
    target.dispatch('click');
    await api.settle(defaultHandler);
    const afterSelect = { studyId: api.state.studyId };
    api.window.hpoToggleHistory();
    const collapsed = { sectionVisible: api.visible('hpoHistorySection') };
    return {
      initial: initial, opened: opened, afterSelect: afterSelect,
      collapsed: collapsed,
      selectableRows: selectableRows,
      lastRowHasButton: lastRowHasButton,
      pageButtons: ['hpoStudyPagePrev', 'hpoStudyPageNext']
        .map((id) => api.all.filter((e) => e.id === id).length),
    };
  },

  /* 最终返修 Task 3：结果操作区只在真正完成后出现（不是只禁用），
     且主界面不再有冗长的技术解释。 */
  async best_result_actions(api) {
    await bootstrap(api);
    const read = () => ({
      region: api.visible('hpoResultActions'),
      openFolder: api.$('hpoOpenResultFolder').disabled,
      download: api.$('hpoDownloadBest').disabled,
      hint: api.text('hpoBestArtifactHint'),
      hintVisible: api.visible('hpoBestArtifactHint'),
      text: api.$('hpoBestArea').textContent,
      labels: collectButtons(api.$('hpoBestArea')).map((b) => b.textContent),
    });
    // RUNNING：仍有阶段性 best，但研究未完成 → 整个操作区不可见
    await selectStudyWithRuns(api, {
      execution_status: 'RUNNING', control_active: true, can_stop: true,
      success_count: 1, has_success: true,
      approved_for_formal_training: false, best: bestPayload(ID_A),
      ranking: [{ rank: 1, trial_number: 2, number: 1, value: 0.9, epoch: 5 }]}, []);
    const staged = read();
    // PAUSED：可恢复但未完成 → 不可见
    await selectStudyWithRuns(api, {
      execution_status: 'PAUSED', can_resume: true, success_count: 1,
      has_success: true, approved_for_formal_training: false,
      best: bestPayload(ID_A)}, []);
    const paused = read();
    // FAILED/BLOCKED：有历史成功结果也不算完成 → 不可见
    await selectStudyWithRuns(api, {
      execution_status: 'BLOCKED', success_count: 1, has_success: true,
      approved_for_formal_training: false, best: bestPayload(ID_A)}, []);
    const failed = read();
    // 已完成但没有 rank-1 成功试验 → 不可见
    await selectStudyWithRuns(api, {
      execution_status: 'COMPLETED', success_count: 0, has_success: false,
      approved_for_formal_training: false, best: null}, []);
    const noBest = read();

    await selectStudyWithRuns(api, {
      execution_status: 'COMPLETED', success_count: 2, has_success: true,
      approved_for_formal_training: true, best: bestPayload(ID_A)}, []);
    const approved = read();
    // 下载只提交研究/试验/白名单产物名身份，绝不提交路径
    api.window.hpoDownloadBestPt();
    const downloadUrl = api.downloads[api.downloads.length - 1] || null;

    // best.pt 缺失：仍可打开结果文件夹，但下载入口禁用且安全 warning 可见
    const missing = bestPayload(ID_A);
    missing.artifacts = { best_pt_available: false, last_pt_available: true };
    await selectStudyWithRuns(api, {
      execution_status: 'COMPLETED', success_count: 2, has_success: true,
      approved_for_formal_training: true, best: missing}, []);
    const noBestPt = read();
    return {
      staged: staged,
      paused: paused,
      failed: failed,
      noBest: noBest,
      approved: approved,
      noBestPt: noBestPt,
      downloadUrl: downloadUrl,
      downloads: api.downloads.length,
      resultActionsCount: api.all.filter((e) => e.id === 'hpoResultActions').length,
    };
  },

  /* 第四轮 Task 3：HPO 模式隐藏训练分析目录输入，其它模式保持原功能。 */
  async training_analysis_card_visibility(api) {
    const display = () => {
      let node = api.$('trainingPathInput');
      while (node && !(node.classList && node.classList.contains('card'))) {
        node = node.parentNode;
      }
      return node ? node.style.display : null;
    };
    api.setInput('tuningModeSelect', 'hpo');
    api.fireDomReady();
    await api.settle(defaultHandler);
    const hpo = { inputDisplay: display(), mode: api.$('tuningModeSelect').value };
    api.window.onTuningModeChange('dry_run');
    const other = { inputDisplay: display() };
    api.window.onTuningModeChange('hpo');
    const backToHpo = { inputDisplay: display() };
    return { hpo: hpo, other: other, backToHpo: backToHpo };
  },

  /* 最终返修 Task 1：HPO 模式只显示 HPO 内容。
     非 HPO 区块（训练总结 / LLM 分析卡 / 视觉分析卡 / 逐项“调优历史” /
     训练目录输入）必须整体不可见，而不是只折叠；切回其它模式后恢复。 */
  async hpo_mode_isolation(api) {
    // 文本定位：取包含该文本的最内层元素，再沿真实祖先链判断是否可见
    function probe(text) {
      let found = null;
      for (const el of api.all) {
        const t = el.textContent;
        if (t && t.indexOf(text) >= 0 &&
            (!found || t.length < found.textContent.length)) {
          found = el;
        }
      }
      if (!found) return { found: false, visible: false };
      let node = found;
      while (node) {
        if (node.classList && node.classList.contains('hidden')) {
          return { found: true, visible: false };
        }
        if (node.style && node.style.display === 'none') {
          return { found: true, visible: false };
        }
        node = node.parentNode;
      }
      return { found: true, visible: true };
    }
    const read = () => ({
      trainingSummary: probe('Training Summary'),
      llmAnalysis: probe('LLM Analysis Report'),
      visionAnalysis: probe('Vision Analysis'),
      tuningHistory: probe('Tuning History'),
      trainingAnalysisInput: probe('Enter the path to a YOLO train directory'),
      historyToggle: api.visible('hpoHistoryToggle'),
      historySection: api.visible('hpoHistorySection'),
      historyRows: api.$('hpoHistoryList').children.map((c) => c.textContent),
    });
    api.setInput('tuningModeSelect', 'hpo');
    api.fireDomReady();
    await api.settle(defaultHandler);
    const hpo = read();
    api.window.hpoToggleHistory();
    await api.settle((url) => {
      if (url.indexOf('/api/hpo/studies?') >= 0) {
        return { status: 200, body: { count: 1, offset: 0, limit: 20, studies: [
          { study_id: ID_A, readable: true, transient: false,
            created_at: '2026-09-15T10:00:00Z', sampler: 'tpe', budget: 2,
            execution_status: 'COMPLETED', execution_error_code: null,
            terminal_count: 1, success_count: 1, dataset_name: 'real-ds',
            evaluation_mode: 'comprehensive' },
        ] } };
      }
      return defaultHandler(url);
    });
    const historyOpen = read();
    api.window.onTuningModeChange('full');
    const full = read();
    api.window.onTuningModeChange('hpo');
    await api.settle(defaultHandler);
    const backToHpo = read();
    return { hpo: hpo, historyOpen: historyOpen, full: full, backToHpo: backToHpo };
  },

  /* 第四轮 Task 5：HPO 正式训练的 202 自动选中同一 runtime run 并订阅监控。 */
  async formal_monitor_reuse(api) {
    api.setInput('tuningModeSelect', 'hpo');
    api.fireDomReady();
    await api.settle(pageHandler);
    await bootstrap(api);
    let formalResolve = null;
    const realFetch = api.window.fetch;
    api.window.fetch = function (url, opts) {
      const u = String(url);
      if (u.indexOf('/train-best') >= 0) {
        return new Promise(function (resolve) { formalResolve = resolve; });
      }
      return realFetch(url, opts);
    };
    await selectStudyWithRuns(api, {
      execution_status: 'COMPLETED', success_count: 2, has_success: true,
      approved_for_formal_training: true, best: bestPayload(ID_A)}, []);
    const hostBefore = api.$('sharedMonitorBlock').parentNode.id;
    api.window.hpoStartFormalTraining();
    await api.drain();
    formalResolve({ ok: true, status: 202, json: () => Promise.resolve({
      status: 'accepted', run_id: RUN_A, train_name: 'train1', mode: 'formal',
      source: {}, training_config: { epochs: 2, batch: 1, imgsz: 96, device: '0' },
      differences: { epochs: { original: 30, requested: 2 } }, links: {},
      error_code: null, error: null, next_action: null }) });
    await api.drain();
    const hostAfter = api.$('sharedMonitorBlock').parentNode.id;
    // 运行身份在 URL 里被 encodeURIComponent 编码（manual%3A…），按前缀判定
    const subscribed = api.pending().some((u) => u.indexOf('/api/runs/') >= 0)
      && api.requestLog.some((u) => u.indexOf('/api/runs/') >= 0
        && u.indexOf('aaaaaaaa-1111-4111-8111') >= 0);
    // 另一条 run 的迟到事件与重复事件都不得污染当前卡片
    await api.respondStream('/api/runs/', [
      sse({ status: 'completed', run_id: RUN_A, event_seq: 1,
            result: finalizedResult('train1', 2) }),
      sse({ status: 'completed',
            run_id: 'manual:bbbbbbbb-2222-4222-8222-222222222222', event_seq: 1,
            result: finalizedResult('train9', 9) }),
      sse({ status: 'completed', run_id: RUN_A, event_seq: 1,
            result: finalizedResult('train1', 7) }),
    ]);
    await api.settle(pageHandler);
    return {
      hostBefore: hostBefore,
      hostAfter: hostAfter,
      subscribed: subscribed,
      cards: readMonitorCards(api),
      monitorIdCounts: ['monitorEpochs', 'monitorMap50', 'monitorMap5095',
                        'monitorPR', 'monitorLog']
        .map((id) => api.all.filter((e) => e.id === id).length),
      monitorInsideFormalHost: (function () {
        let node = api.$('sharedMonitorBlock');
        while (node) {
          if (node.id === 'hpoFormalMonitorHost') return true;
          node = node.parentNode;
        }
        return false;
      })(),
      formalStatus: api.text('hpoFormalStatus'),
    };
  },

  /* 最终返修 Task 4：HPO 正式训练监控必须归属于当前 study/runtime。
     进入 HPO 或切换研究时收敛所有权；有关联正式训练时只认权威投影里的
     合法 runtime 身份并恢复其状态/日志/终态指标；离开 HPO 恢复原投影。 */
  async hpo_monitor_ownership(api) {
    const RUN_C = 'manual:cccccccc-3333-4333-8333-333333333333';
    const OTHER_RUN = 'manual:bbbbbbbb-2222-4222-8222-222222222222';
    // 首屏：全局最近有效训练 run A 已投影到公共监控
    api.fireDomReady();
    await api.settle(pageHandler);
    const firstPaint = {
      cards: readMonitorCards(api),
      log: api.text('monitorLog'),
    };

    // 进入 HPO：立即收敛监控所有权（不保留全局最近运行）
    api.setInput('tuningModeSelect', 'hpo');
    api.window.onTuningModeChange('hpo');
    await api.settle(defaultHandler);
    const hpoEntry = { cards: readMonitorCards(api), log: api.text('monitorLog') };

    // study B：没有关联正式训练 → 保持空状态
    api.window.hpoSelectStudy(ID_A);
    await api.settle(defaultHandler);
    const studyB = { cards: readMonitorCards(api), log: api.text('monitorLog') };

    // study C：有已完成且可恢复的关联正式运行 C1
    api.window.hpoSelectStudy(ID_B);
    await api.drain();
    await api.respond('/studies/' + ID_B, 200, statusPayload(ID_B, {
      execution_status: 'COMPLETED', success_count: 1, has_success: true,
      approved_for_formal_training: true, best: bestPayload(ID_B),
      batch: 1, imgsz: 96, device: '0' }));
    await api.drain();
    await api.respond('/formal-runs', 200, {
      study_id: ID_B, count: 1, truncated: false, warnings: [],
      runs: [{
        train_name: 'train5', runtime_run_id: RUN_C,
        runtime_identity_missing: false, history_run_id: 'manual:train5',
        experiment_run_id: RUN_C, experiment_identity_reason: null,
        status: 'completed', result_available: true,
        metrics: { mAP50: 0.71, mAP50_95: 0.44, precision: 0.62, recall: 0.55 },
        epochs: { configured: 2, completed: 2, best: 2 },
        best_pt_available: true, source_trial_number: 1,
        training_config: { epochs: 2, batch: 1, imgsz: 96, device: '0' },
      }] });
    await api.drain();
    const subscribed = api.requestLog.some((u) => u.indexOf('/api/runs/') >= 0
      && u.indexOf('cccccccc-3333-4333-8333') >= 0);
    await api.respondStream('/api/runs/', [
      sse({ event: 'training_log', run_id: RUN_C, event_seq: 1, message: 'epoch 1/2' }),
      sse({ event: 'training_log', run_id: RUN_C, event_seq: 2, message: 'epoch 2/2' }),
      // 其它 run 的终态事件：绝不改变当前卡片
      sse({ status: 'completed', run_id: OTHER_RUN, event_seq: 1,
            result: finalizedResult('train9', 9) }),
    ]);
    await api.settle(defaultHandler);
    const studyC = { cards: readMonitorCards(api), log: api.text('monitorLog') };

    // 从 C 切回 B（无关联正式训练）：立即清空并收敛，C 的订阅被断开
    api.window.hpoSelectStudy(ID_A);
    await api.settle(defaultHandler);
    const backToB = {
      cards: readMonitorCards(api),
      log: api.text('monitorLog'),
      pendingRunStreams: api.pending().filter((u) => u.indexOf('/api/runs/') >= 0).length,
    };

    // 离开 HPO：其它三模式恢复进入前的公共监控投影
    api.window.onTuningModeChange('full');
    const leftHpo = { cards: readMonitorCards(api), log: api.text('monitorLog') };

    return {
      firstPaint: firstPaint,
      hpoEntry: hpoEntry,
      studyB: studyB,
      subscribed: subscribed,
      studyC: studyC,
      backToB: backToB,
      leftHpo: leftHpo,
      monitorRunId: api.state.monitorRunId,
      monitorIdCounts: ['monitorEpochs', 'monitorMap50', 'monitorMap5095',
                        'monitorPR', 'monitorLog']
        .map((id) => api.all.filter((e) => e.id === id).length),
    };
  },

  /* 第二轮返修 Task 1+2：公共监控以**完整状态**为单位被 HPO 接管、清空与恢复。
     首屏是运行中的普通训练 A（指标 / 状态徽章 / 提示 / 停止按钮 / 报告链接 /
     两套日志）；进入 HPO 并选择没有关联正式训练的研究后，上述每一项都必须收敛
     为空状态；绑定正式运行 B1 后状态、日志、指标只属于 B1；切换研究后 B1 的迟到
     终态无权更新任何字段；离开 HPO 完整恢复 A；第二次进出 HPO 必须恢复“新的 B”
     而不是第一次的旧快照 A。 */
  async hpo_monitor_full_state(api) {
    const RUN_A = 'manual:aaaaaaaa-1111-4111-8111-111111111111';
    const RUN_B = 'manual:bbbbbbbb-2222-4222-8222-222222222222';
    const RUN_B1 = 'manual:dddddddd-4444-4444-8444-444444444444';
    // 全局最近有效训练（普通训练/LLM 调参的事实来源）
    const GLOBAL = { status: 'completed', running: false, run_id: RUN_A,
                     updated_at: '2026-09-15T09:00:00Z' };
    const handler = (url) => {
      if (url.indexOf('/api/training/running') >= 0) return { status: 200, body: GLOBAL };
      return pageHandler(url);
    };
    const read = () => ({
      cards: readMonitorCards(api),
      badge: { text: api.text('statusBadge'), cls: api.$('statusBadge').className },
      note: { text: api.text('runStateNote'),
              display: api.$('runStateNote').style.display },
      stopDisplay: api.$('stopTrainingBtn').style.display,
      log: api.text('monitorLog'),
      fullLog: api.text('monitorFullLog'),
    });
    const monitorIdCounts = () => ['monitorEpochs', 'monitorMap50', 'monitorMap5095',
                                   'monitorPR', 'monitorLog', 'monitorFullLog']
      .map((id) => api.all.filter((e) => e.id === id).length);

    // 首屏：普通训练 A 由真实的断线重连流投影（指标、两套日志、报告链接）
    api.fireDomReady();
    await api.settle(handler);
    api.window.locateTrainingRun(RUN_A);
    await api.drain();
    await api.respondStream('/api/runs/', [
      sse({ event: 'training_log', run_id: RUN_A, event_seq: 1, message: 'epoch 1/2' }),
      sse({ event: 'training_log', run_id: RUN_A, event_seq: 2,
            message: 'epoch 2/2', detail: 'A full detail line' }),
      sse({ status: 'completed', run_id: RUN_A, event_seq: 3,
            result: finalizedResult('train9', 7, { mAP50: 0.31, mAP50_95: 0.11,
                                                   precision: 0.4, recall: 0.5 }) }),
    ]);
    await api.settle(handler);
    // 普通训练 A 仍在进行：徽章与停止按钮按权威运行状态显示
    GLOBAL.status = 'running';
    GLOBAL.running = true;
    api.window.refreshRunState();
    await api.settle(handler);
    const beforeHpo = read();

    // 进入 HPO：以完整状态为单位接管
    api.setInput('tuningModeSelect', 'hpo');
    api.window.onTuningModeChange('hpo');
    await api.settle(handler);
    const hpoEntry = read();

    // 选择一个没有关联正式训练的研究：绝不能留下 A 的任何字段
    api.window.hpoSelectStudy(ID_A);
    await api.settle(handler);
    const emptyStudy = read();
    emptyStudy.pendingRunStreams =
      api.pending().filter((u) => u.indexOf('/api/runs/') >= 0).length;

    // 换到绑定正式运行 B1 的研究（B1 仍在运行）
    api.window.hpoSelectStudy(ID_B);
    await api.drain();
    await api.respond('/studies/' + ID_B, 200, statusPayload(ID_B, {
      execution_status: 'COMPLETED', success_count: 1, has_success: true,
      approved_for_formal_training: true, best: bestPayload(ID_B),
      batch: 1, imgsz: 96, device: '0' }));
    await api.drain();
    await api.respond('/formal-runs', 200, {
      study_id: ID_B, count: 1, truncated: false, warnings: [],
      runs: [{ train_name: 'train5', runtime_run_id: RUN_B1,
        runtime_identity_missing: false, history_run_id: 'manual:train5',
        experiment_run_id: RUN_B1, experiment_identity_reason: null,
        status: 'running', result_available: false, metrics: {},
        best_pt_available: false, source_trial_number: 1,
        training_config: { epochs: 2, batch: 1, imgsz: 96, device: '0' } }] });
    await api.drain();
    // 该 run 的流由测试自己驱动：它必须在切换研究之后仍然“在途”
    const stream = api.respondStreamHold('/api/runs/');
    const b1Bound = read();
    await stream.push(sse({ event: 'training_log', run_id: RUN_B1, event_seq: 1,
                            message: 'B1 epoch 1/2' }));
    // 日志按批渲染（浏览器里由 setTimeout 触发）：这里显式让定时器到期
    await api.flushTimeouts();
    const b1Log = api.text('monitorLog');
    // 第二批日志仍缓冲在旧订阅的渲染器里（浏览器里尚未到期），
    // 它同时也是“旧订阅收尾不得写入当前日志”的探针
    await stream.push(sse({ event: 'training_log', run_id: RUN_B1, event_seq: 2,
                            message: 'B1 epoch 2/2' }));
    // B1 的终态事件：此时全局最近运行仍是 A，徽章必须是 B1 自己的终态
    await stream.push(sse({ status: 'failed', run_id: RUN_B1, event_seq: 3,
                            message: 'B1 训练失败' }));
    await api.settle(handler);
    const b1Terminal = read();

    // 切换到另一个研究（无关联正式训练）：B1 的订阅被断开
    api.window.hpoSelectStudy(ID_C);
    await api.settle(handler);
    const switched = read();
    // B1 的迟到终态（旧订阅仍在途）：不得更新任何监控字段
    await stream.push(sse({ status: 'completed', run_id: RUN_B1, event_seq: 4,
                            message: 'B1 迟到终态',
                            result: finalizedResult('train5', 9, { mAP50: 0.99 }) }));
    await api.settle(handler);
    const afterLateB1 = read();
    // 旧订阅在切换之后才结束：它的收尾也不得把缓冲行写进当前监控日志
    await stream.end();
    await api.settle(handler);
    await api.flushTimeouts();
    const afterStaleEnd = read();

    // 离开 HPO：完整恢复进入前的 A
    api.window.onTuningModeChange('full');
    await api.settle(handler);
    const leftHpo = read();

    // 此后普通监控形成新的事实 B：第二次进入 HPO 必须重新捕获基线
    GLOBAL.status = 'completed';
    GLOBAL.running = false;
    GLOBAL.run_id = RUN_B;
    api.window.locateTrainingRun(RUN_B);
    await api.drain();
    await api.respondStream('/api/runs/', [
      sse({ event: 'training_log', run_id: RUN_B, event_seq: 1, message: 'B epoch 1/30' }),
      sse({ event: 'training_log', run_id: RUN_B, event_seq: 2,
            message: 'B epoch 2/30', detail: 'B full detail line' }),
    ]);
    await api.settle(handler);
    api.window.renderRunState({ status: 'completed', running: false });
    api.window.TrainingMonitor.applyResultToMonitor(
      finalizedResult('train3', 30, { mAP50: 0.5, mAP50_95: 0.4,
                                      precision: 0.6, recall: 0.7 }));
    const beforeHpo2 = read();

    // 第二次进入 HPO，并在 HPO 内连续切换两个研究
    api.window.onTuningModeChange('hpo');
    await api.settle(handler);
    api.window.hpoSelectStudy(ID_A);
    await api.settle(handler);
    api.window.hpoSelectStudy(ID_C);
    await api.settle(handler);
    // 离开 HPO：必须恢复 B，而不是 HPO 空状态或第一次的旧快照 A
    api.window.onTuningModeChange('full');
    await api.settle(handler);
    const leftHpo2 = read();

    return {
      beforeHpo: beforeHpo,
      hpoEntry: hpoEntry,
      emptyStudy: emptyStudy,
      b1Bound: b1Bound,
      b1Log: b1Log,
      b1Terminal: b1Terminal,
      switched: switched,
      afterLateB1: afterLateB1,
      afterStaleEnd: afterStaleEnd,
      leftHpo: leftHpo,
      beforeHpo2: beforeHpo2,
      leftHpo2: leftHpo2,
      monitorIdCounts: monitorIdCounts(),
    };
  },

  /* 第三轮 P1：同一 runtime_run_id 的**权威状态与结果**必须随每次
     /formal-runs 响应收敛，绝不被“监控身份未变化”的提前返回吞掉。
     身份只决定是否重新订阅/清空；同一身份的状态变化只刷新事实与结果投影。 */
  async hpo_formal_monitor_convergence(api) {
    const RUN_A = 'manual:aaaaaaaa-1111-4111-8111-111111111111';
    const RUN_B = 'manual:bbbbbbbb-2222-4222-8222-222222222222';
    const RUN_C = 'manual:cccccccc-3333-4333-8333-333333333333';
    const NEW_RUN_A = 'aaaaaaaa-1111-4111-8111-111111111111';
    const NEW_RUN_B = 'bbbbbbbb-2222-4222-8222-222222222222';
    const NEW_RUN_C = 'cccccccc-3333-4333-8333-333333333333';

    function formalRun(name, runId, status, overrides) {
      const base = {
        train_name: name, runtime_run_id: runId, runtime_identity_missing: false,
        history_run_id: 'manual:' + name, experiment_run_id: runId,
        experiment_identity_reason: null, status: status,
        result_available: status === 'completed', metrics: {}, epochs: null,
        best_pt_available: status === 'completed', source_trial_number: 1,
        training_config: { epochs: 2, batch: 1, imgsz: 96, device: '0' },
      };
      return Object.assign(base, overrides || {});
    }
    function runsPayload(runs) {
      return { study_id: ID_A, count: runs.length, truncated: false,
               warnings: [], runs: runs };
    }
    const runA1 = formalRun('train1', RUN_A, 'completed', {
      metrics: { mAP50: 0.7, mAP50_95: 0.4, precision: 0.6, recall: 0.5 },
      epochs: { configured: 2, completed: 2, best: 2 } });
    const runBRunning = formalRun('train2', RUN_B, 'running');
    const runBCompleted = formalRun('train2', RUN_B, 'completed', {
      result_available: true,
      metrics: { mAP50: 0.66, mAP50_95: 0.33, precision: 0.51, recall: 0.42 },
      epochs: { configured: 5, completed: 5, best: 4 } });
    // 订阅次数按 URL 里的完整 UUID 统计（路径参数被 encodeURIComponent 编码）
    const subscriptions = (uuid) =>
      api.requestLog.filter((u) => u.indexOf('/api/runs/') >= 0 && u.indexOf(uuid) >= 0).length;
    const read = () => ({
      badge: { text: api.text('statusBadge'), cls: api.$('statusBadge').className },
      cards: readMonitorCards(api),
      log: api.text('monitorLog'),
      fullLog: api.text('monitorFullLog'),
      stopDisplay: api.$('stopTrainingBtn').style.display,
      monitorRunId: api.state.monitorRunId,
      streamsA: subscriptions(NEW_RUN_A),
      streamsB: subscriptions(NEW_RUN_B),
      streamsC: subscriptions(NEW_RUN_C),
    });

    const studyStatus = statusPayload(ID_A, {
      execution_status: 'COMPLETED', success_count: 2, has_success: true,
      approved_for_formal_training: true, best: bestPayload(ID_A),
      batch: 1, imgsz: 96, device: '0' });
    // 一次真实的轮询回合：状态与关联结果都来自服务端权威投影
    // （浏览器里由 2s 定时器触发，这里由测试决定何时发生）
    async function pollRound(runs) {
      api.window.hpoRefreshRound(ID_A, true);
      await api.settle((url) => {
        if (url.indexOf('/formal-runs') >= 0) {
          return { status: 200, body: runsPayload(runs) };
        }
        if (url.indexOf('/studies/' + ID_A) >= 0) {
          return { status: 200, body: studyStatus };
        }
        return pageHandler(url);
      });
    }

    api.setInput('tuningModeSelect', 'hpo');
    api.fireDomReady();
    await api.settle(pageHandler);

    // 研究 A：已完成正式运行 A1 已绑定到公共监控（徽章/指标/日志都属于 A1）
    api.window.hpoSelectStudy(ID_A);
    await api.drain();
    await api.respond('/studies/' + ID_A, 200, studyStatus);
    await api.drain();
    await api.respond('/formal-runs', 200, runsPayload([runA1]));
    await api.drain();
    const staleStream = api.respondStreamHold('/api/runs/');
    await staleStream.push(sse({ event: 'training_log', run_id: RUN_A, event_seq: 1,
                                 message: 'A1 epoch 1/2' }));
    await api.flushTimeouts();
    const beforeSubmit = read();

    // 提交新的正式训练 B：202 之后、任何流事件到达之前就必须收敛到 B
    api.setInput('hpoFormalEpochs', '5');
    api.window.hpoStartFormalTraining();
    await api.drain();
    await api.respond('/train-best', 202, {
      status: 'accepted', run_id: RUN_B, train_name: 'train2', mode: 'formal',
      source: {}, training_config: { epochs: 5, batch: 1, imgsz: 96, device: '0' },
      differences: {}, links: {}, error_code: null, error: null, next_action: null });
    await api.drain();
    const accepted = read();

    // B 自己的流：先给一条日志，用于验证同一身份的状态刷新不清空日志
    const streamB = api.respondStreamHold('/api/runs/');
    await streamB.push(sse({ event: 'training_log', run_id: RUN_B, event_seq: 1,
                             message: 'B epoch 1/5' }));
    await api.flushTimeouts();
    const bLogBeforePoll = api.text('monitorLog');

    // 同一身份的权威状态刷新：running/result_available=false
    await pollRound([runBRunning, runA1]);
    const bRunning = read();

    // 同一身份由 running 收敛为 completed + 最终结果
    await pollRound([runBCompleted, runA1]);
    const bCompleted = read();

    // 完全重复的轮询必须幂等：不重复订阅、不重置日志、不重复投影
    await pollRound([runBCompleted, runA1]);
    const repeated = read();

    // 上一订阅（A1）的迟到事件：不得更新任何监控字段
    await staleStream.push(sse({ status: 'completed', run_id: RUN_A, event_seq: 2,
                                 message: 'A1 迟到终态',
                                 result: finalizedResult('train1', 9, { mAP50: 0.99 }) }));
    await api.settle(pageHandler);
    const afterLateA = read();

    // 切换到绑定另一正式运行 C 的研究：B 被清空、B 的订阅失效、只订阅 C
    api.window.hpoSelectStudy(ID_B);
    await api.drain();
    await api.respond('/studies/' + ID_B, 200, statusPayload(ID_B));
    await api.drain();
    await api.respond('/formal-runs', 200, {
      study_id: ID_B, count: 1, truncated: false, warnings: [],
      runs: [formalRun('train7', RUN_C, 'running')] });
    await api.drain();
    const switchedToC = read();
    // B 的迟到终态（旧订阅仍在途）：不得更新任何监控字段
    await streamB.push(sse({ status: 'completed', run_id: RUN_B, event_seq: 9,
                             message: 'B 迟到终态',
                             result: finalizedResult('train2', 9, { mAP50: 0.99 }) }));
    await api.settle(pageHandler);
    const afterLateB = read();

    return {
      beforeSubmit: beforeSubmit,
      accepted: accepted,
      bLogBeforePoll: bLogBeforePoll,
      bRunning: bRunning,
      bCompleted: bCompleted,
      repeated: repeated,
      afterLateA: afterLateA,
      switchedToC: switchedToC,
      afterLateB: afterLateB,
    };
  },

  /* 第三轮 P1：同一 runtime_run_id 由 running → failed 也必须收敛为终态，
     不重新订阅、不清空该 run 已有的日志。 */
  async hpo_formal_monitor_failure(api) {
    const RUN_B = 'manual:bbbbbbbb-2222-4222-8222-222222222222';
    const uuid = 'bbbbbbbb-2222-4222-8222-222222222222';
    const subscriptions = () =>
      api.requestLog.filter((u) => u.indexOf('/api/runs/') >= 0 && u.indexOf(uuid) >= 0).length;
    const read = () => ({
      badge: { text: api.text('statusBadge'), cls: api.$('statusBadge').className },
      cards: readMonitorCards(api),
      log: api.text('monitorLog'),
      stopDisplay: api.$('stopTrainingBtn').style.display,
      monitorRunId: api.state.monitorRunId,
      streams: subscriptions(),
    });

    api.setInput('tuningModeSelect', 'hpo');
    api.fireDomReady();
    await api.settle(pageHandler);
    api.window.hpoSelectStudy(ID_A);
    await api.drain();
    await api.respond('/studies/' + ID_A, 200, statusPayload(ID_A, {
      execution_status: 'COMPLETED', success_count: 2, has_success: true,
      approved_for_formal_training: true, best: bestPayload(ID_A),
      batch: 1, imgsz: 96, device: '0' }));
    await api.drain();
    await api.respond('/formal-runs', 200, {
      study_id: ID_A, count: 1, truncated: false, warnings: [],
      runs: [{ train_name: 'train2', runtime_run_id: RUN_B,
        runtime_identity_missing: false, history_run_id: 'manual:train2',
        experiment_run_id: RUN_B, experiment_identity_reason: null,
        status: 'running', result_available: false, metrics: {},
        epochs: null, best_pt_available: false, source_trial_number: 1,
        training_config: { epochs: 2, batch: 1, imgsz: 96, device: '0' } }] });
    await api.drain();
    const stream = api.respondStreamHold('/api/runs/');
    await stream.push(sse({ event: 'training_log', run_id: RUN_B, event_seq: 1,
                            message: 'B epoch 1/5' }));
    await api.flushTimeouts();
    const running = read();

    // 下一次轮询回合：该 runtime 的权威状态已变为 failed
    const failedRun = { train_name: 'train2', runtime_run_id: RUN_B,
      runtime_identity_missing: false, history_run_id: 'manual:train2',
      experiment_run_id: RUN_B, experiment_identity_reason: null,
      status: 'failed', result_available: false, metrics: {},
      epochs: null, best_pt_available: false, source_trial_number: 1,
      training_config: { epochs: 2, batch: 1, imgsz: 96, device: '0' } };
    api.window.hpoRefreshRound(ID_A, true);
    await api.settle((url) => {
      if (url.indexOf('/formal-runs') >= 0) {
        return { status: 200, body: { study_id: ID_A, count: 1, truncated: false,
                                      warnings: [], runs: [failedRun] } };
      }
      return pageHandler(url);
    });
    const failed = read();
    return { running: running, failed: failed };
  },

  /* 第四轮 P1：从**已完成研究**启动正式训练后，202 必须自动建立唯一的 2 秒轮询，
     并在该 run 收敛为终态后自动停止。全程只推进 fake clock、应答实现自己发出的
     请求，绝不调用任何手动刷新入口（hpoRefreshFormalRuns / hpoRefreshRound）。 */
  async hpo_formal_polling_after_accept(api) {
    const RUN_A1 = 'manual:aaaaaaaa-1111-4111-8111-111111111111';
    const RUN_B = 'manual:bbbbbbbb-2222-4222-8222-222222222222';
    const UUID_B = 'bbbbbbbb-2222-4222-8222-222222222222';

    function formalRun(name, runId, status, overrides) {
      const base = {
        train_name: name, runtime_run_id: runId, runtime_identity_missing: false,
        history_run_id: 'manual:' + name, experiment_run_id: runId,
        experiment_identity_reason: null, status: status,
        result_available: status === 'completed', metrics: {}, epochs: null,
        best_pt_available: status === 'completed', source_trial_number: 1,
        training_config: { epochs: 5, batch: 1, imgsz: 96, device: '0' },
      };
      return Object.assign(base, overrides || {});
    }
    const runA1 = formalRun('train1', RUN_A1, 'completed', {
      metrics: { mAP50: 0.7, mAP50_95: 0.4, precision: 0.6, recall: 0.5 },
      epochs: { configured: 2, completed: 2, best: 2 } });
    const runBRunning = formalRun('train2', RUN_B, 'running');
    const runBDone = formalRun('train2', RUN_B, 'completed', {
      result_available: true, best_pt_available: true,
      metrics: { mAP50: 0.66, mAP50_95: 0.33, precision: 0.51, recall: 0.42 },
      epochs: { configured: 5, completed: 5, best: 4 } });
    const runsBody = (runs) => ({ study_id: ID_A, count: runs.length, truncated: false,
                                  warnings: [], runs: runs });
    const studyStatus = statusPayload(ID_A, {
      execution_status: 'COMPLETED', success_count: 2, has_success: true,
      approved_for_formal_training: true, best: bestPayload(ID_A),
      batch: 1, imgsz: 96, device: '0' });

    const subscriptions = () => api.requestLog.filter((u) =>
      u.indexOf('/api/runs/') >= 0 && u.indexOf(UUID_B) >= 0).length;
    const formalCount = () =>
      api.requestLog.filter((u) => u.indexOf(ID_A + '/formal-runs') >= 0).length;
    const refreshHandler = (url, runs) => {
      if (url.indexOf(ID_A + '/formal-runs') >= 0) {
        return { status: 200, body: runsBody(runs) };
      }
      if (url.indexOf('/api/hpo/studies/' + ID_A) >= 0) {
        return { status: 200, body: studyStatus };
      }
      return pageHandler(url);
    };
    const read = () => {
      const box = api.$('hpoFormalRunsList');
      const first = box.children[0];
      return {
        badge: { text: api.text('statusBadge'), cls: api.$('statusBadge').className },
        cards: readMonitorCards(api),
        log: api.text('monitorLog'),
        stopDisplay: api.$('stopTrainingBtn').style.display,
        list: api.text('hpoFormalRunsList'),
        rowCount: box.children.length,
        buttons: first ? collectButtons(first).map((b) =>
          ({ label: b.textContent, disabled: b.disabled })) : [],
        timers: api.intervalCount(),
        watchFormal: api.state.watchFormal,
        monitorRunId: api.state.monitorRunId,
      };
    };

    api.setInput('tuningModeSelect', 'hpo');
    api.fireDomReady();
    await api.settle(pageHandler);

    // 已完成研究：一条已完成的正式运行 A1 已投影；无活动控制器/正式训练 → 无轮询
    api.window.hpoSelectStudy(ID_A);
    await api.drain();
    await api.respond('/studies/' + ID_A, 200, studyStatus);
    await api.drain();
    await api.respond(ID_A + '/formal-runs', 200, runsBody([runA1]));
    await api.settle(pageHandler);
    const beforeSubmit = read();

    // 提交正式训练（只改轮数）：202 之后不调用任何手动刷新入口
    api.setInput('hpoFormalEpochs', '5');
    api.window.hpoStartFormalTraining();
    await api.drain();
    await api.respond('/train-best', 202, {
      status: 'accepted', run_id: RUN_B, train_name: 'train2', mode: 'formal',
      source: {}, training_config: { epochs: 5, batch: 1, imgsz: 96, device: '0' },
      differences: {}, links: {}, error_code: null, error: null, next_action: null });
    // 202 自己发起的首次关联刷新：测试只做应答
    await api.settleExceptStream((url) => refreshHandler(url, [runBRunning, runA1]));
    // 该 run 的实时流：只订阅一次，并留一条日志用于验证终态轮询不清空日志
    let stream = null;
    if (api.pending().some((u) => u.indexOf('/api/runs/') >= 0)) {
      stream = api.respondStreamHold('/api/runs/');
      await stream.push(sse({ event: 'training_log', run_id: RUN_B, event_seq: 1,
                              message: 'B epoch 1/5' }));
      await api.flushTimeouts();
    }
    const accepted = read();
    const logBeforePoll = api.text('monitorLog');

    // 只推进 fake clock（超过 2 秒）→ 该 run 的权威终态
    await api.tickIntervals();
    await api.settleExceptStream((url) => refreshHandler(url, [runBDone, runA1]));
    const converged = read();

    // 终态后必须自动停止：继续推进时钟不得产生任何新请求
    const afterTerminal = { timers: api.intervalCount(), formal: formalCount(),
                            total: api.requestLog.length };
    await api.tickIntervals();
    await api.settle(pageHandler);
    const afterExtraTick = { timers: api.intervalCount(), formal: formalCount(),
                             total: api.requestLog.length };

    return {
      beforeSubmit: beforeSubmit,
      accepted: accepted,
      logBeforePoll: logBeforePoll,
      converged: converged,
      afterTerminal: afterTerminal,
      afterExtraTick: afterExtraTick,
      subscriptions: subscriptions(),
      streamOpen: !!stream,
    };
  },

  /* 第四轮 P1：202 后的首次关联刷新仍在途时，interval tick 必须走既有的
     refreshing/pendingRefresh 合并机制——不并发发出第二次请求，也不丢最后一次刷新。 */
  async hpo_formal_polling_coalesces_inflight_tick(api) {
    const RUN_B = 'manual:bbbbbbbb-2222-4222-8222-222222222222';

    function formalRun(name, runId, status, overrides) {
      const base = {
        train_name: name, runtime_run_id: runId, runtime_identity_missing: false,
        history_run_id: 'manual:' + name, experiment_run_id: runId,
        experiment_identity_reason: null, status: status,
        result_available: status === 'completed', metrics: {}, epochs: null,
        best_pt_available: status === 'completed', source_trial_number: 1,
        training_config: { epochs: 5, batch: 1, imgsz: 96, device: '0' },
      };
      return Object.assign(base, overrides || {});
    }
    const runBRunning = formalRun('train2', RUN_B, 'running');
    const runsBody = (runs) => ({ study_id: ID_A, count: runs.length, truncated: false,
                                  warnings: [], runs: runs });
    // 研究仍在搜索：活动控制器 → 已存在唯一轮询定时器
    const studyStatus = statusPayload(ID_A, {
      execution_status: 'RUNNING', control_active: true, can_stop: true,
      success_count: 1, has_success: true, approved_for_formal_training: true,
      best: bestPayload(ID_A), batch: 1, imgsz: 96, device: '0' });

    const formalCount = () =>
      api.requestLog.filter((u) => u.indexOf(ID_A + '/formal-runs') >= 0).length;
    const read = () => ({
      badge: { text: api.text('statusBadge'), cls: api.$('statusBadge').className },
      list: api.text('hpoFormalRunsList'),
      timers: api.intervalCount(),
      watchFormal: api.state.watchFormal,
      monitorRunId: api.state.monitorRunId,
    });

    api.setInput('tuningModeSelect', 'hpo');
    api.fireDomReady();
    await api.settle(pageHandler);
    api.window.hpoSelectStudy(ID_A);
    await api.drain();
    await api.respond('/studies/' + ID_A, 200, studyStatus);
    await api.drain();
    await api.respond(ID_A + '/formal-runs', 200, runsBody([]));
    await api.settle(pageHandler);
    const beforeSubmit = read();

    // 提交正式训练：202 后实现自己发起首次关联刷新，此时该刷新仍在途
    api.setInput('hpoFormalEpochs', '5');
    api.window.hpoStartFormalTraining();
    await api.drain();
    await api.respond('/train-best', 202, {
      status: 'accepted', run_id: RUN_B, train_name: 'train2', mode: 'formal',
      source: {}, training_config: { epochs: 5, batch: 1, imgsz: 96, device: '0' },
      differences: {}, links: {}, error_code: null, error: null, next_action: null });
    await api.drain();
    const inflight = { formal: formalCount(), pending: api.pendingCount() };

    // 时钟推进落在在途刷新上：必须被合并，绝不并发
    await api.tickIntervals();
    const afterTick = { formal: formalCount(), pending: api.pendingCount() };

    // 应答在途刷新；被合并的那一次随后补跑（最后一次刷新不丢）
    await api.settleExceptStream((url) => {
      if (url.indexOf(ID_A + '/formal-runs') >= 0) {
        return { status: 200, body: runsBody([runBRunning]) };
      }
      if (url.indexOf('/api/hpo/studies/' + ID_A) >= 0) {
        return { status: 200, body: studyStatus };
      }
      return pageHandler(url);
    });
    const coalesced = read();

    return {
      beforeSubmit: beforeSubmit,
      inflight: inflight,
      afterTick: afterTick,
      coalesced: coalesced,
      formalAfterSettle: formalCount(),
    };
  },

  /* 第四轮 P1：训练期间切换研究——旧研究的定时器与响应必须失效，
     新研究没有活动控制器或关联正式训练时不得继续旧轮询。 */
  async hpo_formal_polling_switch_study(api) {
    const RUN_B = 'manual:bbbbbbbb-2222-4222-8222-222222222222';

    function formalRun(name, runId, status, overrides) {
      const base = {
        train_name: name, runtime_run_id: runId, runtime_identity_missing: false,
        history_run_id: 'manual:' + name, experiment_run_id: runId,
        experiment_identity_reason: null, status: status,
        result_available: status === 'completed', metrics: {}, epochs: null,
        best_pt_available: status === 'completed', source_trial_number: 1,
        training_config: { epochs: 5, batch: 1, imgsz: 96, device: '0' },
      };
      return Object.assign(base, overrides || {});
    }
    const runBRunning = formalRun('train2', RUN_B, 'running');
    const runBDone = formalRun('train2', RUN_B, 'completed', {
      result_available: true, best_pt_available: true,
      metrics: { mAP50: 0.66, mAP50_95: 0.33, precision: 0.51, recall: 0.42 },
      epochs: { configured: 5, completed: 5, best: 4 } });
    const runsBody = (runs) => ({ study_id: ID_A, count: runs.length, truncated: false,
                                  warnings: [], runs: runs });
    const studyStatus = statusPayload(ID_A, {
      execution_status: 'RUNNING', control_active: true, can_stop: true,
      success_count: 1, has_success: true, approved_for_formal_training: true,
      best: bestPayload(ID_A), batch: 1, imgsz: 96, device: '0' });

    const read = () => ({
      badge: { text: api.text('statusBadge'), cls: api.$('statusBadge').className },
      list: api.text('hpoFormalRunsList'),
      timers: api.intervalCount(),
      watchFormal: api.state.watchFormal,
      monitorRunId: api.state.monitorRunId,
      pendingRunStreams: api.pending().filter((u) => u.indexOf('/api/runs/') >= 0).length,
    });

    api.setInput('tuningModeSelect', 'hpo');
    api.fireDomReady();
    await api.settle(pageHandler);

    // 研究 A 仍在搜索（有活动控制器）且绑定运行中的正式 run B → 唯一轮询 + 已订阅
    api.window.hpoSelectStudy(ID_A);
    await api.drain();
    await api.respond('/studies/' + ID_A, 200, studyStatus);
    await api.drain();
    await api.respond(ID_A + '/formal-runs', 200, runsBody([runBRunning]));
    await api.drain();
    const stream = api.respondStreamHold('/api/runs/');
    await api.settle(pageHandler);
    const training = read();

    // 一次 tick：该轮的状态请求被应答，关联结果请求留在途
    await api.tickIntervals();
    await api.respond('/studies/' + ID_A, 200, studyStatus);
    await api.drain();
    const inflightFormal =
      api.pending().some((u) => u.indexOf(ID_A + '/formal-runs') >= 0);

    // 训练期间切换研究：新研究没有活动控制器、也没有关联正式训练
    api.window.hpoSelectStudy(ID_C);
    await api.drain();
    const afterSwitch = read();
    // 旧研究关联结果的迟到响应（本身会改变监控事实）必须被丢弃
    await api.respond(ID_A + '/formal-runs', 200, runsBody([runBDone]));
    await api.drain();
    const afterLateA = read();
    await api.settle(pageHandler);
    const switched = read();

    // 再推进时钟：不得复活旧轮询，也不得产生任何新请求
    const beforeFinalTick = api.requestLog.length;
    await api.tickIntervals();
    await api.settle(pageHandler);
    const afterFinalTick = { timers: api.intervalCount(),
                             total: api.requestLog.length };

    return {
      training: training,
      inflightFormal: inflightFormal,
      afterSwitch: afterSwitch,
      afterLateA: afterLateA,
      switched: switched,
      beforeFinalTick: beforeFinalTick,
      afterFinalTick: afterFinalTick,
      ranStream: !!stream,
    };
  },

  /* 最终返修 Task 5：GPU 默认来自服务端最终事实；defaults 到达前不得把 cpu
     当作可提交暂存值；设备是明确选择控件；创建必须等事实加载完成。 */
  async hpo_device_defaults(api) {
    const deviceEl = () => api.$('hpoDevice');
    const read = () => ({
      tag: deviceEl().tagName,
      value: deviceEl().value,
      options: (deviceEl().options || []).map((o) => o.value),
      createDisabled: api.$('hpoCreateAndStartBtn').disabled,
      notice: api.text('hpoMainSummary'),
      bindingNotice: api.text('hpoBindingNotice'),
    });
    const createCalls = () => api.requestLog.filter((u) => u === '/api/hpo/studies').length;

    api.setInput('tuningModeSelect', 'hpo');
    api.fireDomReady();
    await api.drain();
    // defaults / 快照 / 权重都还没回来：设备没有可提交值，创建按钮必须禁用
    const pending = read();
    // 立刻点击（快速切换 HPO 后马上操作）：绝不发出一次 CPU 创建请求
    api.window.hpoCreateAndStart();
    await api.drain();
    const afterEarlyClick = { createCalls: createCalls(), createDisabled: read().createDisabled };

    await api.settle((url) => {
      if (url.indexOf('/api/hpo/defaults') >= 0) return { status: 200, body: defaultsPayload() };
      return defaultHandler(url);
    });
    const gpu = read();

    // 用户在加载完成后主动改选 CPU：后续无关异步响应不得覆盖
    api.setInput('hpoDevice', 'cpu');
    deviceEl().dispatch('change');
    await api.drain();
    await api.settle(defaultHandler);
    const afterUserChoice = read();

    // 提交值仍是现有合法值（cpu），且创建按钮可用
    api.window.hpoCreateAndStart();
    await api.drain();
    const createRequest = api.queued('/api/hpo/studies');
    await api.settle((url) => (url.indexOf('/start') >= 0)
      ? { status: 202, body: { started: true } } : defaultHandler(url));
    return {
      pending: pending,
      afterEarlyClick: afterEarlyClick,
      gpu: gpu,
      afterUserChoice: afterUserChoice,
      createDevice: createRequest ? createRequest.body.execution_config.device : null,
    };
  },

  /* 无 GPU：允许 CPU 并给出可见、简短的提示；defaults 失败：零创建且不静默用 CPU。 */
  async hpo_device_fallback(api) {
    api.setInput('tuningModeSelect', 'hpo');
    api.fireDomReady();
    await api.settle((url) => {
      if (url.indexOf('/api/hpo/defaults') >= 0) {
        const payload = defaultsPayload();
        payload.search.device = 'cpu';
        payload.devices = { gpus: [], default: 'cpu' };
        payload.device_notice = { code: 'GPU_NOT_AVAILABLE_CPU_FALLBACK',
          message: '未检测到可用 GPU，已使用 CPU。' };
        return { status: 200, body: payload };
      }
      return defaultHandler(url);
    });
    const cpu = {
      tag: api.$('hpoDevice').tagName,
      value: api.$('hpoDevice').value,
      options: (api.$('hpoDevice').options || []).map((o) => o.value),
      createDisabled: api.$('hpoCreateAndStartBtn').disabled,
      notice: api.text('hpoMainSummary'),
      noticeVisible: api.visible('hpoMainSummary'),
    };

    api.window.hpoCreateAndStart();
    await api.drain();
    const request = api.queued('/api/hpo/studies');
    const cpuDevice = request ? request.body.execution_config.device : null;
    return { cpu: cpu, cpuDevice: cpuDevice };
  },

  /* defaults 请求失败：可见可操作的错误 + 零创建，绝不悄悄启用 CPU 创建。 */
  async hpo_device_defaults_failure(api) {
    api.setInput('tuningModeSelect', 'hpo');
    api.fireDomReady();
    await api.settle((url) => {
      if (url.indexOf('/api/hpo/defaults') >= 0) {
        return { status: 500, body: { error: 'x' } };
      }
      return defaultHandler(url);
    });
    const read = () => ({
      value: api.$('hpoDevice').value,
      createDisabled: api.$('hpoCreateAndStartBtn').disabled,
      bindingNotice: api.text('hpoBindingNotice'),
      bindingNoticeVisible: api.visible('hpoBindingNotice'),
    });
    const failed = read();
    api.window.hpoCreateAndStart();
    await api.drain();
    return {
      failed: failed,
      createCalls: api.requestLog.filter((u) => u === '/api/hpo/studies').length,
    };
  },

  /* A2/A3/B1/B2: layout order, single primary action, collapse defaults. */
  async layout_and_defaults(api) {
    await bootstrap(api);
    const draftEl = api.$('hpoCreateDraft');
    const commonEl = api.$('tuningCommonControls');
    const order = {
      modeIsInsideCommon: api.all.indexOf(api.$('tuningModeSelect')) >
        api.all.indexOf(commonEl),
      hpoDraftAfterCommon: api.all.indexOf(draftEl) > api.all.indexOf(commonEl),
      formalRunsListCount: api.all.filter((e) => e.id === 'hpoFormalRunsList').length,
    };
    api.window.onTuningModeChange('hpo');
    await api.settle(defaultHandler);
    const hpoMode = { hpoBtnVisible: api.visible('hpoCreateAndStartBtn'),
      commonBtnVisible: api.visible('startTuningBtn') };
    api.window.onTuningModeChange('dry_run');
    const dryMode = { hpoBtnVisible: api.visible('hpoCreateAndStartBtn'),
      commonBtnVisible: api.visible('startTuningBtn') };
    api.window.onTuningModeChange('full');
    const fullMode = { hpoBtnVisible: api.visible('hpoCreateAndStartBtn'),
      commonBtnVisible: api.visible('startTuningBtn') };
    api.window.onTuningModeChange('hpo');
    await api.settle(defaultHandler);
    const details = {
      searchDetailsOpen: api.$('hpoSearchDetails').attributes.open !== undefined,
      draftDetailsOpen: api.$('hpoDraftDetails').attributes.open !== undefined,
      formalDetailsOpen: api.$('hpoFormalDetails') === null
        ? false : api.$('hpoFormalDetails').attributes.open !== undefined,
      bestDetailsOpen: !!api.$('hpoBestDetails').open,
      // 第四轮：绑定区不再折叠、也没有可编辑路径输入
      inputDetailsCount: api.all.filter((e) => e.id === 'hpoInputDetails').length,
      modelPathCount: api.all.filter((e) => e.id === 'hpoModelPath').length,
      trialsBodyCount: api.all.filter((e) => e.id === 'hpoTrialsBody').length,
      snapshotSelectEnabled: !api.$('hpoSnapshotSelect').disabled,
      modelSelectEnabled: !api.$('hpoModelSelect').disabled,
    };
    // 评价模式默认全面；设备默认来自服务端（本机 GPU 可用时为 "0"）
    const directInputs = {
      evaluationMode: api.$('hpoEvaluationMode').value,
      device: api.$('hpoDevice').value,
    };
    return {
      order: order,
      hpoMode: hpoMode,
      dryMode: dryMode,
      fullMode: fullMode,
      details: details,
      directInputs: directInputs,
      formalInputs: api.all.filter((e) => e.tagName === 'INPUT'
        && e.id && e.id.indexOf('hpoFormal') === 0).map((e) => e.id),
      formalConditions: api.text('hpoFormalConditions'),
      formalEpochs: api.$('hpoFormalEpochs').value,
      draftDefaults: {
        budget: api.$('hpoBudget').value,
        epochs: api.$('hpoEpochs').value,
        seed: api.$('hpoSeed').value,
        modelValue: api.$('hpoModelSelect').value,
      },
      datasetSummary: api.text('hpoDatasetSummary'),
      modelSummary: api.text('hpoModelSummary'),
      mainSummary: api.$('hpoMainSummary').textContent,
    };
  },
};

// Scenarios that drive the whole page must be given a *rendered* single_page.html
// (the training-monitor projections depend on the Jinja context).
for (const name of ['monitor_first_paint', 'monitor_terminal_metrics',
                    'monitor_result_projection', 'monitor_stream_parity',
                    'monitor_epochs_contract', 'formal_monitor_reuse',
                    'hpo_monitor_ownership', 'hpo_monitor_full_state',
                    'hpo_formal_monitor_convergence', 'hpo_formal_monitor_failure',
                    'hpo_formal_polling_after_accept',
                    'hpo_formal_polling_coalesces_inflight_tick',
                    'hpo_formal_polling_switch_study']) {
  SCENARIOS[name].needsPage = true;
}

async function main() {
  const scenario = process.argv[2];
  const uiDir = process.argv[3] || path.resolve(__dirname, '..', '..', 'ui');
  const pageHtml = process.argv[4] || null;
  if (!SCENARIOS[scenario]) {
    process.stderr.write('unknown scenario: ' + scenario + '\n');
    process.exit(2);
  }
  if (SCENARIOS[scenario].needsPage && !pageHtml) {
    process.stderr.write('scenario needs a rendered page: ' + scenario + '\n');
    process.exit(2);
  }
  const api = createHarness({ uiDir, pageHtml });
  let result;
  try {
    result = await SCENARIOS[scenario](api);
  } catch (err) {
    process.stdout.write(JSON.stringify({ error: String(err && err.stack || err) }) + '\n');
    process.exit(1);
  }
  process.stdout.write(JSON.stringify(result, null, 2) + '\n');
}

if (require.main === module) main();
