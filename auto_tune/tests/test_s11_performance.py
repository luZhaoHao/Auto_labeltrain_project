"""S1.1 hard performance-constraint tests (plan section 〇, P3-11).

- 11a: after 10,000 log lines the DOM line counts stay within the caps
      (500 default / 2000 full). The ring-buffer JS is extracted from the
      rendered template and executed directly under the local Node runtime.
- 11b: per-line processing time must not grow quadratically with history.
- 11c: the tuning SSE queue stays bounded under a slow client.
"""

import json
import shutil
import subprocess
import time

import pytest

from auto_tune.modules.agent_engine.training_log import (
    append_training_log,
    classify_training_line,
)

_DOM_MOCK = """
var _mockDoc = (function(){
  function makeEl(tag, isFrag) {
    return {
      tag: tag, className: '', textContent: '', nodeType: isFrag ? 11 : 1,
      childNodes: [], children: [], firstElementChild: null,
      appendChild: function(n) {
        if (n.nodeType === 11) {
          var kids = n.childNodes.slice();
          for (var k = 0; k < kids.length; k++) { this._add(kids[k]); }
          n.childNodes.length = 0; n.children.length = 0;
        } else { this._add(n); }
      },
      _add: function(node) {
        this.childNodes.push(node); this.children.push(node);
        this.firstElementChild = this.children[0] || null;
      },
      removeChild: function(node) {
        var i = this.childNodes.indexOf(node); if (i >= 0) this.childNodes.splice(i, 1);
        var j = this.children.indexOf(node); if (j >= 0) this.children.splice(j, 1);
        this.firstElementChild = this.children[0] || null;
      },
      classList: { contains: function(){return false;}, add: function(){}, remove: function(){}, toggle: function(){} },
      scrollTop: 0, scrollHeight: 0
    };
  }
  return {
    createElement: function(){ return makeEl('div', false); },
    createDocumentFragment: function(){ return makeEl('frag', true); }
  };
})();
var document = _mockDoc;
var TextDecoder = global.TextDecoder;
"""

_DRIVER = """
// Count how many times the scanning helper is called. If the ring buffer
// trimmed by rescaming the DOM per line, this would be ~10,000+; with the
// counter-based buffer it stays at the one-time-per-container init.
var _origLogElementCount = logElementCount;
var _scanCalls = 0;
logElementCount = function(c) { _scanCalls++; return _origLogElementCount(c); };

function bench(n) {
  var dEl = document.createElement('div');
  var fEl = document.createElement('div');
  var r = makeLogRenderer(dEl, fEl, MAX_DEFAULT_LOG_LINES, MAX_FULL_LOG_LINES);
  var t0 = Date.now();
  for (var i = 0; i < n; i++) {
    r.pushDefault('Epoch ' + i + ': box_loss=1.234 cls_loss=0.456', 'info');
    r.pushFull('1/' + n + ' 50%|' + i + ' [00:01<00:01, 4.5it/s]', 'info');
  }
  r.flushAll();
  var t1 = Date.now();
  return {
    n: n, ms: t1 - t0,
    defaultCount: _origLogElementCount(dEl),
    fullCount: _origLogElementCount(fEl)
  };
}
var oneK = bench(1000);
var scansBeforeTenK = _scanCalls;
var tenK = bench(10000);
console.log('RESULT ' + JSON.stringify({
  oneK: oneK, tenK: tenK,
  scansBeforeTenK: scansBeforeTenK, scansAfterTenK: _scanCalls
}));
"""


def _extract_log_helpers_js() -> str:
    from auto_tune.ui.app import _jinja_env
    from auto_tune.ui.i18n import make_translator

    translator = make_translator("zh")
    html = _jinja_env.get_template("single_page.html").render(
        _=translator,
        current_lang="zh",
        active_page="training_monitor",
        experiment_history=[],
        tuning_history=[],
        dataset=None,
        training={
            "summary": {"total_runs_analyzed": 0, "best_mAP50": None,
                        "best_overall_run": None, "average_mAP50": None, "runs_with_issues": 0},
            "runs": {},
            "suggestion": None,
        },
        project={},
        latest_suggestion=None,
        current_args=None,
        dataset_analyzer_config={},
        training_config={},
        llm_analysis=None,
        vision_analysis=None,
        latest_dataset=None,
    )
    start = html.index("// ── S1.1 shared log-layering helpers")
    end = html.index("(function() {\n  var tuningForm", start)
    return html[start:end]


def test_s11_log_renderer_10k_lines_ring_buffer_bounded():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node runtime unavailable; 10k line-cap check deferred to browser acceptance")

    js_block = _extract_log_helpers_js()
    assert "innerHTML" not in js_block  # constraint 1 holds for the log helpers

    script = _DOM_MOCK + "\n" + js_block + "\n" + _DRIVER
    proc = subprocess.run(
        [node, "-e", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"node failed:\n{proc.stdout}\n{proc.stderr}"

    result = json.loads(proc.stdout.strip().splitlines()[-1].split("RESULT ", 1)[1])
    one_k, ten_k = result["oneK"], result["tenK"]

    # 11a: 10,000 lines must stay within the caps.
    assert ten_k["defaultCount"] <= 500, ten_k
    assert ten_k["fullCount"] <= 2000, ten_k
    # Ring buffer does not under-remove: keeps up to the cap, not far below it.
    assert ten_k["defaultCount"] >= 450, ten_k
    assert ten_k["fullCount"] >= 1500, ten_k

    # Strict O(1): the DOM is never rescanned in the hot path. Only the
    # one-time-per-container init may call logElementCount (2 containers per
    # bench x 2 benches = 4). A scan-per-line implementation would reach 10k+.
    assert result["scansAfterTenK"] <= 4, result

    # 11b: per-line cost must not grow quadratically with history.
    one_k_per_line = one_k["ms"] / (2 * one_k["n"])
    ten_k_per_line = ten_k["ms"] / (2 * ten_k["n"])
    assert ten_k_per_line <= one_k_per_line * 5 + 0.1, {
        "oneK_per_line": one_k_per_line,
        "tenK_per_line": ten_k_per_line,
    }


def test_s11_classify_linear_time_not_quadratic(tmp_path):
    """Server-side classify/append stays linear as the log grows."""
    epoch_line = "  1/100  1.20G  1.234  0.456  0.789"

    def bench(n):
        log_path = tmp_path / f"training_{n}.log"
        t0 = time.perf_counter()
        for _ in range(n):
            classify_training_line(epoch_line)
            append_training_log(log_path, epoch_line)
        return time.perf_counter() - t0

    small = bench(1000)
    large = bench(10000)
    large_per_line = large / 10000
    small_per_line = small / 1000
    assert large_per_line <= small_per_line * 5 + 0.01, {
        "small_per_line": small_per_line,
        "large_per_line": large_per_line,
    }


def test_tuning_sse_queue_bounded_under_slow_client():
    import queue

    from auto_tune.ui.app import _enqueue_bounded

    q = queue.Queue(maxsize=10)
    for i in range(100):
        _enqueue_bounded(q, {"status": "running", "message": f"m{i}", "iteration": 1})

    # The bounded producer never lets the queue grow past maxsize.
    assert q.qsize() == 10
    # Terminal events bypass on_progress (emitted by the generator), so the
    # bounded queue can only drop transient progress, never the final status.
    _enqueue_bounded(q, {"status": "running", "message": "m100", "iteration": 1})
    assert q.qsize() == 10
