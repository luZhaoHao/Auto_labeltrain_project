/*
 * 训练监控共用逻辑：把一次训练结果投影到监控卡片。
 *
 * 普通训练原始响应流（startTraining 的 SSE）与页面重连流（_subscribeActiveRun）必须
 * 复用同一份字段转换，否则同一个 run 会因入口不同而显示不一致（例如重连终态事件
 * 不更新指标）。这里只做投影：epochs、mAP50、mAP50-95、Precision/Recall 与报告链接；
 * 日志、状态徽标、停止按钮与 SSE 生命周期仍由页面自己负责。
 *
 * 单独成文件是为了让真实前端逻辑可被自动测试直接执行（tests/js/minidom.js）。
 */
(function () {
  'use strict';

  var METRIC_CARDS = ['monitorEpochs', 'monitorMap50', 'monitorMap5095', 'monitorPR'];

  function text(id, value) {
    var el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  function fixed(value, digits) {
    // 真实 0 是有效数值（必须显示 0.0000），只有缺失值才显示 —
    if (value === null || value === undefined || value === '') return '—';
    var num = Number(value);
    if (!isFinite(num)) return '—';
    return digits == null ? String(num) : num.toFixed(digits);
  }

  // 可转换的数值；boolean 与空串不得被当成 0
  function toNumber(value) {
    if (value === null || value === undefined || value === '' || typeof value === 'boolean') return null;
    var num = Number(value);
    return isFinite(num) ? num : null;
  }

  // finalize_training_run() 的稳定结构是 {configured, completed, best}，不是标量。
  // 结构化对象存在时以“已配置”为准，缺失则依次回退到 completed、params.epochs；
  // 只有 result.epochs 本身是可转换的标量时才按旧格式取值；全部缺失或非法返回 null，
  // 由调用方诚实显示“—”，绝不补造 0。
  function epochsValue(result) {
    var epochs = result.epochs;
    if (epochs !== null && epochs !== undefined && typeof epochs === 'object') {
      var candidates = [epochs.configured, epochs.completed];
      for (var i = 0; i < candidates.length; i++) {
        var value = toNumber(candidates[i]);
        if (value !== null) return value;
      }
      return toNumber((result.params || {}).epochs);
    }
    return toNumber(epochs);
  }

  function resetMetrics() {
    for (var i = 0; i < METRIC_CARDS.length; i++) text(METRIC_CARDS[i], '—');
    var link = document.getElementById('monitorReportLink');
    if (link) link.style.display = 'none';
  }

  function applyResultToMonitor(result) {
    if (!result || typeof result !== 'object') return false;
    var metrics = result.metrics || {};
    text('monitorEpochs', fixed(epochsValue(result), null));
    text('monitorMap50', fixed(metrics.mAP50, 4));
    text('monitorMap5095', fixed(metrics.mAP50_95, 4));
    // Precision/Recall 成对展示：缺任一即整体不可用
    text('monitorPR',
      (metrics.precision === null || metrics.precision === undefined ||
       metrics.recall === null || metrics.recall === undefined)
        ? '—'
        : fixed(metrics.precision, 3) + ' / ' + fixed(metrics.recall, 3));
    var link = document.getElementById('monitorReportLink');
    if (link) {
      if (result.artifacts && result.artifacts.report_path) {
        link.href = '/api/training/report-by-name?name=' +
          encodeURIComponent(result.run_name || '');
        link.style.display = '';
      } else {
        link.style.display = 'none';
      }
    }
    return true;
  }

  window.TrainingMonitor = {
    applyResultToMonitor: applyResultToMonitor,
    resetMetrics: resetMetrics
  };
})();
