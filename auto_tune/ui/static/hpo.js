/*
 * H1.3 HPO Studio front-end (rework).
 *
 * Consumes only the /api/hpo routes (read-only projections plus the controlled
 * formal-training submission). It never builds training commands, never calls
 * /tuning/start for HPO, never sends search parameters, and renders every
 * external value through textContent (no innerHTML with user values).
 *
 * Selection identity is enforced with a monotonically increasing generation:
 * every detail/best reply carries the generation it was requested under, and a
 * reply that no longer belongs to the current selection is dropped instead of
 * being rendered over a newer study. Polling is a single timer bound to one
 * study id; switching studies stops the previous timer first.
 */
(function () {
  'use strict';

  var SEARCH_KEYS = ['optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs'];

  // 评价模式：后端版本化契约里的确定性单一目标；权重由服务端给出。
  // 新建研究只允许这两种产品模式——legacy 只能被读取、恢复与展示（旧研究）。
  var CREATE_EVALUATION_MODES = ['comprehensive', 'quick'];
  var EVALUATION_LABELS = {
    comprehensive: '全面（mAP50 / mAP50-95 / Precision / Recall）',
    quick: '快速（mAP50 / mAP50-95）',
    legacy_map50_95: '旧版 mAP50-95（单指标）'
  };
  var METRIC_LABELS = {
    'metrics/mAP50(B)': 'mAP50',
    'metrics/mAP50-95(B)': 'mAP50-95',
    'metrics/precision(B)': 'Precision',
    'metrics/recall(B)': 'Recall'
  };
  // 运行身份必须是完整的 <kind>:<uuid4>；JSON 历史 ID（manual:trainN）绝不可用
  var RUNTIME_RUN_ID_RE = /^(?:manual|tuning):[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

  function evaluationLabel(mode) {
    return EVALUATION_LABELS[mode] || (mode == null ? '—' : String(mode));
  }

  function isRuntimeRunId(value) {
    return typeof value === 'string' && RUNTIME_RUN_ID_RE.test(value);
  }

  var STATUS_TEXT = {
    READY: '准备就绪',
    RUNNING: '运行中',
    PAUSED: '已暂停',
    INTERRUPTED: '已中断',
    COMPLETED: '已完成',
    BLOCKED: '需处理',
    starting: '启动中',
    running: '运行中',
    stopping: '正在停止',
    completed: '已完成',
    failed: '失败',
    cancelled: '已停止',
    interrupted: '已中断',
    unknown: '状态未知'
  };

  var state = {
    studyId: null,
    selection: 0,
    timer: null,
    pollStudyId: null,
    busy: false,
    refreshing: false,
    pendingRefresh: null,     // coalesced "last target" for a busy refresh round
    best: null,
    searchSpace: null,
    detail: null,
    historyOffset: 0,
    historyLimit: 20,
    historyCount: 0,
    historyOpen: false,       // HPO 历史面板默认收起，由显式按钮展开
    stopPending: false,
    stopPendingAt: 0,
    resumePending: false,     // 恢复请求已提交、等待执行器收敛（页面禁止重复提交）
    watchFormal: false,
    formalRunName: null,
    formalInitialized: false,
    // 正式训练“打开结果文件夹”的逐条状态（train_name → {pending, hint}）：
    // 一条记录的提交/提示绝不影响另一条，切换研究时整体清空。
    formalFolders: {},
    // 公共监控当前归属的 runtime 运行（只认 <kind>:<uuid4>）；null = 无归属，
    // 此时卡片诚实显示空状态，绝不保留全局最近运行。
    monitorRunId: null,
    lastDetailError: null,
    snapshots: [],            // last read-only snapshot listing (labels/details)
    defaults: null,           // server-authoritative defaults (applied once)
    defaultsApplied: false,
    defaultsLoaded: false,    // 服务端权威默认已成功到达（未到达前不允许创建）
    defaultsError: null,      // 读取默认失败时的可见、可操作说明
    operationError: null,     // user-operation failure (create/start/stop/formal)
    statusError: null         // server-reported study status error
  };
  window._hpoState = state;

  function $(id) { return document.getElementById(id); }

  function setHidden(el, hidden) {
    if (!el) return;
    if (hidden) el.classList.add('hidden'); else el.classList.remove('hidden');
  }

  function setText(id, text) {
    var el = $(id);
    if (el) el.textContent = text == null ? '—' : String(text);
  }

  function clearChildren(el) {
    if (!el) return;
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  function disable(btn, flag, label) {
    if (!btn) return;
    btn.disabled = !!flag;
    if (label != null) btn.textContent = label;
  }

  function errText(body) {
    body = body || {};
    var field = body.field ? '[' + body.field + '] ' : '';
    var code = body.error_code ? '[' + body.error_code + '] ' : '';
    var msg = body.error || '';
    var next = body.next_action ? ' ' + body.next_action : '';
    return field + code + msg + next;
  }

  function api(path, options) {
    return fetch(path, options || {}).then(function (resp) {
      return resp.json().catch(function () { return {}; }).then(function (body) {
        return { ok: resp.ok, status: resp.status, body: body };
      });
    });
  }

  function jsonPost(path, body) {
    return api(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {})
    });
  }

  // ── selection identity ───────────────────────────────────────────

  function nextSelection() {
    state.selection += 1;
    return state.selection;
  }

  function isCurrentSelection(generation, studyId) {
    if (studyId && studyId !== state.studyId) return false;
    return generation === state.selection;
  }
  window.hpoIsCurrentSelection = isCurrentSelection;

  function bumpSelection() {
    return nextSelection();
  }

  function clearBestAndErrors() {
    state.best = null;
    state.searchSpace = null;
    state.stopPending = false;
    state.resumePending = false;
    state.watchFormal = false;
    state.formalRunName = null;
    // 上一条正式训练的“打开结果文件夹”提示与在途状态不带进新的研究
    state.formalFolders = {};
    // 正式训练条件默认值由服务端给出，切换研究时重新读取原研究身份，
    // 但绝不把研究的 1epoch/64 当作正式生产默认（见 hpoLoadDefaults）。
    state.formalInitialized = false;
    state.operationError = null;
    state.statusError = null;
    setText('hpoBestSummary', '');
    clearChildren($('hpoBestParamsBody'));
    setHidden($('hpoBestBody'), true);
    setHidden($('hpoBestEmpty'), false);
    setHidden($('hpoBestBadge'), true);
    setText('hpoFieldError', '');
    setHidden($('hpoFieldError'), true);
    renderDetailErrors();
    setText('hpoFormalStatus', '');
    setText('hpoFormalSummary', '');
    setText('hpoFormalConditions', '');
    setText('hpoVerifyHint', '');
    setHidden($('hpoFormalRunsWarning'), true);
    setHidden($('hpoResultActions'), true);
    disable($('hpoOpenResultFolder'), true);
    disable($('hpoDownloadBest'), true);
    setText('hpoBestArtifactHint', '');
    disable($('hpoFormalBestBtn'), true);
    disable($('hpoVerifyBtn'), true);
    // 停止/恢复/启动已有 READY 任务只在拿到本研究的权威状态后才出现；
    // 切换或重载期间先隐藏，绝不留下不适用的禁用按钮。
    setHidden($('hpoStopBtn'), true);
    setHidden($('hpoResumeBtn'), true);
    setHidden($('hpoStartBtn'), true);
    setFormalRunsMessage('正在读取关联的正式训练…');
    // 切换研究必须先收敛监控所有权：清空卡片与日志，等本研究的
    // formal-runs 权威投影到达后再重新绑定。
    resetMonitorOwnership();
  }

  // 监控所有权收敛：断开上一运行并清空公共监控，绝不保留全局最近运行。
  function resetMonitorOwnership() {
    state.monitorRunId = null;
    if (typeof window.detachTrainingMonitor === 'function') {
      window.detachTrainingMonitor();
    }
  }

  function selectStudy(studyId) {
    stopPolling();
    state.studyId = studyId || null;
    state.lastDetailError = null;
    rememberSelection(studyId);
    var generation = bumpSelection();
    clearBestAndErrors();
    setText('hpoStatusText', studyId ? '正在读取任务 ' + shortId(studyId) + ' …' : '未选择任务');
    setText('hpoStudyMeta', '');
    clearChildren($('hpoFrozenSummaryBody'));
    setText('hpoRanking', '');
    setText('hpoRankingNote', '');
    setText('hpoRunNote', '');
    setHidden($('hpoRunNote'), true);
    setHidden($('hpoProgress'), true);
    setHidden($('hpoProgressCard'), false);
    setText('hpoProgressStatus', '未选择任务');
    disable($('hpoStartBtn'), true);
    disable($('hpoStopBtn'), true);
    disable($('hpoResumeBtn'), true);
    if (!studyId) { hpoRefreshHistory(); return Promise.resolve(); }
    return refreshRound(studyId, true);
  }
  window.hpoSelectStudy = selectStudy;

  function shortId(value) {
    return value ? String(value).slice(0, 8) : '—';
  }

  // 统一运行身份 ``<kind>:<uuid4>`` → 用户可读短编号（不展示完整 UUID）
  function shortRuntimeId(value) {
    var text = String(value == null ? '' : value);
    var parts = text.split(':');
    if (parts.length !== 2) return '—';
    return parts[0] + ':' + parts[1].slice(0, 8) + '…';
  }

  // 只记住“上次选择的研究 ID”，绝不保存参数事实/结果；重载后仅用于恢复选择，
  // 是否可用仍由服务端重新判定，且恢复过程绝不启动训练。
  var SELECTION_KEY = 'hpo.selectedStudy';

  function rememberSelection(studyId) {
    try {
      if (studyId) window.localStorage.setItem(SELECTION_KEY, String(studyId));
      else window.localStorage.removeItem(SELECTION_KEY);
    } catch (err) { /* storage disabled: selection restore is optional */ }
  }

  function restoreSelection() {
    var stored = null;
    try { stored = window.localStorage.getItem(SELECTION_KEY); } catch (err) { stored = null; }
    if (!stored || !/^hpo_[0-9a-f]{32}$/.test(stored) || state.studyId) return;
    selectStudy(stored).then(function () {
      // 无效 ID：明确提示并允许重新选择；不启动任何训练
      if (state.lastDetailError) rememberSelection(null);
    });
  }
  window.hpoRestoreSelection = restoreSelection;

  // ── mode switching ───────────────────────────────────────────────

  window.onTuningModeChange = function (mode) {
    var isHpo = mode === 'hpo';
    var hpoOnly = document.querySelectorAll('.hpo-only');
    for (var i = 0; i < hpoOnly.length; i++) {
      hpoOnly[i].style.display = isHpo ? '' : 'none';
    }
    var llmOnly = document.querySelectorAll('.llm-only');
    for (var j = 0; j < llmOnly.length; j++) {
      llmOnly[j].style.display = isHpo ? 'none' : '';
    }
    // 主操作只有一个：HPO 用 hpoCreateAndStartBtn，其他三模式用公共提交按钮，
    // 两者占据共同控制区的同一位置，切换时选项栏与按钮都不会跳动。
    var nonHpo = document.querySelectorAll('.non-hpo-only');
    for (var n = 0; n < nonHpo.length; n++) {
      nonHpo[n].style.display = isHpo ? 'none' : '';
    }
    // .hpo-mode = 该区域在 HPO 模式下始终显示；数据驱动的区域由渲染函数自行
    // 增删 hidden，模式切换只负责隐藏它们，绝不在无数据时凭空显示空卡片。
    var modeEls = document.querySelectorAll('.hpo-mode');
    for (var k = 0; k < modeEls.length; k++) {
      modeEls[k].classList.toggle('hidden', !isHpo);
    }
    placeMonitor();
    if (isHpo) {
      // 进入 HPO 先收敛监控所有权：绝不把全局最近运行留在正式训练监控里
      resetMonitorOwnership();
      if (!state.studyId) {
        setHidden($('hpoProgressCard'), true);
      }
      // 历史面板默认收起；展开状态由用户操作决定，不由模式切换强制打开
      setHidden($('hpoHistorySection'), !state.historyOpen);
      updateHistoryToggle();
      // 服务端权威默认绑定（数据/权重/预算/轮数）；只应用一次，不覆盖用户输入
      hpoLoadDefaults();
      hpoLoadSnapshots();
      hpoLoadLocalModels();
      // 仅恢复“上次选择”的 ID；可用性由服务端重新判定，且绝不自动启动训练
      restoreSelection();
      refreshRound(state.studyId, true);
    } else {
      stopPolling();
      // 离开 HPO：其它三模式继续使用原有的全局最近有效训练投影
      if (typeof window.restoreTrainingMonitorBaseline === 'function') {
        window.restoreTrainingMonitorBaseline();
      }
    }
  };

  // ── 公共监控屏幕的宿主编排 ───────────────────────────────────────
  //
  // 指标卡与日志是普通训练/大模型调参/HPO 正式训练共用的**同一组唯一 ID**：
  // HPO 模式下（且停留在智能训练页时）整体移动到正式训练区域的宿主里，
  // 其余情况（其它模式或切到训练监控页）移回训练监控页宿主。绝不复制节点。
  function placeMonitor() {
    var block = $('sharedMonitorBlock');
    if (!block) return;
    var modeEl = $('tuningModeSelect');
    var isHpo = !!modeEl && modeEl.value === 'hpo';
    var page2 = $('page2');
    var onIntelligentPage = !!page2 && page2.classList.contains('active');
    var host = (isHpo && onIntelligentPage) ? $('hpoFormalMonitorHost')
                                            : $('trainingMonitorHost');
    if (host && block.parentNode !== host) host.appendChild(block);
  }
  window.hpoPlaceMonitor = placeMonitor;

  // ── HPO 历史入口（显式按钮，面板默认收起） ────────────────────────
  function updateHistoryToggle() {
    var btn = $('hpoHistoryToggle');
    if (!btn) return;
    btn.textContent = state.historyOpen ? '收起 HPO 历史记录' : 'HPO 历史记录';
  }
  window.hpoToggleHistory = function () {
    state.historyOpen = !state.historyOpen;
    setHidden($('hpoHistorySection'), !state.historyOpen);
    updateHistoryToggle();
    if (state.historyOpen) hpoRefreshHistory();
  };

  document.addEventListener('DOMContentLoaded', function () {
    var watch = ['hpoSnapshotSelect', 'hpoSampler', 'hpoEvaluationMode',
                 'hpoBudget', 'hpoEpochs', 'hpoSeed', 'hpoBatch', 'hpoImgsz',
                 'hpoDevice', 'hpoTimeout', 'hpoFormalEpochs'];
    for (var i = 0; i < watch.length; i++) {
      var el = $(watch[i]);
      if (el) {
        el.addEventListener('change', updateDraftReadouts);
        el.addEventListener('input', updateDraftReadouts);
      }
    }
    var modelSelect = $('hpoModelSelect');
    if (modelSelect) {
      modelSelect.addEventListener('change', onModelSelectChanged);
    }
    var sel = $('tuningModeSelect');
    if (sel) window.onTuningModeChange(sel.value);
  });

  // ── polling (single timer, bound to one id, cleared on unload) ────

  function stopPolling() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    state.pollStudyId = null;
  }

  // One timer at most. Callers always stopPolling() before switching studies,
  // so the bound id and the timer can never diverge.
  function startPolling(id) {
    if (state.timer) return;
    if (!id) return;
    state.pollStudyId = id;
    state.timer = setInterval(function () {
      var current = state.pollStudyId;
      if (current) refreshRound(current, true);
    }, 2000);
  }
  window.hpoStopPolling = stopPolling;

  window.addEventListener('beforeunload', stopPolling);

  // 轮询只在“搜索控制器活跃”或“有关联的正式训练仍在跑”时进行。
  function syncPolling() {
    var active = !!(state.detail && state.detail.control_active);
    if (active || state.watchFormal) startPolling(state.studyId);
    else stopPolling();
  }

  // ── one serialized refresh round ─────────────────────────────────

  // A refresh round never runs in parallel with itself. If a new selection
  // arrives while one is in flight, its target is **coalesced** (last one wins)
  // and re-run as soon as the current round finishes — the newer study is
  // therefore always refreshed and rendered, instead of being silently dropped
  // and leaving the UI stuck on "正在读取".
  function refreshRound(id, silent) {
    if (state.refreshing) {
      state.pendingRefresh = { id: id, silent: silent };
      return Promise.resolve(false);
    }
    state.refreshing = true;
    var generation = state.selection;
    var chain = id
      ? hpoFetchStatus(id, silent, generation).then(function () {
          if (!isCurrentSelection(generation, id)) return null;
          // 关联记录来自持久化投影：首次选择也必须读取，watchFormal 只决定后续轮询
          return refreshFormalRuns(generation);
        }).then(hpoRefreshHistory)
      : hpoRefreshHistory();
    return chain.then(function () {
      return finishRound(true);
    }, function () {
      return finishRound(false);
    });
  }

  function finishRound(ok) {
    state.refreshing = false;
    var pending = state.pendingRefresh;
    state.pendingRefresh = null;
    // Only the *current* selection may be re-run; a stale queued target is
    // dropped (its generation guard already prevented it from rendering).
    if (pending && pending.id && pending.id === state.studyId) {
      refreshRound(pending.id, pending.silent);
    } else {
      syncPolling();
    }
    return ok;
  }
  window.hpoRefreshRound = refreshRound;

  // ── draft: snapshots / models ────────────────────────────────────

  function option(value, label) {
    var opt = document.createElement('option');
    opt.value = value;
    opt.textContent = label;
    return opt;
  }

  window.hpoLoadSnapshots = function () {
    var select = $('hpoSnapshotSelect');
    if (!select) return Promise.resolve();
    var previous = select.value;
    return api('/api/hpo/snapshots', {}).then(function (res) {
      // 与权重选择器同理：权威默认可能先到，当前值优先
      var wanted = select.value || previous;
      clearChildren(select);
      select.appendChild(option('', '请选择已发布快照'));
      if (!res.ok) {
        setText('hpoSnapshotHint', '快照列表暂不可用，请稍后重试。');
        // 列表读取失败也不能丢掉已有绑定（含服务端刚给出的权威默认）
        var fallback = (state.defaults && state.defaults.dataset &&
          state.defaults.dataset.snapshot &&
          state.defaults.dataset.snapshot.snapshot_id) || previous;
        syncSnapshotSelectToValue(wanted || fallback);
        updateDraftReadouts();
        return;
      }
      var rows = res.body.snapshots || [];
      state.snapshots = rows;
      var selectable = 0;
      for (var i = 0; i < rows.length; i++) {
        var row = rows[i];
        if (!row.selectable) {
          // 损坏快照仍然可见，但不可选：绝不静默替换数据
          select.appendChild(option('', (row.dataset_name || row.short_id || row.snapshot_id) +
            '（记录不可读取，不可选择）'));
          continue;
        }
        selectable += 1;
        // 以原数据集名称为主识别手段；图片总数 = train + val（background 是子集）
        var label = (row.dataset_name || '数据集') + ' · ' +
          'train ' + fmtCount(row.train_count) + ' / val ' + fmtCount(row.val_count) +
          ' · 图片 ' + fmtCount(row.image_count) +
          (row.created_at ? ' · ' + String(row.created_at).slice(0, 19) : '');
        select.appendChild(option(row.snapshot_id, label));
      }
      if (wanted) select.value = wanted;
      if (!selectable) {
        setText('hpoSnapshotHint', '没有可用快照：请先到数据集页创建并发布快照。系统不会自动替代数据集。');
      } else {
        setText('hpoSnapshotHint', '共 ' + selectable + ' 个可用快照。');
      }
      // 默认绑定可能比快照列表先到达；列表就绪后补一次绑定（只绑当前数据集
      // 已登记/同原目录的合法快照，不影响用户已手选的值）。
      applyDefaultSnapshot();
      renderDatasetSummary();
    }).catch(function () {
      setText('hpoSnapshotHint', '读取快照列表失败，请稍后重试。');
    });
  };

  // 主界面直接显示数据集名称、总图片数、训练集数与验证集数（背景是子集，不加）
  function renderDatasetSummary() {
    var id = String(($('hpoSnapshotSelect') || {}).value || '');
    var row = selectedSnapshotRow(id);
    if (!row) {
      setText('hpoDatasetSummary', '数据集：未选择');
      return;
    }
    setText('hpoDatasetSummary', '数据集 ' + (row.dataset_name || '—') +
      '　总图片 ' + fmtCount(row.image_count) +
      '　训练集 ' + fmtCount(row.train_count) +
      '　验证集 ' + fmtCount(row.val_count));
  }

  // 已绑定的快照必须在选择器里可见：列表不可用或没有该项时补一条“当前绑定”，
  // 绝不把已绑定的数据悄悄退回到未选择状态。
  function syncSnapshotSelectToValue(value) {
    var select = $('hpoSnapshotSelect');
    if (!select) return;
    var wanted = String(value == null ? '' : value).trim();
    if (!wanted) { selectOptionByValue(select, ''); return; }
    if (selectOptionByValue(select, wanted)) return;
    var row = selectedSnapshotRow(wanted);
    select.appendChild(option(wanted, (row && row.dataset_name ? row.dataset_name : '数据集') +
      '（当前绑定快照，列表暂不可用）'));
    select.value = wanted;
  }

  function applyDefaultSnapshot() {
    var select = $('hpoSnapshotSelect');
    if (!select || select.value) { updateDraftReadouts(); return; }
    var dataset = (state.defaults && state.defaults.dataset) || {};
    var want = dataset.snapshot && dataset.snapshot.snapshot_id;
    if (want) selectOptionByValue(select, want);
    updateDraftReadouts();
  }

  function fmtCount(value) {
    return (value == null) ? '—' : String(value);
  }

  window.hpoLoadLocalModels = function () {
    var select = $('hpoModelSelect');
    if (!select) return Promise.resolve();
    var previous = select.value;
    return api('/api/hpo/local-models', {}).then(function (res) {
      // 权威默认绑定可能比本请求先完成：以“当前值”为准，否则回退到发起时值，
      // 绝不把已经绑定的权重退回到“未选择”。
      var wanted = select.value || previous;
      clearChildren(select);
      select.appendChild(option('', '请选择本地 .pt 权重'));
      if (!res.ok) {
        // 列表读取失败也不能丢掉当前绑定：选择器仍要回显已绑定的权重名称
        syncModelSelectToPath(wanted);
        updateDraftReadouts();
        return;
      }
      var rows = res.body.models || [];
      for (var i = 0; i < rows.length; i++) {
        var row = rows[i];
        // 区分“模型初始权重”与“训练产物”，不能全部显示同一个 best.pt 标签
        var kind = row.kind === 'training_artifact' ? '训练产物' : '初始权重';
        select.appendChild(option(row.path, row.name +
          (row.size_mb == null ? '' : ' · ' + row.size_mb + ' MB') +
          ' · ' + kind + '（' + (row.origin || '未知来源') + '）'));
      }
      // 绑定的权重必须在选择器里可见，绝不能显示成“未选择”
      syncModelSelectToPath(wanted);
      updateDraftReadouts();
    }).catch(function () { /* the already-bound value stays selected */ });
  };

  // 选择器是**唯一**的权重入口：绑定的路径不在受控列表里时补一条“当前绑定”项
  // （只显示名称，不显示物理路径），绝不留下可编辑的路径输入。
  function syncModelSelectToPath(path) {
    var select = $('hpoModelSelect');
    if (!select) return;
    var value = String(path == null ? '' : path).trim();
    if (!value) {
      selectOptionByValue(select, '');
      renderModelSummary();
      return;
    }
    if (!selectOptionByValue(select, value)) {
      select.appendChild(option(value, '当前绑定：' + modelBasename(value) +
        '（由当前项目/训练配置指定）'));
      select.value = value;
    }
    renderModelSummary();
  }

  function modelBasename(path) {
    var text = String(path || '');
    var parts = text.split(/[\\/]/);
    return parts[parts.length - 1] || text;
  }

  // 主界面只显示模型名称；物理路径既不显示也不接受手工输入
  function renderModelSummary() {
    var select = $('hpoModelSelect');
    var value = String((select && select.value) || '').trim();
    setText('hpoModelSummary', '模型名称：' + (value ? modelBasename(value) : '未选择'));
  }

  function onModelSelectChanged() {
    renderModelSummary();
    updateDraftReadouts();
  }

  // ── 服务端权威默认绑定（少让用户选）─────────────────────────────
  //
  // 默认数据快照 = 当前数据集已登记且有效的绑定；默认权重 = 当前 project/training
  // 配置里的合法本地 .pt（缺失才回退既有 yolov8n.pt）。无法可靠绑定时只提示用户
  // 确认选择，绝不自动换数据/模型。只在首次加载时应用，不覆盖用户后续输入。
  window.hpoLoadDefaults = function () {
    return api('/api/hpo/defaults', {}).then(function (res) {
      if (!res.ok || !res.body || !res.body.search) {
        // 读取失败绝不静默启用 CPU 创建：给出可见、可操作的错误并保持不可创建
        state.defaultsLoaded = false;
        state.defaultsError = '读取服务端默认配置失败：请检查服务端状态后重新进入本模式重试（未使用 CPU 代替）。';
        updateDraftReadouts();
        return null;
      }
      state.defaults = res.body;
      state.defaultsError = null;
      applyDefaults(res.body);
      state.defaultsLoaded = true;
      updateDraftReadouts();
      return res.body;
    }).catch(function () {
      state.defaultsLoaded = false;
      state.defaultsError = '读取服务端默认配置失败：网络暂不可用，请稍后重试（未使用 CPU 代替）。';
      updateDraftReadouts();
      return null;
    });
  };

  // 设备选择器：只列服务端已探测到的 GPU 编号与 CPU，绝不接受自由文本。
  function renderDeviceOptions(devices, fallback) {
    var select = $('hpoDevice');
    if (!select) return;
    var current = String(select.value == null ? '' : select.value).trim();
    var gpus = (devices && devices.gpus) || [];
    clearChildren(select);
    for (var i = 0; i < gpus.length; i++) {
      var index = gpus[i];
      select.appendChild(option(String(index), 'GPU ' + index));
    }
    select.appendChild(option('cpu', 'CPU'));
    // 已经选定的值优先（用户选择不被覆盖）；否则用服务端权威默认
    var wanted = current || ((devices && devices.default) || fallback || '');
    if (wanted) selectOptionByValue(select, String(wanted));
    if (!select.value) select.value = '';
  }

  function applyDefaults(payload) {
    if (state.defaultsApplied) return;
    var search = payload.search || {};
    var formal = payload.formal || {};
    setValue('hpoSampler', search.sampler, true);
    setValue('hpoEvaluationMode', search.evaluation_mode, true);
    setValue('hpoBudget', search.budget);
    setValue('hpoEpochs', search.epochs);
    setValue('hpoSeed', search.seed);
    setValue('hpoTimeout', search.timeout_seconds);
    setValue('hpoBatch', search.batch);
    setValue('hpoImgsz', search.imgsz);
    // 只有服务端探测结果与权威默认都到达后才产生可提交的设备值
    renderDeviceOptions(payload.devices, search.device);
    // 正式训练默认轮数取当前普通训练配置（缺失 100），绝不继承验收研究的 1 轮
    setValue('hpoFormalEpochs', formal.epochs);
    var dataset = payload.dataset || {};
    if (dataset.snapshot && dataset.snapshot.snapshot_id) {
      setValue('hpoSnapshotSelect', dataset.snapshot.snapshot_id, true);
    }
    // 默认初始权重只作为受控列表里的一个选项出现：不从 .pt 列表里自动挑选
    // 搜索产物（避免误选 best.pt），也不自动下载或换模型。
    var model = payload.model || {};
    if (model.available && model.path) {
      syncModelSelectToPath(model.path);
    } else {
      renderModelSummary();
    }
    state.defaultsApplied = true;
  }

  function setValue(id, value, force) {
    var el = $(id);
    if (!el || value == null) return;
    if (force && el.tagName === 'SELECT') {
      selectOptionByValue(el, String(value));
      return;
    }
    if (!el.value || el.value === el.defaultValue) el.value = String(value);
  }

  function selectOptionByValue(select, value) {
    for (var i = 0; i < select.options.length; i++) {
      if (select.options[i].value === value) { select.value = value; return true; }
    }
    return false;
  }

  // 绑定只来自受控选择器：没有可编辑路径输入，也没有任意路径文本。
  function draftInputs() {
    var select = $('hpoSnapshotSelect');
    var modelSelect = $('hpoModelSelect');
    var deviceSelect = $('hpoDevice');
    return {
      snapshot_id: String((select && select.value) || '').trim(),
      model_path: String((modelSelect && modelSelect.value) || '').trim(),
      device: String((deviceSelect && deviceSelect.value) || '').trim()
    };
  }

  // 创建按钮可用性：服务端默认已成功到达，且数据/权重/设备等绑定事实齐全合法。
  function updateCreateAvailability() {
    var btn = $('hpoCreateAndStartBtn');
    if (!btn) return;
    if (state.busy) return;
    btn.disabled = !(state.defaultsLoaded && !!readDraftConfigSafe()
      && bindingAttention().length === 0);
  }

  function selectedSnapshotRow(snapshotId) {
    var select = $('hpoSnapshotSelect');
    if (!select || !snapshotId) return null;
    var rows = (state.snapshots || []);
    for (var i = 0; i < rows.length; i++) {
      if (rows[i].snapshot_id === snapshotId) return rows[i];
    }
    return null;
  }

  // 主摘要保持简短；完整 ID/来源/技术细节放进“高级技术选项”折叠区。
  // 选择不等于可信：创建时仍由服务端重新校验快照与权重文件及其哈希。
  function updateDraftReadouts() {
    var inputs = draftInputs();
    var cfg = readDraftConfigSafe();
    var row = selectedSnapshotRow(inputs.snapshot_id);
    renderDatasetSummary();
    renderModelSummary();
    var short = [];
    short.push('数据：' + (inputs.snapshot_id
      ? ((row && row.dataset_name) || '数据集') + '（图片 ' +
        fmtCount(row && row.image_count) + '）'
      : '未绑定快照，请确认当前数据集快照'));
    // 主摘要只给模型名称；完整内部身份与固定条件在“高级技术选项”里核对
    short.push('初始权重：' + (inputs.model_path ? modelBasename(inputs.model_path) : '未选择'));
    if (cfg) {
      // 编辑控件收进折叠区后，主摘要必须继续如实反映即将提交的条件
      short.push('算法 ' + samplerLabel(cfg.sampler));
      short.push('评价方式 ' + evaluationLabel(cfg.evaluation_mode));
      short.push('试验次数 ' + cfg.budget);
      short.push('每次 ' + cfg.epochs + ' 轮');
      short.push('图像尺寸 ' + cfg.imgsz);
      short.push('Batch ' + cfg.batch);
      short.push('设备 ' + cfg.device);
    }
    setText('hpoCreateConfirm', short.join('　|　'));

    var detail = [];
    detail.push('完整数据快照 ID：' + (inputs.snapshot_id || '未选择'));
    if (row && row.created_at) detail.push('创建时间：' + row.created_at);
    if (row) {
      detail.push('train ' + fmtCount(row.train_count) + ' / val ' + fmtCount(row.val_count) +
        ' / 背景 ' + fmtCount(row.background_count) + '（背景是 train/val 的子集，不重复相加）');
    }
    detail.push('初始权重名称：' + (inputs.model_path ? modelBasename(inputs.model_path) : '未选择'));
    if (cfg) {
      detail.push('种子 ' + cfg.seed + '，Batch ' + cfg.batch +
        '，图像尺寸 ' + cfg.imgsz + '，单试验超时 ' + cfg.timeout_seconds + ' 秒');
    }
    setText('hpoCreateConfirmDetail', detail.join('　|　'));

    // 共同控制区主摘要：实际数据/模型/预算/轮数/设备
    var main = 'HPO：' + short.join('　|　');
    var warnings = (state.defaults && state.defaults.config_warnings) || [];
    if (warnings.length) {
      var fields = [];
      for (var w = 0; w < warnings.length; w++) fields.push(warnings[w].field);
      // 非法已配置值不静默替换：原值照常显示，并明确提示修正
      main += '　|　注意：当前训练配置中的 ' + fields.join('、') +
        ' 不合法，已按原值显示，请修正后再开始。';
    }
    // 无 GPU 时的可操作提示（来自服务端探测，不改变用户已选条件）
    var notice = state.defaults && state.defaults.device_notice;
    if (notice && notice.message) main += '　|　' + notice.message;
    setText('hpoMainSummary', main);
    // 绑定缺失/不合法/需用户确认时只在主区说明原因；选择器本身始终可见可操作，
    // 不需要再展开任何折叠区（面板默认收起）。
    var reasons = bindingAttention();
    if (reasons.length) {
      setText('hpoBindingNotice', reasons.join('　'));
      setHidden($('hpoBindingNotice'), false);
    } else {
      setText('hpoBindingNotice', '');
      setHidden($('hpoBindingNotice'), true);
    }
    updateFormalSummary();
    updateCreateAvailability();
  }

  // 需要用户确认的绑定原因（空数组 = 可靠绑定，主区不显示多余提示）
  function bindingAttention() {
    var inputs = draftInputs();
    var dataset = (state.defaults && state.defaults.dataset) || {};
    var model = (state.defaults && state.defaults.model) || {};
    var reasons = [];
    if (state.defaultsError) {
      reasons.push(state.defaultsError);
      return reasons;
    }
    if (!state.defaultsLoaded) {
      reasons.push('正在读取服务端默认配置与设备列表，完成后才能创建调优任务。');
      return reasons;
    }
    if (!inputs.device) {
      reasons.push('计算设备未就绪：请在“计算设备”中选择一个 GPU 编号或 CPU。');
    }
    if (!inputs.snapshot_id) {
      reasons.push(dataset.reason_code === 'NO_REGISTERED_DATASET'
        ? '当前数据集尚未登记：请先到数据集页确认数据集并创建快照（系统不会自动替代数据）。'
        : '未找到当前数据集的可靠快照绑定：请在上方“已发布数据快照”中选择要使用的快照。');
    }
    if (!inputs.model_path) {
      reasons.push(model.reason_code === 'MODEL_CONFIG_INVALID'
        ? '当前配置里的初始权重不合法：请在上方“本地初始权重”中选择一次合法的本地 .pt 权重。'
        : '未找到合法的本地初始权重：请在上方“本地初始权重”中选择一次本地 .pt 权重。');
    } else if (model.reason_code === 'MODEL_CONFIG_INVALID') {
      reasons.push('当前配置里的初始权重不合法（已按原值显示）：请确认后再开始。');
    }
    return reasons;
  }
  window.hpoUpdateDraftReadouts = updateDraftReadouts;

  function readDraftConfigSafe() {
    try {
      var cfg = readDraftConfig();
      for (var key in cfg) {
        if (cfg[key] == null) return null;
      }
      return cfg;
    } catch (err) {
      return null;
    }
  }

  function numberField(id) {
    var el = $(id);
    if (!el) return { value: null, empty: true };
    var raw = String(el.value == null ? '' : el.value).trim();
    if (raw === '') return { value: null, empty: true };
    if (!/^-?\d+$/.test(raw)) return { value: null, bad: true };
    var value = Number(raw);
    if (!isFinite(value)) return { value: null, bad: true };
    return { value: value, empty: false };
  }

  function rangeError(field, read, min, max) {
    if (read.empty) return { field: field, message: '请填写' + field + '（不能留空）。' };
    if (read.bad) return { field: field, message: field + ' 必须是整数，请修正后重试。' };
    if (read.value < min || read.value > max) {
      return { field: field, message: field + ' 必须在 ' + min + '..' + max + ' 之间。' };
    }
    return null;
  }

  function validateDraft() {
    var budget = numberField('hpoBudget');
    var epochs = numberField('hpoEpochs');
    var seed = numberField('hpoSeed');
    var batch = numberField('hpoBatch');
    var imgsz = numberField('hpoImgsz');
    var timeout = numberField('hpoTimeout');
    var deviceEl = $('hpoDevice');
    var device = String((deviceEl && deviceEl.value) || '').trim();
    var modeEl = $('hpoEvaluationMode');
    var mode = String((modeEl && modeEl.value) || '').trim();
    return (CREATE_EVALUATION_MODES.indexOf(mode) < 0
        ? { field: '评价方式', message: '评价方式必须是全面或快速。' } : null) ||
      rangeError('预算（试验次数）', budget, 1, 100) ||
      rangeError('每次训练轮数', epochs, 1, 1000) ||
      rangeError('种子', seed, 0, 2147483647) ||
      rangeError('Batch', batch, 1, 256) ||
      rangeError('图像尺寸', imgsz, 32, 2048) ||
      (imgsz.value != null && imgsz.value % 32 !== 0
        ? { field: '图像尺寸', message: '图像尺寸必须是 32 的整数倍。' } : null) ||
      rangeError('单试验超时（秒）', timeout, 1, 86400) ||
      (!/^(cpu|[0-9]|[1-5][0-9]|6[0-3])$/.test(device)
        ? { field: '计算设备', message: '计算设备必须是 cpu 或单个 GPU 编号 0..63。' } : null);
  }

  function showFieldError(fieldError, targetId) {
    var el = $(targetId || 'hpoFieldError');
    if (!el) return;
    if (!fieldError) { el.textContent = ''; el.classList.add('hidden'); return; }
    el.textContent = '[' + fieldError.field + '] ' + fieldError.message;
    el.classList.remove('hidden');
  }

  // 采样器/评价方式/计算设备/图像尺寸/Batch 的**编辑控件**默认收在“高级技术选项”
  // 折叠区里；这些字段校验失败时必须自动展开，否则用户看不到出错的地方。
  // 键同时接受前端字段名与服务端字段名，避免因文案不同而漏展开。
  var COLLAPSED_DRAFT_FIELDS = {
    sampler: true, evaluation_mode: true, device: true, imgsz: true, batch: true,
    '采样器': true, '评价方式': true, '计算设备': true, '图像尺寸': true,
    'Batch': true
  };

  function revealCollapsedDraft(fieldError) {
    if (!fieldError || !COLLAPSED_DRAFT_FIELDS[fieldError.field]) return;
    var details = $('hpoDraftDetails');
    if (details) details.open = true;
  }

  function readDraftConfig() {
    return {
      sampler: $('hpoSampler').value,
      evaluation_mode: $('hpoEvaluationMode').value,
      budget: numberField('hpoBudget').value,
      epochs: numberField('hpoEpochs').value,
      seed: numberField('hpoSeed').value,
      batch: numberField('hpoBatch').value,
      imgsz: numberField('hpoImgsz').value,
      device: String($('hpoDevice').value || '').trim(),
      timeout_seconds: numberField('hpoTimeout').value
    };
  }

  // ── create / start / resume / stop ───────────────────────────────

  function createStudy() {
    if (state.busy) return Promise.resolve(false);
    var inputs = draftInputs();
    showFieldError(null);
    setOperationError(null);
    if (!inputs.snapshot_id) {
      showFieldError({ field: '数据快照', message: '请选择已发布的快照。' });
      return Promise.resolve(false);
    }
    if (!inputs.model_path) {
      showFieldError({ field: '本地初始权重', message: '请选择或填写本地 .pt 权重路径。' });
      return Promise.resolve(false);
    }
    var fieldError = validateDraft();
    if (fieldError) {
      // 出错字段的控件在折叠区里时先展开，错误仍显示在折叠区之外的 hpoFieldError
      revealCollapsedDraft(fieldError);
      showFieldError(fieldError);
      return Promise.resolve(false);
    }
    var cfg = readDraftConfig();
    var body = {
      snapshot_id: inputs.snapshot_id,
      model_path: inputs.model_path,
      study_config: {
        sampler: cfg.sampler, evaluation_mode: cfg.evaluation_mode,
        budget: cfg.budget, epochs: cfg.epochs, seed: cfg.seed
      },
      execution_config: {
        batch: cfg.batch, imgsz: cfg.imgsz, device: cfg.device,
        timeout_seconds: cfg.timeout_seconds
      }
    };
    state.busy = true;
    disable($('hpoCreateAndStartBtn'), true, '创建中…');
    return jsonPost('/api/hpo/studies', body).then(function (res) {
      if (res.status === 201 && res.body.study_id) {
        return res.body.study_id;
      }
      if (res.body && res.body.error_code === 'INVALID_HPO_FIELD') {
        var serverField = { field: res.body.field, message: (res.body.error || '') + ' ' + (res.body.next_action || '') };
        revealCollapsedDraft(serverField);
        showFieldError(serverField);
      } else {
        showFieldError({ field: '创建', message: errText(res.body) });
      }
      return null;
    }).catch(function () {
      // 网络结果未知：绝不自动重发创建
      showFieldError({ field: '创建', message: '创建请求结果未知，请到 HPO 历史中确认是否已创建，不要重复点击创建。' });
      return null;
    }).then(function (studyId) {
      state.busy = false;
      var btn = $('hpoCreateAndStartBtn');
      if (btn) btn.textContent = '创建并开始调优';
      // 可用性由事实决定，绝不无条件重新启用
      updateCreateAvailability();
      return studyId;
    });
  }

  function startStudy(id, resume) {
    if (state.busy || !id) return Promise.resolve(false);
    state.busy = true;
    // 明确的用户操作：清掉上一次操作错误（轮询错误保留在状态区）
    setOperationError(null);
    var path = '/api/hpo/studies/' + encodeURIComponent(id) + (resume ? '/resume' : '/start');
    return jsonPost(path, {}).then(function (res) {
      if (res.status === 202 || res.status === 200) {
        setStatus(resume ? '已提交恢复请求，等待执行器收敛…' : '已提交启动请求，等待执行器收敛…');
        return true;
      }
      setOperationError((resume ? '恢复失败 ' : '启动失败 ') + errText(res.body));
      return false;
    }).catch(function () {
      setOperationError('请求结果未知，请稍后刷新查看该任务状态。');
      return false;
    }).then(function (ok) {
      state.busy = false;
      return ok;
    });
  }

  // 序列：创建成功后串行启动；创建结果未知时绝不重发创建。
  window.hpoCreateAndStart = function () {
    if (state.busy) return;
    createStudy().then(function (studyId) {
      if (!studyId) return;
      selectStudy(studyId);
      startStudy(studyId, false);
    });
  };

  window.hpoStartStudy = function () {
    if (state.studyId) startStudy(state.studyId, false);
  };

  // 恢复：使用所选研究的完整 study ID；提交后页面立即进入 pending（按钮禁用 +
  // 明确文案），不依赖 state.busy 静默吞掉重复点击；失败或结果未知时恢复可操作。
  function resumeStudy(id) {
    if (state.resumePending) return Promise.resolve(false);
    state.resumePending = true;
    setOperationError(null);
    disable($('hpoResumeBtn'), true, '正在恢复…');
    setStatus('正在恢复：已提交恢复请求，等待执行器收敛…');
    return jsonPost('/api/hpo/studies/' + encodeURIComponent(id) + '/resume', {})
      .then(function (res) {
        if (res.status === 202 || res.status === 200) {
          // 只刷新状态等待服务端收敛，绝不在这里伪造“已恢复”
          return refreshRound(id, true);
        }
        state.resumePending = false;
        disable($('hpoResumeBtn'), false, '检查并恢复');
        setOperationError('恢复失败 ' + errText(res.body));
        return false;
      })
      .catch(function () {
        state.resumePending = false;
        disable($('hpoResumeBtn'), false, '检查并恢复');
        setOperationError('恢复请求结果未知，请稍后刷新查看。');
        return false;
      });
  }

  window.hpoResumeStudy = function () {
    if (!state.studyId) return;
    resumeStudy(state.studyId);
  };

  window.hpoStopStudy = function () {
    if (!state.studyId) return;
    var id = state.studyId;
    jsonPost('/api/hpo/studies/' + encodeURIComponent(id) + '/stop', {}).then(function (res) {
      if (res.status === 202) {
        state.stopPending = true;
        state.stopPendingAt = Date.now();
        setStatus('正在停止：已提交停止请求，尚未停止…');
        disable($('hpoStopBtn'), true, '正在停止…');
      } else if (res.status === 200) {
        setStatus('该任务已结束，无需停止。');
      } else {
        setOperationError('停止失败 ' + errText(res.body));
      }
      refreshRound(id, true);
    }).catch(function () {
      setOperationError('停止请求结果未知，请稍后刷新查看。');
    });
  };

  window.hpoRefresh = function () {
    refreshRound(state.studyId, false);
  };

  // 共享主按钮在 HPO 模式下等价于“创建并开始调优”
  window.hpoHandleSubmit = function () {
    if (!$('hpoCreateDraft')) {
      // the HPO draft only exists on the Intelligent Analysis page
      window.alert('请在“智能训练”页面使用 HPO 算法调参。');
      return;
    }
    window.hpoCreateAndStart();
  };

  // ── status rendering ─────────────────────────────────────────────

  function setStatus(text) {
    setText('hpoStatusText', text);
  }

  // 用户操作错误与服务端状态错误分开保存、合并显示：轮询成功只清服务端那一半，
  // 用户的操作错误保留到用户修正或发起新的明确操作为止。
  function setOperationError(text) {
    state.operationError = text || null;
    renderDetailErrors();
  }

  function setStatusError(text) {
    state.statusError = text || null;
    renderDetailErrors();
  }

  function renderDetailErrors() {
    var el = $('hpoStudyDetailError');
    if (!el) return;
    var parts = [];
    if (state.operationError) parts.push(state.operationError);
    if (state.statusError) parts.push(state.statusError);
    if (!parts.length) { el.textContent = ''; el.classList.add('hidden'); return; }
    el.textContent = parts.join('　|　');
    el.classList.remove('hidden');
  }

  function hpoFetchStatus(id, silent, generation) {
    if (generation == null) generation = state.selection;
    return api('/api/hpo/studies/' + encodeURIComponent(id), {}).then(function (res) {
      if (!isCurrentSelection(generation, id)) return;   // 旧响应不覆盖新选择
      if (!res.ok) {
        state.lastDetailError = res.body && res.body.error_code;
        if (res.status === 404 || state.lastDetailError === 'HPO_NOT_FOUND') {
          setOperationError('该研究已不可用，请重新选择：' + errText(res.body));
          rememberSelection(null);
        } else if (!silent) {
          setOperationError('查询失败 ' + errText(res.body));
        } else {
          showRunNote('状态暂不可用，稍后自动重试。');
        }
        return;
      }
      setHidden($('hpoRunNote'), true);
      renderStatus(res.body, generation);
    }).catch(function () {
      if (!isCurrentSelection(generation, id)) return;
      showRunNote('网络暂不可用：显示最近一次可信状态，正在重试…');
    });
  }
  window.hpoFetchStatus = hpoFetchStatus;

  function showRunNote(text) {
    var el = $('hpoRunNote');
    if (!el) return;
    el.textContent = text;
    el.classList.remove('hidden');
  }

  function renderStatus(body, generation) {
    if (!isCurrentSelection(generation, body.study_id)) return;
    var status = body.execution_status || 'READY';
    var counts = (body.terminal_count == null ? '—' : body.terminal_count) + '/' +
      (body.budget == null ? '—' : body.budget);
    setStatus('状态：' + (STATUS_TEXT[status] || status) + '（' + status + '）　已完成试验：' +
      counts + '　成功：' + (body.success_count == null ? '—' : body.success_count) +
      (body.control_active ? '　（执行中）' : ''));
    setText('hpoStudyMeta', '研究 ' + shortId(body.study_id) + '　算法 ' +
      samplerLabel(body.sampler) + '　种子 ' + (body.seed == null ? '—' : body.seed));
    // 服务端状态错误单独保存；轮询成功绝不抹掉用户操作错误（创建/启动/停止/
    // 正式参数），只有用户发起新操作时才会清掉那一条。
    if (body.error_code) {
      setStatusError('[' + body.error_code + '] ' + (body.next_action || ''));
    } else {
      setStatusError(null);
    }

    state.detail = body;
    renderFrozenSummary(body);
    renderSearchConfig(body.search_space);
    setHidden($('hpoBestArea'), false);

    var active = !!body.control_active;
    // 停止 pending：收到 202 只代表请求已被接受，尚未停止
    if (state.stopPending && active) {
      disable($('hpoStopBtn'), true, '正在停止…');
    } else {
      state.stopPending = false;
      disable($('hpoStopBtn'), !body.can_stop, '停止');
    }
    // 停止/恢复只在适用时显示；不适用就隐藏，不留下永远禁用的按钮
    setHidden($('hpoStopBtn'), !(body.can_stop || (state.stopPending && active)));
    // 恢复入口：can_resume 时可见且**可点击**（仅在适用时隐藏，绝不留下无意义的
    // 禁用按钮）；请求 pending 期间保持不可重复点击，直到服务端收敛或明确失败。
    if (state.resumePending && body.can_resume) {
      setHidden($('hpoResumeBtn'), false);
      disable($('hpoResumeBtn'), true, '正在恢复…');
    } else {
      state.resumePending = false;
      setHidden($('hpoResumeBtn'), !body.can_resume);
      disable($('hpoResumeBtn'), !body.can_resume, '检查并恢复');
    }
    // “启动此任务”：仅当已创建(READY)、无活动控制器时开放显式启动
    var canExplicitStart = !active && status === 'READY';
    setHidden($('hpoStartBtn'), !canExplicitStart);
    disable($('hpoStartBtn'), !canExplicitStart);

    renderRanking(body.ranking || [], body.evaluation_mode);
    setHidden($('hpoProgress'), false);
    setHidden($('hpoProgressCard'), false);
    setHidden($('hpoHistorySection'), !state.historyOpen);

    renderProgress(body, counts);

    // 有成功结果时展示排名第一的六参数；无成功结果时展示明确的空状态，
    // 但绝不呈现任何模型生成的理由或修改动作。
    renderBest(body.best, body);
  }

  // ── 进度：只投影服务端权威事实，绝不靠前端定时器伪造 ────────────────
  function renderProgress(body, counts) {
    var status = body.execution_status || 'READY';
    var budget = (body.budget == null) ? null : body.budget;
    var terminal = (body.terminal_count == null) ? null : body.terminal_count;
    setText('hpoProgressStatus', STATUS_TEXT[status] || status);
    setText('hpoProgressCounts', counts);

    var percent = null;
    if (budget && terminal != null) {
      percent = Math.max(0, Math.min(100, Math.round(terminal * 100 / budget)));
    }
    setText('hpoProgressPercent', percent == null ? '—' : percent + '%');
    var bar = $('hpoProgressBar');
    if (bar) bar.style.width = (percent == null ? 0 : percent) + '%';

    // 当前执行第几项：没有权威当前项时诚实显示“—”，绝不猜测
    setText('hpoProgressCurrent', '当前执行：' +
      (body.current_trial_number == null
        ? '—'
        : '第 ' + body.current_trial_number + ' 项'));

    setText('hpoProgressTally',
      '成功 ' + fmtCount(body.success_count) +
      '　失败 ' + fmtCount(body.failed_count) +
      '　已取消/中断 ' + (body.cancelled_count == null && body.interrupted_count == null
        ? '—'
        : String((body.cancelled_count || 0) + (body.interrupted_count || 0))) +
      '　运行中 ' + fmtCount(body.running_count) +
      '　剩余 ' + fmtCount(body.remaining_count));

    var best = body.best;
    setText('hpoProgressBest', best
      ? ('当前最佳：' + evaluationLabel(body.evaluation_mode) + ' 综合分数 ' +
         fmt(best.value) + '　试验 ' + best.trial_display_number)
      : '当前最佳：—（还没有成功的试验）');

    // 终态文案：完成/暂停必须有明确、可操作的说明
    if (status === 'COMPLETED') {
      setText('hpoProgressMessage', '调优已完成');
    } else if (status === 'PAUSED') {
      setText('hpoProgressMessage', '已暂停，可恢复');
    } else if (status === 'INTERRUPTED') {
      setText('hpoProgressMessage', '已中断，可恢复');
    } else if (status === 'BLOCKED') {
      setText('hpoProgressMessage', '需先处理才能继续');
    } else {
      setText('hpoProgressMessage', '');
    }
  }

  function samplerLabel(value) {
    if (value === 'tpe') return 'TPE';
    if (value === 'random') return '随机搜索';
    return value == null ? '—' : String(value);
  }

  function addRow(tbody, cells) {
    var tr = document.createElement('tr');
    for (var i = 0; i < cells.length; i++) {
      var td = document.createElement('td');
      td.textContent = cells[i] == null ? '—' : String(cells[i]);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
    return tr;
  }

  function renderFrozenSummary(body) {
    var tbody = $('hpoFrozenSummaryBody');
    if (!tbody) return;
    clearChildren(tbody);
    var space = body.search_space || {};
    var fixed = space.fixed || {};
    addRow(tbody, ['创建时间', body.created_at || '—']);
    addRow(tbody, ['算法', samplerLabel(body.sampler)]);
    addRow(tbody, ['评价方式', evaluationLabel(body.evaluation_mode)]);
    addRow(tbody, ['试验次数（包含失败/取消槽）', body.budget]);
    addRow(tbody, ['每次训练轮数', body.study_epochs]);
    addRow(tbody, ['Batch', body.batch]);
    addRow(tbody, ['图像尺寸', body.imgsz]);
    addRow(tbody, ['计算设备', body.device]);
    addRow(tbody, ['随机种子', body.seed]);
    // 内部身份只给可读短编号；完整 ID 不进入主界面
    addRow(tbody, ['数据快照', body.snapshot_short_id || '—']);
    addRow(tbody, ['初始权重', body.model_display]);
    addRow(tbody, ['单试验超时（秒）', body.timeout_seconds]);
    if (fixed.imgsz != null && body.imgsz == null) {
      addRow(tbody, ['图像尺寸', fixed.imgsz]);
    }
  }

  function rangeText(param) {
    if (!param) return '—';
    if (param.kind === 'choice') return (param.choices || []).join(' / ');
    if (param.ranges) {
      var parts = [];
      for (var name in param.ranges) {
        if (!Object.prototype.hasOwnProperty.call(param.ranges, name)) continue;
        var r = param.ranges[name];
        parts.push((name === 'SGD' ? 'SGD' : 'AdamW') + ' ' + r.low + '..' + r.high);
      }
      return parts.join('；');
    }
    var unit = param.kind === 'int' ? '' : (param.log ? '（对数）' : '');
    return param.low + '..' + param.high + unit;
  }

  // 评价方式与权重完全来自服务端；前端绝不自行重算或改写口径
  function scoreText(space) {
    var score = (space && space.score) || {};
    var weights = score.weights || {};
    var parts = [];
    var order = ['metrics/mAP50(B)', 'metrics/mAP50-95(B)',
                 'metrics/precision(B)', 'metrics/recall(B)'];
    for (var k = 0; k < order.length; k++) {
      var key = order[k];
      if (weights[key] == null) continue;
      parts.push((METRIC_LABELS[key] || key) + ' × ' + Number(weights[key]).toFixed(2));
    }
    return parts.join(' + ') || '—';
  }

  function renderSearchConfig(space) {
    var body = $('hpoSearchParamsBody');
    if (!body || !space) return;
    state.searchSpace = space;
    clearChildren(body);
    var params = space.parameters || {};
    for (var i = 0; i < SEARCH_KEYS.length; i++) {
      var key = SEARCH_KEYS[i];
      var param = params[key] || {};
      addRow(body, [key, rangeText(param),
        param.conditional ? (param.condition_label || '条件参数') : '独立采样']);
    }
    setText('hpoObjectiveText', '评价方式：' + evaluationLabel(space.evaluation_mode) +
      '　综合分数 = ' + scoreText(space) +
      '（' + (space.direction === 'maximize' ? '越大越好' : '越小越好') + '）；' +
      '固定条件与搜索参数分开列出。');
    setText('hpoSearchBadge', samplerLabel(space.sampler) + ' · ' +
      (space.search_space_version || '—'));
    var fixed = space.fixed || {};
    setText('hpoFixedConditions', '每次训练轮数 ' + fixed.epochs + '；Batch ' + fixed.batch +
      '；图像尺寸 ' + fixed.imgsz + '；计算设备 ' + fixed.device +
      '；种子 ' + fixed.seed + '；快照 ' + fixed.snapshot + '；初始权重 ' + fixed.model);
    setHidden($('hpoSearchConfig'), false);
    setHidden($('hpoSearchError'), true);
    applyFormalNotes(space);
  }

  // 六项搜索参数由服务端从最佳试验重建并锁定（只读展示）。正式训练只允许调整
  // 轮数：batch/imgsz/device 与搜索阶段完全一致，由服务端重新读取执行事实并
  // 严格校验，前端绝不自行改写。主界面只保留简洁事实摘要（评价模式/综合分数/
  // 组成指标/最佳轮次/正式 epochs/batch/imgsz/device），不再堆技术解释。
  function applyFormalNotes(space) {
    updateFormalSummary();
  }

  // 正式训练主摘要：轮数来自输入框，其余三项来自当前研究的权威执行条件
  function updateFormalSummary() {
    var epochs = numberField('hpoFormalEpochs').value;
    var detail = state.detail || {};
    setText('hpoFormalConditions', 'Batch ' + fmtCount(detail.batch) +
      '　图像尺寸 ' + fmtCount(detail.imgsz) +
      '　计算设备 ' + fmtCount(detail.device));
    setText('hpoFormalSummary', '本次正式训练：epochs=' + fmtCount(epochs) +
      '　batch=' + fmtCount(detail.batch) + '　imgsz=' + fmtCount(detail.imgsz) +
      '　device=' + fmtCount(detail.device));
  }
  window.hpoUpdateFormalSummary = updateFormalSummary;

  function renderRanking(ranking, mode) {
    var el = $('hpoRanking');
    if (!el) return;
    clearChildren(el);
    if (!ranking.length) {
      el.textContent = '暂无可用排名（仅成功的试验参与排名）。';
      setText('hpoRankingNote', '');
      return;
    }
    var parts = [];
    for (var i = 0; i < ranking.length; i++) {
      var r = ranking[i];
      parts.push('#' + r.rank + ' 试验' + r.trial_number + ' 综合分数 ' +
        (r.value == null ? '—' : Number(r.value).toFixed(4)) +
        '（最佳指标所在轮次 ' + (r.epoch == null ? '—' : r.epoch) + '）');
    }
    el.textContent = '排名（' + evaluationLabel(mode) + '）：' + parts.join('　|　');
    var tie = ranking.length > 1 && ranking[0].value != null &&
      ranking[0].value === ranking[1].value;
    setText('hpoRankingNote', tie
      ? '存在同分结果：按试验编号较小者优先，不代表精度更高。'
      : '搜索运行中的阶段性最佳与用户停止后的结果为不同事实，详情中保留原状态码。');
  }

  // 组成指标：键只显示映射后的短名，值按 4 位小数；缺失显示“—”
  function metricsText(metrics) {
    if (!metrics) return '—';
    var order = ['metrics/mAP50(B)', 'metrics/mAP50-95(B)',
                 'metrics/precision(B)', 'metrics/recall(B)'];
    var parts = [];
    for (var i = 0; i < order.length; i++) {
      if (metrics[order[i]] == null) continue;
      parts.push((METRIC_LABELS[order[i]]) + '=' + fmt(metrics[order[i]]));
    }
    return parts.length ? parts.join('　') : '—';
  }

  // ── best result ─────────────────────────────────────────────────

  function renderBest(best, body) {
    state.best = best || null;
    if (!best) {
      setHidden($('hpoBestBody'), true);
      setHidden($('hpoBestEmpty'), false);
      disable($('hpoFormalBestBtn'), true);
      setHidden($('hpoResultActions'), true);
      disable($('hpoOpenResultFolder'), true);
      disable($('hpoDownloadBest'), true);
      setText('hpoBestArtifactHint', '');
      return;
    }
    setHidden($('hpoBestEmpty'), true);
    setHidden($('hpoBestBody'), false);
    var badge = $('hpoBestBadge');
    if (badge) {
      badge.textContent = '排名第一 · 试验 ' + best.trial_display_number;
      badge.classList.remove('hidden');
    }
    // 综合分数必须与其评价模式、组成指标一起展示，绝不标成单独的 mAP50-95
    setText('hpoBestSummary', '评价方式 ' +
      evaluationLabel(best.evaluation_mode || (body && body.evaluation_mode)) +
      '　综合分数 ' + (best.value == null ? '—' : Number(best.value).toFixed(4)) +
      '　组成指标：' + metricsText(best.metrics) +
      '　最佳指标所在轮次 ' + (best.epoch == null ? '—' : best.epoch) +
      '　来源试验 ' + best.trial_display_number);
    var tbody = $('hpoBestParamsBody');
    clearChildren(tbody);
    for (var i = 0; i < SEARCH_KEYS.length; i++) {
      var key = SEARCH_KEYS[i];
      addRow(tbody, [key, best.search ? best.search[key] : null, '试验 ' + best.trial_display_number]);
    }
    var artifacts = best.artifacts || {};
    var approved = !!(body && body.approved_for_formal_training);
    // 结果操作只在“研究已完成 + rank-1 成功 + 无活动搜索”时**出现**：
    // 条件不满足时整个区域隐藏，不留下可见但禁用的按钮与占位提示。
    setHidden($('hpoResultActions'), !approved);
    disable($('hpoOpenResultFolder'), !approved);
    disable($('hpoDownloadBest'), !(approved && artifacts.best_pt_available));
    // best.pt 缺失时沿用可见的安全 warning，绝不伪装可下载
    setText('hpoBestArtifactHint', (approved && !artifacts.best_pt_available)
      ? '当前无法读取排名第一试验的 best.pt，请确认该试验产物是否仍然存在。'
      : '');
    disable($('hpoFormalBestBtn'), !approved);
    disable($('hpoVerifyBtn'), false);
    setText('hpoVerifyHint', '同条件验证（次要入口）：与搜索阶段完全相同的条件，只用于复核。');
  }

  // 下载只提交研究/试验/白名单产物名身份，绝不提交路径；服务端从冻结输出根与
  // 权威 attempt 重建路径。搜索阶段的 last.pt 与非最佳 Trial 没有入口。
  window.hpoDownloadBestPt = function () {
    if (!state.studyId || !state.best) return;
    if (state.best.artifacts && state.best.artifacts.best_pt_available === false) {
      return;   // 服务端已明确不可用：不发起一次注定失败的下载
    }
    var url = '/api/hpo/studies/' + encodeURIComponent(state.studyId) +
      '/trials/' + encodeURIComponent(state.best.trial_id) +
      '/artifacts/best.pt';
    window.location.href = url;
  };

  // 打开调优结果文件夹：客户端只提交当前研究身份，服务端重新解析当前 rank-1
  // 试验的权威运行目录并在本地打开；响应不含任何绝对路径。
  window.hpoOpenResultFolder = function () {
    if (!state.studyId) return;
    var studyId = state.studyId;
    var generation = state.selection;
    setOperationError(null);
    disable($('hpoOpenResultFolder'), true);
    jsonPost('/api/hpo/studies/' + encodeURIComponent(studyId) + '/best/open-folder', {})
      .then(function (res) {
        if (!isCurrentSelection(generation, studyId)) return;
        disable($('hpoOpenResultFolder'), false);
        if (res.ok) {
          setText('hpoBestArtifactHint', '已在该研究排名第一试验的结果目录中打开文件管理器。');
        } else {
          setOperationError('打开结果文件夹失败 ' + errText(res.body));
        }
      })
      .catch(function () {
        if (!isCurrentSelection(generation, studyId)) return;
        disable($('hpoOpenResultFolder'), false);
        setOperationError('打开结果文件夹请求结果未知，请稍后重试或从调优结果目录手动查看。');
      });
  };

  window.hpoDownloadFormalArtifact = function (trainName, name) {
    if (!state.studyId || !trainName) return;
    var url = '/api/hpo/studies/' + encodeURIComponent(state.studyId) +
      '/formal-runs/' + encodeURIComponent(trainName) +
      '/artifacts/' + encodeURIComponent(name);
    window.location.href = url;
  };

  // 正式训练“打开结果文件夹”：只提交当前页面的完整 studyId 与当前行的 train_name，
  // 目标目录一律由服务端从受控 detect 根重建（客户端不发任何路径）。每条记录的状态
  // 互相独立，迟到/切换研究后的响应不得改写当前研究的提示。
  function formalFolderEntry(trainName) {
    if (!state.formalFolders) state.formalFolders = {};
    var key = String(trainName == null ? '' : trainName);
    if (!state.formalFolders[key]) {
      state.formalFolders[key] = { pending: false, hint: '' };
    }
    return state.formalFolders[key];
  }

  // 只有已完成、且来源事实经服务端受控扫描合法的正式训练才给可点的目录入口；
  // best.pt 是否存在不影响能否打开结果目录。
  function canOpenFormalFolder(run) {
    return !!run && run.status === 'completed' && run.source_readable !== false;
  }

  window.hpoOpenFormalFolder = function (trainName, run, button, hintEl) {
    if (!state.studyId || !trainName || !canOpenFormalFolder(run)) return;
    var studyId = state.studyId;
    var generation = state.selection;
    var entry = formalFolderEntry(trainName);
    if (entry.pending) return;                 // pending 期间不可重复提交
    entry.pending = true;
    entry.hint = '';
    if (button) button.disabled = true;
    if (hintEl) hintEl.textContent = '';
    setOperationError(null);
    var url = '/api/hpo/studies/' + encodeURIComponent(studyId) +
      '/formal-runs/' + encodeURIComponent(trainName) + '/open-folder';
    jsonPost(url, {}).then(function (res) {
      if (!isCurrentSelection(generation, studyId)) return;
      entry.pending = false;
      if (button) button.disabled = false;
      if (res.ok) {
        entry.hint = '已打开训练编号 ' + trainName + ' 的结果文件夹。';
      } else {
        entry.hint = '';
        setOperationError('打开正式训练结果文件夹失败 ' + errText(res.body));
      }
      if (hintEl) hintEl.textContent = entry.hint;
    }).catch(function () {
      if (!isCurrentSelection(generation, studyId)) return;
      entry.pending = false;
      if (button) button.disabled = false;
      entry.hint = '';
      if (hintEl) hintEl.textContent = '';
      setOperationError('打开正式训练结果文件夹请求结果未知，请稍后重试或手动到 Detect 目录查看。');
    });
  };

  // ── formal training (best parameters) ───────────────────────────

  function validateFormal() {
    var epochs = numberField('hpoFormalEpochs');
    return rangeError('训练轮数', epochs, 1, 1000);
  }

  window.hpoStartFormalTraining = function () {
    if (!state.studyId || !state.best) return;
    showFieldError(null, 'hpoFormalStatus');
    var fieldError = validateFormal();
    if (fieldError) {
      var badEl = $('hpoFormalStatus');
      if (badEl) badEl.textContent = '[' + fieldError.field + '] ' + fieldError.message;
      return;
    }
    // 提交必须携带选择代号：A 的响应绝不覆盖切换到 B 之后的提示/关联/按钮状态
    var studyId = state.studyId;
    var generation = state.selection;
    var trialId = state.best.trial_id;
    // batch/imgsz/device 一律取自当前研究的权威状态投影；服务端仍会逐项比对
    // execution 事实，任何失配都零启动拒绝。前端绝不提供这三项的编辑入口。
    var detail = state.detail || {};
    var body = {
      trial_id: trialId,
      training_config: {
        epochs: numberField('hpoFormalEpochs').value,
        batch: detail.batch,
        imgsz: detail.imgsz,
        device: detail.device
      }
    };
    disable($('hpoFormalBestBtn'), true, '正在提交…');
    jsonPost('/api/hpo/studies/' + encodeURIComponent(studyId) + '/train-best', body)
      .then(function (res) {
        if (!isCurrentSelection(generation, studyId)) return;  // 旧提交不覆盖新选择
        disable($('hpoFormalBestBtn'), false, '使用最佳参数正式训练');
        if (res.status === 202) {
          // 202 = 已接受但尚未完成：明确区分“接受/运行/完成”，不当作完成
          state.watchFormal = true;
          state.formalRunName = res.body.train_name;
          setText('hpoFormalStatus', '已接受（尚未完成）：训练编号 ' + res.body.train_name +
            '　本次条件 epoch=' + res.body.training_config.epochs +
            '、batch=' + res.body.training_config.batch +
            '、imgsz=' + res.body.training_config.imgsz +
            '、device=' + res.body.training_config.device +
            '；进度在下方正式训练监控中显示。');
          // 202 后立即绑定完整 runtime UUID 并订阅监控；JSON 历史 ID 绝不使用。
          // 只在身份真正变化时订阅：关联结果的权威投影若先到，绝不重复订阅。
          var acceptedRunId = res.body.run_id;
          if (isRuntimeRunId(acceptedRunId)) {
            if (state.monitorRunId !== acceptedRunId) {
              state.monitorRunId = acceptedRunId;
              if (typeof window.locateTrainingRun === 'function') {
                window.locateTrainingRun(acceptedRunId);
              }
            }
            // 服务端已接受（accepted，尚未完成）：立即按活跃事实收敛监控，
            // 绝不继续显示上一 run 的终态，也不显示“未选择”。
            applyFormalMonitorFacts({ status: res.body.status, result_available: false });
          }
          // 首次关联刷新必须走既有 refreshRound 生命周期：它负责串行化并与定时器
          // tick 合并，并在结束时按 watchFormal 启动唯一的 2 秒轮询——直接调用
          // refreshFormalRuns() 会绕过 finishRound，使已完成研究在 202 后再也不轮询，
          // 关联列表只能靠用户手动刷新才收敛。
          refreshRound(studyId, true);
          return;
        }
        var el = $('hpoFormalStatus');
        if (el) el.textContent = errText(res.body);
      }).catch(function () {
        if (!isCurrentSelection(generation, studyId)) return;
        disable($('hpoFormalBestBtn'), false, '使用最佳参数正式训练');
        var el = $('hpoFormalStatus');
        if (el) el.textContent = '正式训练请求结果未知，请在 HPO 关联结果与训练监控中确认，不要重复提交。';
      });
  };

  window.hpoStartVerification = function () {
    if (!state.studyId || !state.best) return;
    var body = { source_hpo: { study_id: state.studyId, trial_id: state.best.trial_id } };
    setText('hpoVerifyHint', '正在以相同条件提交一次普通训练验证…');
    fetch('/api/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }).then(function (resp) {
      if (resp.ok) {
        setText('hpoVerifyHint', '同条件验证训练已提交，请到训练监控查看（该入口保持原固定配置验证契约）。');
      } else {
        setText('hpoVerifyHint', '验证启动失败，请在训练监控或历史中确认；系统未修改最佳结果。');
      }
    }).catch(function () {
      setText('hpoVerifyHint', '验证请求结果未知，请在历史/训练监控中确认是否已启动。');
    });
  };

  // ── linked formal runs ──────────────────────────────────────────

  function refreshFormalRuns(generation) {
    if (!state.studyId) return Promise.resolve();
    var id = state.studyId;
    return api('/api/hpo/studies/' + encodeURIComponent(id) + '/formal-runs', {})
      .then(function (res) {
        if (!isCurrentSelection(generation, id)) return;
        if (!res.ok) {
          setFormalRunsMessage('正式训练关联结果暂不可用。');
          return;
        }
        renderFormalRuns(res.body.runs || [], res.body.warnings || []);
      }).catch(function () {
        if (!isCurrentSelection(generation, id)) return;
        setFormalRunsMessage('正式训练关联结果暂不可用。');
      });
  }
  window.hpoRefreshFormalRuns = function () {
    return refreshFormalRuns(state.selection);
  };

  function setFormalRunsMessage(text) {
    var box = $('hpoFormalRunsList');
    if (!box) return;
    clearChildren(box);
    box.textContent = text;
  }

  var ACTIVE_RUN_STATES = { starting: true, running: true, stopping: true };

  // 详情身份不可用时的稳定原因文案（对应服务端稳定错误码），让用户知道为什么
  // 按钮不可用，而不是点开一个必然 404 的入口。
  function experimentIdentityReason(code) {
    if (code === 'EXPERIMENT_NOT_INDEXED') {
      return '未在实验索引中找到对应记录（运行未收尾或索引缺失）。';
    }
    if (code === 'EXPERIMENT_AMBIGUOUS') {
      return '实验索引中存在多条同名记录，无法确定唯一结果。';
    }
    if (code === 'LOCAL_INDEX_UNAVAILABLE') {
      return '实验索引当前不可用，暂不能打开结果详情。';
    }
    return '暂无可用的实验详情身份。';
  }


  // 公共监控的所有权只来自当前研究的 formal-runs 权威投影：最新一条合法
  // runtime 身份（<kind>:<uuid4>）即当前正式训练。JSON 历史 ID 与短编号绝不
  // 作为监控身份，没有合法运行就保持空状态。
  //
  // 「身份生命周期」与「事实投影」是两件事，必须分开：
  //   * runtime_run_id 真正变化时才算换了一个被监控的运行——清空指标/日志并
  //     新建订阅（locateTrainingRun）；
  //   * 每次权威投影到达都要按该身份刷新状态徽章与可用结果，即使身份没变。
  // 因此“身份相同”只能表示“不需要重新订阅”，绝不表示“不需要收敛状态”。
  function syncFormalMonitor(runs) {
    var target = null;
    var targetRun = null;
    for (var i = 0; i < runs.length; i++) {
      if (isRuntimeRunId(runs[i].runtime_run_id)) {
        target = runs[i].runtime_run_id;
        targetRun = runs[i];
        break;
      }
    }
    var rebind = (target !== state.monitorRunId);
    state.monitorRunId = target;
    if (!target) return;
    if (rebind && typeof window.locateTrainingRun === 'function') {
      window.locateTrainingRun(target);
    }
    applyFormalMonitorFacts(targetRun);
  }

  // 该 formal run 的权威事实 → 公共监控的状态徽章/提示/停止按钮与结果卡片。
  // 指标与结果只来自服务端给出的该 run 事实（result_available 为真才投影），
  // 复用共用的 applyResultToMonitor，绝不复制解析逻辑、绝不补造指标。
  function applyFormalMonitorFacts(run) {
    if (!run) return;
    if (typeof window.renderRunState === 'function') {
      window.renderRunState({
        status: monitorRunStatus(run.status),
        running: isActiveRunStatus(run.status),
      });
    }
    if (run.result_available && window.TrainingMonitor) {
      window.TrainingMonitor.applyResultToMonitor(formalRunResult(run));
    }
  }

  // 关联结果里的 starting/stopping 是活跃阶段；正式训练提交的 202 响应
  // （status=accepted）同样表示服务端已接受这次正式训练，都按活跃呈现。
  function isActiveRunStatus(status) {
    return status === 'accepted' || !!ACTIVE_RUN_STATES[status];
  }

  function monitorRunStatus(status) {
    if (status === 'starting' || status === 'stopping' || status === 'accepted') {
      return 'running';
    }
    return status;
  }

  // 正式 run 的权威事实 → 共用监控屏的输入形状（键与 monitor.js 的既有契约一致）
  function formalRunResult(run) {
    var metrics = run.metrics || {};
    return {
      run_name: run.train_name,
      params: {},
      epochs: (run.epochs && typeof run.epochs === 'object') ? run.epochs : null,
      metrics: {
        mAP50: metrics.mAP50, mAP50_95: metrics.mAP50_95,
        precision: metrics.precision, recall: metrics.recall
      },
      artifacts: {}
    };
  }

  function renderFormalRuns(runs, warnings) {
    var box = $('hpoFormalRunsList');
    if (!box) return;
    clearChildren(box);
    renderFormalWarnings(warnings || []);
    if (!runs.length) {
      box.textContent = '暂无由该研究发起的正式训练。';
      state.watchFormal = false;
      syncFormalMonitor([]);
      return;
    }
    var anyActive = false;
    for (var i = 0; i < runs.length; i++) {
      var run = runs[i];
      var active = !!ACTIVE_RUN_STATES[run.status];
      if (active) anyActive = true;
      var line = document.createElement('div');
      line.style.marginBottom = '6px';
      line.style.paddingBottom = '6px';
      line.style.borderBottom = '1px solid var(--muted)';
      var statusText = STATUS_TEXT[run.status] || run.status;
      var metrics = run.metrics || {};
      var summary = '训练编号 ' + run.train_name + '　状态 ' + statusText +
        '（' + run.status + '）' +
        (run.source_trial_number == null ? '' : '　来源试验 ' + (run.source_trial_number + 1));

      if (i === 0) {
        // 最近一次正式训练突出显示（列表按 trainN 倒序）
        summary = '【最近一次】' + summary;
        line.style.borderLeft = '3px solid var(--primary)';
        line.style.paddingLeft = '6px';
      }
      // 指标只来自对应正式 run 的事实：只有容器的终态指标才叫“最终结果”；
      // 运行中的末行只能作为“阶段指标”，绝不冒充最终模型指标。
      var hasMetrics = (metrics.mAP50 != null || metrics.mAP50_95 != null);
      var metricsLine;
      if (run.result_available && hasMetrics) {
        metricsLine = '最终结果 mAP50=' + fmt(metrics.mAP50) +
          '　mAP50-95=' + fmt(metrics.mAP50_95);
      } else if (hasMetrics) {
        metricsLine = '阶段指标（尚未完成，非最终结果）mAP50=' + fmt(metrics.mAP50) +
          '　mAP50-95=' + fmt(metrics.mAP50_95);
      } else if (run.status === 'unknown') {
        metricsLine = '未找到该运行的终态事实，此处不伪造记录。';
      } else {
        metricsLine = '尚未完成，暂无最终结果。';
      }
      var head = document.createElement('div');
      head.textContent = summary + '　' + metricsLine;
      line.appendChild(head);

      // 主界面只给可读短编号与稳定原因：完整运行身份/JSON 历史 ID 等内部身份
      // 不在这里展示。入口能否使用由服务端身份解析决定，原因在这里说清。
      var identity = document.createElement('div');
      identity.className = 'text-muted';
      var identityParts = [];
      if (run.runtime_run_id) {
        identityParts.push('运行编号 ' + shortRuntimeId(run.runtime_run_id));
      } else if (run.runtime_identity_missing) {
        identityParts.push('运行编号缺失（旧记录不补造身份），监控入口不可用');
      }
      if (!run.experiment_run_id) {
        identityParts.push('实验详情：' + experimentIdentityReason(run.experiment_identity_reason));
      }
      identity.textContent = identityParts.join('　');
      if (identity.textContent) line.appendChild(identity);

      if (run.training_config) {
        var cfg = document.createElement('div');
        cfg.textContent = '正式条件：epochs=' + run.training_config.epochs +
          ' batch=' + run.training_config.batch +
          ' imgsz=' + run.training_config.imgsz +
          ' device=' + run.training_config.device;
        line.appendChild(cfg);
      }
      var actions = document.createElement('div');
      actions.style.marginTop = '2px';
      var monitor = document.createElement('button');
      monitor.type = 'button';
      monitor.className = 'btn btn-sm';
      monitor.textContent = active ? '查看监控' : '监控已结束';
      monitor.disabled = !active || !run.runtime_run_id;
      monitor.addEventListener('click', (function (runtimeId) {
        return function () {
          // 定位到这条 run 的事件流，而不是跳通用页面让用户再找
          if (typeof window.locateTrainingRun === 'function') {
            window.locateTrainingRun(runtimeId);
          }
        };
      })(run.runtime_run_id));
      actions.appendChild(monitor);

      // “查看结果”只认权威实验详情身份：传入 JSON 历史 ID 会直接 404。
      var results = document.createElement('button');
      results.type = 'button';
      results.className = 'btn btn-sm';
      results.style.marginLeft = '6px';
      results.textContent = '查看结果';
      results.disabled = !run.experiment_run_id;
      if (!run.experiment_run_id) results.title = experimentIdentityReason(run.experiment_identity_reason);
      results.addEventListener('click', (function (detailId) {
        return function () {
          if (detailId && typeof window.showExperimentDetail === 'function') {
            window.showExperimentDetail(detailId);
          } else {
            setFormalRunsMessage('该运行暂时没有可用的实验详情身份，未打开结果页。');
          }
        };
      })(run.experiment_run_id));
      actions.appendChild(results);

      // “打开结果文件夹”只打开**这条正式训练**的受控 detect/trainN 目录；服务端
      // 会重新解析并校验来源事实，因此这里只提交身份，且每条记录独立提交。
      var openable = canOpenFormalFolder(run);
      var folderState = formalFolderEntry(run.train_name);
      var folder = document.createElement('button');
      folder.type = 'button';
      folder.className = 'btn btn-sm';
      folder.style.marginLeft = '6px';
      folder.textContent = '打开结果文件夹';
      folder.disabled = !openable || folderState.pending;
      if (!openable) {
        folder.title = '只有已完成、且来源事实合法的正式训练才能打开结果目录。';
      }
      var folderHint = document.createElement('span');
      folderHint.className = 'text-muted';
      folderHint.style.marginLeft = '6px';
      folderHint.textContent = folderState.hint || '';
      folder.addEventListener('click', (function (name, facts, button, hintEl) {
        return function () {
          window.hpoOpenFormalFolder(name, facts, button, hintEl);
        };
      })(run.train_name, run, folder, folderHint));
      actions.appendChild(folder);
      actions.appendChild(folderHint);

      if (run.best_pt_available) {
        var weights = document.createElement('button');
        weights.type = 'button';
        weights.className = 'btn btn-sm';
        weights.style.marginLeft = '6px';
        weights.textContent = '下载最终 best.pt';
        weights.addEventListener('click', (function (name) {
          return function () { window.hpoDownloadFormalArtifact(name, 'best.pt'); };
        })(run.train_name));
        actions.appendChild(weights);
      } else {
        var pending = document.createElement('span');
        pending.style.marginLeft = '6px';
        pending.className = 'text-muted';
        // 文件存在也可能是搜索阶段的 best.pt：必须在服务端校验过来源后才声称可用
        pending.textContent = run.status === 'completed'
          ? '最终 best.pt 不可用' : '最终 best.pt 尚未生成';
        actions.appendChild(pending);
      }
      line.appendChild(actions);
      box.appendChild(line);
    }
    state.watchFormal = anyActive;
    syncFormalMonitor(runs);
  }

  function renderFormalWarnings(warnings) {
    var el = $('hpoFormalRunsWarning');
    if (!el) return;
    if (!warnings.length) { el.textContent = ''; el.classList.add('hidden'); return; }
    var parts = [];
    for (var i = 0; i < warnings.length; i++) {
      var code = warnings[i].code;
      var name = warnings[i].train_name;
      if (code === 'FORMAL_RUNS_TRUNCATED') {
        parts.push('正式训练目录超出扫描上限，可能还有更早的记录未列出。');
      } else if (code === 'FORMAL_SOURCE_UNREADABLE') {
        parts.push('存在无法读取的来源记录' + (name ? '（' + name + '）' : '') + '，未计入上方结果。');
      } else {
        parts.push('存在身份不完整的来源记录' + (name ? '（' + name + '）' : '') + '，未计入上方结果。');
      }
    }
    el.textContent = parts.join('　|　');
    el.classList.remove('hidden');
  }

  function fmt(value) {
    return value == null ? '—' : Number(value).toFixed(4);
  }

  // ── history (paged, current item highlighted, bad rows visible) ──

  function hpoRefreshHistory() {
    var box = $('hpoHistoryList');
    if (!box) return Promise.resolve();
    var offset = state.historyOffset;
    var limit = state.historyLimit;
    return api('/api/hpo/studies?offset=' + offset + '&limit=' + limit, {})
      .then(function (res) {
        if (!res.ok) { setText('hpoHistoryList', 'HPO 历史暂不可用。'); return; }
        renderHistory(res.body.studies || []);
        state.historyCount = res.body.count || 0;
        var from = state.historyCount ? offset + 1 : 0;
        setText('hpoStudyPageInfo', from + '-' + (offset + (res.body.studies || []).length) +
          ' / 共 ' + state.historyCount);
        // 历史面板默认收起：只有用户显式展开过才保持可见（拉取不改变可见性）
        setHidden($('hpoHistorySection'), !state.historyOpen);
      }).catch(function () {
        setText('hpoHistoryList', 'HPO 历史暂不可用。');
      });
  }
  window.hpoRefreshHistory = hpoRefreshHistory;

  window.hpoHistoryPage = function (delta) {
    var next = state.historyOffset + delta * state.historyLimit;
    if (next < 0) next = 0;
    if (next >= state.historyCount && delta > 0) return;
    state.historyOffset = next;
    hpoRefreshHistory();
  };

  function renderHistory(rows) {
    var box = $('hpoHistoryList');
    if (!box) return;
    clearChildren(box);
    if (!rows.length) {
      box.textContent = '暂无 HPO 研究记录。';
      return;
    }
    for (var i = 0; i < rows.length; i++) {
      var row = rows[i];
      var line = document.createElement('div');
      line.style.marginBottom = '4px';
      line.style.padding = '4px 6px';
      if (row.study_id === state.studyId) {
        line.style.border = '1px solid var(--primary)';
        line.style.borderRadius = '6px';
      }
      var summary = document.createElement('span');
      if (!row.readable) {
        if (row.transient) {
          summary.textContent = shortId(row.study_id) + ' — 暂忙（正在读写该研究），稍后刷新会自动恢复。';
        } else {
          summary.textContent = shortId(row.study_id) + ' — 记录不可读取 [' +
            (row.execution_error_code || 'HPO_CORRUPT_STUDY') + ']';
        }
        line.appendChild(summary);
      } else {
        // 研究级历史：时间、数据集名称、评价模式、完成数/预算、状态与“查看”
        var statusText = STATUS_TEXT[row.execution_status] ||
          (row.transient ? '暂忙' : (row.execution_status || 'READY'));
        // 数据集名称只来自权威事实；解析不出时诚实说明缺失，绝不用
        // study/snapshot 短身份冒充数据集名称。
        summary.textContent = (row.created_at ? String(row.created_at).slice(0, 19) : '—') +
          '　' + (row.dataset_name || '数据集不可用') +
          '　' + evaluationLabel(row.evaluation_mode) +
          '　' + statusText +
          '　完成 ' + (row.terminal_count || 0) + '/' + (row.budget == null ? '—' : row.budget);
        line.appendChild(summary);
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-sm';
        btn.style.marginLeft = '6px';
        btn.textContent = row.study_id === state.studyId ? '当前' : '查看';
        btn.addEventListener('click', (function (sid) {
          return function () { selectStudy(sid); };
        })(row.study_id));
        line.appendChild(btn);
      }
      box.appendChild(line);
    }
  }
})();
