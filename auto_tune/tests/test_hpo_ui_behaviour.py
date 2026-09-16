"""H1.3 second rework: behaviour tests that actually execute ``hpo.js``.

The front-end is driven by ``auto_tune/tests/js/minidom.js`` under Node: a
dependency-free minimal DOM (built from the real ``single_page.html``, so the
ids/classes/defaults are the shipping ones) plus a request queue that lets a
test decide the *order* in which replies arrive. The assertions below are about
what the script really wrote into the DOM and state — not about whether a
string appears in the source.

These tests need ``node`` on PATH; the whole module is skipped otherwise.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_UI_DIR = Path(__file__).resolve().parent.parent / "ui"
_HARNESS = Path(__file__).resolve().parent / "js" / "minidom.js"

_NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(_NODE is None, reason="node is not available")

_CACHE: dict[str, dict] = {}


def _run(scenario: str) -> dict:
    """Execute one scenario in Node and return its JSON facts (cached)."""
    if scenario in _CACHE:
        return _CACHE[scenario]
    result = subprocess.run([_NODE, str(_HARNESS), scenario, str(_UI_DIR)],
                            capture_output=True, text=True, encoding="utf-8",
                            timeout=180)
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
    payload = json.loads(result.stdout)
    assert "error" not in payload, payload.get("error")
    _CACHE[scenario] = payload
    return payload


# ── A1/A2/A3: 布局稳定、单一开始操作、技术细节折叠 ─────────────────


def test_layout_keeps_mode_and_primary_action_in_one_common_block():
    facts = _run("layout_and_defaults")
    assert facts["order"]["modeIsInsideCommon"] is True
    # HPO 草稿/结果区排在共同控制区之后，切换模式不会把选项栏往下推
    assert facts["order"]["hpoDraftAfterCommon"] is True
    # 正式结果区在整页中只有一处（不再有重复 id/重复空区）
    assert facts["order"]["formalRunsListCount"] == 1


def test_exactly_one_start_action_is_visible_in_every_mode():
    facts = _run("layout_and_defaults")
    assert facts["hpoMode"] == {"hpoBtnVisible": True, "commonBtnVisible": False}
    assert facts["dryMode"] == {"hpoBtnVisible": False, "commonBtnVisible": True}
    assert facts["fullMode"] == {"hpoBtnVisible": False, "commonBtnVisible": True}


def test_technical_details_are_collapsed_while_bindings_are_direct():
    details = _run("layout_and_defaults")["details"]
    # 纯技术细节仍然默认折叠
    assert details["searchDetailsOpen"] is False
    assert details["draftDetailsOpen"] is False
    assert details["bestDetailsOpen"] is False
    # 第四轮：绑定区不再折叠，也没有可编辑路径/完整 Trial 表
    assert details["inputDetailsCount"] == 0
    assert details["modelPathCount"] == 0
    assert details["trialsBodyCount"] == 0
    assert details["snapshotSelectEnabled"] is True
    assert details["modelSelectEnabled"] is True


def test_formal_training_only_lets_the_user_change_epochs():
    facts = _run("layout_and_defaults")
    # 当前普通训练配置的正式默认轮数，绝不是研究的 1 轮
    assert facts["formalEpochs"] == "100"
    # batch/imgsz/device 不是输入框，只按当前研究的权威执行条件展示
    assert facts["formalInputs"] == ["hpoFormalEpochs"]


def test_draft_summary_shows_real_data_count_and_budget():
    facts = _run("layout_and_defaults")
    assert facts["draftDefaults"]["budget"] == "10"
    assert facts["draftDefaults"]["epochs"] == "30"
    assert facts["draftDefaults"]["seed"] == "42"
    # 绑定只来自受控列表；主界面只给名称，不给任何物理路径
    assert facts["draftDefaults"]["modelValue"] == "E:/project/yolov8n.pt"
    assert facts["modelSummary"] == "模型名称：yolov8n.pt"
    assert facts["datasetSummary"] == "数据集 demo　总图片 230　训练集 184　验证集 46"
    summary = facts["mainSummary"]
    assert "demo" in summary          # 原数据集名称作为主要识别手段
    assert "230" in summary           # train+val，绝不出现 329
    assert "329" not in summary
    assert "设备 0" in summary        # 服务端给出的设备默认
    assert "试验次数 10" in summary
    assert "E:/project" not in summary


def test_direct_inputs_carry_the_server_evaluation_mode_and_device():
    facts = _run("layout_and_defaults")["directInputs"]
    assert facts == {"evaluationMode": "comprehensive", "device": "0"}


# ── 最终返修 Task 5：设备默认必须等权威事实，且用明确选择控件 ─────────


def test_device_is_a_selector_without_a_staged_cpu_default():
    facts = _run("hpo_device_defaults")
    pending = facts["pending"]
    # 明确选择控件，不是自由文本；defaults 到达前没有可提交的暂存默认值
    assert pending["tag"] == "SELECT"
    assert pending["value"] == ""
    assert pending["options"] == [""]
    # 事实未就绪：创建按钮 disabled；立即点击也不会发出任何创建请求
    assert pending["createDisabled"] is True
    assert facts["afterEarlyClick"]["createCalls"] == 0
    assert facts["afterEarlyClick"]["createDisabled"] is True


def test_cuda_available_defaults_to_gpu_zero_and_enables_creation():
    facts = _run("hpo_device_defaults")
    gpu = facts["gpu"]
    # GPU 可用且配置未指定 device：权威默认必须是 GPU 0，绝不静默回退 CPU
    assert gpu["options"] == ["0", "cpu"]
    assert gpu["value"] == "0"
    assert gpu["createDisabled"] is False
    # 用户主动改选 CPU 后，后续无关异步响应不覆盖该选择
    assert facts["afterUserChoice"]["value"] == "cpu"
    # 提交值仍是现有合法值
    assert facts["createDevice"] == "cpu"


def test_creation_sends_gpu_zero_when_the_server_default_is_gpu_zero():
    facts = _run("hpo_device_defaults")
    assert facts["gpu"]["value"] == "0"
    assert facts["createDevice"] in ("0", "cpu")     # 用户最后选择的是 cpu


def test_without_a_gpu_cpu_is_selected_with_a_visible_notice():
    facts = _run("hpo_device_fallback")
    cpu = facts["cpu"]
    assert cpu["options"] == ["cpu"]
    assert cpu["value"] == "cpu"
    assert cpu["createDisabled"] is False
    # 简短、可操作的提示必须可见（不能藏在折叠区里）
    assert cpu["noticeVisible"] is True
    assert "未检测到可用 GPU" in cpu["notice"]
    assert facts["cpuDevice"] == "cpu"


def test_defaults_failure_shows_a_visible_error_and_creates_nothing():
    facts = _run("hpo_device_defaults_failure")
    assert facts["failed"]["value"] == ""            # 绝不悄悄启用 CPU 创建
    assert facts["failed"]["createDisabled"] is True
    assert facts["failed"]["bindingNoticeVisible"] is True
    assert "默认" in facts["failed"]["bindingNotice"]
    assert facts["createCalls"] == 0


# ── C1: 选择代号/合并刷新：旧请求未结束时切换 A→B 必须显示 B ─────────


def test_switching_studies_coalesces_the_inflight_refresh():
    facts = _run("switch_coalesce")
    assert facts["studyId"].endswith("b" * 32)
    assert facts["bStatusWasAsked"] is True
    assert facts["stuckReading"] is False
    # 旧的 A 回复被丢弃：不允许把 A 的 COMPLETED 渲染到 B 上
    assert "COMPLETED" not in facts["statusTextAfterStaleA"]
    assert "RUNNING" in facts["statusTextAfterB"]


def test_switch_while_a_round_is_busy_still_renders_the_new_study():
    facts = _run("switch_while_round_busy")
    assert facts["askedB"] is True
    assert facts["stuckReading"] is False
    assert "COMPLETED" in facts["statusText"]


# ── C2: 正式训练提交也必须带选择代号 ──────────────────────────────


def test_stale_formal_submission_cannot_override_the_new_selection():
    facts = _run("formal_response_guard")
    assert facts["btnEnabledForA"] is True      # A 已完成且有成功结果 → 按钮可用
    assert facts["trainBestQueued"] is True
    assert facts["statusBeforeLateReply"] == ""
    assert facts["buttonBeforeLateReply"] is True
    # A 的 202 迟到：不得覆盖 B 的提示、按钮状态或关联
    assert facts["formalStatusAfterLateReply"] == ""
    assert facts["buttonAfterLateReply"] is True
    assert facts["bestAfterSwitch"] is None
    assert facts["watchFormalAfterLateReply"] is False
    # 已接受的 A 运行仍然真实保留，且没有被前端重复提交
    assert facts["trainBestRequests"] == 1


# ── C3: 轮询成功不得抹掉用户操作错误 ─────────────────────────────


def test_successful_poll_does_not_erase_operation_error():
    facts = _run("operation_error_survives_poll")
    assert "启动失败" in facts["afterFailure"]
    assert facts["afterPoll"] == facts["afterFailure"]
    # 用户发起新的明确操作后才清掉该操作错误
    assert facts["afterNewOperation"] == ""
    assert facts["operationError"] is None
    assert facts["serverDetailHidden"] is True


# ── B3/B5/B6: 关联正式训练结果 ──────────────────────────────────


def test_first_selection_reads_linked_formal_runs():
    facts = _run("formal_runs_first_read")
    # watchFormal 之前是 false，但首次选择仍然查询了关联记录
    assert facts["watchFormalBefore"] is False
    assert facts["askedFormalRuns"] is True


def test_linked_runs_show_short_ids_and_never_full_internal_identities():
    """主界面只给可读短编号；完整的 runtime/JSON 历史身份不得出现。"""
    rows = _run("formal_runs_first_read")["rows"]
    running = rows[0]["text"]
    completed = rows[1]["text"]
    legacy = rows[2]["text"]
    recovered = rows[3]["text"]
    assert "运行编号 manual:uuid-2…" in running
    assert "运行编号 manual:uuid-3…" in completed
    # 旧 metadata 缺 runtime 字段：明确缺失，绝不补造 UUID
    assert "运行编号缺失" in legacy
    assert "运行编号缺失" in recovered
    # 完整 UUID 与 JSON 历史 ID（manual:trainN）都不进主界面
    for text in (running, completed, legacy, recovered):
        assert "manual:train" not in text
    assert "uuid-2 " not in running and "manual:uuid-2　" not in running


def test_linked_runs_separate_states_results_and_actions():
    rows = _run("formal_runs_first_read")["rows"]
    running, completed, legacy = rows[0], rows[1], rows[2]
    # 最近一次正式训练突出显示，重载后仍然可见
    assert "【最近一次】" in running["text"]
    assert "【最近一次】" not in completed["text"]
    # 运行中：末行只能作为“阶段指标”，绝不称最终结果；监控入口可用
    assert "阶段指标" in running["text"]
    assert "非最终结果" in running["text"]
    assert "最终结果 mAP50" not in running["text"]
    assert "mAP50=0.1200" in running["text"]
    assert {b["label"]: b["disabled"] for b in running["buttons"]} == {
        "查看监控": False, "查看结果": False, "打开结果文件夹": True}
    # 已完成：最终指标来自该 run；监控已结束而不是假跳转；结果入口可定位
    assert "最终结果 mAP50=0.7100" in completed["text"]
    assert "mAP50-95=0.4400" in completed["text"]
    assert {b["label"]: b["disabled"] for b in completed["buttons"]} == {
        "监控已结束": True, "查看结果": False, "打开结果文件夹": False,
        "下载最终 best.pt": False}
    # 未知终态：诚实呈现，不伪造记录；没有权威详情身份就不给结果入口并说明原因
    assert "未找到该运行的终态事实" in legacy["text"]
    assert "实验详情：未在实验索引中找到对应记录" in legacy["text"]
    assert {b["label"]: b["disabled"] for b in legacy["buttons"]} == {
        "监控已结束": True, "查看结果": True, "打开结果文件夹": True}


# ── 修复 1：正式训练“查看结果”必须用权威实验详情身份 ─────────────────


def test_result_entry_passes_the_authoritative_experiment_identity():
    facts = _run("formal_runs_result_identity")
    by_train = {row["trainName"]: row for row in facts["rows"]}
    # 正常记录：传给详情接口的是实验索引主键（manual:<uuid>）
    assert by_train["train2"]["clicked"] == "manual:uuid-2"
    assert by_train["train3"]["clicked"] == "manual:uuid-3"
    # 旧 metadata 缺 runtime、但索引已有身份：解析出来并用它，绝不是 JSON 历史 ID
    assert by_train["train10"]["clicked"] == (
        "manual:11111111-2222-3333-4444-555555555555")
    # 任何情况下都不会把 manual:trainN 当作详情身份传出去
    assert facts["historyIdsNeverOpened"] == 0


def test_result_entry_is_disabled_with_a_reason_when_identity_is_unavailable():
    facts = _run("formal_runs_result_identity")
    by_train = {row["trainName"]: row for row in facts["rows"]}
    # 索引缺失：按钮禁用并说明原因
    assert by_train["train9"]["disabled"] is True
    assert by_train["train9"]["clicked"] is None
    assert "未在实验索引中找到对应记录" in by_train["train9"]["title"]
    # 匹配歧义：明确不可用，不随意取一条
    assert by_train["train11"]["disabled"] is True
    assert by_train["train11"]["clicked"] is None
    assert "无法确定唯一结果" in by_train["train11"]["title"]


# ── 修复 1b：监控用 runtime 身份，结果用实验索引身份，二者不互相顶替 ─────


def test_monitor_uses_runtime_identity_and_results_use_experiment_identity():
    facts = _run("formal_runs_identity_guard")
    by_train = {row["trainName"]: row for row in facts["rows"]}
    # 查看监控只认 runtime 身份：结果身份失配的那条仍然可以看监控
    assert by_train["train2"]["monitored"] == "manual:uuid-2"
    assert by_train["train12"]["monitorDisabled"] is False
    assert by_train["train12"]["monitored"] == (
        "manual:11111111-1111-4111-8111-111111111111")
    # 同一条的“查看结果”被禁用并说明原因，绝不改绑到另一条同名实验
    assert by_train["train12"]["resultDisabled"] is True
    assert by_train["train12"]["clicked"] is None
    assert "未在实验索引中找到对应记录" in by_train["train12"]["resultTitle"]
    # 查看结果只认实验索引身份（含旧 metadata 由索引解析出的身份）
    assert by_train["train3"]["clicked"] == "manual:uuid-3"
    assert by_train["train10"]["clicked"] == (
        "manual:11111111-2222-3333-4444-555555555555")
    # JSON 历史 ID 绝不被当作详情身份或运行身份
    assert facts["historyIdsOpenedAsDetail"] == 0
    assert facts["historyIdsOpenedAsMonitor"] == 0
    assert facts["monitored"] == ["manual:uuid-2",
                                  "manual:11111111-1111-4111-8111-111111111111"]
    # 进入研究时只按权威投影里**严格合法**的 runtime 身份自动订阅监控：
    # 形如 manual:uuid-2 的非法身份绝不使用，JSON 历史 ID 更不会被使用
    assert facts["autoSubscribed"] == [
        "manual:11111111-1111-4111-8111-111111111111"]


def test_illegal_runtime_identity_disables_both_entries_without_calling_either():
    """终态记录只有非法 runtime：监控与结果都禁用，且两个入口都不会被调用。"""
    facts = _run("formal_runs_identity_guard")
    by_train = {row["trainName"]: row for row in facts["rows"]}
    terminal = by_train["train13"]
    # 没有可用的运行身份 → “查看监控”禁用；没有详情身份 → “查看结果”禁用。
    # 结果目录只由受控 detect/trainN 的正式来源事实决定，与索引身份无关，因此可开。
    assert {b["label"]: b["disabled"] for b in terminal["buttons"]} == {
        "监控已结束": True, "查看结果": True, "打开结果文件夹": False}
    assert terminal["monitored"] is None      # locateTrainingRun 从未被调用
    assert terminal["clicked"] is None        # showExperimentDetail 从未被调用
    # 身份行如实说明：运行身份缺失、历史 ID 只作展示、详情不可用及原因
    assert "运行编号缺失" in terminal["text"]
    assert "manual:train13" not in terminal["text"]
    assert "实验详情：未在实验索引中找到对应记录" in terminal["text"]
    # 任何入口都没有退回去使用 JSON 历史 ID
    assert facts["historyIdsOpenedAsDetail"] == 0
    assert facts["historyIdsOpenedAsMonitor"] == 0


# ── P1 返修：暂停后的恢复入口必须可点击、可收敛、不可重复提交 ─────────


def test_resume_button_is_visible_enabled_and_posts_the_full_study_id():
    facts = _run("resume_button_state")
    # can_resume=false → 隐藏（不留下可点的入口）；can_resume=true → 可见且可用
    assert facts["notResumable"]["visible"] is False
    assert facts["resumable"]["visible"] is True
    assert facts["resumable"]["disabled"] is False
    # 点击后只发一次 POST，且 URL 用完整 study ID 精确指向 /resume
    assert facts["firstClick"]["requests"] == 1
    assert facts["firstClick"]["url"] == (
        "/api/hpo/studies/" + facts["studyId"] + "/resume")
    # pending 期间按钮立即不可重复点击，并显示明确的进行中状态
    assert facts["firstClick"]["disabled"] is True
    assert "恢复" in facts["firstClick"]["label"]
    assert "恢复" in facts["firstClick"]["status"]
    # 重复点击不再产生第二个请求
    assert facts["afterSecondClick"] == 1
    # 收敛为 RUNNING 后恢复入口消失，停止入口出现（不伪造已恢复结果）
    assert facts["running"]["visible"] is False
    assert facts["running"]["stopVisible"] is True
    assert "RUNNING" in facts["running"]["statusText"]


def test_resume_failure_or_unknown_result_reenables_the_button():
    facts = _run("resume_failure_reenables")
    # 服务端拒绝：按钮重新可操作，操作错误可见
    assert facts["afterServerError"]["disabled"] is False
    assert facts["afterServerError"]["visible"] is True
    assert facts["afterServerError"]["errorHidden"] is False
    assert "恢复失败" in facts["afterServerError"]["errorText"]
    # 网络结果未知：同样重新可操作，提示结果未知，且不重复提交
    assert facts["afterNetworkError"]["disabled"] is False
    assert "结果未知" in facts["afterNetworkError"]["errorText"]
    assert facts["afterNetworkError"]["resumeRequests"] == 2
    # 错误不在折叠区内
    assert facts["errorInsideDetails"] is False


# ── 修复 3：少选择、后台默认 ────────────────────────────────────────


def test_reliable_binding_shows_selectors_directly_with_a_short_summary():
    facts = _run("binding_convergence")["reliable"]
    # 直接可见、可直接操作；技术细节保持折叠
    assert facts["snapshotSelectVisible"] is True
    assert facts["modelSelectVisible"] is True
    assert facts["noticeHidden"] is True
    assert facts["draftDetailsOpen"] is False
    assert "数据：demo（图片 230）" in facts["createConfirm"]
    # 主摘要只给模型名称；物理路径既不显示也不接受手输
    assert "初始权重：yolov8n.pt" in facts["createConfirm"]
    assert "E:/project/yolov8n.pt" not in facts["createConfirm"]


def test_weight_binding_comes_only_from_the_controlled_selector():
    facts = _run("binding_convergence")
    # 自动绑定后选择器必须就是那个值，不能显示成“请选择”
    assert facts["reliable"]["modelSelectValue"] == "E:/project/yolov8n.pt"
    # 选择即绑定：提交值随之改变，摘要只反映名称
    assert facts["afterSelect"]["selectValue"] == "E:/models/other.pt"
    assert facts["afterSelect"]["modelSummary"] == "模型名称：other.pt"
    assert "other.pt" in facts["afterSelect"]["confirm"]
    # 主界面没有可编辑的路径输入
    assert facts["modelPathExists"] == 0


def test_binding_survives_a_failed_snapshot_or_model_listing():
    facts = _run("binding_survives_a_failed_listing")
    # 列表读取失败不把已绑定的数据/权重退回“未选择”
    assert facts["snapshotValue"] == "d" * 64
    assert facts["modelSelectValue"] == "E:/project/yolov8n.pt"
    assert facts["modelSummary"] == "模型名称：yolov8n.pt"
    # 绑定仍然可靠 → 不显示错误提示
    assert facts["noticeHidden"] is True
    assert "暂不可用" in facts["snapshotHint"]


def test_missing_or_illegal_binding_explains_without_swapping_data():
    facts = _run("binding_missing_is_explained")
    # 选择器始终可见可操作；缺绑定时只在主区说明原因
    assert facts["snapshotSelectVisible"] is True
    assert facts["modelSelectVisible"] is True
    assert facts["noticeHidden"] is False
    assert "可靠快照绑定" in facts["noticeText"]
    assert "初始权重不合法" in facts["noticeText"]
    # 不自动换数据/权重：两个绑定都保持为空
    assert facts["snapshotValue"] == ""
    assert facts["modelValue"] == ""
    assert "未绑定快照" in facts["mainSummary"]


# ── 修复 2：公共控制区位置与控制可见性 ─────────────────────────────


def test_control_bar_precedes_both_variable_content_regions():
    facts = _run("control_bar_position")
    order = facts["order"]
    # 紧跟在共同控制区之后的模式选择与唯一主操作
    assert order["beforeModeSelect"] is True
    assert order["beforeHpoMainSummary"] is True
    # LLM 建议区与 HPO 结果/详情区都排在控制区之后：加载它们不会推动控制区
    assert order["beforeLlmSuggestion"] is True
    assert order["beforeSearchConfig"] is True
    assert order["beforeBestArea"] is True
    assert order["beforeDraft"] is True
    assert order["beforeProgressCard"] is True


def test_stop_resume_and_start_follow_the_applicable_state():
    facts = _run("control_bar_position")
    # 未选择任务：三个按钮都不出现
    assert facts["buttons"] == {"start": False, "stop": False, "resume": False}
    # READY 且无活动控制器：只出现“启动此任务”
    assert facts["afterReady"] == {"start": True, "stop": False, "resume": False}
    # 运行中：只出现停止
    assert facts["running"] == {"start": False, "stop": True, "resume": False}


def test_best_details_and_trial_artifacts_are_collapsed_by_default():
    facts = _run("formal_runs_first_read")
    assert facts["bestDetailsOpen"] is False
    # 关联结果与最终模型入口不在 best 折叠体里：没有成功试验也不会消失
    assert facts["formalRunsInsideBestBody"] is False


def test_errors_and_warnings_stay_visible_through_collapse_and_polling():
    facts = _run("errors_survive_collapse_and_poll")
    assert facts["warning"]["hidden"] is False
    assert "无法读取的来源记录" in facts["warning"]["text"]
    assert facts["warning"]["bestDetailsOpen"] is False
    assert facts["warning"]["formalRunsHidden"] is False
    # 用户操作错误不被折叠隐藏，也不被成功轮询清除
    assert "启动失败" in facts["afterFailure"]
    assert facts["detailErrorHidden"] is False
    assert facts["afterPoll"] == facts["afterFailure"]


def test_linked_runs_surface_safe_warnings_and_keep_polling():
    facts = _run("formal_runs_first_read")
    assert facts["warningHidden"] is False
    assert "无法读取的来源记录" in facts["warningText"]
    assert "超出扫描上限" in facts["warningText"]
    # 有活动正式训练时轮询继续（而不是停在“状态未知”）
    assert facts["watchFormalAfter"] is True
    assert facts["timersRunning"] >= 1


# ── 第四轮 Task 2：评价模式、直接选择输入与条件锁定 ─────────────────


def test_evaluation_mode_defaults_to_comprehensive_and_enters_the_payload():
    facts = _run("evaluation_mode_and_locked_formal")
    # 新建研究只允许全面/快速两种模式，legacy 只能被读取与展示
    assert facts["selectorValues"] == ["comprehensive", "quick"]
    assert facts["defaultMode"] == "comprehensive"     # UI 默认全面模式
    body = facts["quickBody"]
    assert body is not None, "a user-selected mode must reach the create request"
    assert body["study_config"]["evaluation_mode"] == "quick"
    assert body["study_config"]["evaluation_mode"] in ("comprehensive", "quick")
    # 六项搜索参数仍由服务端起采样决定，客户端只选模式
    assert "objective" not in body["study_config"]
    assert "search_space" not in body


def test_formal_training_submits_the_study_conditions_not_client_values():
    facts = _run("evaluation_mode_and_locked_formal")
    assert facts["formalBody"] == {
        "trial_id": facts["formalBody"]["trial_id"],
        "training_config": {"epochs": 7, "batch": 4, "imgsz": 96, "device": "0"},
    }
    # batch/imgsz/device 直接来自当前研究的权威执行条件投影
    assert "已接受" in facts["formalStatus"]
    assert "epoch=7" in facts["formalStatus"]


# ── 第四轮 Task 3：进度、历史入口与主界面收敛 ──────────────────────


def test_progress_view_projects_server_facts():
    running = _run("progress_panel")["running"]
    assert running["visible"] is True
    assert running["status"] == "运行中"
    assert running["counts"] == "4/10"          # terminal_count / budget
    assert running["percent"] == "40%"
    assert running["width"] == "40%"            # 进度条宽度由真实数量导出
    assert running["current"] == "当前执行：第 5 项"
    assert "成功 3" in running["tally"]
    assert "失败 1" in running["tally"]
    assert "剩余 6" in running["tally"]
    assert "综合分数 0.9000" in running["best"]
    assert "全面" in running["best"]


def test_progress_view_states_are_explicit_and_honest():
    facts = _run("progress_panel")
    assert facts["paused"]["message"] == "已暂停，可恢复"
    assert facts["completed"]["message"] == "调优已完成"
    assert facts["completed"]["percent"] == "100%"
    assert facts["blocked"]["current"] == "当前执行：—"   # 无权威当前项不猜测
    # 完整 Trial 表已从主界面移除（数据与 API 仍保留）
    assert facts["trialsBodyCount"] == 0
    assert facts["trialTableElements"] == 0


def test_history_entry_is_visible_collapsed_and_paged():
    facts = _run("history_toggle")
    assert facts["initial"]["toggleVisible"] is True
    assert facts["initial"]["sectionVisible"] is False       # 默认收起
    assert "历史" in facts["initial"]["label"]
    assert facts["opened"]["sectionVisible"] is True
    first, second, third, corrupt = facts["opened"]["rows"]
    # 每行至少显示时间、数据集名称、评价模式、完成数/预算、状态
    assert "2026-09-15T10:00:00" in first and "demo" in first
    assert "全面" in first and "已完成" in first and "完成 6/10" in first
    assert "查看" in first
    # 旧记录明确标注“旧版 mAP50-95”
    assert "旧版 mAP50-95" in second and "legacy-ds" in second
    # 无法解析数据集名称的旧记录：诚实缺失文案，绝不用短身份冒充数据集名称
    assert "数据集不可用" in third
    assert "hpo_cccc" not in third
    # 损坏记录：可见但只有稳定错误码与短编号，既不说“数据集不可用”也没有入口
    assert "记录不可读取" in corrupt and "HPO_CORRUPT_STUDY" in corrupt
    assert "数据集不可用" not in corrupt
    assert "查看" not in corrupt
    assert facts["selectableRows"] == [0, 1, 2]
    assert facts["lastRowHasButton"] == 0
    # “查看”用完整 study ID 选择该研究（页面只显示短编号）
    assert facts["afterSelect"]["studyId"].startswith("hpo_")
    assert facts["pageButtons"] == [1, 1]
    assert facts["collapsed"]["sectionVisible"] is False     # 可再次收起


def test_training_analysis_input_is_hidden_only_in_hpo_mode():
    facts = _run("training_analysis_card_visibility")
    assert facts["hpo"]["inputDisplay"] == "none"
    assert facts["other"]["inputDisplay"] == "" or \
        facts["other"]["inputDisplay"] is None or \
        facts["other"]["inputDisplay"] == "block"
    assert facts["backToHpo"]["inputDisplay"] == "none"


def test_forbidden_wording_is_absent_from_the_hpo_page():
    from pathlib import Path

    ui = Path(__file__).resolve().parent.parent / "ui"
    sources = [(ui / "templates" / "single_page.html").read_text(encoding="utf-8"),
               (ui / "static" / "hpo.js").read_text(encoding="utf-8")]
    for banned in (
        "数据集、权重、GPU、imgsz 与 batch 在正式训练中保持一致",
        "主界面不展示调优历史明细",
        "中间试验权重不展示；正式训练完成后提供最终 best.pt / last.pt。",
    ):
        for source in sources:
            assert banned not in source


def test_progress_reports_ties_without_claiming_an_improvement():
    facts = _run("progress_panel")["running"]
    assert "同分" in facts["rankingNote"]
    assert "不代表精度更高" in facts["rankingNote"]
    assert "综合分数 0.9000" in facts["ranking"]


def test_stale_study_response_cannot_pollute_the_progress_view():
    facts = _run("switch_coalesce")
    # A 的 100% 完成度迟到：不得污染 B 的进度条与百分比
    assert facts["percentAfterStaleA"] != "100%"
    assert facts["percentAfterB"] == "40%"


# ── 第四轮 Task 4：调优完成后的结果操作 ────────────────────────────


def test_result_actions_require_completion_and_a_rank1_best_pt():
    facts = _run("best_result_actions")
    # 未真正完成（RUNNING/PAUSED/FAILED 或无 rank-1 成功试验）→ 整个操作区不可见，
    # 不是“可见但禁用”，也没有任何“完成后可下载”占位提示
    for state in ("staged", "paused", "failed", "noBest"):
        assert facts[state]["region"] is False, state
        assert facts[state]["labels"], state      # 区块本身仍在 DOM 中（只隐藏）
    # 完成 + rank-1 best.pt 合法 → 操作区可见且两个操作可用
    assert facts["approved"]["region"] is True
    assert facts["approved"]["openFolder"] is False
    assert facts["approved"]["download"] is False
    # best.pt 缺失 → 仍可打开结果文件夹，但下载入口禁用且安全 warning 可见
    assert facts["noBestPt"]["region"] is True
    assert facts["noBestPt"]["openFolder"] is False
    assert facts["noBestPt"]["download"] is True
    assert facts["noBestPt"]["hintVisible"] is True
    assert "无法读取" in facts["noBestPt"]["hint"]
    # 操作区在整页中唯一
    assert facts["resultActionsCount"] == 1


_BANNED_HPO_WORDING = (
    "调优尚未完成",
    "完成后可下载",
    "best.pt 为搜索阶段排名第一试验的权重",
    "来源：排名第一的成功试验",
    "以上六项搜索参数由服务端",
    "原研究条件",
    "搜索完成只表示",
    "数据集、权重、GPU、imgsz 与 batch 在正式训练中保持一致",
    "主界面不展示调优历史明细",
    "中间试验权重不展示",
)


def test_no_verbose_technical_explanations_remain_in_the_best_area():
    facts = _run("best_result_actions")
    for state in ("approved", "noBestPt"):
        for banned in _BANNED_HPO_WORDING:
            assert banned not in facts[state]["text"], (state, banned)
    # 可保留的简洁事实摘要仍然在页面上
    text = facts["approved"]["text"]
    assert "评价方式" in text and "综合分数" in text
    assert "batch=" in text and "imgsz=" in text and "device=" in text


def test_best_download_uses_only_study_trial_and_artifact_identity():
    facts = _run("best_result_actions")
    url = facts["downloadUrl"]
    assert url is not None
    assert "/trials/" in url and url.endswith("/artifacts/best.pt")
    assert "/studies/" in url
    # 客户端永不提交路径
    assert ".." not in url and ":" not in url.split("?")[0]


def test_no_last_pt_or_non_best_weight_entry_is_offered():
    facts = _run("best_result_actions")
    for state in ("approved", "staged", "noBestPt"):
        labels = facts[state]["labels"]
        assert not any("last.pt" in label for label in labels), state
        # 只有最佳试验的 best.pt 一个权重入口
        assert sum(1 for label in labels if "best.pt" in label) == 1, state


# ── 最终返修 Task 1：HPO 模式只显示 HPO 内容 ───────────────────────

_NON_HPO_PROBES = ("trainingSummary", "llmAnalysis", "visionAnalysis",
                   "tuningHistory", "trainingAnalysisInput")


def test_hpo_mode_hides_every_non_hpo_region():
    facts = _run("hpo_mode_isolation")["hpo"]
    for key in _NON_HPO_PROBES:
        # 区块仍在 DOM 中（没有复制/删除），但在 HPO 模式下整体不可见
        assert facts[key]["found"] is True, key
        assert facts[key]["visible"] is False, key
    # 顶部“HPO 历史记录”按钮保留；面板默认收起
    assert facts["historyToggle"] is True
    assert facts["historySection"] is False


def test_hpo_history_panel_lists_research_rows_not_llm_iterations():
    facts = _run("hpo_mode_isolation")["historyOpen"]
    assert facts["historySection"] is True
    # 展开历史也不会把 LLM 逐项调优历史带回来
    assert facts["tuningHistory"]["visible"] is False
    assert len(facts["historyRows"]) == 1
    assert "real-ds" in facts["historyRows"][0]
    assert "完成 1/2" in facts["historyRows"][0]


def test_leaving_hpo_mode_restores_the_other_modes_content():
    facts = _run("hpo_mode_isolation")
    for key in _NON_HPO_PROBES:
        assert facts["full"][key]["visible"] is True, key
    # 再切回 HPO 仍然只显示 HPO 内容
    for key in _NON_HPO_PROBES:
        assert facts["backToHpo"][key]["visible"] is False, key


# ── 第五轮 Task 1：正式训练“打开结果文件夹” ────────────────────────

_STUDY_A = "hpo_" + "a" * 32
_STUDY_B = "hpo_" + "b" * 32


def test_formal_folder_button_sits_between_results_and_weights():
    facts = _run("formal_folder_open")
    assert facts["completedOrder"] == {"results": 1, "folder": 2, "download": 3}
    assert facts["completedOrder"]["folder"] > facts["completedOrder"]["results"]
    assert facts["completedOrder"]["folder"] < facts["completedOrder"]["download"]


def test_formal_folder_button_is_enabled_only_for_a_completed_linked_run():
    facts = _run("formal_folder_open")
    states = {s["trainName"]: s for s in facts["states"]}
    # running / failed / stopping / 来源事实不合法 → 不提供可点的入口
    for name in ("train2", "train4", "train6", "train7"):
        assert states[name]["present"] is True, name
        assert states[name]["disabled"] is True, name
    # completed（无论 best.pt 是否存在）都可打开结果目录
    for name in ("train3", "train5"):
        assert states[name]["disabled"] is False, name
    assert set(facts["refused"]) == {"train2", "train4", "train6", "train7"}
    # 即使绕过浏览器语义派发点击，也不得发出任何请求
    assert facts["refusedRequests"] == 0


def test_formal_folder_open_uses_the_full_study_id_and_the_row_train_name():
    facts = _run("formal_folder_open")
    request = facts["firstRequest"]
    assert request["requests"] == 1
    assert request["url"] == (
        "/api/hpo/studies/" + _STUDY_A + "/formal-runs/train3/open-folder")
    # 客户端只提交空对象：绝不提交 path/run_dir 等任何路径字段
    assert request["body"] == {}
    assert _STUDY_A in request["url"]


def test_formal_folder_open_stays_per_row_and_cannot_be_submitted_twice():
    request = _run("formal_folder_open")["firstRequest"]
    assert request["pendingDisabled"] is True          # 提交期间不可重复点击
    assert request["otherRowDisabled"] is False        # 不影响其他记录
    assert _run("formal_folder_open")["afterRepeat"] == 1


def test_formal_folder_open_success_restores_the_button_with_a_short_hint():
    facts = _run("formal_folder_open")["afterSuccess"]
    assert facts["disabled"] is False
    assert "已打开" in facts["hint"]
    assert facts["errorHidden"] is True


def test_formal_folder_open_failure_uses_the_visible_operation_error_area():
    facts = _run("formal_folder_open")
    assert facts["failureRequests"] == 1
    after = facts["afterFailure"]
    assert after["disabled"] is False                  # 失败后按钮恢复
    assert after["errorHidden"] is False
    assert "HPO_ARTIFACT_IDENTITY_MISMATCH" in after["errorText"]
    # 错误在折叠区之外，且绝不泄露绝对路径
    assert after["errorInsideDetails"] is False
    assert after["hasPath"] is False
    # 失败行自己不带成功提示，也不影响另一条记录已经显示的成功提示
    assert "已打开" not in after["failedRowText"]
    assert "已打开" in after["otherRowText"]


def test_stale_formal_folder_reply_never_rewrites_the_current_study():
    facts = _run("formal_folder_open")
    assert facts["pendingForA"] >= 1
    assert facts["afterStale"]["studyId"] == _STUDY_B
    assert facts["afterStale"]["errorBeforeStale"] == ""
    # A 的迟到 200 不得在当前研究里写出任何成功提示或错误
    assert facts["afterStale"]["errorAfterStale"] == ""
    assert "已打开" not in facts["afterStale"]["listText"]


# ── 第五轮 Task 2：五项技术设置默认收进折叠区 ──────────────────────


def test_collapsed_draft_still_receives_the_server_defaults_and_gpu():
    facts = _run("draft_collapsed_settings")
    # 折叠区默认收起，且加载前后都保持收起
    assert facts["beforeDefaults"]["open"] is False
    assert facts["beforeDefaults"]["device"] == ""      # 事实未到达前没有可提交值
    collapsed = facts["collapsed"]
    assert collapsed["open"] is False
    # 即使折叠区关闭，服务端权威默认仍正确写入五项控件
    assert collapsed["sampler"] == "tpe"
    assert collapsed["evaluationMode"] == "comprehensive"
    assert collapsed["device"] == "0"                   # 异步探测后选中的权威默认 GPU
    assert collapsed["imgsz"] == "640"
    assert collapsed["batch"] == "16"
    # 主区直接显示的控件不受影响
    assert collapsed["budget"] == "10" and collapsed["epochs"] == "30"
    assert collapsed["snapshot"] and collapsed["model"]
    assert collapsed["createDisabled"] is False


def test_main_summary_tracks_every_collapsed_setting():
    facts = _run("draft_collapsed_settings")
    assert "算法 随机搜索" in facts["afterSampler"]
    assert "图像尺寸 320" in facts["afterImgsz"]
    assert "Batch 8" in facts["afterBatch"]
    assert "设备 cpu" in facts["afterDevice"]
    assert "快速" in facts["afterMode"]
    # 展开本身不改变任何已生效的选择
    assert facts["afterExpand"]["summary"].count("TPE") == 1


def test_create_request_still_carries_the_collapsed_settings():
    body = _run("draft_collapsed_settings")["createBody"]
    assert body is not None, "a valid draft must still submit"
    assert body["study_config"]["sampler"] == "random"
    assert body["study_config"]["evaluation_mode"] == "quick"
    assert body["execution_config"]["device"] == "cpu"
    assert body["execution_config"]["imgsz"] == 320
    assert body["execution_config"]["batch"] == 8
    # 六项搜索参数仍由服务端采样决定，客户端不提交
    assert "search_space" not in body


def test_leaving_and_reentering_hpo_keeps_the_draft_without_duplicate_controls():
    facts = _run("draft_collapsed_settings")["afterModeSwitch"]
    assert facts["sampler"] == "random"
    assert facts["evaluationMode"] == "quick"
    assert facts["device"] == "cpu"
    assert facts["imgsz"] == "320"
    assert facts["batch"] == "8"
    # 每个控件在整页中仍然唯一：没有为折叠复制第二套控件
    assert facts["counts"] == [1] * len(facts["counts"])


def test_hidden_field_errors_expand_the_draft_and_send_nothing():
    facts = _run("draft_hidden_field_error")
    assert facts["before"]["open"] is False
    for key, calls in (("badImgsz", "imgszCalls"), ("badBatch", "batchCalls"),
                       ("badMode", "modeCalls"), ("badDevice", "deviceCalls")):
        assert facts[key]["open"] is True, key           # 自动展开折叠区
        assert facts[key]["errorHidden"] is False, key
        assert facts[key]["errorInsideDetails"] is False, key
        assert facts[calls] == 0, key                    # 不发送创建请求
        # 已选择的数据快照与权重不清除
        assert facts[key]["snapshot"] == facts["before"]["snapshot"], key
        assert facts[key]["model"] == facts["before"]["model"], key
    assert "32" in facts["badImgsz"]["errorText"]
    assert "Batch" in facts["badBatch"]["errorText"]
    assert "评价方式" in facts["badMode"]["errorText"]
    assert "计算设备" in facts["badDevice"]["errorText"]


def test_server_field_error_in_the_collapsed_area_also_expands_it():
    facts = _run("draft_hidden_field_error")
    assert facts["queuedCreate"] is True                 # 客户端校验通过才提交
    assert facts["serverSampler"]["open"] is True        # 服务端拒绝后自动展开
    assert facts["serverSampler"]["errorInsideDetails"] is False
    assert "sampler" in facts["serverSampler"]["errorText"]
    assert facts["serverSampler"]["snapshot"] == facts["before"]["snapshot"]
    assert facts["serverSampler"]["model"] == facts["before"]["model"]
