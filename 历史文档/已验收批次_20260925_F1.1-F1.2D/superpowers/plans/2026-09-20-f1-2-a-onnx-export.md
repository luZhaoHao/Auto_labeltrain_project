# F1.2-A ONNX Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not dispatch subagents for this small batch.

**Goal:** Let an operator upload a trusted YOLOv8 Detect `.pt` and manually export a validated `.onnx` beside it in the controlled model library.

**Architecture:** Reuse the existing `ModelStore` identity, upload route and single page. One small export service validates a managed `model_id`, runs the existing Ultralytics exporter in a bounded child process, checks the temporary ONNX and publishes it without overwriting another export. The UI shows progress and provides a download; training paths remain untouched.

**Tech Stack:** Existing Python 3.10 `auto_tune` Conda environment, FastAPI, Ultralytics 8.3.253, ONNX, ONNX Runtime, existing vanilla JS.

**Spec:** `docs/f1_2_windows_docker_onnx_spec_20260918.md`, section 4 and F1.2-A.

## Global Constraints

- Scope: only an explicitly uploaded `origin=managed` `.pt`, FP32 default, same-directory `.onnx`, no arbitrary path, no training result/automatic conversion.
- Fixed export settings: `imgsz=640`, `opset=17`, `dynamic=False`, `simplify=False`, `nms=False`; FP16 is an advanced option only if real export and inference pass.
- Keep upload opaque; only user-triggered export may load a trusted weight. A subprocess isolates crashes but does not sandbox a malicious pickle.
- No overwrite, no persistent jobs framework, no external public jobs API, no new database and no modifications to the training state machine.
- Missing dependencies: enumerate exact packages/versions and request 艾卡's approval before installing or adding them to dependency manifests.
- Claude Code changes business code and tests only; Codex handles authoritative docs. Do not commit or push.
- Use `D:\Program Files\anaconda3\envs\auto_tune\python.exe` for every Python/test command; first print `sys.executable` and version. If the interpreter exits abnormally, report it instead of switching to system Python.

## Review Focus

- A `legacy` record with a valid `model_id` must be rejected on the server even if the button is hidden.
- A `.pt` replaced after listing must fail identity verification before export; a changed source during conversion must not publish output.
- Simultaneous clicks and an existing target must not overwrite a valid ONNX or leave a partial public file.
- A broken checkpoint or hanging child process must return a bounded error without taking down the Studio server; remove only this request's temporary files.
- An FP16 request must never silently return an FP32 file after an upstream warning.

---

### Task 1: Export service and file boundary

**Files:**
- Create: `auto_tune/modules/model_store/onnx_export.py` (service and bounded subprocess invocation).
- Create: `auto_tune/modules/model_store/onnx_export_worker.py` (child entry, the only place that imports Ultralytics to load `.pt`).
- Modify: `auto_tune/modules/model_store/service.py` (small method to resolve a managed record and revalidate its SHA-256; keep `resolve()` semantics unchanged).
- Test: `auto_tune/tests/test_model_onnx_export.py`.

**Interfaces:** Service takes `(store: ModelStore, model_id: str, precision: Literal['fp32','fp16'])` and returns only a safe output name, size and source `model_id`; typed export errors carry stable `error_code`, public message and HTTP status. Worker accepts internally generated paths and fixed export arguments, never a client-supplied path.

- [ ] Write failing service tests: managed valid ID succeeds with a stub worker; legacy ID, tampered hash, invalid precision, existing output and concurrent same-source requests fail with stable codes. Assert the original `.pt` and any prior `.onnx` bytes remain unchanged.
- [ ] Implement only the identity, allowed-origin, conflict and single-flight rules needed for those tests. Hold the per-target reservation until a checked final file exists; verify source identity again before publishing. Do not expose filesystem paths to clients.
- [ ] Write failing worker tests using a fake exporter: fixed arguments, subprocess failure and timeout, temporary cleanup, ONNX structural check failure, FP16 output-type failure. Implement bounded child invocation and temp-to-final commit. Never invoke Ultralytics in the API process; never install dependencies automatically.
- [ ] Run `test_model_onnx_export.py` and existing `test_model_store.py`. Report actual RED then GREEN output and any tests blocked by interpreter startup.

### Task 2: Internal UI API and download

**Files:**
- Modify: `auto_tune/ui/model_store_api.py` (new export, status and download routes under existing `/api/models`; reuse existing CSRF/origin gate on mutations).
- Modify: `auto_tune/ui/app.py` (wire the service only, no training routes).
- Test: `auto_tune/tests/test_model_onnx_api.py`.

**Interfaces:** `POST /api/models/export` accepts `{model_id, precision}`; a small status read endpoint supports UI polling only if export is asynchronous; download accepts an opaque export identity, revalidates its managed source and streams the controlled output. Response contains only safe name, size, state/error, and internal download URL. No arbitrary file path parameter.

- [ ] Write failing HTTP tests for success, CSRF/origin rejection, legacy ID, malformed ID, conflict, duplicate running export, missing output and safe errors; verify downloaded bytes match the committed ONNX.
- [ ] Implement a non-blocking UI request: bounded background worker with minimal in-memory status, or a request pattern that demonstrably leaves other UI requests responsive. Keep status temporary; do not create a general jobs framework.
- [ ] Run `test_model_onnx_api.py`, `test_model_store_api.py` and `test_upload_security.py` in the pinned interpreter.

### Task 3: Single-page control and real acceptance

**Files:**
- Modify: `auto_tune/ui/templates/single_page.html` (one export control near existing upload block, FP16 under advanced options, no duplicate model input).
- Modify: `auto_tune/ui/static/hpo.js` (reuse model list/model_id; disabled while active; status and download).
- Modify: `auto_tune/ui/i18n.py` only for new text actually used.
- Test: relevant existing UI tests and one focused UI behavior test in `auto_tune/tests/`.

- [ ] Write failing UI tests for managed-only availability, FP32 default, explicit trusted-source confirmation, progress/failure/conflict, download, duplicate click and no arbitrary path field. Make the test exercise behavior rather than only search strings.
- [ ] Add the controls and client wiring; keep upload behavior and direct-training/HPO selectors unchanged.
- [ ] Run focused UI and model-store tests, `node --check auto_tune/ui/static/hpo.js`, then full `auto_tune/tests` regression and `git diff --check` once related changes pass.
- [ ] With a **trusted local small YOLOv8 Detect `.pt`**, run FP32 export, `onnx.checker.check_model` and one fixed-input PyTorch/ONNX Runtime inference. Require matching output shapes and `np.allclose(..., rtol=1e-3, atol=1e-4)` for raw FP32 outputs; document any exporter-dependent output mismatch rather than weaken the test silently. Offer FP16 only after its dtype and compatible-runtime inference have passed; otherwise keep it hidden/disabled with a clear reason.
- [ ] Report exact files, interpreter path/version, RED/GREEN, regression and real-export evidence, deviations and remaining risks to Codex; stop without a commit or push.

## Self-check before handing off

Confirm the user can download the ONNX after page refresh (derived from the validated target, not only transient memory), that unsupported FP16 cannot silently succeed, that an interrupted conversion cannot leave an apparently complete output, and that no unrelated workflow changed.
