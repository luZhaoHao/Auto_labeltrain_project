# F1.2 B Docker Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the smallest production-shaped Linux container that runs the existing Auto Tune Studio UI without changing training semantics or exposing a new external API.

**Architecture:** Keep the current FastAPI/Jinja2 product intact. Add a small delivery boundary for host, port and configuration path, then package the current application in one CUDA-capable Linux image whose persistent application directories are mounted from the host. The image runs the full Studio UI only; `/healthz` is an operational probe, not a public scheduling API.

**Tech Stack:** Python 3.10, FastAPI, Uvicorn, PyTorch 2.5.1 CUDA 12.1, Ultralytics, Docker Desktop Linux containers, Docker Compose.

**Spec:** `docs/f1_2_windows_docker_onnx_spec_20260918.md`

## Global Constraints

- F1.2 is one Studio product operated through the Web UI; do not add `/api/v1/jobs` or Engine mode.
- Do not change direct training, HPO, LLM tuning, run identity, dataset snapshot, scoring or persistence semantics.
- F1.2-A ONNX export must already be independently accepted before this plan starts; do not extend it in F1.2-B.
- Do not include real `auto_tune/config.yaml`, credentials, datasets, model weights, logs, audit files or training outputs in the image or build context.
- Use `D:\Program Files\anaconda3\envs\auto_tune\python.exe` for Python tests outside Docker.
- Do not modify README, roadmap, handoff, implementation-plan DOCX/Markdown, images or historical documents.
- Do not commit or push.

---

### Task 1: Freeze the delivery configuration boundary

**Files:**
- Create: `auto_tune/delivery/__init__.py`
- Create: `auto_tune/delivery/runtime.py`
- Modify: `auto_tune/ui/app.py:580`
- Modify: `auto_tune/ui/app.py:4325`
- Modify: `auto_tune/main.py:20`
- Test: `auto_tune/tests/test_delivery_runtime.py`

**Interfaces:**
- Produces: `resolve_config_path(default: Path) -> Path`
- Produces: `resolve_server_bind(default_host: str = "127.0.0.1", default_port: int = 8000) -> tuple[str, int]`
- Environment: `AUTO_TUNE_CONFIG_PATH`, `AUTO_TUNE_HOST`, `AUTO_TUNE_PORT`

- [ ] **Step 1: Write failing tests for default and environment-controlled values**

```python
def test_resolve_server_bind_keeps_desktop_defaults(monkeypatch):
    monkeypatch.delenv("AUTO_TUNE_HOST", raising=False)
    monkeypatch.delenv("AUTO_TUNE_PORT", raising=False)
    assert resolve_server_bind() == ("127.0.0.1", 8000)


def test_resolve_server_bind_accepts_container_values(monkeypatch):
    monkeypatch.setenv("AUTO_TUNE_HOST", "0.0.0.0")
    monkeypatch.setenv("AUTO_TUNE_PORT", "18000")
    assert resolve_server_bind() == ("0.0.0.0", 18000)


def test_resolve_config_path_uses_controlled_environment_path(tmp_path, monkeypatch):
    target = tmp_path / "config.yaml"
    monkeypatch.setenv("AUTO_TUNE_CONFIG_PATH", str(target))
    assert resolve_config_path(tmp_path / "default.yaml") == target.resolve()
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_delivery_runtime.py -v -p no:cacheprovider
```

Expected: collection or import failure because `auto_tune.delivery.runtime` does not exist.

- [ ] **Step 3: Implement strict parsing without changing defaults**

`resolve_server_bind` must reject an empty host, non-integer port, and ports outside 1–65535 with `ValueError`. `resolve_config_path` must expand variables and `~`, resolve to an absolute path and preserve the existing package config path when the environment variable is absent.

- [ ] **Step 4: Connect both existing server entry points to the resolver**

`auto_tune/main.py` and `auto_tune/ui/app.py` must use the same configuration path. `start_server()` must keep `127.0.0.1:8000` on Windows when no variables are set and use the resolved environment values in Docker.

- [ ] **Step 5: Run the focused tests and existing configuration tests**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_delivery_runtime.py auto_tune\tests\test_ai_settings_api.py -q -p no:cacheprovider
```

Expected: all pass.

### Task 2: Add an operational health probe

**Files:**
- Modify: `auto_tune/ui/app.py`
- Test: `auto_tune/tests/test_delivery_runtime.py`

**Interfaces:**
- Produces: `GET /healthz`
- Response: status 200 with exactly `{"status": "ok", "product": "auto-tune-studio"}`

- [ ] **Step 1: Write a failing route test**

```python
def test_healthz_is_a_minimal_operational_probe():
    response = TestClient(app).get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "product": "auto-tune-studio"}
```

- [ ] **Step 2: Run the test and verify a 404**

Run the single test with the approved `auto_tune` interpreter.

- [ ] **Step 3: Add the minimal route**

The route must not read datasets, models, credentials or training state. It must not expose versions, paths, environment values or exception details. It is an infrastructure probe and must not be added to the user navigation.

- [ ] **Step 4: Run the focused test and UI lifecycle tests**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_delivery_runtime.py auto_tune\tests\test_hpo_ui_lifecycle.py -q -p no:cacheprovider
```

Expected: all pass.

### Task 3: Define the controlled Docker runtime dependencies

**Files:**
- Create: `docker/requirements-runtime.txt`
- Create: `auto_tune/tests/test_docker_delivery_files.py`

**Interfaces:**
- Produces: a pinned Linux runtime dependency file consumed only by `Dockerfile`

- [ ] **Step 1: Write failing static contract tests**

The tests must assert that the runtime file pins Python packages required by imports used in production, pins `torch==2.5.1` and `torchvision==0.20.1`, pins the accepted Ultralytics version, excludes pytest and Windows-only `pyreadline3`, and contains no VCS URL, local path or credential-like token.

- [ ] **Step 2: Run the static tests and verify RED**

Run `test_docker_delivery_files.py`; expect failure because the file is absent.

- [ ] **Step 3: Create the runtime dependency list**

Use the verified F1.1 environment and `environment.yml` as evidence. Do not merge the incompatible root and package requirements blindly. Keep CUDA-specific Torch installation out of this file; the Dockerfile installs the PyTorch CUDA 12.1 wheels from the official PyTorch index before installing this file.

- [ ] **Step 4: Run the static contract tests**

Expected: all pass.

### Task 4: Add the single-image Docker delivery files

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `docker/entrypoint.sh`
- Create: `compose.yaml`
- Modify: `auto_tune/tests/test_docker_delivery_files.py`

**Interfaces:**
- Image: `auto-tune:local`
- Container port: `8000`
- Probe: `GET http://127.0.0.1:8000/healthz`
- Persistent mounts: `/data/config`, `/opt/auto-tune/log`, `/opt/auto-tune/detect`, `/opt/auto-tune/runs`, `/opt/auto-tune/models/weights`
- Dataset mount: `/data/datasets`

- [ ] **Step 1: Extend static tests and verify RED**

The tests must verify one final image, a pinned base image, no `latest`, a non-root runtime user, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`, the correct `WORKDIR`, a health check, and an entrypoint that initializes `/data/config/config.yaml` from `config.template.yaml` only when absent.

The `.dockerignore` test must require exclusion of `.git`, `.claude`, `.codex*`, `auto_tune/config.yaml`, `dataset*`, `detect`, `runs`, `log`, `models/weights`, `*.pt`, `*.onnx`, `*.zip`, caches, render output and local virtual environments.

- [ ] **Step 2: Create the Dockerfile**

Use `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`, install Python 3.10 and the minimum Linux system libraries needed by OpenCV and Ultralytics, install `torch==2.5.1` and `torchvision==0.20.1` from the CUDA 12.1 PyTorch wheel index, then install `docker/requirements-runtime.txt`. Copy only lightweight source and templates. Do not copy ignored local artifacts.

- [ ] **Step 3: Create the entrypoint**

The entrypoint must:

1. refuse to start if required persistent directories are not writable;
2. create `/data/config/config.yaml` from the committed template only when missing;
3. set `AUTO_TUNE_CONFIG_PATH=/data/config/config.yaml`;
4. execute `python -m auto_tune.main` with `exec` so stop signals reach the application;
5. never print configuration contents or credentials.

- [ ] **Step 4: Create one-container Compose configuration**

Publish `127.0.0.1:${AUTO_TUNE_PORT:-8000}:8000`, set `AUTO_TUNE_HOST=0.0.0.0`, mount the six controlled host directories, and request GPU access only through a documented optional Compose override or `docker run --gpus all`. Do not create Web and Worker services.

- [ ] **Step 5: Run static delivery tests and shell syntax validation**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_docker_delivery_files.py -v -p no:cacheprovider
docker compose config
```

Expected: tests pass and Compose renders one service with no secret values.

### Task 5: Build and execute the CPU container smoke test

**Files:**
- Modify only files from Tasks 1–4 if failures prove a defect.

**Interfaces:**
- Consumes: `Dockerfile`, `compose.yaml`, persistent test directories
- Produces: recorded build, health, UI and persistence evidence

- [ ] **Step 1: Check Docker Engine availability**

Run `docker info`. If the Linux Engine is unavailable, stop this task and report the exact blocker. Do not claim Docker validation passed.

- [ ] **Step 2: Build without injecting secrets**

Run:

```powershell
docker build --pull=false -t auto-tune:local .
```

Expected: successful build; build output must not contain a real configuration file or credential.

- [ ] **Step 3: Inspect the image content**

Confirm `/opt/auto-tune/auto_tune/config.yaml`, `.git`, datasets, `*.pt`, logs and training results are absent from the image layers and final filesystem.

- [ ] **Step 4: Start the CPU container and verify health and UI**

Use isolated temporary host directories. Verify `/healthz` returns the exact contract and `/` returns the Studio HTML. Confirm the created config is the sanitized template.

- [ ] **Step 5: Verify persistence**

Write a harmless marker into the mounted log directory, restart the container, and confirm the marker remains. Confirm container removal does not delete the host directories.

### Task 6: Run regression checks and prepare the review report

**Files:**
- No additional production files.

- [ ] **Step 1: Run all focused delivery tests**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_delivery_runtime.py auto_tune\tests\test_docker_delivery_files.py -q -p no:cacheprovider
```

- [ ] **Step 2: Run existing startup, security and training-state tests**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_upload_security.py auto_tune\tests\test_training_gate.py auto_tune\tests\test_run_state_training_api.py auto_tune\tests\test_hpo_ui_lifecycle.py -q -p no:cacheprovider
```

- [ ] **Step 3: Run the full automated suite**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests -q -p no:cacheprovider
```

Expected baseline: no regression from 2738 passed and the two existing sklearn PCA warnings; test count may increase only by the new delivery tests.

- [ ] **Step 4: Run repository hygiene checks**

Run `git diff --check`, list every modified/untracked file, and prove that no config, credential, dataset, weight, log, training output or Docker test data is included.

- [ ] **Step 5: Stop and report for Codex review**

Report changed files, RED/GREEN evidence, Docker build and smoke results, deviations, unresolved GPU validation, and final `git status --short`. Do not modify project documentation, commit or push.
