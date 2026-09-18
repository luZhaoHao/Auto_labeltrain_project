# Studio S1.3 API Credential Security Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create commits or push; Codex will review S1.2 and S1.3 together after acceptance.

**Goal:** Remove plaintext API keys from project configuration and give single-machine Studio customers safe credential management with constrained custom endpoints.

**Architecture:** Keep the existing `llm`, `vision`, `APP_CONFIG`, Module B/C and UI architecture. Add one narrow `security` package for credential resolution, endpoint policy and redaction, connect the three existing HTTP call sites to it, then add a small settings/migration API and UI block.

**Tech Stack:** Python 3.10+, FastAPI, Jinja2, PyYAML, `requests`, standard-library `ctypes`, pytest, existing browser verification workflow.

**Spec:** `docs/superpowers/specs/2026-08-21-studio-s1-3-api-credential-security-design.md`

## Global Constraints

- Studio is single-user and single-machine; do not add login, RBAC, tenant or Cloud code.
- Do not add dependencies or a new frontend framework.
- Do not modify LLM prompts, LLM action space, Guardrails, KPI definitions, training execution, snapshot behavior or history schemas.
- Windows Credential Manager is the only persistent secret store in this batch; non-Windows supports environment variables only.
- `AUTO_TUNE_TEXT_API_KEY` and `AUTO_TUNE_VISION_API_KEY` override the OS store and are read-only from UI.
- Resolved secrets never enter `APP_CONFIG`, YAML, URL, response, SSE, history, audit or logs.
- Custom public endpoints require HTTPS. Local/private HTTP(S) requires the explicit `allow_private_endpoint` switch.
- No real customer key may be used in tests or output.
- Claude Code changes business code and tests only. Codex owns README, roadmap, handoff, DOCX and release notes; PDF is no longer a required maintained format.
- Do not commit or push. Report changed files, commands/results, deviations and risks to Codex.

---

## File Structure

**Create**

- `auto_tune/modules/security/__init__.py` — stable exports only.
- `auto_tune/modules/security/credentials.py` — fixed purpose mapping, environment/Windows store, status and five-minute cache.
- `auto_tune/modules/security/endpoint_policy.py` — URL normalization, IP/DNS policy and safe request settings.
- `auto_tune/modules/security/redaction.py` — shared recursive/text redaction and provider error classification.
- `auto_tune/tests/test_credentials.py` — credential service and migration primitives.
- `auto_tune/tests/test_endpoint_policy.py` — endpoint and redaction policy.
- `auto_tune/tests/test_ai_settings_api.py` — settings, credential and migration API tests.

**Modify**

- `auto_tune/modules/agent_engine/audit.py` — delegate redaction to the shared implementation while preserving imports and output.
- `auto_tune/modules/agent_engine/decision_agent.py` — resolve `text` credential and validate endpoint immediately before HTTP.
- `auto_tune/modules/train_analyzer/llm_analyzer.py` — same `text` boundary.
- `auto_tune/modules/train_analyzer/vision_analyzer.py` — same `vision` boundary.
- `auto_tune/ui/app.py` — safe config writes, CSRF/same-origin guard, APIs and migration orchestration.
- `auto_tune/ui/templates/single_page.html` — two compact AI service cards.
- `auto_tune/ui/i18n.py` — Chinese/English strings for the new UI.
- `auto_tune/config.template.yaml` — remove `api_key`, add non-sensitive references and endpoint switch.
- Existing related tests only where regression assertions belong.

---

### Task 1: Credential Store and Resolver

**Files:**

- Create: `auto_tune/modules/security/__init__.py`
- Create: `auto_tune/modules/security/credentials.py`
- Create: `auto_tune/tests/test_credentials.py`

**Interfaces:**

- Produces `CredentialPurpose`, `CredentialStatus`, `CredentialError`.
- Produces `get_credential_status(purpose)`, `resolve_credential(purpose)`, `store_credential(purpose, value)`, `delete_credential(purpose)`, `invalidate_credential_cache(purpose=None)`.
- The only valid purposes are `text` and `vision`; callers cannot supply an arbitrary Windows target.

- [ ] **Step 1: Write failing tests for source priority, safe status and cache**

Use fake backend functions so tests never read or modify the developer's real Windows Credential Manager:

```python
def test_environment_wins_and_status_never_contains_secret(monkeypatch):
    monkeypatch.setenv("AUTO_TUNE_TEXT_API_KEY", "env-secret-123")
    monkeypatch.setattr(credentials, "_read_windows_credential", lambda target: "vault-secret")
    credentials.invalidate_credential_cache()
    assert credentials.resolve_credential("text") == "env-secret-123"
    status = credentials.get_credential_status("text")
    assert status.configured is True
    assert status.source == "environment"
    assert status.writable is False
    assert "secret" not in repr(status).lower()


def test_cache_expires_and_delete_invalidates(monkeypatch):
    clock = iter([0.0, 1.0, 301.0, 302.0])
    values = iter(["first", "second"])
    monkeypatch.delenv("AUTO_TUNE_TEXT_API_KEY", raising=False)
    monkeypatch.setattr(credentials.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(credentials, "_read_windows_credential", lambda target: next(values))
    assert credentials.resolve_credential("text") == "first"
    assert credentials.resolve_credential("text") == "first"
    assert credentials.resolve_credential("text") == "second"
```

- [ ] **Step 2: Run Task 1 tests and verify they fail because the module does not exist**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_credentials.py -v -p no:cacheprovider
```

- [ ] **Step 3: Implement the narrow public contract**

Use frozen statuses and fixed mappings:

```python
CredentialPurpose = Literal["text", "vision"]

_PURPOSES = {
    "text": ("AUTO_TUNE_TEXT_API_KEY", "AutoTuneStudio/text/deepseek"),
    "vision": ("AUTO_TUNE_VISION_API_KEY", "AutoTuneStudio/vision/qwen"),
}
_CACHE_TTL_SECONDS = 300.0

@dataclass(frozen=True)
class CredentialStatus:
    purpose: CredentialPurpose
    configured: bool
    source: Literal["environment", "windows_credential_manager", "missing"]
    writable: bool
    last_tested_at: str | None = None
    last_test_result: str | None = None
```

Validate purpose before every operation. Resolve environment first, then cached/Windows value. Never expose a method that accepts a raw target name.

- [ ] **Step 4: Implement Windows backend using standard-library `ctypes`**

Requirements:

- use Generic Credentials scoped to the current Windows user;
- define exact `CREDENTIALW` structures and `CredReadW`, `CredWriteW`, `CredDeleteW`, `CredFree` signatures;
- copy only the credential blob needed by the resolver and always call `CredFree` in `finally`;
- treat Windows “not found” as missing and other failures as `CredentialError` with a safe numeric error code;
- on non-Windows, read returns missing and write/delete raise a safe unsupported-platform error;
- never enumerate Credential Manager entries.

- [ ] **Step 5: Add mutation and platform failure tests**

Cover fixed target mapping, write/read/replace, idempotent deletion, environment-write conflict, non-Windows behavior, Windows API failure redaction and immediate cache invalidation. Windows functions remain mocked in unit tests.

- [ ] **Step 6: Run Task 1 tests**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_credentials.py -q -p no:cacheprovider
```

Expected: all pass, no real Credential Manager target created.

---

### Task 2: Endpoint Policy and Shared Redaction

**Files:**

- Create: `auto_tune/modules/security/endpoint_policy.py`
- Create: `auto_tune/modules/security/redaction.py`
- Create: `auto_tune/tests/test_endpoint_policy.py`
- Modify: `auto_tune/modules/agent_engine/audit.py`
- Modify: `auto_tune/tests/test_audit.py`

**Interfaces:**

- Produces `EndpointPolicyError`, `validate_endpoint(endpoint, allow_private_endpoint, resolver=socket.getaddrinfo) -> str`.
- Produces `REDACTED`, `redact_sensitive(value, known_secrets=())`, `safe_provider_error(status_code) -> str`.
- Preserves the existing `audit.redact_sensitive` import path.

- [ ] **Step 1: Write the endpoint policy matrix as failing parametrized tests**

```python
@pytest.mark.parametrize("url", [
    "file:///tmp/key",
    "https://user:pass@example.com/v1/chat/completions",
    "https://example.com/v1?key=x",
    "https://example.com/v1#fragment",
    "http://public.example/v1/chat/completions",
])
def test_public_endpoint_rejects_unsafe_urls(url, public_dns):
    with pytest.raises(EndpointPolicyError):
        validate_endpoint(url, False, resolver=public_dns)


@pytest.mark.parametrize("host", ["127.0.0.1", "10.0.0.8", "169.254.1.2", "::1"])
def test_private_targets_require_explicit_switch(host):
    url = f"http://[{host}]/v1/chat/completions" if ":" in host else f"http://{host}/v1/chat/completions"
    with pytest.raises(EndpointPolicyError):
        validate_endpoint(url, False)
    assert validate_endpoint(url, True).startswith("http")
```

Also test a DNS response containing both a public and a private address: the entire endpoint must be rejected.

- [ ] **Step 2: Write failing redaction tests**

```python
def test_redaction_handles_keys_headers_and_free_text_without_mutation():
    source = {
        "Authorization": "Bearer top-secret",
        "nested": {"client_secret_value": "top-secret"},
        "message": "provider echoed Bearer top-secret",
    }
    result = redact_sensitive(source, known_secrets=("top-secret",))
    assert "top-secret" not in repr(result)
    assert source["Authorization"] == "Bearer top-secret"
```

- [ ] **Step 3: Run the new tests and verify failure**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_endpoint_policy.py auto_tune\tests\test_audit.py -q -p no:cacheprovider
```

- [ ] **Step 4: Implement endpoint validation**

Parse with `urllib.parse.urlsplit`, reject user info/query/fragment/invalid port, resolve immediately before use, normalize each result with `ipaddress.ip_address`, and reject the whole public endpoint when any resolved address is loopback/private/link-local/multicast/unspecified/reserved. Public endpoints require HTTPS. With the private switch enabled, local/private HTTP(S) is allowed but all other URL restrictions remain.

Return the normalized endpoint only; do not perform the HTTP call in this module.

- [ ] **Step 5: Implement shared redaction and preserve audit compatibility**

Move the current sensitive-key logic into `security/redaction.py`, add Cookie/auth header matching and known-secret free-text replacement, then import it from `audit.py`:

```python
from auto_tune.modules.security.redaction import REDACTED, redact_sensitive
```

`atomic_write_json` must continue calling `redact_sensitive(payload)`. Do not change audit schema or fatal persistence behavior.

- [ ] **Step 6: Run Task 2 and existing audit tests**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_endpoint_policy.py auto_tune\tests\test_audit.py -q -p no:cacheprovider
```

---

### Task 3: Bind All Existing Model Calls to the Security Boundary

**Files:**

- Modify: `auto_tune/modules/agent_engine/decision_agent.py`
- Modify: `auto_tune/modules/train_analyzer/llm_analyzer.py`
- Modify: `auto_tune/modules/train_analyzer/vision_analyzer.py`
- Modify: `auto_tune/tests/test_decision_agent.py`
- Modify: `auto_tune/tests/test_train_analyzer.py`
- Create or modify the smallest appropriate vision analyzer test file.

**Interfaces:**

- Consumes `resolve_credential("text"|"vision")` and `validate_endpoint(...)`.
- Produces unchanged successful return structures and safe categorized failures.

- [ ] **Step 1: Add failing tests proving YAML keys are ignored**

For each of the three call paths, inject `api_key: yaml-secret-must-not-be-used`, monkeypatch `resolve_credential` to return `resolved-secret`, and capture `requests.post`. Assert:

```python
assert sent_headers["Authorization"] == "Bearer resolved-secret"
assert "yaml-secret-must-not-be-used" not in repr(captured_request)
assert captured_kwargs["allow_redirects"] is False
```

Assert decision and text use purpose `text`; vision uses purpose `vision`.

- [ ] **Step 2: Add failing safe-error tests**

Return a fake 401/429/500 whose body contains both `resolved-secret` and `provider-private-body`. Assert the public exception/result contains only the stable category and status code; neither secret nor raw body may appear in exception, report, audit-compatible payload or captured log.

- [ ] **Step 3: Run focused tests and confirm failure**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_decision_agent.py auto_tune\tests\test_train_analyzer.py -q -p no:cacheprovider
```

- [ ] **Step 4: Apply the same minimal call sequence to all three files**

Immediately before `requests.post`:

```python
purpose = "vision"  # "text" in decision_agent.py and llm_analyzer.py
api_key = resolve_credential(purpose)
endpoint = validate_endpoint(
    section.get("endpoint", default_endpoint),
    bool(section.get("allow_private_endpoint", False)),
)
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
response = requests.post(
    endpoint,
    headers=headers,
    json=payload,
    timeout=(10, 120),
    allow_redirects=False,
)
```

Use the existing configuration for provider/model/temperature/max tokens. Do not alter prompts or successful output structures. Convert failures to stable categories without reading raw error text into returned values.

- [ ] **Step 5: Run model call tests and tuning regressions**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_decision_agent.py auto_tune\tests\test_train_analyzer.py auto_tune\tests\test_tuning_loop.py auto_tune\tests\test_audit.py -q -p no:cacheprovider
```

---

### Task 4: Safe Settings API, Explicit Legacy Migration and Minimal UI

**Files:**

- Create: `auto_tune/tests/test_ai_settings_api.py`
- Modify: `auto_tune/ui/app.py`
- Modify: `auto_tune/ui/templates/single_page.html`
- Modify: `auto_tune/ui/i18n.py`
- Modify: `auto_tune/config.template.yaml`
- Modify: `auto_tune/tests/test_ui_training_results.py` only for shared page regressions.

**Interfaces:**

- Produces the six route operations specified in design sections 12.1–12.6.
- Produces an in-memory `migration_required` state derived from legacy YAML without exposing its value.
- Consumes the credential and endpoint modules from Tasks 1–2.

- [ ] **Step 1: Write failing read/update security tests**

Test that `GET /api/ai-settings` contains only non-sensitive fields and statuses. Recursively serialize the response and assert it contains none of the injected secrets. Test that settings update rejects `api_key`, arbitrary fields, invalid purpose, unsafe endpoint, cross-origin request and missing/invalid CSRF token.

- [ ] **Step 2: Write failing credential lifecycle tests**

Cover:

- environment source returns conflict on write/delete;
- tested replacement writes only after a successful minimal probe;
- failed probe retains the previous credential;
- `test_before_replace=false` stores with `last_test_result="untested"`;
- delete requires confirmation and invalidates cache;
- test response is one of the fixed safe categories and contains no provider body.

- [ ] **Step 3: Write failing migration transaction tests**

Use a temporary YAML file and mocked credential backend:

```python
def test_migration_removes_yaml_key_only_after_verified_store(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("llm:\n  api_key: legacy-secret\n  enabled: true\n", encoding="utf-8")
    # patch app.config_path, store and read-back verification
    response = client.post("/api/credentials/text/migrate", headers=valid_security_headers())
    assert response.status_code == 200
    assert "legacy-secret" not in config_path.read_text(encoding="utf-8")
    assert "api_key" not in yaml.safe_load(config_path.read_text(encoding="utf-8"))["llm"]
```

Add failure cases for store failure, read-back mismatch, YAML replace failure, active environment source and placeholder values. Startup must detect but never silently migrate or use the legacy key.

- [ ] **Step 4: Run API tests and verify failure**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_ai_settings_api.py -v -p no:cacheprovider
```

- [ ] **Step 5: Implement safe config and request guards in `app.py`**

- Add one atomic YAML update helper using a temporary file in the same directory, `flush`, `os.fsync` and `os.replace`.
- Whitelist non-sensitive settings; explicitly reject secret fields.
- Generate one process-session CSRF token with `secrets.token_urlsafe(32)` and render it into the existing page context.
- For mutation endpoints, require same-origin host/origin agreement and `X-CSRF-Token` equality using `secrets.compare_digest`.
- Do not log request bodies.
- Keep legacy secret values out of `APP_CONFIG`: load into a short-lived migration holder/state, then remove the `api_key` keys from the in-memory public configuration.
- Do not change project/dataset/training APIs beyond reusing the atomic helper where necessary for migration safety.

- [ ] **Step 6: Implement the six route operations**

Use fixed purposes and fixed credential targets. Connection test sends a minimal OpenAI-compatible message such as `Reply with OK` and never includes project, training or image data. Keep public results to the categories defined in the spec.

For legacy migration: confirm request → secure store → secure read-back equality → atomic YAML removal → reload sanitized config → clear holder/cache. If YAML rewrite fails after secure storage, retain the secure credential, keep `migration_required`, and return a retryable safe error.

- [ ] **Step 7: Add the two compact UI cards**

Use the existing page styles and escaping helpers. The password input must never receive a `value` from the server. Clear it after submit/close. Show only configured/source/writable/test status. Require confirmation for delete, legacy migration and enabling private endpoints; show a persistent warning for private HTTP endpoints.

Do not display key suffixes, lengths or hashes. Do not use unsafe `innerHTML` for provider error content.

- [ ] **Step 8: Update the configuration template**

Remove both `api_key` fields. Preserve the other existing fields and add:

```yaml
credential_ref: AutoTuneStudio/text/deepseek
allow_private_endpoint: false
```

and the corresponding `AutoTuneStudio/vision/qwen` reference. Comments direct users to the Studio UI or environment variables, never to plaintext YAML or command-line arguments.

- [ ] **Step 9: Run API and UI tests**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_ai_settings_api.py auto_tune\tests\test_ui_training_results.py -q -p no:cacheprovider
```

---

### Task 5: Integrated Security and Regression Verification

**Files:**

- Modify tests only if a genuine uncovered requirement is found; do not broaden business scope.

**Interfaces:**

- Consumes all previous tasks.
- Produces verification evidence for Codex review, not a commit.

- [ ] **Step 1: Run the S1.3 focused suite**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_credentials.py auto_tune\tests\test_endpoint_policy.py auto_tune\tests\test_ai_settings_api.py auto_tune\tests\test_decision_agent.py auto_tune\tests\test_train_analyzer.py auto_tune\tests\test_audit.py auto_tune\tests\test_ui_training_results.py -q -p no:cacheprovider
```

- [ ] **Step 2: Run critical compatibility tests**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_executor.py auto_tune\tests\test_guardrails.py auto_tune\tests\test_tuning_loop.py auto_tune\tests\test_training_finalizer.py auto_tune\tests\test_dataset_snapshot.py auto_tune\tests\test_dataset_snapshot_api.py -q -p no:cacheprovider
```

- [ ] **Step 3: Run the complete suite**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests -q -p no:cacheprovider
```

- [ ] **Step 4: Run a repository secret-pattern scan**

Scan tracked/source candidates for plaintext credential fields, Bearer tokens, common provider key prefixes and the test canary. Exclude `.git`, ignored runtime logs, datasets, weights and history documents only when the exclusion is justified. Report every remaining hit and distinguish placeholders/test fixtures from real secrets. Do not print a discovered full secret into the report.

- [ ] **Step 5: Hand off to Codex without modifying documentation or Git history**

Report:

1. changed/created business and test files;
2. every test command and exact result;
3. any deviation from this plan;
4. residual risks, including DNS rebinding limits and Windows API behavior;
5. confirmation that no real key, commit or push was made.

Leave these independent Codex acceptance activities unclaimed: real Chromium interaction, controlled Windows test-target lifecycle, simulated provider redirect/error leakage inspection, documentation update, S1.2+S1.3 combined pre-commit review and final acceptance wording.

---

## Scope Guard for Claude Code

Stop and report instead of expanding the implementation if any of the following appears necessary:

- adding an authentication framework or database;
- adding a third-party credential package;
- redesigning `APP_CONFIG` across the application;
- changing prompts, model analysis content or tuning decisions;
- adding Cloud/multi-user behavior or Linux Secret Service;
- changing S1.2 snapshot contracts, training command construction or history schemas;
- deleting an existing real credential or permanently deleting project files.

The goal is a narrow security boundary around existing calls, not a general provider platform.

---

## Implementation Status (2026-08-21)

All five tasks were implemented and independently accepted. Final evidence: S1.3 nine-file suite `180 passed`; complete suite `396 passed, 2 existing PCA warnings, 0 skipped`; simulated provider redirect/error leakage, Windows cross-process credential lifecycle, Chromium UI, and real rotated-key connection/analysis all passed. No commit or push has been made; S1.2 and S1.3 remain pending combined pre-commit review.
