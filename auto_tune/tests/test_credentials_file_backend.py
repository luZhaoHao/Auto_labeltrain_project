"""The Linux/Docker credential backend: a persisted JSON credential file.

Windows keeps the Credential Manager; the container has no OS credential store,
so a key typed into the page must survive a container restart. These tests drive
the real module with ``_is_windows`` forced to False and a path inside
``tmp_path``; no test reads or writes a real credential, and every value here is
a fabricated string.
"""

from __future__ import annotations

import json
import os
import stat
import threading

import pytest

from auto_tune.modules.security import credentials
from auto_tune.modules.security.credentials import (
    CredentialError,
    credentials_file_path,
    delete_credential,
    get_credential_status,
    invalidate_credential_cache,
    known_credentials,
    resolve_credential,
    store_credential,
    supports_file_credential_store,
    supports_os_credential_store,
)

TEXT_KEY = "text-key-aaaaaaaa"
VISION_KEY = "vision-key-bbbbbbbb"


@pytest.fixture(autouse=True)
def _linux_file_store(monkeypatch, tmp_path):
    """Every test runs as the Linux container, against a controlled path."""
    monkeypatch.delenv("AUTO_TUNE_TEXT_API_KEY", raising=False)
    monkeypatch.delenv("AUTO_TUNE_VISION_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "_is_windows", lambda: False)
    path = tmp_path / "secrets" / "credentials.json"
    monkeypatch.setenv("AUTO_TUNE_CREDENTIALS_PATH", str(path))
    invalidate_credential_cache()
    yield path
    invalidate_credential_cache()


def _stored(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ── first run: no file, a writable backend ──────────────────────────────────


def test_the_file_backend_replaces_the_os_store_off_windows():
    assert supports_os_credential_store() is False
    assert supports_file_credential_store() is True


def test_a_first_run_reports_missing_and_writable(_linux_file_store):
    status = get_credential_status("text")

    assert status.configured is False
    assert status.source == "missing"
    assert status.writable is True, "a key must be typeable before any file exists"
    assert resolve_credential("text") is None
    assert _linux_file_store.exists() is False, "reading creates nothing"


def test_the_configured_path_wins_over_the_container_default(monkeypatch):
    monkeypatch.setenv("AUTO_TUNE_CREDENTIALS_PATH", "/tmp/auto-tune-test/keys.json")
    assert credentials_file_path() == credentials.Path("/tmp/auto-tune-test/keys.json")


def test_the_container_default_is_the_documented_mount(monkeypatch):
    monkeypatch.delenv("AUTO_TUNE_CREDENTIALS_PATH", raising=False)
    assert credentials_file_path() == credentials.Path("/data/secrets/credentials.json")


# ── save, replace, delete ───────────────────────────────────────────────────


def test_a_saved_key_is_resolvable_immediately(_linux_file_store):
    store_credential("text", TEXT_KEY)

    assert resolve_credential("text") == TEXT_KEY
    status = get_credential_status("text")
    assert status.configured is True
    assert status.source == "file_credential_store"
    assert status.writable is True
    assert TEXT_KEY not in repr(status)


def test_only_the_configured_purpose_is_written(_linux_file_store):
    store_credential("text", TEXT_KEY)

    assert _stored(_linux_file_store) == {"text": TEXT_KEY}
    # UTF-8 by contract, not by locale
    assert _linux_file_store.read_bytes().decode("utf-8") == _linux_file_store.read_text(
        encoding="utf-8"
    )


def test_saving_vision_does_not_drop_text(_linux_file_store):
    store_credential("text", TEXT_KEY)
    store_credential("vision", VISION_KEY)

    assert _stored(_linux_file_store) == {"text": TEXT_KEY, "vision": VISION_KEY}
    assert resolve_credential("text") == TEXT_KEY
    assert resolve_credential("vision") == VISION_KEY


def test_replacing_text_keeps_vision(_linux_file_store):
    store_credential("text", TEXT_KEY)
    store_credential("vision", VISION_KEY)
    store_credential("text", "text-key-cccccccc")

    assert _stored(_linux_file_store) == {"text": "text-key-cccccccc", "vision": VISION_KEY}
    assert resolve_credential("vision") == VISION_KEY


def test_deleting_one_purpose_keeps_the_other(_linux_file_store):
    store_credential("text", TEXT_KEY)
    store_credential("vision", VISION_KEY)

    delete_credential("text")

    assert _stored(_linux_file_store) == {"vision": VISION_KEY}
    assert resolve_credential("text") is None
    assert resolve_credential("vision") == VISION_KEY


def test_deleting_the_last_purpose_removes_the_empty_file(_linux_file_store):
    store_credential("text", TEXT_KEY)

    delete_credential("text")

    assert _linux_file_store.exists() is False, (
        "the same rule applies to the last purpose: no purpose left, no file")
    assert get_credential_status("text").configured is False


def test_deleting_a_missing_purpose_is_idempotent(_linux_file_store):
    delete_credential("vision")
    delete_credential("vision")

    assert _linux_file_store.exists() is False


def test_deleting_an_absent_purpose_still_drops_a_stale_cached_value(_linux_file_store):
    """A cached value must not outlive a delete, whatever the file now holds."""
    store_credential("text", TEXT_KEY)
    assert resolve_credential("text") == TEXT_KEY
    _linux_file_store.unlink()  # the file went away behind the process

    delete_credential("text")

    assert resolve_credential("text") is None


# ── environment keeps the highest priority and stays read-only ──────────────


def test_the_environment_wins_over_the_persisted_file(_linux_file_store, monkeypatch):
    store_credential("text", TEXT_KEY)
    monkeypatch.setenv("AUTO_TUNE_TEXT_API_KEY", "env-key-dddddddd")

    assert resolve_credential("text") == "env-key-dddddddd"
    status = get_credential_status("text")
    assert status.source == "environment"
    assert status.writable is False


def test_the_environment_source_can_not_be_replaced_or_deleted(_linux_file_store, monkeypatch):
    monkeypatch.setenv("AUTO_TUNE_TEXT_API_KEY", "env-key-dddddddd")

    with pytest.raises(CredentialError):
        store_credential("text", "new-key-eeeeeeee")
    with pytest.raises(CredentialError):
        delete_credential("text")
    assert _linux_file_store.exists() is False


def test_removing_the_environment_falls_back_to_the_file(_linux_file_store, monkeypatch):
    store_credential("text", TEXT_KEY)
    monkeypatch.setenv("AUTO_TUNE_TEXT_API_KEY", "env-key-dddddddd")
    assert resolve_credential("text") == "env-key-dddddddd"

    monkeypatch.delenv("AUTO_TUNE_TEXT_API_KEY")

    assert resolve_credential("text") == TEXT_KEY, "the cache must not keep the env value"


# ── Windows is untouched ────────────────────────────────────────────────────


def test_windows_keeps_the_credential_manager_and_never_writes_a_file(
    monkeypatch, tmp_path
):
    written: dict[str, str] = {}
    monkeypatch.setenv("AUTO_TUNE_CREDENTIALS_PATH", str(tmp_path / "secrets" / "credentials.json"))
    monkeypatch.setattr(credentials, "_is_windows", lambda: True)
    monkeypatch.setattr(credentials, "_write_windows_credential", lambda t, v: written.update({t: v}))
    monkeypatch.setattr(credentials, "_read_windows_credential", lambda t: written.get(t))

    store_credential("text", TEXT_KEY)

    assert written == {"AutoTuneStudio/text/deepseek": TEXT_KEY}
    assert not (tmp_path / "secrets").exists(), "Windows must not use the plaintext file"
    assert get_credential_status("text").source == "windows_credential_manager"


def test_windows_status_is_unaffected_by_the_file_backend(monkeypatch):
    monkeypatch.setattr(credentials, "_is_windows", lambda: True)
    monkeypatch.setattr(credentials, "_read_windows_credential", lambda t: None)

    assert supports_file_credential_store() is False
    status = get_credential_status("vision")
    assert status.source == "missing"
    assert status.writable is True


# ── value validation ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "YOUR_DEEPSEEK_API_KEY", "x" * 513, None, 7, ["a"], {"a": 1}],
)
def test_invalid_values_are_rejected_without_touching_the_file(_linux_file_store, bad):
    store_credential("text", TEXT_KEY)

    with pytest.raises(CredentialError):
        store_credential("vision", bad)

    assert _stored(_linux_file_store) == {"text": TEXT_KEY}


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all",
        json.dumps(["text"]),
        json.dumps({"text": ""}),
        json.dumps({"text": "YOUR_DEEPSEEK_API_KEY"}),
        json.dumps({"text": 12345}),
        json.dumps({"text": "x" * 513}),
        json.dumps({"unknown": "value"}),
    ],
)
def test_a_file_that_is_not_a_valid_store_is_never_trusted(_linux_file_store, payload):
    _linux_file_store.parent.mkdir(parents=True, exist_ok=True)
    _linux_file_store.write_text(payload, encoding="utf-8")

    assert resolve_credential("text") is None
    status = get_credential_status("text")
    assert status.configured is False
    assert status.writable is False, "an unreadable store can not be written through"


def test_a_corrupt_file_is_never_overwritten(_linux_file_store):
    _linux_file_store.parent.mkdir(parents=True, exist_ok=True)
    _linux_file_store.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(CredentialError):
        store_credential("text", TEXT_KEY)
    with pytest.raises(CredentialError):
        delete_credential("text")

    assert _linux_file_store.read_text(encoding="utf-8") == "{ this is not json"


def test_an_oversized_file_is_refused(_linux_file_store):
    _linux_file_store.parent.mkdir(parents=True, exist_ok=True)
    _linux_file_store.write_text(
        json.dumps({"text": "x" * (credentials._MAX_FILE_BYTES + 1)}), encoding="utf-8"
    )

    assert resolve_credential("text") is None
    assert get_credential_status("text").writable is False
    with pytest.raises(CredentialError):
        store_credential("text", TEXT_KEY)


def test_the_error_messages_never_carry_a_value_or_a_path(_linux_file_store):
    _linux_file_store.parent.mkdir(parents=True, exist_ok=True)
    _linux_file_store.write_text("{ broken", encoding="utf-8")

    with pytest.raises(CredentialError) as excinfo:
        store_credential("text", TEXT_KEY)

    message = str(excinfo.value)
    assert TEXT_KEY not in message
    assert "credentials.json" not in message
    assert str(_linux_file_store.parent) not in message


# ── durable, concurrent, restricted writes ──────────────────────────────────


def test_an_atomic_replace_failure_keeps_the_previous_credentials(
    _linux_file_store, monkeypatch
):
    store_credential("text", TEXT_KEY)

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(credentials.os, "replace", fail_replace)

    with pytest.raises(CredentialError):
        store_credential("text", "text-key-cccccccc")

    assert _stored(_linux_file_store) == {"text": TEXT_KEY}
    assert resolve_credential("text") == TEXT_KEY


def test_a_failed_write_leaves_no_staging_file_behind(_linux_file_store, monkeypatch):
    store_credential("text", TEXT_KEY)

    monkeypatch.setattr(
        credentials.os, "replace", lambda source, destination: (_ for _ in ()).throw(OSError())
    )
    with pytest.raises(CredentialError):
        store_credential("vision", VISION_KEY)

    leftovers = [item.name for item in _linux_file_store.parent.iterdir()
                 if item.name != "credentials.json"]
    assert leftovers == [], f"a temporary credential file survived: {leftovers}"


def test_a_stored_key_survives_a_fresh_process_level_read(_linux_file_store):
    store_credential("text", TEXT_KEY)
    invalidate_credential_cache()

    assert resolve_credential("text") == TEXT_KEY


def test_a_successful_save_invalidates_the_cached_value(_linux_file_store):
    store_credential("text", TEXT_KEY)
    assert resolve_credential("text") == TEXT_KEY

    store_credential("text", "text-key-cccccccc")

    assert resolve_credential("text") == "text-key-cccccccc"


def test_a_successful_delete_invalidates_the_cached_value(_linux_file_store):
    store_credential("text", TEXT_KEY)
    assert resolve_credential("text") == TEXT_KEY

    delete_credential("text")

    assert resolve_credential("text") is None


def test_concurrent_purpose_writes_do_not_lose_each_other(_linux_file_store, monkeypatch):
    """Read-modify-write is one critical section: the two purposes share a file.

    The barrier only releases once *both* writers have read the file, so the
    write of one purpose begins while the other's read is still open. Without a
    lock around the whole read-modify-write the second write would start from
    the same snapshot and drop the first purpose.
    """
    original_read = credentials._read_file_store
    barrier = threading.Barrier(2, timeout=3)
    first_reader = threading.local()

    def slow_read():
        data = original_read()
        if not getattr(first_reader, "seen", False):
            first_reader.seen = True
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
        return data

    monkeypatch.setattr(credentials, "_read_file_store", slow_read)
    failures: list[BaseException] = []

    def worker(purpose: str, value: str) -> None:
        try:
            store_credential(purpose, value)
        except BaseException as exc:  # noqa: BLE001 - reported as a failure below
            failures.append(exc)

    threads = [
        threading.Thread(target=worker, args=("text", TEXT_KEY)),
        threading.Thread(target=worker, args=("vision", VISION_KEY)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert failures == []
    assert all(not thread.is_alive() for thread in threads)
    assert _stored(_linux_file_store) == {"text": TEXT_KEY, "vision": VISION_KEY}


def test_the_credentials_file_is_restricted_to_the_running_user(_linux_file_store):
    if os.name != "posix":
        pytest.skip("POSIX file modes only")
    store_credential("text", TEXT_KEY)

    mode = stat.S_IMODE(_linux_file_store.stat().st_mode)
    assert mode == 0o600, f"the credential file is group/world readable: {oct(mode)}"


def test_a_chmod_failure_does_not_lose_the_key(_linux_file_store, monkeypatch):
    """A bind-mounted volume may not support chmod: the key still has to be saved."""
    def fail_chmod(path, mode):
        raise OSError("chmod is not supported on this mount")

    monkeypatch.setattr(credentials.os, "chmod", fail_chmod)

    store_credential("text", TEXT_KEY)

    assert resolve_credential("text") == TEXT_KEY


def test_an_unwritable_directory_reports_not_writable(_linux_file_store, monkeypatch):
    monkeypatch.setattr(credentials, "_directory_writable", lambda directory: False)

    status = get_credential_status("text")

    assert status.configured is False
    assert status.writable is False


def test_saving_into_an_unwritable_directory_reports_a_fixed_error(
    _linux_file_store, monkeypatch
):
    monkeypatch.setattr(credentials, "_directory_writable", lambda directory: False)

    with pytest.raises(CredentialError) as excinfo:
        store_credential("text", TEXT_KEY)

    message = str(excinfo.value)
    assert TEXT_KEY not in message
    assert str(_linux_file_store.parent) not in message
    assert _linux_file_store.exists() is False


# ── redaction ───────────────────────────────────────────────────────────────


def test_known_credentials_covers_the_file_store(_linux_file_store):
    store_credential("text", TEXT_KEY)
    store_credential("vision", VISION_KEY)

    known = known_credentials()

    assert TEXT_KEY in known
    assert VISION_KEY in known


def test_known_credentials_is_empty_when_nothing_is_configured(_linux_file_store):
    assert known_credentials() == ()


def test_a_status_never_reveals_the_value(_linux_file_store):
    store_credential("vision", VISION_KEY)

    status = get_credential_status("vision")

    assert VISION_KEY not in repr(status)
    assert VISION_KEY not in str(status)


def test_resolving_a_corrupt_store_never_raises(_linux_file_store):
    """The Studio must keep working without an LLM credential."""
    _linux_file_store.parent.mkdir(parents=True, exist_ok=True)
    _linux_file_store.write_text("<<<", encoding="utf-8")

    assert resolve_credential("text") is None
    assert resolve_credential("vision") is None
    assert known_credentials() == ()
