"""F1.2-D: the privileged container start-up and the drop to ``studio``.

A missing bind-mount host directory is created by the container engine owned by
``root``, so a first ``docker compose up -d`` without any preparation script only
works if something hands those directories to the unprivileged runtime account
before the delivery preflight runs. That something is
``auto_tune/delivery/container_entrypoint.py``, and these tests pin its whole
contract: which directories it may touch, that it touches nothing but their own
inode, that it never widens the permissions of existing data, that the identity
really becomes 10001:10001, and that the preflight and the Studio only ever run
after that.

Every filesystem and kernel call is injectable, so the rules are exercised
without a Linux container, without a root shell and without a real GPU. What
cannot be tested here is stated instead of implied: see the module tests for the
real-filesystem paths, and the report for what was left to a real Docker host.
"""

import os
import re
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from auto_tune.delivery import container_entrypoint as ce
from auto_tune.delivery.container_entrypoint import (
    ALLOWED_DIRECTORIES,
    CONTAINER_DATASETS_UNREADABLE,
    CONTAINER_DIR_NOT_CREATABLE,
    CONTAINER_DIR_NOT_WRITABLE,
    CONTAINER_DIR_PERMISSION_INSUFFICIENT,
    CONTAINER_ENVIRONMENT_UNSUPPORTED,
    CONTAINER_INPUT_ROOTS_INVALID,
    CONTAINER_PATH_NOT_ALLOWED,
    CONTAINER_PERMISSION_CHANGE_FAILED,
    CONTAINER_PREFLIGHT_FAILED,
    CONTAINER_PRIVILEGE_DROP_FAILED,
    CONTAINER_ROOT_REQUIRED,
    CONTAINER_TARGET_NOT_DIRECTORY,
    CONTAINER_TARGET_SYMLINK,
    CONTAINER_TARGET_UNAVAILABLE,
    DIRECTORY_MODE,
    STUDIO_GID,
    STUDIO_UID,
    DeliveryError,
)
from auto_tune.delivery.preflight import DeliveryPaths

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "auto_tune" / "delivery" / "container_entrypoint.py"
ENTRYPOINT = REPO_ROOT / "docker" / "entrypoint.sh"
DOCKERFILE = REPO_ROOT / "Dockerfile"
COMPOSE = REPO_ROOT / "compose.yaml"

CONFIG_DIR = Path("/data/config")
SECRETS_DIR = Path("/data/secrets")
DATASETS_DIR = Path("/data/datasets")

# The six directories the Studio writes into. The dataset share is not one of
# them: it is an input, and nothing about it is ever changed.
OUTPUT_DIRECTORIES = tuple(
    directory for directory in ALLOWED_DIRECTORIES if directory != DATASETS_DIR)

# A host account that is neither root nor the runtime user — the operator's own.
HOST_UID = 1000


def _key(directory) -> str:
    """The path exactly as the module passes it to the filesystem calls."""
    return os.fspath(directory)


def _stat(mode, uid=0, gid=0):
    return os.stat_result((mode, 0, 0, 1, uid, gid, 0, 0, 0, 0))


def _container_layout(**overrides) -> DeliveryPaths:
    layout = DeliveryPaths(
        app_root=Path("/opt/auto-tune"),
        config_path=CONFIG_DIR / "config.yaml",
        template_path=Path("/opt/auto-tune/auto_tune/config.template.yaml"),
        datasets_dir=DATASETS_DIR,
        credentials_path=SECRETS_DIR / "credentials.json",
    )
    return replace(layout, **overrides) if overrides else layout


class _VirtualFilesystem:
    """A mode/owner table standing in for the container's mounted directories.

    ``chown`` is what the container engine would do to a fresh bind directory:
    the directory appears owned by root, and the one privileged step is expected
    to hand exactly that inode to the runtime account.
    """

    def __init__(self, entries=None, deny_chown=()):
        # The caller's dict is kept live: the tests assert the state the module
        # left behind, not a copy taken before it ran.
        self.entries = {} if entries is None else entries
        self.deny_chown = set(deny_chown)
        self.calls: list[tuple] = []

    def lstat(self, path):
        if path not in self.entries:
            raise FileNotFoundError(path)
        return _stat(*self.entries[path])

    def mkdir(self, path, mode):
        self.calls.append(("mkdir", path, mode))
        if path in self.entries:
            raise FileExistsError(path)
        self.entries[path] = (stat.S_IFDIR | mode, 0, 0)

    def chmod(self, path, mode):
        self.calls.append(("chmod", path, mode))
        current, uid, gid = self.entries[path]
        self.entries[path] = (stat.S_IFMT(current) | mode, uid, gid)

    def chown(self, path, uid, gid):
        self.calls.append(("chown", path, uid, gid))
        if path in self.deny_chown:
            raise PermissionError(1, "operation not permitted")
        current, _, _ = self.entries[path]
        self.entries[path] = (current, uid, gid)

    # ── assertions helpers ────────────────────────────────────────────────

    def paths_touched(self) -> set[str]:
        return {call[1] for call in self.calls}

    def calls_of(self, kind: str) -> list[tuple]:
        return [call for call in self.calls if call[0] == kind]


class _RecordingProbes(ce._FilesystemProbes):
    """The real filesystem, with ownership recorded instead of applied.

    ``os.chown`` does not exist on every host this suite runs on, and changing
    the owner of a real test directory would be a side effect the suite has no
    business causing. The mode is still set for real, so the created-directory
    tests exercise the shipping ``mkdir``/``chmod`` calls.
    """

    def __init__(self):
        self.calls: list[tuple] = []
        self._handed: set[str] = set()

    def mkdir(self, path, mode):
        self.calls.append(("mkdir", path, mode))
        super().mkdir(path, mode)

    def chmod(self, path, mode):
        self.calls.append(("chmod", path, mode))
        super().chmod(path, mode)

    def chown(self, path, uid, gid):
        self.calls.append(("chown", path, uid, gid))
        self._handed.add(os.path.normpath(path))

    def lstat(self, path):
        info = super().lstat(path)
        if os.path.normpath(path) in self._handed:
            info = _stat(info.st_mode, STUDIO_UID, STUDIO_GID)
        return info


def _all_directories(*, mode=0o755, uid=0, gid=0) -> dict:
    return {_key(directory): (stat.S_IFDIR | mode, uid, gid)
            for directory in ALLOWED_DIRECTORIES}


def _initialise(filesystem, **kwargs):
    return ce.initialise_directories(
        ALLOWED_DIRECTORIES, datasets_directory=DATASETS_DIR,
        probes=filesystem, **kwargs)


# ── what the privileged step is allowed to touch ─────────────────────────────


def test_the_container_targets_are_exactly_the_declared_directories():
    assert ce.container_targets(_container_layout()) == ALLOWED_DIRECTORIES
    assert tuple(directory.as_posix() for directory in ALLOWED_DIRECTORIES) == (
        "/data/config",
        "/data/secrets",
        "/data/datasets",
        "/opt/auto-tune/log",
        "/opt/auto-tune/detect",
        "/opt/auto-tune/runs",
        "/opt/auto-tune/models/weights",
    )


def test_the_declared_directories_are_the_compose_mount_targets():
    """One layout, defined once: the whitelist cannot drift from compose."""
    service = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]["studio"]
    mounted = {str(volume).split(":")[-1] for volume in service["volumes"]}

    assert mounted == {directory.as_posix() for directory in ALLOWED_DIRECTORIES}


@pytest.mark.parametrize("field, value", [
    ("app_root", Path("/opt/other")),
    ("datasets_dir", Path("/mnt/host-data")),
    ("config_path", Path("/etc/other/config.yaml")),
    ("credentials_path", Path("/home/operator/credentials.json")),
])
def test_a_layout_that_leaves_the_whitelist_is_refused(field, value):
    """An environment variable must never widen what the root step may touch."""
    with pytest.raises(DeliveryError) as excinfo:
        ce.container_targets(_container_layout(**{field: value}))

    assert excinfo.value.code == CONTAINER_PATH_NOT_ALLOWED


def test_a_layout_that_names_one_directory_twice_is_refused():
    """Pointing the dataset share at another declared directory would make that
    directory's read-only contract apply to one the Studio writes into."""
    with pytest.raises(DeliveryError) as excinfo:
        ce.container_targets(_container_layout(datasets_dir=CONFIG_DIR))

    assert excinfo.value.code == CONTAINER_PATH_NOT_ALLOWED


def test_a_layout_without_a_credential_file_drops_the_secrets_directory():
    """The desktop backend has no credential file; the container always has one,
    because the entrypoint supplies the documented default."""
    layout = _container_layout(credentials_path=None)

    assert SECRETS_DIR not in ce.container_targets(layout)
    assert ce.container_targets(layout) == tuple(
        directory for directory in ALLOWED_DIRECTORIES if directory != SECRETS_DIR)


# ── the container start-up context ───────────────────────────────────────────


def test_the_container_start_up_context_is_accepted_on_linux_as_root():
    assert ce.verify_environment(
        platform_name="linux", geteuid=lambda: 0,
        account=lambda: (STUDIO_UID, STUDIO_GID)) is None


@pytest.mark.parametrize("platform_name", ["win32", "darwin"])
def test_a_start_up_outside_the_container_is_refused(platform_name):
    with pytest.raises(DeliveryError) as excinfo:
        ce.verify_environment(platform_name=platform_name, geteuid=lambda: 0,
                              account=lambda: (STUDIO_UID, STUDIO_GID))

    assert excinfo.value.code == CONTAINER_ENVIRONMENT_UNSUPPORTED


def test_a_start_up_without_root_is_refused_instead_of_half_done():
    """An operator override that removed the privileges this step needs must
    stop the start, not leave the directories half-initialised."""
    with pytest.raises(DeliveryError) as excinfo:
        ce.verify_environment(platform_name="linux", geteuid=lambda: 1000,
                              account=lambda: (STUDIO_UID, STUDIO_GID))

    assert excinfo.value.code == CONTAINER_ROOT_REQUIRED


@pytest.mark.parametrize("account", [lambda: None, lambda: (1000, 1000), lambda: (10001, 1000)])
def test_a_missing_or_wrong_runtime_account_is_refused(account):
    with pytest.raises(DeliveryError) as excinfo:
        ce.verify_environment(platform_name="linux", geteuid=lambda: 0, account=account)

    assert excinfo.value.code == CONTAINER_ENVIRONMENT_UNSUPPORTED


def test_the_shipping_account_lookup_answers_with_an_id_pair_or_nothing():
    identity = ce.studio_account()

    assert identity is None or (isinstance(identity, tuple) and len(identity) == 2)


# ── directory initialisation: created, or handed over inode by inode ─────────


def test_directories_that_do_not_exist_are_created_owned_by_the_runtime_user():
    filesystem = _VirtualFilesystem()

    created = _initialise(filesystem)

    assert created == list(ALLOWED_DIRECTORIES)
    for directory in OUTPUT_DIRECTORIES:
        _, uid, gid = filesystem.entries[_key(directory)]
        assert (uid, gid) == (STUDIO_UID, STUDIO_GID)


def test_a_created_dataset_share_is_only_created_never_handed_over():
    """The dataset share is created when it is missing, and that is all: no
    chown, no chmod, and read/traverse is the only contract it has to meet."""
    filesystem = _VirtualFilesystem()

    _initialise(filesystem)

    assert filesystem.entries[_key(DATASETS_DIR)] == (stat.S_IFDIR | DIRECTORY_MODE, 0, 0)
    assert [call for call in filesystem.calls if call[1] == _key(DATASETS_DIR)] == [
        ("mkdir", _key(DATASETS_DIR), DIRECTORY_MODE)]


def test_every_created_directory_uses_the_minimal_mode():
    filesystem = _VirtualFilesystem()

    _initialise(filesystem)

    assert filesystem.calls_of("mkdir") == [
        ("mkdir", _key(directory), DIRECTORY_MODE) for directory in ALLOWED_DIRECTORIES]
    assert DIRECTORY_MODE == 0o755
    assert filesystem.calls_of("chmod") == [
        ("chmod", _key(directory), DIRECTORY_MODE) for directory in OUTPUT_DIRECTORIES]


def test_an_existing_directory_the_runtime_user_cannot_write_is_handed_over():
    """The engine-created case: root-owned and otherwise healthy.

    Only ``root``-owned directories are handed over, and the dataset share is
    never among them."""
    filesystem = _VirtualFilesystem(_all_directories(mode=0o755))

    _initialise(filesystem)

    assert filesystem.calls_of("chown") == [
        ("chown", _key(directory), STUDIO_UID, STUDIO_GID)
        for directory in OUTPUT_DIRECTORIES]
    assert filesystem.calls_of("chmod") == []
    assert filesystem.calls_of("mkdir") == []


def test_an_existing_directory_the_runtime_user_can_write_is_left_alone():
    """Nothing is changed that does not need changing."""
    filesystem = _VirtualFilesystem(
        _all_directories(mode=0o755, uid=STUDIO_UID, gid=STUDIO_GID))

    _initialise(filesystem)

    assert filesystem.calls == []


def test_a_dataset_share_owned_by_a_host_user_is_never_taken_over():
    """A dataset folder a host account owns, with the usual 0755, is readable and
    traversable: the start is allowed and nothing about it is changed."""
    entries = _all_directories(mode=0o755)
    entries[_key(DATASETS_DIR)] = (stat.S_IFDIR | 0o755, HOST_UID, HOST_UID)
    filesystem = _VirtualFilesystem(entries)

    assert _initialise(filesystem) == []
    assert filesystem.entries[_key(DATASETS_DIR)] == \
        (stat.S_IFDIR | 0o755, HOST_UID, HOST_UID)
    assert [call for call in filesystem.calls if call[1] == _key(DATASETS_DIR)] == []


def test_a_read_only_dataset_share_is_a_correct_setup():
    """A dataset share mounted read-only is an input used as intended."""
    entries = _all_directories(mode=0o755)
    entries[_key(DATASETS_DIR)] = (stat.S_IFDIR | 0o555, HOST_UID, HOST_UID)
    filesystem = _VirtualFilesystem(entries)

    assert _initialise(filesystem) == []
    assert filesystem.entries[_key(DATASETS_DIR)] == \
        (stat.S_IFDIR | 0o555, HOST_UID, HOST_UID)
    assert [call for call in filesystem.calls if call[1] == _key(DATASETS_DIR)] == []


@pytest.mark.parametrize("mode", [0o000, 0o100, 0o300, 0o444, 0o500])
def test_a_dataset_share_that_cannot_be_read_or_traversed_stops_the_start(mode):
    """Read *and* traverse are both required; the share is reported, not changed."""
    entries = _all_directories(mode=0o755)
    entries[_key(DATASETS_DIR)] = (stat.S_IFDIR | mode, HOST_UID, HOST_UID)
    filesystem = _VirtualFilesystem(entries)

    with pytest.raises(DeliveryError) as excinfo:
        _initialise(filesystem)

    assert excinfo.value.code == CONTAINER_DATASETS_UNREADABLE
    assert filesystem.entries[_key(DATASETS_DIR)] == \
        (stat.S_IFDIR | mode, HOST_UID, HOST_UID)
    assert [call for call in filesystem.calls if call[1] == _key(DATASETS_DIR)] == []


def test_a_host_owned_directory_the_runtime_user_cannot_use_is_reported_not_taken():
    """An *output* directory a host account owns and the runtime user cannot use
    stays that account's directory: it is refused, not chowned away from it."""
    entries = _all_directories(mode=0o755)
    entries[_key(CONFIG_DIR)] = (stat.S_IFDIR | 0o755, HOST_UID, HOST_UID)
    filesystem = _VirtualFilesystem(entries)

    with pytest.raises(DeliveryError) as excinfo:
        _initialise(filesystem)

    assert excinfo.value.code == CONTAINER_DIR_PERMISSION_INSUFFICIENT
    assert filesystem.entries[_key(CONFIG_DIR)] == \
        (stat.S_IFDIR | 0o755, HOST_UID, HOST_UID)
    assert filesystem.calls == [], "the host account's directory is not touched at all"


def test_a_host_owned_directory_the_runtime_user_can_use_is_left_alone():
    """Already usable by the runtime user, whatever owns it: nothing to do."""
    entries = {_key(DATASETS_DIR): (stat.S_IFDIR | 0o755, HOST_UID, HOST_UID)}
    for directory in OUTPUT_DIRECTORIES:
        entries[_key(directory)] = (stat.S_IFDIR | 0o770, HOST_UID, STUDIO_GID)
    filesystem = _VirtualFilesystem(entries)

    assert _initialise(filesystem) == []
    assert filesystem.calls == []


def test_only_the_directory_inode_is_ever_touched():
    """A populated share is the normal case: the datasets, the training results
    and the weights inside keep the ownership and mode their owner gave them."""
    entries = _all_directories(mode=0o755)
    entries.update({
        f"{_key(DATASETS_DIR)}/img.jpg": (stat.S_IFREG | 0o644, 1234, 1234),
        f"{_key(DATASETS_DIR)}/labels": (stat.S_IFDIR | 0o750, 1234, 1234),
        "/opt/auto-tune/detect/train8/results.csv": (stat.S_IFREG | 0o600, 1234, 1234),
        "/opt/auto-tune/models/weights/best.pt": (stat.S_IFREG | 0o644, 1234, 1234),
        "/opt/auto-tune/log/tuning_audit_s.json": (stat.S_IFREG | 0o600, 1234, 1234),
    })
    filesystem = _VirtualFilesystem(entries)

    _initialise(filesystem)

    assert filesystem.paths_touched() == {_key(d) for d in OUTPUT_DIRECTORIES}
    assert entries[f"{_key(DATASETS_DIR)}/img.jpg"] == (stat.S_IFREG | 0o644, 1234, 1234)
    assert entries[f"{_key(DATASETS_DIR)}/labels"] == (stat.S_IFDIR | 0o750, 1234, 1234)
    assert entries["/opt/auto-tune/detect/train8/results.csv"] == \
        (stat.S_IFREG | 0o600, 1234, 1234)
    assert entries["/opt/auto-tune/models/weights/best.pt"] == \
        (stat.S_IFREG | 0o644, 1234, 1234)
    assert entries["/opt/auto-tune/log/tuning_audit_s.json"] == \
        (stat.S_IFREG | 0o600, 1234, 1234)


def test_no_recursive_or_unbounded_call_is_used():
    """The proof of the two tests above is structural: there is no way to walk
    into a directory, so no call can reach a file inside one."""
    source = MODULE_PATH.read_text(encoding="utf-8")

    for forbidden in ("os.walk", "rglob", "scandir", "shutil.chown", "shutil.copytree",
                      "os.removedirs", "os.rmdir", "os.unlink", "os.remove",
                      "os.rename", "os.replace", "shutil.rmtree", "-R", "glob("):
        assert forbidden not in source, f"the privileged step must not use {forbidden!r}"


def test_a_symlinked_persistent_directory_is_refused():
    entries = _all_directories()
    entries[_key(CONFIG_DIR)] = (stat.S_IFLNK | 0o777, 0, 0)
    filesystem = _VirtualFilesystem(entries)

    with pytest.raises(DeliveryError) as excinfo:
        _initialise(filesystem)

    assert excinfo.value.code == CONTAINER_TARGET_SYMLINK
    assert filesystem.calls_of("chown") == [], (
        "a symlink must be refused before anything is chowned through it")


def test_a_symlinked_persistent_directory_is_refused_before_anything_else():
    """The link is checked on the directory itself, not on a path resolved
    through it, so nothing is created or chowned through a link either."""
    filesystem = _VirtualFilesystem({
        _key(DATASETS_DIR): (stat.S_IFLNK | 0o777, 0, 0)})

    with pytest.raises(DeliveryError) as excinfo:
        ce.initialise_directories([DATASETS_DIR], datasets_directory=None,
                                  probes=filesystem)

    assert excinfo.value.code == CONTAINER_TARGET_SYMLINK
    assert filesystem.calls == []


def test_a_file_standing_in_for_a_persistent_directory_is_refused():
    entries = _all_directories()
    entries[_key(CONFIG_DIR)] = (stat.S_IFREG | 0o644, 0, 0)
    filesystem = _VirtualFilesystem(entries)

    with pytest.raises(DeliveryError) as excinfo:
        _initialise(filesystem)

    assert excinfo.value.code == CONTAINER_TARGET_NOT_DIRECTORY
    assert filesystem.calls == []


def test_a_directory_that_cannot_be_inspected_is_refused():
    """Anything the root step cannot even look at safely stops the start."""
    class _Unreadable(_VirtualFilesystem):
        def lstat(self, path):
            raise PermissionError(13, "permission denied")

    with pytest.raises(DeliveryError) as excinfo:
        _initialise(_Unreadable(_all_directories()))

    assert excinfo.value.code == CONTAINER_TARGET_UNAVAILABLE


def test_a_directory_that_cannot_be_created_is_refused():
    class _Uncreatable(_VirtualFilesystem):
        def mkdir(self, path, mode):
            raise OSError(28, "no space left on device")

    with pytest.raises(DeliveryError) as excinfo:
        _initialise(_Uncreatable())

    assert excinfo.value.code == CONTAINER_DIR_NOT_CREATABLE


def test_a_failed_ownership_change_stops_the_start():
    filesystem = _VirtualFilesystem(_all_directories(mode=0o755),
                                    deny_chown=[_key(CONFIG_DIR)])

    with pytest.raises(DeliveryError) as excinfo:
        _initialise(filesystem)

    assert excinfo.value.code == CONTAINER_PERMISSION_CHANGE_FAILED
    assert filesystem.calls_of("chown") == [
        ("chown", _key(CONFIG_DIR), STUDIO_UID, STUDIO_GID)], (
        "the start stops at the first directory that cannot be handed over")


def test_a_directory_still_unwritable_after_the_handover_stops_the_start():
    """A directory whose owner bits carry no write permission is reported, never
    loosened: the operator's own read-only choice must not be overridden."""
    entries = _all_directories(mode=0o555)
    filesystem = _VirtualFilesystem(entries)

    with pytest.raises(DeliveryError) as excinfo:
        _initialise(filesystem)

    assert excinfo.value.code == CONTAINER_DIR_NOT_WRITABLE
    assert filesystem.entries[_key(CONFIG_DIR)] == \
        (stat.S_IFDIR | 0o555, STUDIO_UID, STUDIO_GID)
    assert filesystem.calls_of("chmod") == []


def test_an_unreadable_dataset_share_is_reported_without_loosening_it():
    """Write and search are not enough to *read* a dataset: the share is an input,
    and a share the runtime user cannot read is reported rather than opened up."""
    entries = _all_directories()
    entries[_key(DATASETS_DIR)] = (stat.S_IFDIR | 0o300, 0, 0)
    filesystem = _VirtualFilesystem(entries)

    with pytest.raises(DeliveryError) as excinfo:
        _initialise(filesystem)

    assert excinfo.value.code == CONTAINER_DATASETS_UNREADABLE
    assert stat.S_IMODE(filesystem.entries[_key(DATASETS_DIR)][0]) == 0o300, (
        "an unreadable dataset share is reported, not opened up")


def test_readability_is_only_enforced_for_the_dataset_share():
    """The dataset share is the one input the product reads without owning, so it
    is the one directory whose readability is checked here for a reason of its
    own. A directory without read permission elsewhere is still handed over and
    reported by the delivery preflight after the drop."""
    entries = _all_directories()
    entries[_key(CONFIG_DIR)] = (stat.S_IFDIR | 0o300, 0, 0)
    filesystem = _VirtualFilesystem(entries)

    assert _initialise(filesystem) == []
    assert filesystem.entries[_key(CONFIG_DIR)] == \
        (stat.S_IFDIR | 0o300, STUDIO_UID, STUDIO_GID)


def test_a_directory_without_search_permission_is_handed_over_and_then_reported():
    """A directory that can be written into but not entered is unusable: it is
    handed over like any other, and the start stops instead of pretending."""
    entries = _all_directories()
    entries[_key(CONFIG_DIR)] = (stat.S_IFDIR | 0o600, 0, 0)
    filesystem = _VirtualFilesystem(entries)

    with pytest.raises(DeliveryError) as excinfo:
        _initialise(filesystem)

    assert excinfo.value.code == CONTAINER_DIR_NOT_WRITABLE
    assert filesystem.entries[_key(CONFIG_DIR)] == \
        (stat.S_IFDIR | 0o600, STUDIO_UID, STUDIO_GID), "the mode is left alone"


# ── the same rules against a real filesystem ─────────────────────────────────


def test_the_shipping_probes_read_the_real_filesystem(tmp_path):
    probes = ce._FilesystemProbes()

    with pytest.raises(FileNotFoundError):
        probes.lstat(str(tmp_path / "missing"))
    probes.mkdir(str(tmp_path / "made"), DIRECTORY_MODE)
    assert (tmp_path / "made").is_dir()


def test_a_created_directory_on_the_real_filesystem_is_empty_and_handed_over(tmp_path):
    directory = tmp_path / "log"
    probes = _RecordingProbes()

    created = ce.initialise_directories([directory], datasets_directory=None,
                                        probes=probes)

    assert created == [directory]
    assert directory.is_dir()
    assert list(directory.iterdir()) == [], "a created directory is empty by definition"
    assert probes.calls == [
        ("mkdir", str(directory), DIRECTORY_MODE),
        ("chmod", str(directory), DIRECTORY_MODE),
        ("chown", str(directory), STUDIO_UID, STUDIO_GID),
    ]


def test_a_real_directory_the_runtime_user_can_write_is_left_alone(tmp_path):
    directory = tmp_path / "detect"
    directory.mkdir()
    inside = directory / "results.csv"
    inside.write_text("epoch,metrics/mAP50(B)\n", encoding="utf-8")
    os.chmod(directory, 0o777)
    probes = _RecordingProbes()

    created = ce.initialise_directories([directory], datasets_directory=None,
                                        probes=probes)

    assert created == []
    assert probes.calls == [], "a world-writable directory needs no privileged step"
    assert inside.read_text(encoding="utf-8") == "epoch,metrics/mAP50(B)\n"


# ── the drop to the runtime account ─────────────────────────────────────────


def _drop_recorder(state=None):
    state = {"uid": 0, "gid": 0} if state is None else state
    calls: list[tuple] = []

    def setgroups(groups):
        calls.append(("setgroups", groups))

    def setgid(gid):
        calls.append(("setgid", gid))
        state["gid"] = gid

    def setuid(uid):
        calls.append(("setuid", uid))
        state["uid"] = uid

    return calls, {
        "setgroups": setgroups,
        "setgid": setgid,
        "setuid": setuid,
        "getuid": lambda: state["uid"],
        "getgid": lambda: state["gid"],
    }


def test_the_drop_drops_supplementary_groups_then_gid_then_uid():
    """Order matters: supplementary groups can only be dropped while the process
    is still privileged, and the GID must be set before the UID."""
    calls, handlers = _drop_recorder()

    ce.drop_privileges(**handlers)

    assert calls == [("setgroups", []), ("setgid", STUDIO_GID), ("setuid", STUDIO_UID)]


def test_the_drop_is_refused_when_the_identity_did_not_change():
    """A setuid that quietly did nothing would otherwise leave business code
    running as root."""
    _, handlers = _drop_recorder(state={"uid": 0, "gid": 0})
    handlers["setuid"] = lambda uid: None
    handlers["setgid"] = lambda gid: None

    with pytest.raises(DeliveryError) as excinfo:
        ce.drop_privileges(**handlers)

    assert excinfo.value.code == CONTAINER_PRIVILEGE_DROP_FAILED


@pytest.mark.parametrize("step", ["setgroups", "setgid", "setuid"])
def test_the_drop_is_refused_when_the_kernel_rejects_it(step):
    _, handlers = _drop_recorder()

    def denied(*args):
        raise PermissionError(1, "operation not permitted")

    handlers[step] = denied

    with pytest.raises(DeliveryError) as excinfo:
        ce.drop_privileges(**handlers)

    assert excinfo.value.code == CONTAINER_PRIVILEGE_DROP_FAILED


def test_the_drop_uses_only_what_the_image_already_has():
    source = MODULE_PATH.read_text(encoding="utf-8")

    for forbidden in ("gosu", "su-exec", "suexec"):
        assert forbidden not in source
    assert "setgroups" in source and "setgid" in source and "setuid" in source


# ── hand-over: preflight, then the Studio, both unprivileged ────────────────


def test_the_preflight_runs_only_after_the_drop():
    seen: list[tuple] = []

    def runner(argv, check):
        seen.append((tuple(argv), check))
        return subprocess.CompletedProcess(argv, 0)

    ce.run_preflight(runner=runner, geteuid=lambda: STUDIO_UID)

    assert seen == [(tuple(ce.build_preflight_argv()), False)], (
        "the preflight runs as the dropped identity, in this process's child")


def test_the_preflight_is_refused_from_a_privileged_process():
    def runner(argv, check):  # pragma: no cover - must not be reached
        raise AssertionError("a root start must not reach the preflight")

    with pytest.raises(DeliveryError) as excinfo:
        ce.run_preflight(runner=runner, geteuid=lambda: 0)

    assert excinfo.value.code == CONTAINER_PRIVILEGE_DROP_FAILED


def test_a_failed_preflight_stops_the_start():
    def runner(argv, check):
        return subprocess.CompletedProcess(argv, 1)

    with pytest.raises(DeliveryError) as excinfo:
        ce.run_preflight(runner=runner, geteuid=lambda: STUDIO_UID)

    assert excinfo.value.code == CONTAINER_PREFLIGHT_FAILED


def test_the_studio_is_never_started_from_a_privileged_process():
    executed: list[tuple] = []

    with pytest.raises(DeliveryError) as excinfo:
        ce.exec_studio(execv=lambda path, argv: executed.append((path, argv)),
                       geteuid=lambda: 0)

    assert excinfo.value.code == CONTAINER_PRIVILEGE_DROP_FAILED
    assert executed == [], "the business process must not run as root"


def test_the_studio_is_executed_with_a_fixed_argument_array():
    executed: list[tuple] = []

    ce.exec_studio(execv=lambda path, argv: executed.append((path, tuple(argv))),
                   geteuid=lambda: STUDIO_UID)

    assert executed == [(sys.executable, (sys.executable, "-m", "auto_tune.main"))]


def test_the_commands_are_fixed_arrays_and_never_shell_strings():
    assert ce.build_studio_argv() == [sys.executable, "-m", "auto_tune.main"]
    assert ce.build_preflight_argv() == [
        sys.executable, "-m", "auto_tune.delivery.preflight",
        "--require-mounts", "--require-gpu", "--bootstrap-config"]
    assert all(isinstance(argument, str) for argument in ce.build_preflight_argv())

    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in ("shell=True", "os.system", "os.popen", "Popen", "eval(", "exec("):
        assert forbidden not in source, f"the privileged step must not use {forbidden!r}"


def test_the_studio_replaces_the_process_so_pid_one_keeps_its_signals():
    """``execv`` keeps the process identity: the container's PID 1 stays the
    Python process, so ``docker stop`` reaches the Studio's signal handling
    instead of being swallowed by a shell parent."""
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "os.execv" in source
    assert "Popen" not in source
    assert ENTRYPOINT.read_text(encoding="utf-8").count("exec ") == 1


def test_the_studio_is_not_started_when_the_initialisation_failed(monkeypatch):
    monkeypatch.setattr(ce, "run", lambda argv=None: 1)
    monkeypatch.setattr(ce, "exec_studio",
                        lambda: pytest.fail("the Studio must not start"))

    assert ce.main([]) == 1


def test_the_start_up_order_is_initialise_drop_preflight_studio(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(ce, "verify_environment", lambda: events.append("verify"))
    monkeypatch.setattr(ce, "verify_input_roots",
                        lambda: events.append("input_roots"))
    monkeypatch.setattr(ce, "resolve_paths", lambda: _container_layout())
    monkeypatch.setattr(ce, "container_targets",
                        lambda layout: (events.append("targets"), ALLOWED_DIRECTORIES)[1])
    monkeypatch.setattr(ce, "initialise_directories",
                        lambda targets, **kwargs: events.append("initialise") or [])
    monkeypatch.setattr(ce, "drop_privileges", lambda: events.append("drop"))
    monkeypatch.setattr(ce, "run_preflight", lambda: events.append("preflight"))
    monkeypatch.setattr(ce, "exec_studio", lambda: events.append("studio"))

    assert ce.main([]) == 0

    assert events == ["verify", "input_roots", "targets", "initialise", "drop",
                      "preflight", "studio"]


# ── the browse roots the start refuses to run with ──────────────────────────


def _set_browse_roots(monkeypatch, value: str) -> None:
    from auto_tune.delivery import runtime

    monkeypatch.setenv(runtime.INPUT_ALLOWED_ROOTS_ENV, value)


def test_the_declared_browse_roots_are_accepted(monkeypatch):
    from auto_tune.delivery.runtime import CONTAINER_INPUT_ROOTS

    _set_browse_roots(monkeypatch, ";".join(CONTAINER_INPUT_ROOTS))

    assert ce.verify_input_roots() is None


@pytest.mark.parametrize("value", [
    "/data/datasets",
    "/data/datasets/",
    "/opt/auto-tune/detect",
    "/opt/auto-tune/detect/",
    "/data/datasets;/data/datasets/",
])
def test_a_container_missing_either_browse_root_stops_the_start(monkeypatch, value):
    """Both declared roots are required, not any subset of them.

    A container that declares only one — or the same one twice — would ship a
    picker that can never reach the other directory the delivery declares, so
    the start stops instead of running half-configured.
    """
    _set_browse_roots(monkeypatch, value)

    with pytest.raises(DeliveryError) as excinfo:
        ce.verify_input_roots()

    assert excinfo.value.code == CONTAINER_INPUT_ROOTS_INVALID


def test_a_resolver_that_lost_a_root_stops_the_start():
    """The start-up checks the accepted set itself, so a resolver that dropped a
    root is refused here rather than believed."""
    from auto_tune.delivery.runtime import CONTAINER_INPUT_ROOTS

    with pytest.raises(DeliveryError) as excinfo:
        ce.verify_input_roots(resolver=lambda: (Path(CONTAINER_INPUT_ROOTS[0]),))

    assert excinfo.value.code == CONTAINER_INPUT_ROOTS_INVALID


def test_a_resolver_returning_both_roots_in_any_order_is_accepted():
    from auto_tune.delivery.runtime import CONTAINER_INPUT_ROOTS

    roots = tuple(Path(root) for root in reversed(CONTAINER_INPUT_ROOTS))

    assert ce.verify_input_roots(resolver=lambda: roots) is None


def test_a_resolver_returning_nothing_stops_the_start():
    with pytest.raises(DeliveryError) as excinfo:
        ce.verify_input_roots(resolver=lambda: ())

    assert excinfo.value.code == CONTAINER_INPUT_ROOTS_INVALID


@pytest.mark.parametrize("value", [
    "/opt/auto-tune/runs",
    "/data/secrets",
    "/opt/auto-tune/log",
    "/etc",
    "relative/path",
])
def test_a_browse_root_the_product_would_refuse_stops_the_start(monkeypatch, value):
    """A value the picker would refuse at click time must stop the start instead,
    and the message must never repeat the value."""
    _set_browse_roots(monkeypatch, value)

    with pytest.raises(DeliveryError) as excinfo:
        ce.verify_input_roots()

    assert excinfo.value.code == CONTAINER_INPUT_ROOTS_INVALID
    assert value not in excinfo.value.message
    assert value not in str(excinfo.value)


def test_a_container_that_declares_no_browse_root_stops_the_start(monkeypatch):
    """The container always declares them (compose and the image entrypoint both
    do), so an absent value is a misconfiguration rather than the desktop
    default, and it would leave the operator with a picker that lists nothing."""
    from auto_tune.delivery import runtime

    monkeypatch.delenv(runtime.INPUT_ALLOWED_ROOTS_ENV, raising=False)

    with pytest.raises(DeliveryError) as excinfo:
        ce.verify_input_roots()

    assert excinfo.value.code == CONTAINER_INPUT_ROOTS_INVALID


def test_the_browse_roots_are_checked_before_anything_is_touched(monkeypatch):
    """The check runs before the layout is resolved, so an invalid value can
    never reach the one privileged step."""
    monkeypatch.setattr(ce, "verify_environment", lambda: None)
    monkeypatch.setattr(ce, "resolve_paths",
                        lambda: pytest.fail("no layout may be resolved"))
    monkeypatch.setattr(ce, "container_targets",
                        lambda layout: pytest.fail("no target may be inspected"))
    monkeypatch.setattr(ce, "initialise_directories",
                        lambda *args, **kwargs: pytest.fail("nothing may be touched"))
    _set_browse_roots(monkeypatch, "/opt/auto-tune/runs")

    assert ce.run([]) == 1


def test_the_shipped_entrypoint_declares_the_browse_roots_the_check_accepts(monkeypatch):
    """The value the image exports is the value the start-up accepts: a default
    that drifted from the whitelist would make every container start fail."""
    from auto_tune.delivery.runtime import CONTAINER_INPUT_ROOTS

    text = ENTRYPOINT.read_text(encoding="utf-8")
    declared = re.search(
        r'AUTO_TUNE_INPUT_ALLOWED_ROOTS="\$\{AUTO_TUNE_INPUT_ALLOWED_ROOTS:-([^}]*)\}"',
        text)
    assert declared, "the entrypoint must default the browse roots"
    assert declared.group(1).split(";") == list(CONTAINER_INPUT_ROOTS)

    _set_browse_roots(monkeypatch, declared.group(1))

    assert ce.verify_input_roots() is None


# ── the start that is not the container's, and what it prints ───────────────


@pytest.mark.skipif(
    sys.platform.startswith("linux") and getattr(os, "geteuid", lambda: -1)() == 0,
    reason="this host runs the suite as root on Linux: the real start would modify directories")
def test_the_container_start_refuses_to_run_outside_the_container(capsys):
    """On a developer host the module must refuse, exit 1 and print exactly one
    stable code — never a traceback and never a partial initialisation."""
    assert ce.run([]) == 1

    captured = capsys.readouterr()
    assert CONTAINER_ENVIRONMENT_UNSUPPORTED in captured.err
    assert "Traceback" not in captured.err
    assert "credentials.json" not in captured.out + captured.err
    assert "Traceback" not in captured.out


def test_the_failure_codes_are_a_fixed_documented_set():
    """Every stop is reported as one of these codes; a new ad-hoc string would
    change the operator-facing contract silently."""
    declared = set(re.findall(r'^(CONTAINER_[A-Z_]+) = "', MODULE_PATH.read_text(
        encoding="utf-8"), re.MULTILINE))

    assert declared == {
        "CONTAINER_ENVIRONMENT_UNSUPPORTED",
        "CONTAINER_ROOT_REQUIRED",
        "CONTAINER_PATH_NOT_ALLOWED",
        "CONTAINER_TARGET_UNAVAILABLE",
        "CONTAINER_TARGET_SYMLINK",
        "CONTAINER_TARGET_NOT_DIRECTORY",
        "CONTAINER_DIR_NOT_CREATABLE",
        "CONTAINER_DIR_NOT_WRITABLE",
        "CONTAINER_DIR_PERMISSION_INSUFFICIENT",
        "CONTAINER_DATASETS_UNREADABLE",
        "CONTAINER_INPUT_ROOTS_INVALID",
        "CONTAINER_PERMISSION_CHANGE_FAILED",
        "CONTAINER_PRIVILEGE_DROP_FAILED",
        "CONTAINER_PREFLIGHT_FAILED",
    }


def test_a_failure_message_never_carries_a_credential_or_a_traceback():
    with pytest.raises(DeliveryError) as excinfo:
        ce.container_targets(_container_layout(app_root=Path("/opt/elsewhere")))

    message = f"{excinfo.value.code}: {excinfo.value.message}"
    assert "credentials.json" not in message
    assert "Traceback" not in message


def test_the_delivery_preflight_still_owns_the_rules_that_follow_the_drop():
    """The privileged step must not re-implement a delivery rule: mount contract,
    write permission, configuration bootstrap and the GPU stay in the preflight,
    which the entrypoint runs after dropping privileges."""
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "auto_tune.delivery.preflight" in source
    for rule in ("--require-mounts", "--require-gpu", "--bootstrap-config"):
        assert rule in source


# ── hygiene ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", [MODULE_PATH, ENTRYPOINT, DOCKERFILE, COMPOSE])
def test_no_world_writable_mode_is_ever_set(path):
    assert "777" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", [MODULE_PATH, ENTRYPOINT, DOCKERFILE, COMPOSE,
                                 REPO_ROOT / "docker" / "requirements-runtime.txt"])
def test_no_gosu_or_su_exec_dependency_is_added(path):
    text = path.read_text(encoding="utf-8").lower()

    assert "gosu" not in text
    assert "su-exec" not in text


def test_the_privileged_step_never_reads_a_credential_or_writes_a_file():
    source = MODULE_PATH.read_text(encoding="utf-8")

    for forbidden in ("credentials.json", "read_text", "open(", "shutil.copy",
                      "json.load"):
        assert forbidden not in source, f"the privileged step must not {forbidden!r}"
    assert "credentials_dir" in source, "the credential *directory* is initialised"
