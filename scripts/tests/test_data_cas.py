"""Tests for signed content-addressed data activation."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path

import pytest

from scripts import data_cas


def _manifest(file_path: str, content: bytes, bundle_path: str = "manifest.sigstore.json") -> dict:
    bundle_sha = hashlib.sha256(b"verified-bundle").hexdigest()
    return {
        "schema_version": "skinscout.data_cas.v1",
        "bundle": "skinscout-data-core",
        "version": "1.0.0",
        "image_compatibility": ["skinscout-app@sha256:test"],
        "signature": {
            "policy": "cosign-keyless-or-key-pinned",
            "certificate_identity": "release@skinscout.example",
            "issuer": "https://token.actions.githubusercontent.com",
            "bundle_path": bundle_path,
            "bundle_sha256": bundle_sha,
        },
        "packs": [
            {
                "name": "core",
                "version": "1.0.0",
                "files": [
                    {
                        "path": file_path,
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "size": len(content),
                        "license_spdx": "CC-BY-4.0",
                        "redistribution": True,
                    }
                ],
            }
        ],
    }


def _verifier(calls: list[bytes] | None = None):
    def verify(
        manifest_bytes: bytes,
        signature: dict,
        manifest_path: Path | None = None,
    ) -> None:
        assert b'"signature"' not in manifest_bytes
        assert signature["certificate_identity"] == "release@skinscout.example"
        if calls is not None:
            calls.append(manifest_bytes)

    return verify


def _trust_policy() -> dict:
    return {
        "schema_version": "skinscout.data_trust_policy.v1",
        "fail_closed": True,
        "verification": "cosign",
        "trusted_signers": [
            {
                "type": "keyless",
                "certificate_identity": "release@skinscout.example",
                "issuer": "https://token.actions.githubusercontent.com",
            }
        ],
    }


def test_manifest_validation_fails_closed_without_signature() -> None:
    manifest = _manifest("models/a.txt", b"abc")
    del manifest["signature"]

    with pytest.raises(data_cas.CasError, match="missing required fields"):
        data_cas.validate_manifest(manifest)


def test_manifest_validation_requires_redistribution_metadata() -> None:
    manifest = _manifest("models/a.txt", b"abc")
    del manifest["packs"][0]["files"][0]["license_spdx"]

    with pytest.raises(data_cas.CasError, match="redistribution/license"):
        data_cas.validate_manifest(manifest, verifier=_verifier())


def test_manifest_validation_rejects_placeholder_digests() -> None:
    manifest = _manifest("models/a.txt", b"abc")
    manifest["packs"][0]["files"][0]["sha256"] = "0" * 64

    with pytest.raises(data_cas.CasError, match="invalid sha256"):
        data_cas.validate_manifest(manifest, verifier=_verifier())


def test_load_manifest_verifies_canonical_bytes_and_bundle_digest(tmp_path: Path) -> None:
    manifest = _manifest("models/a.txt", b"abc")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    calls: list[bytes] = []

    loaded = data_cas.load_manifest(manifest_path, verifier=_verifier(calls))

    assert loaded["bundle"] == "skinscout-data-core"
    assert len(calls) == 1
    assert calls[0] == data_cas._canonical_manifest_bytes(manifest)


def test_default_manifest_verification_requires_external_trust_policy(
    tmp_path: Path,
) -> None:
    manifest = _manifest("models/a.txt", b"abc")
    bundle = tmp_path / "manifest.sigstore.json"
    bundle.write_bytes(b"verified-bundle")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(data_cas.CasError, match="external trust policy is required"):
        data_cas.load_manifest(manifest_path)


def test_manifest_cannot_replace_trusted_keyless_identity(tmp_path: Path) -> None:
    manifest = _manifest("models/a.txt", b"abc")
    manifest["signature"]["certificate_identity"] = "<개인 주소>"
    bundle = tmp_path / "manifest.sigstore.json"
    bundle.write_bytes(b"verified-bundle")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(data_cas.CasError, match="signer is not trusted"):
        data_cas.load_manifest(
            manifest_path,
            trust_policy=_trust_policy(),
            trust_policy_path=tmp_path / "trust-policy.json",
        )


def test_manifest_cannot_install_attacker_controlled_public_key(tmp_path: Path) -> None:
    manifest = _manifest("models/a.txt", b"abc")
    attacker_key = tmp_path / "attacker.pub"
    attacker_key.write_text("attacker key\n", encoding="utf-8")
    manifest["signature"].pop("certificate_identity")
    manifest["signature"].pop("issuer")
    manifest["signature"]["public_key"] = attacker_key.name
    manifest["signature"]["public_key_sha256"] = hashlib.sha256(
        attacker_key.read_bytes()
    ).hexdigest()
    bundle = tmp_path / "manifest.sigstore.json"
    bundle.write_bytes(b"verified-bundle")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(data_cas.CasError, match="signer is not trusted"):
        data_cas.load_manifest(
            manifest_path,
            trust_policy=_trust_policy(),
            trust_policy_path=tmp_path / "trust-policy.json",
        )


def test_cosign_uses_external_trusted_identity(tmp_path: Path, monkeypatch) -> None:
    manifest = _manifest("models/a.txt", b"abc")
    bundle = tmp_path / "manifest.sigstore.json"
    bundle.write_bytes(b"verified-bundle")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, "verified", "")

    monkeypatch.setattr(data_cas.subprocess, "run", fake_run)

    loaded = data_cas.load_manifest(
        manifest_path,
        trust_policy=_trust_policy(),
        trust_policy_path=tmp_path / "trust-policy.json",
    )

    assert loaded["bundle"] == "skinscout-data-core"
    assert len(commands) == 1
    command = commands[0]
    identity_index = command.index("--certificate-identity")
    issuer_index = command.index("--certificate-oidc-issuer")
    assert command[identity_index + 1] == _trust_policy()["trusted_signers"][0][
        "certificate_identity"
    ]
    assert command[issuer_index + 1] == _trust_policy()["trusted_signers"][0]["issuer"]


def test_offline_import_verifies_before_atomic_activation_and_rollback(
    tmp_path: Path,
) -> None:
    content_v1 = b"version-one"
    manifest_v1 = _manifest("models/a.txt", content_v1)
    offline_v1 = tmp_path / "offline-v1"
    (offline_v1 / "models").mkdir(parents=True)
    (offline_v1 / "models/a.txt").write_bytes(content_v1)

    store = tmp_path / "store"
    staged_v1 = data_cas.stage_pack(
        manifest_v1,
        "core",
        store=store,
        offline_dir=offline_v1,
        verifier=_verifier(),
    )
    active_v1 = data_cas.activate(staged_v1, store=store, verifier=_verifier())
    assert (store / "current").is_symlink()
    assert (store / "current").resolve() == active_v1

    content_v2 = b"version-two"
    manifest_v2 = _manifest("models/a.txt", content_v2)
    manifest_v2["packs"][0]["version"] = "1.0.1"
    offline_v2 = tmp_path / "offline-v2"
    (offline_v2 / "models").mkdir(parents=True)
    (offline_v2 / "models/a.txt").write_bytes(content_v2)
    staged_v2 = data_cas.stage_pack(
        manifest_v2,
        "core",
        store=store,
        offline_dir=offline_v2,
        verifier=_verifier(),
    )
    active_v2 = data_cas.activate(staged_v2, store=store, verifier=_verifier())
    assert (store / "current").resolve() == active_v2

    rolled_back = data_cas.rollback(store=store)
    assert rolled_back == active_v1
    assert ((store / "current").resolve() / "models/a.txt").read_bytes() == content_v1


def test_reactivating_identical_pack_never_deletes_active_destination(
    tmp_path: Path,
    monkeypatch,
) -> None:
    content = b"stable-version"
    manifest = _manifest("models/a.txt", content)
    offline = tmp_path / "offline"
    (offline / "models").mkdir(parents=True)
    (offline / "models/a.txt").write_bytes(content)
    store = tmp_path / "store"
    first_staged = data_cas.stage_pack(
        manifest,
        "core",
        store=store,
        offline_dir=offline,
        verifier=_verifier(),
    )
    active = data_cas.activate(first_staged, store=store, verifier=_verifier())
    second_staged = data_cas.stage_pack(
        manifest,
        "core",
        store=store,
        offline_dir=offline,
        verifier=_verifier(),
    )

    def fail_replace(*_args, **_kwargs):
        raise AssertionError("identical active pack must be reused without replacement")

    monkeypatch.setattr(data_cas.os, "replace", fail_replace)

    reused = data_cas.activate(second_staged, store=store, verifier=_verifier())

    assert reused == active
    assert (store / "current").resolve() == active
    assert (active / "models/a.txt").read_bytes() == content


def test_failed_current_link_swap_preserves_previous_active_pack(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = tmp_path / "store"
    content_v1 = b"version-one"
    manifest_v1 = _manifest("models/a.txt", content_v1)
    offline_v1 = tmp_path / "offline-v1"
    (offline_v1 / "models").mkdir(parents=True)
    (offline_v1 / "models/a.txt").write_bytes(content_v1)
    active_v1 = data_cas.activate(
        data_cas.stage_pack(
            manifest_v1,
            "core",
            store=store,
            offline_dir=offline_v1,
            verifier=_verifier(),
        ),
        store=store,
        verifier=_verifier(),
    )

    content_v2 = b"version-two"
    manifest_v2 = _manifest("models/a.txt", content_v2)
    manifest_v2["packs"][0]["version"] = "2.0.0"
    offline_v2 = tmp_path / "offline-v2"
    (offline_v2 / "models").mkdir(parents=True)
    (offline_v2 / "models/a.txt").write_bytes(content_v2)
    staged_v2 = data_cas.stage_pack(
        manifest_v2,
        "core",
        store=store,
        offline_dir=offline_v2,
        verifier=_verifier(),
    )
    real_replace = data_cas.os.replace

    def fail_current_swap(source, destination):
        if Path(destination) == store / "current":
            raise OSError("simulated current-link swap failure")
        return real_replace(source, destination)

    monkeypatch.setattr(data_cas.os, "replace", fail_current_swap)

    with pytest.raises(OSError, match="simulated current-link swap failure"):
        data_cas.activate(staged_v2, store=store, verifier=_verifier())

    assert (store / "current").is_symlink()
    assert (store / "current").resolve() == active_v1
    assert (active_v1 / "models/a.txt").read_bytes() == content_v1


def test_offline_import_rejects_hash_mismatch_before_activation(tmp_path: Path) -> None:
    manifest = _manifest("models/a.txt", b"expected")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    offline = tmp_path / "offline"
    (offline / "models").mkdir(parents=True)
    (offline / "models/a.txt").write_bytes(b"tampered")

    with pytest.raises(data_cas.CasError, match="size mismatch|sha256 mismatch"):
        data_cas.stage_pack(
            data_cas.load_manifest(manifest_path, verifier=_verifier()),
            "core",
            store=tmp_path / "store",
            offline_dir=offline,
            verifier=_verifier(),
        )

    assert not (tmp_path / "store/current").exists()


def test_activate_rejects_symlink_or_non_staging_pack(tmp_path: Path) -> None:
    store = tmp_path / "store"
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(data_cas.CasError, match="store/staging"):
        data_cas.activate(outside, store=store, verifier=_verifier())

    staging = store / "staging"
    staging.mkdir(parents=True)
    link = staging / "core@1.0.0"
    link.symlink_to(outside)
    with pytest.raises(data_cas.CasError, match="non-symlink staging"):
        data_cas.activate(link, store=store, verifier=_verifier())


def test_activate_rejects_symlink_file_in_verified_staging(tmp_path: Path) -> None:
    content = b"real-data"
    manifest = _manifest("models/a.txt", content)
    staging = tmp_path / "store/staging/core@1.0.0"
    (staging / "models").mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(content)
    (staging / "models/a.txt").symlink_to(outside)
    (staging / ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(data_cas.CasError, match="must not be a symlink"):
        data_cas.activate(staging, store=tmp_path / "store", verifier=_verifier())


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, *, status: int, content_range: str | None = None):
        super().__init__(payload)
        self.status = status
        self.headers = {}
        if content_range is not None:
            self.headers["Content-Range"] = content_range

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False

    def getcode(self) -> int:
        return self.status


def test_resume_requires_206_content_range_and_restarts_on_200(
    tmp_path: Path,
    monkeypatch,
) -> None:
    content = b"abcdef"
    entry = {
        "path": "models/a.txt",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
        "urls": ["https://example.test/a.txt"],
    }
    staging = tmp_path / "staging"
    partial = staging / "models/a.txt.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"abc")
    requests: list[object] = []
    responses = [
        _Response(content, status=200),
        _Response(content, status=200),
    ]

    def fake_urlopen(request, timeout=60):
        requests.append(request)
        return responses.pop(0)

    monkeypatch.setattr(data_cas.urllib.request, "urlopen", fake_urlopen)

    target = data_cas.download_entry(entry, staging=staging)

    assert target.read_bytes() == content
    assert len(requests) == 2
    assert requests[0].headers["Range"] == "bytes=3-"
    assert "Range" not in requests[1].headers


def test_download_rejects_external_partial_symlink_without_touching_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    content = b"abcdef"
    entry = {
        "path": "models/a.txt",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
        "urls": ["https://example.test/a.txt"],
    }
    staging = tmp_path / "staging"
    partial = staging / "models/a.txt.part"
    partial.parent.mkdir(parents=True)
    outside = tmp_path / "outside.part"
    outside.write_bytes(b"external-secret")
    partial.symlink_to(outside)

    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("symlinked partial must be rejected before download")

    monkeypatch.setattr(data_cas.urllib.request, "urlopen", fail_urlopen)

    with pytest.raises(data_cas.CasError, match="partial download must not be a symlink"):
        data_cas.download_entry(entry, staging=staging)

    assert partial.is_symlink()
    assert outside.read_bytes() == b"external-secret"


def test_resume_rejects_mismatched_206_content_range(
    tmp_path: Path,
    monkeypatch,
) -> None:
    entry = {
        "path": "models/a.txt",
        "sha256": hashlib.sha256(b"abcdef").hexdigest(),
        "size": 6,
        "urls": ["https://example.test/a.txt"],
    }
    partial = tmp_path / "staging/models/a.txt.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"abc")

    monkeypatch.setattr(
        data_cas.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(
            b"def",
            status=206,
            content_range="bytes 2-5/6",
        ),
    )

    with pytest.raises(data_cas.CasError, match="206 with matching Content-Range"):
        data_cas.download_entry(entry, staging=tmp_path / "staging")


def test_activate_rejects_external_current_symlink_before_preserving(
    tmp_path: Path,
) -> None:
    content = b"real-data"
    manifest = _manifest("models/a.txt", content)
    store = tmp_path / "store"
    staging = store / "staging/core@1.0.0"
    (staging / "models").mkdir(parents=True)
    (staging / "models/a.txt").write_bytes(content)
    (staging / ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outside = tmp_path / "outside-pack"
    outside.mkdir()
    (outside / "sentinel.txt").write_text("external", encoding="utf-8")
    current = store / "current"
    current.symlink_to(outside)

    with pytest.raises(data_cas.CasError, match="current data pack target must remain under store/packs"):
        data_cas.activate(staging, store=store, verifier=_verifier())

    assert current.is_symlink()
    assert current.resolve() == outside
    assert not (store / "previous").exists()
    assert (outside / "sentinel.txt").read_text(encoding="utf-8") == "external"


def test_rollback_rejects_external_previous_symlink_without_touching_current(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    valid_pack = store / "packs/core@1.0.0"
    valid_pack.mkdir(parents=True)
    current = store / "current"
    current.symlink_to(valid_pack)
    outside = tmp_path / "outside-pack"
    outside.mkdir()
    (outside / "sentinel.txt").write_text("external", encoding="utf-8")
    previous = store / "previous"
    previous.symlink_to(outside)

    with pytest.raises(data_cas.CasError, match="previous data pack target must remain under store/packs"):
        data_cas.rollback(store=store)

    assert current.is_symlink()
    assert current.resolve() == valid_pack
    assert previous.is_symlink()
    assert previous.resolve() == outside
    assert (outside / "sentinel.txt").read_text(encoding="utf-8") == "external"


def test_activate_rejects_external_previous_symlink_before_replacing_it(
    tmp_path: Path,
) -> None:
    content = b"real-data"
    manifest = _manifest("models/a.txt", content)
    store = tmp_path / "store"
    staging = store / "staging/core@1.0.0"
    (staging / "models").mkdir(parents=True)
    (staging / "models/a.txt").write_bytes(content)
    (staging / ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outside = tmp_path / "outside-pack"
    outside.mkdir()
    previous = store / "previous"
    previous.symlink_to(outside)

    with pytest.raises(data_cas.CasError, match="previous data pack target must remain under store/packs"):
        data_cas.activate(staging, store=store, verifier=_verifier())

    assert previous.is_symlink()
    assert previous.resolve() == outside
    assert staging.exists()


def test_rollback_rejects_external_current_symlink_without_switching_it(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    valid_pack = store / "packs/core@1.0.0"
    valid_pack.mkdir(parents=True)
    previous = store / "previous"
    previous.symlink_to(valid_pack)
    outside = tmp_path / "outside-pack"
    outside.mkdir()
    current = store / "current"
    current.symlink_to(outside)

    with pytest.raises(data_cas.CasError, match="current data pack target must remain under store/packs"):
        data_cas.rollback(store=store)

    assert current.is_symlink()
    assert current.resolve() == outside
    assert previous.is_symlink()
    assert previous.resolve() == valid_pack


def test_rollback_rejects_symlinked_packs_root_even_when_target_resolves_below_it(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    store.mkdir()
    outside_packs = tmp_path / "outside-packs"
    previous_pack = outside_packs / "core@1.0.0"
    current_pack = outside_packs / "core@2.0.0"
    previous_pack.mkdir(parents=True)
    current_pack.mkdir()
    (store / "packs").symlink_to(outside_packs)
    previous = store / "previous"
    previous.symlink_to(previous_pack)
    current = store / "current"
    current.symlink_to(current_pack)

    with pytest.raises(data_cas.CasError, match="store/packs must not be a symlink"):
        data_cas.rollback(store=store)

    assert current.resolve() == current_pack
    assert previous.resolve() == previous_pack


def test_repair_rejects_external_current_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    content = b"external-data"
    manifest = _manifest("models/a.txt", content)
    store = tmp_path / "store"
    store.mkdir()
    outside = tmp_path / "core@1.0.0"
    (outside / "models").mkdir(parents=True)
    external_file = outside / "models/a.txt"
    external_file.write_bytes(content)
    (store / "current").symlink_to(outside)

    with pytest.raises(data_cas.CasError, match="current data pack target must remain under store/packs"):
        data_cas.repair(manifest, store=store, verifier=_verifier())

    assert external_file.read_bytes() == content
