#!/usr/bin/env python3
"""Content-addressed data bundle installer for SkinScout runtime assets."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = Path.home() / ".local/share/skinscout/data"
DEFAULT_TRUST_POLICY = ROOT / "compose/data-trust-policy.json"
REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "bundle",
    "version",
    "image_compatibility",
    "signature",
    "packs",
}
REQUIRED_FILE_FIELDS = {"path", "sha256", "size"}
SHA256_HEX = set("0123456789abcdef")


class CasError(RuntimeError):
    """Fail-closed CAS validation or activation error."""


class ManifestVerifier(Protocol):
    def __call__(
        self,
        manifest_bytes: bytes,
        signature: Mapping[str, Any],
        manifest_path: Path | None = None,
    ) -> None:
        """Verify canonical manifest bytes against a detached signature bundle."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_valid_sha256(value: object) -> bool:
    text = value if isinstance(value, str) else ""
    return len(text) == 64 and all(ch in SHA256_HEX for ch in text) and len(set(text)) > 1


def _canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    unsigned = dict(manifest)
    unsigned.pop("signature", None)
    return (
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def load_manifest(
    path: Path,
    *,
    verifier: ManifestVerifier | None = None,
    trust_policy: Mapping[str, Any] | None = None,
    trust_policy_path: Path | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CasError(f"cannot read manifest: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CasError("manifest must be a JSON object")
    validate_manifest(
        payload,
        manifest_path=path,
        verifier=verifier,
        trust_policy=trust_policy,
        trust_policy_path=trust_policy_path,
    )
    return payload


def load_trust_policy(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CasError(f"cannot read external data trust policy: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CasError("external data trust policy must be a JSON object")
    validate_trust_policy(payload, policy_path=path)
    return payload


def _trusted_public_key_path(signer: Mapping[str, Any], policy_path: Path | None) -> Path:
    key_path_text = signer.get("public_key")
    if not isinstance(key_path_text, str) or not key_path_text:
        raise CasError("trusted public-key signer requires public_key")
    key_path = Path(key_path_text)
    if not key_path.is_absolute():
        if policy_path is None:
            raise CasError("relative trusted public_key requires trust policy path context")
        key_path = policy_path.parent / key_path
    if key_path.is_symlink() or not key_path.is_file():
        raise CasError(f"trusted public key is missing or unsafe: {key_path}")
    expected_sha = signer.get("public_key_sha256")
    if not _is_valid_sha256(expected_sha):
        raise CasError("trusted public-key signer requires public_key_sha256")
    if sha256_file(key_path) != expected_sha:
        raise CasError("trusted public key digest mismatch")
    return key_path


def validate_trust_policy(
    policy: Mapping[str, Any],
    *,
    policy_path: Path | None = None,
) -> None:
    if policy.get("schema_version") != "skinscout.data_trust_policy.v1":
        raise CasError("external data trust policy schema is unsupported")
    if policy.get("fail_closed") is not True or policy.get("verification") != "cosign":
        raise CasError("external data trust policy must be fail-closed Cosign")
    signers = policy.get("trusted_signers")
    if not isinstance(signers, list) or not signers:
        raise CasError("external data trust policy must list trusted_signers")
    identities: set[tuple[str, ...]] = set()
    for signer in signers:
        if not isinstance(signer, Mapping):
            raise CasError("each trusted signer must be an object")
        signer_type = signer.get("type")
        if signer_type == "keyless":
            identity = signer.get("certificate_identity")
            issuer = signer.get("issuer")
            if not isinstance(identity, str) or not identity:
                raise CasError("trusted keyless signer requires certificate_identity")
            if not isinstance(issuer, str) or not issuer:
                raise CasError("trusted keyless signer requires issuer")
            signer_identity = ("keyless", identity, issuer)
        elif signer_type == "public_key":
            key_path = _trusted_public_key_path(signer, policy_path)
            signer_identity = (
                "public_key",
                str(signer["public_key_sha256"]),
                str(key_path.resolve()),
            )
        else:
            raise CasError(f"unsupported trusted signer type: {signer_type!r}")
        if signer_identity in identities:
            raise CasError("external data trust policy contains duplicate trusted signer")
        identities.add(signer_identity)


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path | None = None,
    verifier: ManifestVerifier | None = None,
    trust_policy: Mapping[str, Any] | None = None,
    trust_policy_path: Path | None = None,
) -> None:
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    if missing:
        raise CasError(f"manifest missing required fields: {', '.join(missing)}")
    signature = manifest.get("signature")
    if not isinstance(signature, Mapping):
        raise CasError("manifest signature must be an object")
    if signature.get("policy") != "cosign-keyless-or-key-pinned":
        raise CasError("manifest signature policy is absent or unsupported")
    has_identity = bool(signature.get("certificate_identity"))
    has_public_key = bool(signature.get("public_key_sha256"))
    if has_identity == has_public_key:
        raise CasError("manifest signature must pin identity or public key")
    if has_identity and not signature.get("issuer"):
        raise CasError("manifest signature identity requires issuer")
    if has_public_key and not signature.get("public_key"):
        raise CasError("manifest signature public_key_sha256 requires public_key path")
    if not _is_valid_sha256(signature.get("bundle_sha256")):
        raise CasError("manifest signature must include bundle_sha256")
    if not signature.get("bundle_path"):
        raise CasError("manifest signature must include detached bundle_path")
    packs = manifest.get("packs")
    if not isinstance(packs, list) or not packs:
        raise CasError("manifest packs must be a non-empty list")
    for pack in packs:
        if not isinstance(pack, Mapping):
            raise CasError("each pack must be an object")
        for field in ("name", "version", "files"):
            if field not in pack:
                raise CasError(f"pack missing required field: {field}")
        _validate_pack_component(pack.get("name"), "name")
        _validate_pack_component(pack.get("version"), "version")
        files = pack.get("files")
        if not isinstance(files, list) or not files:
            raise CasError(f"pack {pack.get('name')} must contain files")
        for entry in files:
            if not isinstance(entry, Mapping):
                raise CasError("each file entry must be an object")
            missing_file = sorted(REQUIRED_FILE_FIELDS - set(entry))
            if missing_file:
                raise CasError(
                    f"file entry missing required fields: {', '.join(missing_file)}"
                )
            if not _is_valid_sha256(entry["sha256"]):
                raise CasError(f"invalid sha256 for {entry.get('path')}")
            if int(entry["size"]) < 0:
                raise CasError(f"invalid size for {entry.get('path')}")
            if not entry.get("license_spdx") or entry.get("redistribution") is not True:
                raise CasError(f"redistribution/license metadata missing for {entry.get('path')}")
    verify_manifest_signature(
        manifest,
        manifest_path=manifest_path,
        verifier=verifier,
        trust_policy=trust_policy,
        trust_policy_path=trust_policy_path,
    )


def _signature_bundle_path(
    signature: Mapping[str, Any],
    manifest_path: Path | None,
) -> Path:
    bundle_path = Path(str(signature["bundle_path"]))
    if not bundle_path.is_absolute():
        if manifest_path is None:
            raise CasError("relative signature bundle_path requires manifest path context")
        bundle_path = manifest_path.parent / bundle_path
    return bundle_path


def _manifest_signer_matches(
    signature: Mapping[str, Any],
    signer: Mapping[str, Any],
) -> bool:
    if signer.get("type") == "keyless":
        return (
            signature.get("certificate_identity") == signer.get("certificate_identity")
            and signature.get("issuer") == signer.get("issuer")
            and not signature.get("public_key_sha256")
        )
    if signer.get("type") == "public_key":
        return (
            signature.get("public_key_sha256") == signer.get("public_key_sha256")
            and not signature.get("certificate_identity")
        )
    return False


def _trusted_signer_args(
    signer: Mapping[str, Any],
    trust_policy_path: Path | None,
) -> list[str]:
    if signer.get("type") == "keyless":
        return [
            "--certificate-identity",
            str(signer["certificate_identity"]),
            "--certificate-oidc-issuer",
            str(signer["issuer"]),
        ]
    key_path = _trusted_public_key_path(signer, trust_policy_path)
    return ["--key", str(key_path)]


def _cosign_verify_manifest(
    manifest_bytes: bytes,
    signature: Mapping[str, Any],
    trusted_signers: list[Mapping[str, Any]],
    manifest_path: Path | None = None,
    trust_policy_path: Path | None = None,
) -> None:
    bundle_path = _signature_bundle_path(signature, manifest_path)
    if not bundle_path.is_file():
        raise CasError(f"manifest signature bundle is missing: {bundle_path}")
    if sha256_file(bundle_path) != str(signature["bundle_sha256"]):
        raise CasError("manifest signature bundle digest mismatch")
    failures: list[str] = []
    with tempfile.NamedTemporaryFile(prefix="skinscout-manifest-", suffix=".json") as temp:
        temp.write(manifest_bytes)
        temp.flush()
        for signer in trusted_signers:
            command = [
                "cosign",
                "verify-blob",
                "--bundle",
                str(bundle_path),
                *_trusted_signer_args(signer, trust_policy_path),
                temp.name,
            ]
            try:
                result = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise CasError("cosign is required for data manifest verification") from exc
            if result.returncode == 0:
                return
            failures.append(
                result.stderr.strip()
                or result.stdout.strip()
                or "cosign verify-blob failed"
            )
    raise CasError("manifest signature verification failed: " + " | ".join(failures))


def verify_manifest_signature(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path | None = None,
    verifier: ManifestVerifier | None = None,
    trust_policy: Mapping[str, Any] | None = None,
    trust_policy_path: Path | None = None,
) -> None:
    signature = manifest.get("signature")
    if not isinstance(signature, Mapping):
        raise CasError("manifest signature must be an object")
    canonical_bytes = _canonical_manifest_bytes(manifest)
    if verifier is not None:
        verifier(canonical_bytes, signature, manifest_path)
        return
    if trust_policy is None:
        raise CasError("external trust policy is required for manifest signature verification")
    validate_trust_policy(trust_policy, policy_path=trust_policy_path)
    trusted_signers = [
        signer
        for signer in trust_policy["trusted_signers"]
        if _manifest_signer_matches(signature, signer)
    ]
    if not trusted_signers:
        raise CasError("manifest signature signer is not trusted by external policy")
    _cosign_verify_manifest(
        canonical_bytes,
        signature,
        trusted_signers,
        manifest_path,
        trust_policy_path,
    )


def _safe_relative(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts:
        raise CasError(f"unsafe manifest path: {path_text}")
    return path


def _validate_pack_component(value: object, field: str) -> str:
    """Pack name/version become one path component under store/staging and
    store/packs. Reject anything that could escape either directory before the
    staging path is built."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise CasError(f"pack {field} must be a non-empty path-safe string")
    if any(character in value for character in ("/", "\\", "\x00")):
        raise CasError(f"unsafe pack {field}: {value!r}")
    path = Path(value)
    if value in {".", ".."} or path.is_absolute() or len(path.parts) != 1:
        raise CasError(f"unsafe pack {field}: {value!r}")
    return value


def _file_sources(entry: Mapping[str, Any], mirror: str | None) -> Iterable[str]:
    if mirror:
        yield mirror.rstrip("/") + "/" + str(entry["path"]).lstrip("/")
    for url in entry.get("urls", []) or []:
        yield str(url)


def _stat_regular_nofollow(path: Path, label: str) -> os.stat_result | None:
    try:
        stat_result = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(stat_result.st_mode):
        raise CasError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(stat_result.st_mode):
        raise CasError(f"{label} must be a regular file: {path}")
    return stat_result


def _open_partial_append_nofollow(path: Path):
    if not hasattr(os, "O_NOFOLLOW"):
        raise CasError("this platform cannot safely open partial downloads without following symlinks")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise CasError(f"cannot open partial download safely: {path}: {exc}") from exc
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise CasError(f"partial download must be a regular file: {path}")
    return os.fdopen(fd, "ab")


class DownloadCapExceeded(CasError):
    """The response streamed past the size trusted by the signed manifest."""


def _read_response_chunk(source: Any, size: int) -> bytes:
    if isinstance(source, http.client.HTTPResponse):
        return source.read1(size)
    return source.read(size)


def _copy_response_with_limit(
    source: Any,
    destination: Any,
    *,
    max_bytes: int,
    deadline: float,
    label: str,
) -> int:
    """Copy at most ``max_bytes`` and abort before the next byte is written."""

    written = 0
    while True:
        if time.monotonic() > deadline:
            raise CasError(f"download timed out for {label}")
        remaining = max_bytes - written
        chunk = _read_response_chunk(source, min(1024 * 1024, max(remaining, 0) + 1))
        if not chunk:
            return written
        written += len(chunk)
        if written > max_bytes:
            raise DownloadCapExceeded(
                f"downloaded size for {label} exceeds expected {max_bytes}"
            )
        destination.write(chunk)


def download_entry(
    entry: Mapping[str, Any],
    *,
    staging: Path,
    mirror: str | None = None,
    timeout: float = 60.0,
) -> Path:
    relative = _safe_relative(str(entry["path"]))
    target = staging / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(entry["size"])
    expected_sha = str(entry["sha256"])
    if target.exists() and target.stat().st_size == expected_size:
        if target.is_symlink():
            raise CasError(f"CAS file must not be a symlink: {relative}")
        if sha256_file(target) == expected_sha:
            return target
        target.unlink()
    partial = target.with_suffix(target.suffix + ".part")
    last_error: Exception | None = None
    for url in _file_sources(entry, mirror):
        # 미러마다 현재 partial 크기로 Range를 다시 계산한다. 루프 밖에서 한 번만
        # 계산하면, 첫 미러의 잘린 응답 뒤에 둘째 미러의 전체 파일이 이어 붙는다.
        partial_stat = _stat_regular_nofollow(partial, "partial download")
        resume_from = partial_stat.st_size if partial_stat is not None else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())
                if resume_from and status == 200:
                    # 서버가 Range를 무시하고 전체 파일을 보냈다. 기존 조각에
                    # 이어 붙이면 길이가 두 배가 되므로, 조각을 버리고 같은
                    # URL에 Range 없이 다시 요청해 처음부터 쓴다.
                    partial.unlink()
                    response.close()
                    deadline = time.monotonic() + timeout
                    with urllib.request.urlopen(
                        urllib.request.Request(url), timeout=timeout
                    ) as fresh:
                        with _open_partial_append_nofollow(partial) as out:
                            _copy_response_with_limit(
                                fresh,
                                out,
                                max_bytes=expected_size,
                                deadline=deadline,
                                label=str(entry["path"]),
                            )
                else:
                    if resume_from:
                        content_range = response.headers.get("Content-Range")
                        expected_prefix = f"bytes {resume_from}-"
                        if (
                            status != 206
                            or not content_range
                            or not content_range.startswith(expected_prefix)
                        ):
                            raise CasError(
                                f"resume response for {entry['path']} must be 206 "
                                "with matching Content-Range"
                            )
                    deadline = time.monotonic() + timeout
                    with _open_partial_append_nofollow(partial) as out:
                        _copy_response_with_limit(
                            response,
                            out,
                            max_bytes=expected_size - resume_from,
                            deadline=deadline,
                            label=str(entry["path"]),
                        )
            completed_stat = _stat_regular_nofollow(partial, "partial download")
            if completed_stat is None:
                raise CasError(f"partial download disappeared: {partial}")
            if completed_stat.st_size != expected_size:
                # 끝까지 읽은 응답의 크기가 다르면 그 조각은 이 파일의 앞부분이
                # 아니다. 이어받기의 씨앗으로 남기면 다음 미러가 뒤에 붙인다.
                partial.unlink(missing_ok=True)
                raise CasError(
                    f"downloaded size mismatch for {entry['path']}: "
                    f"{completed_stat.st_size} != {expected_size}"
                )
            if sha256_file(partial) != expected_sha:
                partial.unlink(missing_ok=True)
                raise CasError(f"downloaded sha256 mismatch for {entry['path']}")
            partial.replace(target)
            return target
        except DownloadCapExceeded as exc:
            partial.unlink(missing_ok=True)
            last_error = exc
        except Exception as exc:  # noqa: BLE001 - keep mirror fallback simple.
            last_error = exc
            partial_stat = _stat_regular_nofollow(partial, "partial download")
            if partial_stat is not None and partial_stat.st_size > expected_size:
                partial.unlink()
    raise CasError(f"download failed for {entry['path']}: {last_error}")


def verify_pack_files(pack: Mapping[str, Any], root: Path) -> None:
    expected_files: set[str] = set()
    for entry in pack["files"]:
        relative = _safe_relative(str(entry["path"]))
        relative_text = relative.as_posix()
        if relative_text in expected_files:
            raise CasError(f"duplicate CAS file declaration: {relative_text}")
        expected_files.add(relative_text)
        path = root / relative
        if not path.exists():
            raise CasError(f"missing CAS file: {relative}")
        if path.is_symlink():
            raise CasError(f"CAS file must not be a symlink: {relative}")
        if path.stat().st_size != int(entry["size"]):
            raise CasError(f"size mismatch for {relative}")
        if sha256_file(path) != str(entry["sha256"]):
            raise CasError(f"sha256 mismatch for {relative}")
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CasError(f"CAS pack must not contain symlinks: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative != ".manifest.json":
                actual_files.add(relative)
        elif not path.is_dir():
            raise CasError(f"CAS pack contains unsupported entry: {path}")
    if actual_files != expected_files:
        undeclared = sorted(actual_files - expected_files)
        missing = sorted(expected_files - actual_files)
        detail = []
        if undeclared:
            detail.append("undeclared=" + ",".join(undeclared[:10]))
        if missing:
            detail.append("missing=" + ",".join(missing[:10]))
        raise CasError("CAS pack file inventory mismatch: " + "; ".join(detail))


def _pack_id(pack: Mapping[str, Any]) -> str:
    return f"{pack['name']}@{pack['version']}"


def _pack_digest(pack: Mapping[str, Any]) -> str:
    canonical = json.dumps(pack, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _active_pack_basename(pack: Mapping[str, Any]) -> str:
    return f"{_pack_id(pack)}+sha256.{_pack_digest(pack)}"


def _fsync_file(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CasError(f"cannot open CAS file for durable publish: {path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CasError(f"CAS publish source must be a regular file: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CasError(f"cannot open CAS directory for durable publish: {path}: {exc}") from exc
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CasError(f"CAS publish tree must not contain symlinks: {path}")
        if path.is_file():
            _fsync_file(path)
        elif path.is_dir():
            directories.append(path)
        else:
            raise CasError(f"CAS publish tree contains unsupported entry: {path}")
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _atomic_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.parent / (
        f".{link.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    try:
        os.symlink(str(target.resolve(strict=True)), temporary)
        _fsync_directory(link.parent)
        os.replace(temporary, link)
        _fsync_directory(link.parent)
    finally:
        if temporary.is_symlink() or temporary.exists():
            temporary.unlink()


def _path_is_under(path: Path, root: Path) -> bool:
    return root == path or root in path.parents


def _non_symlink_store_dir(store: Path, name: str) -> Path:
    path = store / name
    if path.is_symlink():
        raise CasError(f"store/{name} must not be a symlink")
    if path.exists() and not path.is_dir():
        raise CasError(f"store/{name} must be a directory")
    return path.resolve()


def _verified_pack_symlink_target(link: Path, store: Path, label: str) -> Path:
    if not link.is_symlink():
        raise CasError(f"{label} must be a symlink")
    try:
        target = link.resolve(strict=True)
    except OSError as exc:
        raise CasError(f"cannot resolve {label} data pack: {link}: {exc}") from exc
    packs_root = _non_symlink_store_dir(store, "packs")
    if target == packs_root or not _path_is_under(target, packs_root):
        raise CasError(f"{label} data pack target must remain under store/packs")
    if not target.is_dir():
        raise CasError(f"{label} data pack target must be a directory")
    return target


def stage_pack(
    manifest: Mapping[str, Any],
    pack_name: str,
    *,
    store: Path = DEFAULT_STORE,
    mirror: str | None = None,
    offline_dir: Path | None = None,
    verifier: ManifestVerifier | None = None,
    manifest_path: Path | None = None,
    trust_policy: Mapping[str, Any] | None = None,
    trust_policy_path: Path | None = None,
) -> Path:
    validate_manifest(
        manifest,
        manifest_path=manifest_path,
        verifier=verifier,
        trust_policy=trust_policy,
        trust_policy_path=trust_policy_path,
    )
    pack = next((item for item in manifest["packs"] if item["name"] == pack_name), None)
    if pack is None:
        raise CasError(f"pack not found in manifest: {pack_name}")
    pack_id = _pack_id(pack)
    staging = store / "staging" / pack_id
    staging.mkdir(parents=True, exist_ok=True)
    if offline_dir is not None:
        for entry in pack["files"]:
            relative = _safe_relative(str(entry["path"]))
            source = offline_dir / relative
            destination = staging / relative
            if not source.exists():
                raise CasError(f"offline import missing file: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    else:
        for entry in pack["files"]:
            download_entry(entry, staging=staging, mirror=mirror)
    verify_pack_files(pack, staging)
    (staging / ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return staging


def activate(
    staged_pack: Path,
    *,
    store: Path = DEFAULT_STORE,
    verifier: ManifestVerifier | None = None,
    trust_policy: Mapping[str, Any] | None = None,
    trust_policy_path: Path | None = None,
) -> Path:
    if not staged_pack.exists():
        raise CasError(f"staged pack does not exist: {staged_pack}")
    staging_root = _non_symlink_store_dir(store, "staging")
    try:
        staged_resolved = staged_pack.resolve(strict=True)
    except OSError as exc:
        raise CasError(f"cannot resolve staged pack: {staged_pack}: {exc}") from exc
    if staged_pack.is_symlink() or staging_root not in [staged_resolved, *staged_resolved.parents]:
        raise CasError("activation requires verified non-symlink staging under store/staging")
    manifest_path = staged_pack / ".manifest.json"
    manifest = load_manifest(
        manifest_path,
        verifier=verifier,
        trust_policy=trust_policy,
        trust_policy_path=trust_policy_path,
    )
    active_name = staged_pack.name
    pack = next((_pack for _pack in manifest["packs"] if _pack_id(_pack) == active_name), None)
    if pack is None:
        raise CasError(f"staged pack is not in manifest: {active_name}")
    verify_pack_files(pack, staged_pack)
    current = store / "current"
    if current.exists() and not current.is_symlink():
        raise CasError("current must be a symlink; refusing non-atomic activation")
    current_target: Path | None = None
    if current.is_symlink():
        current_target = _verified_pack_symlink_target(current, store, "current")
    previous = store / "previous"
    if previous.is_symlink():
        _verified_pack_symlink_target(previous, store, "previous")
    elif previous.exists():
        raise CasError("previous must be a symlink; refusing non-atomic activation")
    active_root = _non_symlink_store_dir(store, "packs")
    active_root.mkdir(parents=True, exist_ok=True)
    destination_base = active_root / _active_pack_basename(pack)
    destination = destination_base
    reused_existing = False
    invalid_existing_destination = False
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise CasError(f"content-addressed CAS destination is unsafe: {destination}")
        try:
            existing_manifest = load_manifest(
                destination / ".manifest.json",
                verifier=verifier,
                trust_policy=trust_policy,
                trust_policy_path=trust_policy_path,
            )
            existing_pack = next(
                (
                    item
                    for item in existing_manifest["packs"]
                    if _pack_digest(item) == _pack_digest(pack)
                ),
                None,
            )
            if existing_pack is None:
                raise CasError("existing CAS destination manifest does not bind expected pack")
            verify_pack_files(existing_pack, destination)
        except CasError:
            invalid_existing_destination = True
            destination = active_root / (
                f"{destination_base.name}.repair-{secrets.token_hex(8)}"
            )
        else:
            reused_existing = True

    if reused_existing:
        shutil.rmtree(staged_pack)
        _fsync_directory(staging_root)
    else:
        _fsync_tree(staged_pack)
        os.replace(staged_pack, destination)
        _fsync_directory(active_root)
        _fsync_directory(staging_root)

    if current_target == destination:
        return destination
    if current_target is not None and not (
        invalid_existing_destination and current_target == destination_base
    ):
        _atomic_symlink(previous, current_target)
    _atomic_symlink(current, destination)
    return destination


def rollback(*, store: Path = DEFAULT_STORE) -> Path:
    current = store / "current"
    previous = store / "previous"
    if not previous.is_symlink():
        raise CasError("no previous data pack is available for rollback")
    previous_target = _verified_pack_symlink_target(previous, store, "previous")
    current_target: Path | None = None
    if current.is_symlink():
        current_target = _verified_pack_symlink_target(current, store, "current")
    elif current.exists():
        raise CasError("current must be a symlink; refusing non-atomic rollback")
    _atomic_symlink(current, previous_target)
    if current_target is not None and current_target != previous_target:
        _atomic_symlink(previous, current_target)
    return current.resolve(strict=True)


def repair(
    manifest: Mapping[str, Any],
    *,
    store: Path = DEFAULT_STORE,
    verifier: ManifestVerifier | None = None,
    manifest_path: Path | None = None,
    trust_policy: Mapping[str, Any] | None = None,
    trust_policy_path: Path | None = None,
) -> None:
    validate_manifest(
        manifest,
        manifest_path=manifest_path,
        verifier=verifier,
        trust_policy=trust_policy,
        trust_policy_path=trust_policy_path,
    )
    current = store / "current"
    if not current.is_symlink():
        raise CasError("no active current data pack")
    target = _verified_pack_symlink_target(current, store, "current")
    active_name = target.name
    pack = next(
        (
            item
            for item in manifest["packs"]
            if active_name == _pack_id(item)
            or active_name == _active_pack_basename(item)
            or active_name.startswith(_active_pack_basename(item) + ".repair-")
        ),
        None,
    )
    if pack is None:
        raise CasError(f"active pack is not in manifest: {active_name}")
    verify_pack_files(pack, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--trust-policy", type=Path, default=DEFAULT_TRUST_POLICY)
    sub = parser.add_subparsers(dest="command", required=True)
    validate_cmd = sub.add_parser("validate")
    validate_cmd.add_argument("manifest", type=Path)
    stage_cmd = sub.add_parser("stage")
    stage_cmd.add_argument("manifest", type=Path)
    stage_cmd.add_argument("pack")
    stage_cmd.add_argument("--mirror")
    stage_cmd.add_argument("--offline-dir", type=Path)
    activate_cmd = sub.add_parser("activate")
    activate_cmd.add_argument("staged_pack", type=Path)
    repair_cmd = sub.add_parser("repair")
    repair_cmd.add_argument("manifest", type=Path)
    sub.add_parser("rollback")
    args = parser.parse_args(argv)
    try:
        trust_policy = (
            None
            if args.command == "rollback"
            else load_trust_policy(args.trust_policy)
        )
        if args.command == "validate":
            load_manifest(
                args.manifest,
                trust_policy=trust_policy,
                trust_policy_path=args.trust_policy,
            )
            print("manifest ok")
        elif args.command == "stage":
            manifest = load_manifest(
                args.manifest,
                trust_policy=trust_policy,
                trust_policy_path=args.trust_policy,
            )
            staged = stage_pack(
                manifest,
                args.pack,
                store=args.store,
                mirror=args.mirror,
                offline_dir=args.offline_dir,
                manifest_path=args.manifest,
                trust_policy=trust_policy,
                trust_policy_path=args.trust_policy,
            )
            print(staged)
        elif args.command == "activate":
            print(activate(
                args.staged_pack,
                store=args.store,
                trust_policy=trust_policy,
                trust_policy_path=args.trust_policy,
            ))
        elif args.command == "repair":
            manifest = load_manifest(
                args.manifest,
                trust_policy=trust_policy,
                trust_policy_path=args.trust_policy,
            )
            repair(
                manifest,
                store=args.store,
                manifest_path=args.manifest,
                trust_policy=trust_policy,
                trust_policy_path=args.trust_policy,
            )
            print("repair ok")
        elif args.command == "rollback":
            print(rollback(store=args.store))
    except CasError as exc:
        print(f"[SkinScout][ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
