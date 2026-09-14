"""Regression tests for immutable third-party tool bootstrap inputs."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_autodock_gpu_setup_uses_a_full_pinned_commit() -> None:
    script = (ROOT / "scripts" / "setup_autodock_gpu.sh").read_text()

    assert "SKINSCOUT_AUTODOCK_REPO_COMMIT" in script
    assert "6b150b35d0c615bc8dcec40ae09ca855227cc66f" in script
    assert "REPO_BRANCH" not in script
    assert "rev-parse HEAD" in script
    assert "status --porcelain --untracked-files=all" in script
    assert "source_manifest.json" in script


def test_p2rank_setup_verifies_release_sha256() -> None:
    script = (ROOT / "scripts" / "setup_p2rank.sh").read_text()

    assert "SKINSCOUT_P2RANK_SHA256" in script
    assert "9c21755967450300f2eb052d059ef128d38021626e52e135561c7c4687891751" in script
    assert "sha256sum --check --status" in script
    assert 'TMP_ARCHIVE="${ARCHIVE}.tmp"' in script
    assert 'mv "${TMP_ARCHIVE}" "${ARCHIVE}"' in script
    assert "source_manifest.json" in script
