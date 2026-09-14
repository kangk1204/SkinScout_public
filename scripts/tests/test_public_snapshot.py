"""What reaches the public repository is a decision, so it is tested.

The public repo is a snapshot rather than a mirror: pushing history would
expose everything ever committed, and deleting a file later does not un-index
it. This pins the things that would be damaging to get wrong — the excluded
files staying out, third-party database records staying out (they cannot be
republished under this repo's Apache-2.0 licence), the install URL pointing
somewhere that exists, and the signing policies *not* being rewritten.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_public_snapshot.sh"

EXCLUDED = ("scripts/report_email.sh", "review_dump_01.txt", "review_dump_02.txt")

# Internal development specifications and review-coordination records: they
# describe how the repository was assembled, not what the tool does.
DEV_SPECS_EXCLUDED = (
    "INSTRUCTIONS_v2.md",
    "INSTRUCTIONS_v2_bootstrap.md",
    "docs/SKINSCOUT_EXPERT_LINE_BY_LINE_AUDIT_20260904.md",
)

# Row-level records from ChEMBL / BindingDB / PubChem / the TDC skin_reaction
# benchmark. The repository is Apache-2.0 and the panels cannot carry the
# attribution the source licences require, so they never reach the public tree.
PUBLIC_DATA_EXCLUDED = (
    "data/pocket_cold_panel_202608",
    "data/transfer_reachable_panel_202608",
    "data/validation/rerank_band_20260828",
    "data/validation/skin_sens_benchmark_20260830",
    "data/curation/pubchem_probe_cache.json",
    "data/curation/pubchem_probe_cache_medium.json",
)

# Self-authored / permissively licensed data that readers and tests rely on:
# the CosIng-derived name dictionary (workbench name search), the curation
# worklists, and the CC-BY-4.0 AlphaFold-derivative fixture whose attribution
# the README carries.
PUBLIC_DATA_KEPT = (
    "data/compound_names/name_index.csv",
    "data/compound_names/korean_aliases.csv",
    "data/curation/skin_target_worklist.csv",
    "data/holo_transplant/P14679_with_Cu.pdb",
)


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("public") / "tree"
    result = subprocess.run(
        ["bash", str(SCRIPT), str(out)], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return out


# 텍스트로 읽어 볼 만한 확장자. 바이너리 파켓·이미지까지 훑을 이유는 없다.
TEXT_SUFFIXES = {".md", ".py", ".sh", ".json", ".yaml", ".yml", ".txt", ".html", ".js", ".css"}
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# 공개 저장소에 정상적으로 등장하는 주소는 개인 식별자가 아니다.
KEEP_EMAIL_RE = re.compile(r"@(?:users\.noreply\.github\.com|noreply\.|example\.(?:com|org|test|net)|[a-z0-9-]*\.)?skinscout\.example")


def test_the_excluded_files_do_not_reach_it(snapshot: Path) -> None:
    """report_email.sh carries a recipient address, hostname and Tailscale IP.
    The review dumps describe defects that are largely fixed."""
    for path in EXCLUDED:
        assert not (snapshot / path).exists(), path


def test_third_party_derived_data_stays_out(snapshot: Path) -> None:
    """ChEMBL/BindingDB/PubChem/TDC 유래 레코드는 Apache-2.0 공개본에 실지 않는다."""
    for path in PUBLIC_DATA_EXCLUDED:
        assert not (snapshot / path).exists(), path


def test_internal_development_specs_stay_out(snapshot: Path) -> None:
    for path in DEV_SPECS_EXCLUDED:
        assert not (snapshot / path).exists(), path


def test_no_spreadsheet_workbook_reaches_the_delivery(snapshot: Path) -> None:
    """개인 작업 스프레드시트는 공개본에 실리지 않는다.

    제외 목록에 파일명을 적으면 이 테스트 자체가 그 이름을 게시하게 되므로,
    파일이 아니라 확장자 규칙으로 막는다.
    """
    workbooks = [
        str(path.relative_to(snapshot))
        for path in snapshot.rglob("*")
        if path.is_file() and path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}
    ]
    assert not workbooks, workbooks[:5]


def test_a_local_exclusion_list_is_honoured(tmp_path: Path) -> None:
    """공개 스크립트에 이름을 적을 수 없는 파일은 로컬 목록으로 뺀다."""
    out = tmp_path / "tree"
    exclude = tmp_path / "public-exclude"
    exclude.write_text("ATTRIBUTION.md\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(SCRIPT), str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "SKINSCOUT_LOCAL_EXCLUDE": str(exclude)},
    )

    assert result.returncode == 0, result.stderr
    assert not (out / "ATTRIBUTION.md").exists(), "local exclusion list was ignored"
    assert (out / "README.md").is_file()


def test_the_name_dictionary_and_curation_stay(snapshot: Path) -> None:
    """이름 검색 사전(CosIng)과 자체 큐레이션, AlphaFold 예외 픽스처는 남는다."""
    for path in PUBLIC_DATA_KEPT:
        assert (snapshot / path).is_file(), path


def test_it_still_carries_what_a_reader_needs(snapshot: Path) -> None:
    for path in ("README.md", "LICENSE", "ATTRIBUTION.md", "LICENSE_POLICY.md"):
        assert (snapshot / path).exists(), path
    # The audit reports are deliberately kept: they record what the tool cannot do.
    assert list((snapshot / "docs").glob("*AUDIT*")), "audit reports were dropped"


def test_no_identifier_from_a_dropped_file_survives_anywhere(snapshot: Path) -> None:
    """파일을 지우는 것만으로는 부족했다.

    report_email.sh는 제외 목록에 있었지만, 감사 문서가 그 파일을 지적하면서 개인
    주소를 본문에 그대로 인용해 두었다. 지운 파일의 *내용*이 남은 파일에 살아 있는지
    까지 봐야 한다.

    가릴 값을 이 테스트에 적지 않는 이유도 같다 - 적으면 이 파일이 공개본에 실리면서
    가리려던 것을 자기가 게시한다. 제외 대상 파일에서 직접 읽어 와 대조한다.
    """
    secrets = set()
    for name in EXCLUDED:
        source = ROOT / name
        texts = [source.read_text(encoding="utf-8", errors="ignore")] if source.is_file() else []
        archived = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"HEAD:{name}"],
            capture_output=True, text=True, check=False,
        )
        if archived.returncode == 0:
            texts.append(archived.stdout)
        for text in texts:
            secrets.update(
                hit for hit in EMAIL_RE.findall(text) if not KEEP_EMAIL_RE.search(hit)
            )
    if not secrets:
        pytest.skip("제외 대상 파일에 식별자가 없습니다")

    offenders = []
    for path in snapshot.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for secret in secrets:
            if secret in text:
                offenders.append(str(path.relative_to(snapshot)))
                break
    assert not offenders, f"제외 파일의 식별자가 공개본에 남아 있습니다: {offenders[:5]}"


def test_no_personal_identifier_survives_any_text(snapshot: Path) -> None:
    """제외 파일이 청소되면 위 가드는 조용히 꺼진다 - 그래서 출력 트리를 직접 훑는다.

    제외 대상 파일의 값이 변수·정규식으로 바뀌면 '제외 파일에서 읽은 식별자'는 0개가
    되어 아무것도 가리지 않는다. 그런데 감사 문서는 그 주소를 본문에 인용해 두었을
    수 있어, 그 순간 주소가 공개본에 실린다. 빌드 스크립트가 출력 트리를 직접
    스캔해 가리도록 한 것과 같은 기준으로 여기서도 확인한다. 의도적으로 심어 둔
    placeholder(example.*, @skinscout.example)는 예외다.
    """
    offenders = []
    for path in snapshot.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for hit in EMAIL_RE.findall(text):
            if not KEEP_EMAIL_RE.search(hit):
                offenders.append((str(path.relative_to(snapshot)), hit))
    assert not offenders, f"개인 식별자가 공개본에 남아 있습니다: {offenders[:5]}"


def test_the_redaction_tooling_does_not_publish_what_it_redacts(snapshot: Path) -> None:
    """가림 코드가 가릴 값을 자기 안에 적어 두면, 스냅샷에 그 값이 그대로 실린다.

    처음 구현이 정확히 그랬다. 스크립트와 이 테스트 둘 다 주소를 정규식 리터럴로
    갖고 있었고, 둘 다 공개본에 포함되는 파일이다.
    """
    for name in ("scripts/build_public_snapshot.sh", "scripts/tests/test_public_snapshot.py"):
        path = snapshot / name
        if not path.is_file():
            continue
        # 정규식 안에 적힌 주소는 `gmail\.com`처럼 이스케이프돼 있어서, 그대로
        # 훑으면 탐지기가 자기 눈앞의 것을 놓친다. 실제로 처음에 그렇게 통과했다.
        text = path.read_text(encoding="utf-8").replace("\\", "")
        found = [hit for hit in EMAIL_RE.findall(text) if not KEEP_EMAIL_RE.search(hit)]
        assert not found, f"{name}이 개인 주소를 리터럴로 담고 있습니다"


def test_the_redaction_keeps_the_finding_it_redacts(snapshot: Path) -> None:
    """값만 가리고 사실은 남긴다.

    그 문단은 이 도구가 무엇을 잘못하고 있었는지 적은 것이라, 통째로 지우면
    감사 보고서를 일부러 남겨 둔 이유가 사라진다. 문서가 새로 추가되어 감사본이
    여럿이 되므로, 글롭 첫 번째가 아니라 report_email 위험을 기록한 쪽을 찾는다.
    """
    findings = []
    for path in (snapshot / "docs").glob("*LINE_BY_LINE_AUDIT*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if "report_email.sh" in text or "수신자" in text or "recipient" in text:
            findings.append((path.name, text))
    assert findings, "report_email.sh 위험을 기록한 감사 문서가 없습니다"
    name, text = findings[0]
    assert "<개인 주소>" in text, f"{name}에서 값이 가려지지 않았습니다"


def test_the_install_url_points_at_the_public_repo(snapshot: Path) -> None:
    """The one-line install is the first thing a collaborator runs. Pointing it
    at the private repo makes it 404 for exactly the person it is written for."""
    readme = (snapshot / "README.md").read_text(encoding="utf-8")
    installer = (snapshot / "install_skinscout.sh").read_text(encoding="utf-8")

    assert "kangk1204/SkinScout_public/main/install_skinscout.sh" in readme
    assert "kangk1204/SkinScout_public.git" in installer
    assert "kangk1204/SkinScout_public/" not in readme


def test_the_signing_policies_are_left_alone(snapshot: Path) -> None:
    """Sigstore certificate identities pin which workflow may sign a release.
    Repointing them at a repo that is not the signer weakens the policy; it
    does not fix anything."""
    for name in ("data-trust-policy.json", "qualification-trust-policy.json"):
        text = (snapshot / "compose" / name).read_text(encoding="utf-8")
        assert "kangk1204/SkinScout_public/" in text, name
        assert "SkinScout_public" not in text, name


def test_it_says_it_is_the_public_copy(snapshot: Path) -> None:
    readme = (snapshot / "README.md").read_text(encoding="utf-8")

    assert "이 저장소는 공개본입니다" in readme
    # And the same note must not accumulate on a rebuild.
    assert readme.count("이 저장소는 공개본입니다") == 1


# ------------------------------------------------- 출력 경로 삭제 안전장치


def test_the_script_refuses_to_delete_a_directory_it_did_not_create(
    tmp_path: Path,
) -> None:
    """`rm -rf "$OUT"` 는 인자를 그대로 지운다.

    `build_public_snapshot.sh ~/문서` 는 그 디렉터리를 영구 삭제한다. 되돌릴 수
    없는 일이라, 지우기 전에 그것이 우리가 만든 스냅샷인지 확인해야 한다.
    """
    victim = tmp_path / "someones_work"
    victim.mkdir()
    keepsake = victim / "important.txt"
    keepsake.write_text("소중한 파일\n")

    res = subprocess.run(
        ["bash", str(SCRIPT), str(victim)],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert res.returncode != 0
    assert "not a SkinScout snapshot" in res.stderr
    assert keepsake.exists(), "남의 디렉터리를 지웠습니다"
    assert keepsake.read_text() == "소중한 파일\n"


@pytest.mark.parametrize("path", ["/", "/usr", "/home", "/etc"])
def test_the_script_refuses_a_system_path(path: str) -> None:
    """오타 하나가 시스템 디렉터리를 지우면 안 된다."""
    res = subprocess.run(
        ["bash", str(SCRIPT), path],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert res.returncode != 0, f"{path} 를 받아들였습니다"
    assert "system path" in res.stderr, path


def test_a_snapshot_directory_carries_the_marker_and_can_be_rebuilt(
    tmp_path: Path,
) -> None:
    """표식이 있어야 다음 실행이 덮어쓸 수 있다."""
    out = tmp_path / "snap"
    first = subprocess.run(
        ["bash", str(SCRIPT), str(out)],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert first.returncode == 0, first.stderr
    assert (out / ".skinscout-public-snapshot").is_file()

    again = subprocess.run(
        ["bash", str(SCRIPT), str(out)],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert again.returncode == 0, again.stderr
    backups = list(tmp_path.glob("snap.previous.*"))
    assert len(backups) == 1
    assert (backups[0] / "README.md").is_file()


def test_snapshot_uses_script_repository_from_another_cwd(tmp_path: Path) -> None:
    out = tmp_path / "outside"
    result = subprocess.run(["bash", str(SCRIPT), str(out)], cwd=tmp_path,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert (out / "scripts/run_skinscout.py").is_file()


@pytest.mark.parametrize("through_parent", [False, True])
def test_snapshot_rejects_symlink_output_ancestors(tmp_path: Path, through_parent: bool) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    sentinel = victim / "important.txt"
    sentinel.write_text("keep")
    (victim / ".skinscout-public-snapshot").write_text(
        "Generated by scripts/build_public_snapshot.sh. Safe to delete.\n"
    )
    link = tmp_path / "alias"
    link.symlink_to(victim, target_is_directory=True)
    out = link / "child" if through_parent else link
    result = subprocess.run(["bash", str(SCRIPT), str(out)], cwd=tmp_path,
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "symlink" in result.stderr
    assert sentinel.read_text() == "keep"


def test_snapshot_rejects_initialized_repository_even_with_marker(tmp_path: Path) -> None:
    out = tmp_path / "published"
    out.mkdir()
    (out / ".git").mkdir()
    (out / ".skinscout-public-snapshot").write_text(
        "Generated by scripts/build_public_snapshot.sh. Safe to delete.\n"
    )
    result = subprocess.run(["bash", str(SCRIPT), str(out)], cwd=tmp_path,
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert (out / ".git").is_dir()
