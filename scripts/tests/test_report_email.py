"""Mail delivery is opt-in; tests use a local capture executable only."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def _mail_environment(tmp_path):
    binary = tmp_path / "msmtp"
    binary.write_text('#!/bin/sh\ncat > "$SKINSCOUT_TEST_MAIL_CAPTURE"\n')
    binary.chmod(0o755)
    config = tmp_path / "mail.conf"
    config.write_text("# test config\n")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SKINSCOUT_REPORT_")}
    environment.update(PATH=str(tmp_path) + os.pathsep + os.environ.get("PATH", ""),
                       SKINSCOUT_REPORT_CONFIG=str(config),
                       SKINSCOUT_TEST_MAIL_CAPTURE=str(tmp_path / "capture"))
    return environment


def test_email_never_sends_without_explicit_recipient(tmp_path):
    result = subprocess.run(["bash", str(ROOT / "scripts/report_email.sh"), "Report"],
                            input="body", text=True, capture_output=True, env=_mail_environment(tmp_path))
    assert result.returncode == 2
    assert not (tmp_path / "capture").exists()


def test_email_omits_host_metadata_by_default(tmp_path):
    env = _mail_environment(tmp_path)
    env["SKINSCOUT_REPORT_TO"] = "<개인 주소>"
    result = subprocess.run(["bash", str(ROOT / "scripts/report_email.sh"), "Report"],
                            input="report body", text=True, capture_output=True, env=env)
    assert result.returncode == 0, result.stderr
    message = (tmp_path / "capture").read_text()
    assert "To: <개인 주소>" in message
    assert message.endswith("report body")
    assert "Host:" not in message


def test_email_rejects_header_injection(tmp_path):
    env = _mail_environment(tmp_path)
    env["SKINSCOUT_REPORT_TO"] = "<개인 주소>"
    result = subprocess.run(["bash", str(ROOT / "scripts/report_email.sh"), "Report\nBcc: <개인 주소>"],
                            input="body", text=True, capture_output=True, env=env)
    assert result.returncode == 2
    assert not (tmp_path / "capture").exists()
