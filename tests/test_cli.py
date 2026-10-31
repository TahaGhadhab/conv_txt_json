"""The documented entry point: ``python -m drone_voice`` (called in-process here)."""

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from drone_voice.cli import main
from drone_voice.jsonio import read_json_file

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "examples" / "transcripts_simulated.jsonl"


def run(*argv):
    out = io.StringIO()
    code = main(list(argv), out=out)
    return code, out.getvalue()


def calls_section(output: str):
    block = output.split("== Simulated adapter calls ==\n", 1)[1]
    return json.loads(block.rsplit("\nSIMULATION ONLY", 1)[0])


def test_demo_valid_command(tmp_path):
    saved = tmp_path / "m.json"
    code, output = run("demo", "تقدّم", "--now", "2026-01-01T12:00:00Z", "--save-json", str(saved))
    assert code == 0
    calls = calls_section(output)
    assert [(c["method"], c.get("direction"), c.get("cause")) for c in calls] == [
        ("move", "FORWARD", None), ("stop", None, "duration_elapsed")]
    assert read_json_file(saved)["intent"] == "MOVE_FORWARD"
    # never overwrite a previous output
    code, _ = run("demo", "وقف", "--save-json", str(saved))
    assert code == 2
    assert read_json_file(saved)["intent"] == "MOVE_FORWARD"


@pytest.mark.parametrize("argv", [["demo", "ما تتقدّمش"], ["demo", "دور"], ["demo", "تقدّم", "--partial"]])
def test_demo_without_movement(argv):
    code, output = run(*argv)
    assert code == 0
    assert calls_section(output) == []


def test_batch_fixture_then_rerun_does_not_reexecute(tmp_path):
    state = tmp_path / "ids.jsonl"
    out1 = tmp_path / "out1.jsonl"
    code, _ = run("batch", str(FIXTURE), "--out", str(out1), "--state", str(state),
                  "--now", "2026-01-01T12:00:00Z")
    assert code == 0
    records = [json.loads(line) for line in out1.read_text(encoding="utf-8").splitlines()]
    calls = records[-1]["simulated_adapter_calls"]
    assert [(c["method"], c.get("direction") or c.get("cause")) for c in calls] == [
        ("move", "FORWARD"),            # sim-001 (the partial before it did nothing)
        ("stop", "stop_command"),       # sim-002 STOP during the movement
        ("move", "RIGHT"),              # sim-006
        ("stop", "duration_elapsed"),   # end of sim-006
        ("move", "FORWARD"),            # sim-007: same text as sim-001, distinct utterance
        ("move", "UP"),                 # sim-008 replaces sim-007
        ("stop", "duration_elapsed"),   # end of sim-008
    ]
    reasons = {r["line"]: r["dispatch"]["reason"] for r in records if "dispatch" in r and r["line"]}
    assert reasons[3] == "DUPLICATE"
    assert reasons[5] == reasons[6] == reasons[7] == "UNKNOWN_EXPRESSION"

    # same file, same state: nothing is executed again
    out2 = tmp_path / "out2.jsonl"
    code, _ = run("batch", str(FIXTURE), "--out", str(out2), "--state", str(state))
    assert code == 0
    records = [json.loads(line) for line in out2.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["simulated_adapter_calls"] == []

    # existing output file is never overwritten
    before = out1.read_bytes()
    code, _ = run("batch", str(FIXTURE), "--out", str(out1), "--state", str(tmp_path / "other.jsonl"))
    assert code == 2 and out1.read_bytes() == before


def test_batch_refuses_invalid_input_before_acting(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"utterance_id": "a", "text": "وقف", "is_final": true}\n{"utterance_id": "b"}\n',
                   encoding="utf-8")
    code, _ = run("batch", str(bad), "--out", str(tmp_path / "o.jsonl"), "--state", str(tmp_path / "s.jsonl"))
    assert code == 2
    assert not (tmp_path / "o.jsonl").exists()


def test_replaying_an_old_message_file_is_rejected(tmp_path):
    saved = tmp_path / "old.json"
    run("demo", "اطلع", "--now", "2020-01-01T00:00:00Z", "--save-json", str(saved))
    code, output = run("dispatch", str(saved), "--state", str(tmp_path / "s.jsonl"))
    assert code == 0
    assert '"reason": "EXPIRED"' in output
    assert calls_section(output) == []


def test_module_entry_point_prints_arabic_as_utf8():
    proc = subprocess.run([sys.executable, "-m", "drone_voice", "demo", "وقف"], cwd=REPO,
                          capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert "وقف".encode("utf-8") in proc.stdout
