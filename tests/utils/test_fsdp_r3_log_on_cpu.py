# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Audit genuine controller output and reject missing/altered replay evidence."""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.check_fsdp_r3_log import R3AuditError, audit_r3_log, main
from verl.utils.fsdp_router_replay import FSDPRouterReplay, validate_vllm_r3_routes
from verl.utils.fsdp_router_replay_logging import r3_warning


def _run(caplog):
    caplog.set_level("WARNING")
    lengths = [4, 7]
    offsets = torch.tensor([0, 4, 11])
    ids = torch.nested.nested_tensor_from_jagged(torch.arange(11), offsets)
    values = (torch.arange(11 * 2).view(11, 2, 1) + torch.tensor([2, 3])) % 4
    routes = torch.nested.nested_tensor_from_jagged(values, offsets)
    for i, piece in enumerate(values.split(lengths)):
        validate_vllm_r3_routes(piece[:-1], len(piece) - 2, 2, 2, 2, 4, f"sample-{i}")
    controller = FSDPRouterReplay(2, 4, 2, 0, True)
    # Real install() itself is tested through the production block tests.
    import logging

    logging.getLogger(__name__).warning("[R3][INSTALL] rank=0 layers=2 experts=4 top_k=2 checkpointing=True")
    logits = torch.randn(11, 4)
    native = torch.tensor([0, 1]).expand(11, -1)
    for forward_only in (True, False):
        with controller.batch(forward_only):
            controller.set_microbatch(ids, routes, torch.ones(2, 3))
            for layer in range(2):
                controller.route(layer, logits, native, True)
            if not forward_only:
                controller.mark_backward()
                for layer in (1, 0):
                    controller.route(layer, logits, native.flip(-1), True)
            controller.finish_microbatch(forward_only)
    return caplog.text


def test_complete_log_matches_capture_to_both_training_phases(caplog):
    result = audit_r3_log(_run(caplog), 1, 1, require_native_diff=True)
    assert result["status"] == "PASS"
    assert result["unique_rollout_signatures"] == 2
    assert result["verified_batches"] == 2


@pytest.mark.parametrize("damage", ["capture", "transport", "verify", "rank", "mismatch", "fingerprint", "failed"])
def test_audit_rejects_missing_or_corrupt_evidence(caplog, damage):
    text = _run(caplog)
    if damage in ("capture", "transport", "verify"):
        text = "\n".join(line for line in text.splitlines() if f"[R3][{damage.upper()}]" not in line)
    elif damage == "mismatch":
        text = text.replace("replay_mismatch_slots=0", "replay_mismatch_slots=1")
    elif damage == "fingerprint":
        text = text.replace("sample_fingerprints=[", "sample_fingerprints=[999999999, ")
    elif damage == "failed":
        text += "\nWARNING [R3][FAILED] rank=0 phase=actor_update\n"
    with pytest.raises(ValueError, match="FAILED|mismatch"):
        audit_r3_log(text, 2 if damage == "rank" else 1, 1)


def test_audit_merges_mirrored_records_without_double_counting(caplog):
    text = _run(caplog)
    mirror = "\n".join("WARNING:pid=1234:" + line[line.index("[R3]") :] for line in text.splitlines() if "[R3]" in line)
    assert audit_r3_log(text + "\n" + mirror, 1, 1) == audit_r3_log(text, 1, 1)


@pytest.mark.parametrize("tag", ["VERIFY", "TRANSPORT", "CAPTURE"])
def test_audit_rejects_conflicting_mirrored_records(caplog, tag):
    text = _run(caplog)
    line = next(line for line in text.splitlines() if f"[R3][{tag}]" in line)
    if tag == "VERIFY":
        line = line.replace("native_diff_slots=", "native_diff_slots=9")
    elif tag == "TRANSPORT":
        line = line.replace("sample_fingerprints=[", "sample_fingerprints=[9")
    else:
        line = line.replace("fingerprint=", "fingerprint=9")
    with pytest.raises(R3AuditError, match=f"Conflicting {tag}") as error:
        audit_r3_log(text + "\n" + line, 1, 1)
    assert error.value.result["failure_kind"] == "INVALID_EVIDENCE"


def test_missing_capture_is_not_certified_by_training_transport(caplog):
    text = "\n".join(line for line in _run(caplog).splitlines() if "[R3][CAPTURE]" not in line)
    with pytest.raises(R3AuditError) as error:
        audit_r3_log(text, 1, 1)
    assert error.value.result["failure_kind"] == "MISSING_EVIDENCE"
    assert error.value.result["observed"]["verified_batches"] == 2
    assert "lack matching CAPTURE" in str(error.value)


@pytest.mark.parametrize("complete", [True, False])
def test_cli_combines_files_and_always_writes_json(caplog, capsys, monkeypatch, tmp_path, complete):
    text = _run(caplog)
    main_log = tmp_path / "train.log"
    main_log.write_text("\n".join(line for line in text.splitlines() if "[R3][CAPTURE]" not in line), encoding="utf-8")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    if complete:
        (evidence / "r3-host-123.log").write_text(text, encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["check", str(main_log), "--ranks", "1", "--min-batches", "1", "--evidence-dir", str(evidence)]
    )
    assert main() == (0 if complete else 1)
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == ("PASS" if complete else "FAILED")
    assert len(result["log_files"]) == (2 if complete else 1)


def test_cli_missing_log_produces_json_failure(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["check", str(tmp_path / "missing.log")])
    assert main() == 1
    assert json.loads(capsys.readouterr().out)["failure_kind"] == "LOG_READ_OR_PARSE_ERROR"


def test_warning_evidence_survives_disabled_worker_logging(monkeypatch, tmp_path):
    monkeypatch.setenv("VERL_R3_LOG_DIR", str(tmp_path))
    logger = logging.getLogger("verl.r3")
    monkeypatch.setattr(logger, "disabled", True)
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(logger, "level", logging.CRITICAL)
    previous = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        r3_warning("[R3][VERIFY] rank=%s status=PASS", 0)
        try:
            raise RuntimeError("test failure detail")
        except RuntimeError:
            r3_warning("[R3][FAILED] rank=0", exc_info=True)
    finally:
        logging.disable(previous)
    files = list(tmp_path.glob("r3-*.log"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "WARNING:" in text and "[R3][VERIFY] rank=0 status=PASS" in text
    assert "RuntimeError: test failure detail" in text


def test_evidence_file_errors_are_not_silently_ignored(monkeypatch, tmp_path):
    target = tmp_path / "not-a-directory"
    target.write_text("occupied", encoding="utf-8")
    monkeypatch.setenv("VERL_R3_LOG_DIR", str(target))
    with pytest.raises(OSError):
        r3_warning("[R3][CAPTURE] request=test")


def test_production_capture_and_controller_persist_auditable_evidence(caplog, monkeypatch, tmp_path):
    monkeypatch.setenv("VERL_R3_LOG_DIR", str(tmp_path))
    text = _run(caplog)
    # _run simulates INSTALL separately from the real controller, so include it
    # in the merged driver log just as production does.
    evidence = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.glob("r3-*.log"))
    installs = "\n".join(line for line in text.splitlines() if "[R3][INSTALL]" in line)
    assert audit_r3_log(installs + "\n" + evidence, 1, 1)["status"] == "PASS"


@pytest.mark.parametrize("training_status,complete", [(0, True), (134, True), (0, False), (134, False)])
def test_launcher_audits_after_training_failure_and_preserves_exit_code(caplog, tmp_path, training_status, complete):
    bash = shutil.which("bash")
    if os.name == "nt":
        git = shutil.which("git")
        bash = str(Path(git).parents[1] / "bin/bash.exe") if git else None
    if not bash or not Path(bash).is_file():
        pytest.skip("A working Bash is required to exercise the launcher")
    repo = Path(__file__).parents[2]
    launcher = tmp_path / "examples/grpo_trainer/run_qwen3_moe_r3_fsdp2.sh"
    launcher.parent.mkdir(parents=True)
    shutil.copyfile(repo / "examples/grpo_trainer/run_qwen3_moe_r3_fsdp2.sh", launcher)
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(repo / "scripts/check_fsdp_r3_log.py", tmp_path / "scripts/check_fsdp_r3_log.py")
    text = _run(caplog)
    full_log = "\n".join(
        re.sub(
            r"batch=(\d+)",
            lambda m, cycle=cycle: f"batch={int(m[1]) + 2 * cycle}",
            text.replace("rank=0", f"rank={rank}"),
        )
        for rank in range(8)
        for cycle in range(2)
    )
    if not complete:
        full_log = "\n".join(line for line in full_log.splitlines() if "[R3][CAPTURE]" not in line)
    source_log = tmp_path / "source.log"
    source_log.write_text(full_log, encoding="utf-8")
    baseline = tmp_path / "baseline.sh"
    baseline.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\ncat "$R3_TEST_SOURCE_LOG" > "$LOG_DIR/train.log"\n'
        'exit "$R3_TEST_TRAIN_STATUS"\n',
        encoding="utf-8",
    )
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    python_shim = binary_dir / "python3"
    # Skip only the environment inventory, which imports server-only packages.
    # The launcher and its post-training audit execute their actual code.
    python_shim.write_text(
        '#!/usr/bin/env bash\nif [[ $1 == - ]]; then cat >/dev/null; exit 0; fi\nexec "$R3_TEST_PYTHON" "$@"\n',
        encoding="utf-8",
    )
    python_shim.chmod(0o755)
    log_dir = tmp_path / "logs"
    env = {
        **os.environ,
        "PATH": str(binary_dir) + os.pathsep + os.environ["PATH"],
        "LOG_DIR": str(log_dir),
        "RUN_ROOT": str(tmp_path / "run"),
        "R3_TEST_PYTHON": sys.executable,
        "R3_TEST_SOURCE_LOG": str(source_log),
        "R3_TEST_TRAIN_STATUS": str(training_status),
    }
    result = subprocess.run(
        [bash, str(launcher), "--smoke", str(baseline)], env=env, capture_output=True, text=True, timeout=30
    )
    audit_status = 0 if complete else 1
    assert result.returncode == (training_status or audit_status), result.stdout + result.stderr
    assert json.loads((log_dir / "r3_launcher_status.json").read_text()) == {
        "training_exit_code": training_status,
        "audit_exit_code": audit_status,
    }
    assert json.loads((log_dir / "r3_log_audit.json").read_text())["status"] == ("PASS" if complete else "FAILED")


def test_evidence_directory_is_forwarded_to_ray_workers(monkeypatch):
    # Execute the real env builder without requiring Ray or an NPU runtime.
    import ast

    source = Path(__file__).parents[2] / "verl/trainer/constants_ppo.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    tree.body = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_ppo_ray_runtime_env"
    ]
    namespace = {
        "os": os,
        "json": json,
        "RAY_JOB_CONFIG_JSON_ENV_VAR": "R3_TEST_RAY_JOB",
        "PPO_RAY_RUNTIME_ENV": {"env_vars": {}},
    }
    exec(compile(tree, str(source), "exec"), namespace)
    monkeypatch.setenv("VERL_R3_LOG_DIR", "/run/logs/evidence")
    assert namespace["get_ppo_ray_runtime_env"]()["env_vars"]["VERL_R3_LOG_DIR"] == "/run/logs/evidence"
