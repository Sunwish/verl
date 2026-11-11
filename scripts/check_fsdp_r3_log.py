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
"""Audit WARNING R3 logs from a single FSDP2 + vLLM-Ascend run.

This checks the observations present in a log; it does not certify NPU kernel
numerics. Missing, truncated, or deduplicated logs fail instead of passing.
"""

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path


def audit_r3_log(text: str, ranks: int, min_batches: int, require_native_diff: bool = False) -> dict:
    """Match every training sample signature to a validated rollout capture."""
    errors = []
    captures = set()
    transports = []
    verifications = {}
    installs = set()
    native_diff_total = 0
    for line in text.splitlines():
        if "[R3][FAILED]" in line or "[R3][CAPTURE_INVALID]" in line:
            errors.append(line)
        if "[R3][CAPTURE]" in line:
            shape = re.search(r"shape=(\([^)]*\))", line)
            fp = re.search(r"fingerprint=(\d+)", line)
            if shape is None or fp is None or "status=PASS" not in line:
                errors.append(f"Incomplete capture log: {line}")
            else:
                captures.add((ast.literal_eval(shape[1])[0], int(fp[1])))
        if "[R3][INSTALL]" in line:
            installs.add(int(re.search(r"rank=(\d+)", line)[1]))
        if "[R3][TRANSPORT]" in line:
            lengths = re.search(r"captured_lengths=(\[[^]]*\])", line)
            fps = re.search(r"sample_fingerprints=(\[[^]]*\])", line)
            scope = re.search(r"rank=(\d+) phase=(\w+) batch=(\d+) microbatch=(\d+)", line)
            if lengths is None or fps is None or scope is None:
                errors.append(f"Incomplete transport log: {line}")
                continue
            lengths, fps = ast.literal_eval(lengths[1]), ast.literal_eval(fps[1])
            if len(lengths) != len(fps):
                errors.append(f"Transport signature count mismatch: {line}")
                continue
            transports.append((scope.groups(), list(zip(lengths, fps, strict=True))))
        if "[R3][VERIFY]" in line:
            fields = dict(re.findall(r"(\w+)=([\w.]+)", line))
            required = {
                "rank",
                "phase",
                "batch",
                "microbatches",
                "layers",
                "forward_calls",
                "recompute_calls",
                "replayed_slots",
                "native_diff_slots",
                "replay_mismatch_slots",
                "coverage",
                "status",
            }
            if required - fields.keys():
                errors.append(f"Incomplete verification log: {line}")
                continue
            key = (int(fields["rank"]), fields["phase"], int(fields["batch"]))
            expected_calls = int(fields["microbatches"]) * int(fields["layers"])
            if (
                fields["status"] != "PASS"
                or int(fields["replay_mismatch_slots"]) != 0
                or float(fields["coverage"]) != 1.0
                or int(fields["replayed_slots"]) <= 0
                or int(fields["forward_calls"]) != expected_calls
                or (fields["phase"] == "actor_update" and int(fields["recompute_calls"]) < expected_calls)
            ):
                errors.append(f"Invalid verification: {line}")
            verifications[key] = fields
    for rank in range(ranks):
        if rank not in installs:
            errors.append(f"rank={rank}: missing INSTALL")
        for phase in ("old_log_prob", "actor_update"):
            count = sum(key[:2] == (rank, phase) for key in verifications)
            if count < min_batches:
                errors.append(f"rank={rank} phase={phase}: only {count} verified batches, require {min_batches}")
    if not captures:
        errors.append("No validated rollout CAPTURE records")
    if not transports:
        errors.append("No TRANSPORT records")
    seen = Counter()
    for scope, signatures in transports:
        rank, phase, batch, micro = scope
        key = (int(rank), phase, int(batch))
        seen[key] += 1
        if key not in verifications:
            errors.append(f"Unfinished training batch: {scope}")
        for signature in signatures:
            if signature not in captures:
                errors.append(
                    f"Unknown rollout signature {signature}: rank={rank} phase={phase} batch={batch} mb={micro}"
                )
    for key, fields in verifications.items():
        if seen[key] != int(fields["microbatches"]):
            errors.append(f"{key}: expected {fields['microbatches']} TRANSPORT records, got {seen[key]}")
        native_diff_total += int(fields["native_diff_slots"])
    if require_native_diff and native_diff_total == 0:
        errors.append("No native route differs from rollout; this run did not exercise an actual routing override")
    if errors:
        raise ValueError("R3 audit FAILED:\n" + "\n".join(errors[:30]))
    return {
        "status": "PASS",
        "ranks": ranks,
        "unique_rollout_signatures": len(captures),
        "transport_microbatches": len(transports),
        "verified_batches": len(verifications),
        "native_diff_slots": native_diff_total,
        "replay_mismatch_slots": 0,
        "note": "Log invariants verified; NPU kernel numerics and convergence require separate validation.",
    }


def main():
    """Read an existing run log and return a nonzero exit code on missing evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--min-batches", type=int, default=2)
    parser.add_argument("--require-native-diff", action="store_true")
    args = parser.parse_args()
    try:
        result = audit_r3_log(
            args.log.read_text(encoding="utf-8", errors="replace"),
            args.ranks,
            args.min_batches,
            args.require_native_diff,
        )
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
