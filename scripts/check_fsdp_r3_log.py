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
import sys
from collections import Counter
from pathlib import Path


class R3AuditError(ValueError):
    """Expose missing evidence separately from contradictory or invalid records."""

    def __init__(self, missing, invalid, observed):
        self.result = {
            "status": "FAILED",
            "failure_kind": "INVALID_EVIDENCE" if invalid else "MISSING_EVIDENCE",
            "missing_evidence": missing,
            "invalid_evidence": invalid,
            "observed": observed,
            "numerical_equivalence": "NOT_CERTIFIED",
        }
        errors = invalid + missing
        super().__init__("R3 audit FAILED:\n" + "\n".join(errors[:30]))


def audit_r3_log(text: str, ranks: int, min_batches: int, require_native_diff: bool = False) -> dict:
    """Match every training sample signature to a validated rollout capture."""
    errors = []
    missing = []
    captures = set()
    capture_requests = {}
    transports = {}
    verifications = {}
    installs = set()
    native_diff_total = 0
    for line in text.splitlines():
        # Ray, the driver tee, and the direct per-process sink have different
        # timestamp/PID prefixes. Compare only the actual R3 record payload.
        start = line.find("[R3]")
        if start < 0:
            continue
        line = line[start:]
        if "[R3][FAILED]" in line or "[R3][CAPTURE_INVALID]" in line:
            errors.append(line)
        if "[R3][CAPTURE]" in line:
            shape = re.search(r"shape=(\([^)]*\))", line)
            fp = re.search(r"fingerprint=(\d+)", line)
            if shape is None or fp is None or "status=PASS" not in line:
                errors.append(f"Incomplete capture log: {line}")
            else:
                signature = (ast.literal_eval(shape[1])[0], int(fp[1]))
                request = re.search(r"request=(\S+)", line)
                if request is not None:
                    if request[1] in capture_requests and capture_requests[request[1]] != signature:
                        errors.append(f"Conflicting CAPTURE records for request={request[1]}")
                    capture_requests[request[1]] = signature
                captures.add(signature)
        if "[R3][INSTALL]" in line:
            rank = re.search(r"rank=(\d+)", line)
            if rank is None:
                errors.append(f"Incomplete install log: {line}")
            else:
                installs.add(int(rank[1]))
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
            key = scope.groups()
            signatures = list(zip(lengths, fps, strict=True))
            if key in transports and transports[key] != signatures:
                errors.append(f"Conflicting TRANSPORT records for {key}")
            transports[key] = signatures
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
            if key in verifications and verifications[key] != fields:
                errors.append(f"Conflicting VERIFY records for {key}")
            verifications[key] = fields
    for rank in range(ranks):
        if rank not in installs:
            missing.append(f"rank={rank}: missing INSTALL")
        for phase in ("old_log_prob", "actor_update"):
            count = sum(key[:2] == (rank, phase) for key in verifications)
            if count < min_batches:
                missing.append(f"rank={rank} phase={phase}: only {count} verified batches, require {min_batches}")
    if not captures:
        missing.append("No validated rollout CAPTURE records")
    if not transports:
        missing.append("No TRANSPORT records")
    seen = Counter()
    unknown = Counter()
    for scope, signatures in transports.items():
        rank, phase, batch, micro = scope
        key = (int(rank), phase, int(batch))
        seen[key] += 1
        if key not in verifications:
            missing.append(f"Missing VERIFY for training batch: {scope}")
        for signature in signatures:
            if signature not in captures:
                unknown[signature] += 1
    if unknown:
        missing.append(
            f"{len(unknown)} rollout signatures ({sum(unknown.values())} sample uses) lack matching CAPTURE evidence; "
            f"examples={list(unknown)[:5]}. Supply this run's rollout worker logs; "
            "TRANSPORT cannot certify its own source."
        )
    for key, fields in verifications.items():
        if seen[key] != int(fields["microbatches"]):
            missing.append(f"{key}: expected {fields['microbatches']} TRANSPORT records, got {seen[key]}")
        native_diff_total += int(fields["native_diff_slots"])
    if require_native_diff and native_diff_total == 0:
        missing.append("No native route differs from rollout; this run did not exercise an actual routing override")
    observed = {
        "installed_ranks": sorted(installs),
        "verified_ranks": sorted({key[0] for key in verifications}),
        "unique_rollout_signatures": len(captures),
        "transport_microbatches": len(transports),
        "verified_batches": len(verifications),
    }
    if errors or missing:
        raise R3AuditError(missing, errors, observed)
    return {
        "status": "PASS",
        "validation_scope": "capture_transport_dispatch_only",
        "numerical_equivalence": "NOT_CERTIFIED",
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
    parser.add_argument("--extra-log", type=Path, action="append", default=[], help="Additional log from the SAME run")
    parser.add_argument("--evidence-dir", type=Path, help="This run's VERL_R3_LOG_DIR containing r3-*.log files")
    args = parser.parse_args()
    paths = [args.log, *args.extra_log]
    if args.evidence_dir:
        paths.extend(sorted(args.evidence_dir.glob("r3-*.log")))
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    try:
        if args.evidence_dir and not args.evidence_dir.is_dir():
            raise OSError(f"Evidence directory does not exist: {args.evidence_dir}")
        result = audit_r3_log(
            "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in paths),
            args.ranks,
            args.min_batches,
            args.require_native_diff,
        )
    except R3AuditError as error:
        result = error.result
        print(error, file=sys.stderr)
    except (OSError, ValueError, SyntaxError, TypeError, IndexError) as error:
        result = {"status": "FAILED", "failure_kind": "LOG_READ_OR_PARSE_ERROR", "error": str(error)}
        print(f"R3 audit FAILED: {error}", file=sys.stderr)
    result["log_files"] = [str(path) for path in paths]
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
