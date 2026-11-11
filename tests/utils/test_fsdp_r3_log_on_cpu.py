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

import pytest
import torch

from scripts.check_fsdp_r3_log import audit_r3_log
from verl.utils.fsdp_router_replay import FSDPRouterReplay, validate_vllm_r3_routes


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
