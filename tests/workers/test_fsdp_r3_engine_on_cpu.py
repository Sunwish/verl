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
"""Exercise the production FSDP micro-batch driver without an NPU process group."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.checkpoint import checkpoint

from verl.utils import tensordict_utils as tu
from verl.utils.config import validate_config
from verl.utils.fsdp_router_replay import FSDPRouterReplay


@pytest.mark.parametrize("forward_only", [True, False])
@pytest.mark.parametrize("enabled", [True, False])
def test_engine_driver_scopes_targets_through_backward(monkeypatch, forward_only, enabled, caplog):
    path = Path(__file__).parents[2] / "verl/workers/engine/fsdp/transformer_impl.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FSDPEngine")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_backward_batch")
    tree.body = [method]
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **kw: None)
    replay = FSDPRouterReplay(1, 4, 2, 1, True) if enabled else None
    microbatches = []
    for length in (4, 7):
        offsets = torch.tensor([0, length])
        ids = torch.nested.nested_tensor_from_jagged(torch.arange(length), offsets)
        routes = torch.nested.nested_tensor_from_jagged(torch.tensor([[[2, 3]]]).expand(length, -1, -1), offsets)
        microbatches.append({"input_ids": ids, "routed_experts": routes, "response_mask": torch.ones(1, 3)})
    weight = torch.randn(4, 4, requires_grad=True)
    outputs = []

    def forward_step(micro_batch, **kwargs):
        if enabled:
            replay.set_microbatch(micro_batch["input_ids"], micro_batch["routed_experts"], micro_batch["response_mask"])
        length = micro_batch["input_ids"].values().numel()
        x = torch.randn(length, 4, requires_grad=not forward_only)

        def moe(x):
            logits = x @ weight
            native = logits.topk(2).indices
            if enabled:
                _, probs = replay.route(0, logits, native, True)
            else:
                probs = logits.softmax(-1).gather(-1, native)
            return probs.square().sum()

        loss = moe(x) if forward_only else checkpoint(moe, x, use_reentrant=False)
        outputs.append(loss.detach())
        return loss, {"tokens": length}

    ns = {
        "torch": torch,
        "tu": tu,
        "TensorDict": TensorDict,
        "Callable": object,
        "nullcontext": nullcontext,
        "get_device_id": lambda: "cpu",
        "prepare_micro_batches": lambda **kw: (microbatches, [0, 1]),
        "postprocess_batch_func": lambda **kw: kw["output_lst"],
    }
    exec(compile(tree, str(path), "exec"), ns)
    engine = SimpleNamespace(
        ulysses_sequence_parallel_size=1,
        _router_replay=replay,
        get_data_parallel_group=lambda: None,
        get_data_parallel_size=lambda: 1,
        forward_step=forward_step,
    )
    data = TensorDict({"loss_mask": torch.ones(2, 3)}, batch_size=[2])
    tu.assign_non_tensor_data(data, "enable_routing_replay", enabled)
    with caplog.at_level("WARNING"):
        result = ns["forward_backward_batch"](engine, data, None, forward_only=forward_only)
    assert result == [{"tokens": 4}, {"tokens": 7}]
    assert len(outputs) == 2
    assert (weight.grad is None) == forward_only
    if enabled:
        assert not replay.active and replay._targets is None
        assert "microbatches=2" in caplog.text
        assert f"recompute_calls={0 if forward_only else 2}" in caplog.text
        assert f"phase={'old_log_prob' if forward_only else 'actor_update'}" in caplog.text
    else:
        assert "[R3]" not in caplog.text


def test_r3_rejects_bypassed_old_log_prob_before_worker_launch():
    config = OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 8, "nnodes": 1},
            "actor_rollout_ref": {"actor": {"strategy": "fsdp2", "fsdp_config": {"router_replay": {"mode": "R3"}}}},
            "algorithm": {"rollout_correction": {"bypass_mode": True}},
        }
    )
    with pytest.raises(ValueError, match="bypass_mode=False"):
        validate_config(config, use_reference_policy=True, use_critic=False)


def test_production_forward_step_compares_native_without_replacing_replay_probabilities(caplog):
    from verl.workers.utils.padding import no_padding_2_padding

    path = Path(__file__).parents[2] / "verl/workers/engine/fsdp/transformer_impl.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FSDPEngineWithLMHead")
    tree.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_step"]
    ns = {
        "torch": torch,
        "TensorDict": TensorDict,
        "ContextManager": object,
        "nullcontext": nullcontext,
        "get_device_name": lambda: "cpu",
        "get_device_id": lambda: "cpu",
    }
    exec(compile(tree, str(path), "exec"), ns)

    offsets = torch.tensor([0, 5, 9])
    ids = torch.nested.nested_tensor_from_jagged(torch.arange(9), offsets)
    targets = torch.tensor([2, 3]).expand(9, 1, 2).clone()
    routes = torch.nested.nested_tensor_from_jagged(targets, offsets)
    replay = FSDPRouterReplay(1, 4, 2, 1, False, debug_native_compare=True)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    replay_logs = torch.linspace(-2, -0.1, 9)
    native_logs = replay_logs.flip(0).clone()
    data = TensorDict(
        {
            "input_ids": ids,
            "routed_experts": routes,
            "response_mask": mask,
            "prompts": torch.ones(2, 2, dtype=torch.long),
            "responses": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
        },
        batch_size=[2],
    )
    expected = no_padding_2_padding(torch.nested.nested_tensor_from_jagged(replay_logs, offsets), data)
    data["rollout_log_probs"] = no_padding_2_padding(native_logs, data)
    calls = []

    def model(use_cache):
        assert not use_cache
        calls.append(replay.active)
        native = torch.tensor([0, 1]).expand(9, 2)
        if replay.active:
            replay.route(0, torch.zeros(9, 4), native, True)
            logs = replay_logs
        else:
            assert not torch.is_grad_enabled()
            replay.record_native(0, native)
            logs = native_logs
        return torch.nested.nested_tensor_from_jagged(logs, offsets)

    engine = SimpleNamespace(
        _router_replay=replay,
        _autocast_dtype=torch.float32,
        module=model,
        prepare_model_inputs=lambda **kwargs: ({}, {}),
        prepare_model_outputs=lambda **kwargs: {"log_probs": kwargs["output"]},
    )
    with caplog.at_level("WARNING"), replay.batch(True):
        _, output = ns["forward_step"](engine, data, loss_function=None, forward_only=True)
        torch.testing.assert_close(no_padding_2_padding(output["model_output"]["log_probs"], data), expected)
        replay.finish_microbatch(True)
    assert calls == [True, False]
    assert "'native_rollout_pearson': 1.0" in caplog.text
    assert "[R3][VERIFY]" in caplog.text and "forward_calls=1 recompute_calls=0" in caplog.text
