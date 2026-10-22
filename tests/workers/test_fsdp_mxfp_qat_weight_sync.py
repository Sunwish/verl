from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("peft")

from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


@pytest.mark.parametrize("mode", ["mxfp_w8a8", "mxfp_w4a8", "mxfp_w4a4"])
def test_mxfp_weight_sync_keeps_high_precision_weights(mode, monkeypatch):
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.module = torch.nn.Linear(32, 32)
    engine.model_config = SimpleNamespace(lora={})
    engine.engine_config = SimpleNamespace(mixed_precision=None)
    engine._qat_enabled = True
    engine._qat_config = SimpleNamespace(mode=mode, group_size=32, ignore_patterns=[])
    engine._is_offload_param = False
    engine._uses_fsdp2_cpu_offload_policy = True
    engine.rank = 0

    monkeypatch.setattr(transformer_impl, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: torch.device("cpu"))

    original_params = engine.module.state_dict()
    params, peft_config = engine.get_per_tensor_param()
    synced_params = dict(params)

    assert peft_config is None
    assert synced_params.keys() == original_params.keys()
    for name, tensor in original_params.items():
        torch.testing.assert_close(synced_params[name], tensor)
