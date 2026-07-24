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

import sys
import types
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf
from unittest.mock import patch

from verl.workers.config import RolloutConfig
from verl.workers.rollout.utils import get_rollout_bootstrap_model_path


def _make_rollout_config(name: str):
    config = RolloutConfig(
        name=name,
        mode="async",
        max_model_len=128,
        max_num_seqs=8,
        max_num_batched_tokens=256,
        response_length=16,
        tensor_model_parallel_size=1,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_parallel_size=1,
        moe_tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        disable_log_stats=True,
        enable_sleep_mode=False,
        engine_kwargs={},
        load_format="auto",
    )
    return OmegaConf.create(asdict(config))


def test_get_rollout_bootstrap_model_path_defaults_to_shared_model_path():
    model_config = SimpleNamespace(local_path="/tmp/bf16-model", use_shm=False)
    rollout_config = SimpleNamespace(bootstrap_model_path=None)

    assert get_rollout_bootstrap_model_path(model_config, rollout_config) == "/tmp/bf16-model"


def test_get_rollout_bootstrap_model_path_materializes_override(monkeypatch):
    import verl.workers.rollout.utils as rollout_utils

    captured = {}

    def fake_copy_to_local(path, use_shm=False):
        captured["path"] = path
        captured["use_shm"] = use_shm
        return "/tmp/local-bootstrap"

    monkeypatch.setattr(rollout_utils, "copy_to_local", fake_copy_to_local)

    model_config = SimpleNamespace(local_path="/tmp/bf16-model", use_shm=True)
    rollout_config = SimpleNamespace(bootstrap_model_path="/mnt/quantized-model")

    resolved = get_rollout_bootstrap_model_path(model_config, rollout_config)

    assert resolved == "/tmp/local-bootstrap"
    assert captured == {"path": "/mnt/quantized-model", "use_shm": True}


def test_vllm_launch_server_uses_bootstrap_model_path(monkeypatch):
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout import vllm_async_server as vllm_server

    captured = {}

    class FakeParser:
        def __init__(self, *args, **kwargs):
            pass

        def add_subparsers(self, **kwargs):
            return SimpleNamespace()

        def parse_args(self, args):
            captured["argv"] = args
            return SimpleNamespace(model_tag=args[1], subparser=None)

    server = object.__new__(vllm_server.vLLMHttpServer)
    server.config = _make_rollout_config("vllm")
    server.model_config = SimpleNamespace(local_path="/tmp/bf16-model", trust_remote_code=False, lora_rank=0, lora={})
    server.rollout_bootstrap_model_path = "/tmp/quantized-model"
    server.node_rank = 0
    server.replica_rank = 0
    server.rollout_mode = vllm_server.RolloutMode.STANDALONE
    server.gpus_per_node = 1
    server.nnodes = 1
    server.workers = []
    server._master_address = None
    server._master_port = None
    server._dp_rpc_port = None
    server._dp_master_port = None
    server._server_address = "127.0.0.1"
    server.profiler_controller = SimpleNamespace(config=None, tool_config=None)
    server._get_cli_modules = lambda: []
    server.run_server = AsyncMock()
    server.run_headless = AsyncMock()
    server._apply_quantization = lambda: (None, {})

    monkeypatch.setattr(vllm_server, "build_cli_args_from_config", lambda config: [])
    monkeypatch.setattr(vllm_server, "FlexibleArgumentParser", FakeParser)

    import asyncio

    asyncio.run(vllm_server.vLLMHttpServer.launch_server(server))

    assert captured["argv"][:2] == ["serve", "/tmp/quantized-model"]
    server.run_server.assert_awaited_once()
    server.run_headless.assert_not_called()


def test_sglang_launch_server_uses_bootstrap_model_path(monkeypatch):
    pytest.importorskip("sglang")
    from verl.workers.rollout.sglang_rollout import async_sglang_server as sglang_server
    import sglang.srt.entrypoints.http_server as http_server

    captured = {}

    class FakeServerArgs:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.enable_metrics = False

    def fake_launch_subprocesses(**kwargs):
        captured["launch_kwargs"] = kwargs
        return MagicMock(), MagicMock(), MagicMock(), None

    server = object.__new__(sglang_server.SGLangHttpServer)
    server.config = _make_rollout_config("sglang")
    server.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(max_position_embeddings=128),
        trust_remote_code=False,
        lora_rank=0,
        lora={},
    )
    server.rollout_bootstrap_model_path = "/tmp/quantized-model"
    server.node_rank = 1
    server.nnodes = 2
    server._master_address = "127.0.0.1"
    server._master_port = 2345
    server._server_address = "127.0.0.2"
    server._server_port = None
    server.base_gpu_id = 0
    server.rollout_mode = sglang_server.RolloutMode.STANDALONE
    server.replica_rank = 0
    server.workers = []
    server.profiler_controller = SimpleNamespace(config=None, tool_config=None)
    server._disaggregation_role = "null"
    server._disaggregation_bootstrap_port = None

    monkeypatch.setattr(sglang_server, "ServerArgs", FakeServerArgs)
    monkeypatch.setattr(http_server, "_launch_subprocesses", fake_launch_subprocesses, raising=False)

    class FakeEngine:
        _launch_subprocesses = staticmethod(fake_launch_subprocesses)

    monkeypatch.setattr(http_server, "Engine", FakeEngine, raising=False)

    import asyncio

    asyncio.run(sglang_server.SGLangHttpServer.launch_server(server, master_address="127.0.0.1", master_port=2345))

    assert captured["kwargs"]["model_path"] == "/tmp/quantized-model"


def test_sglang_server_adapter_uses_bootstrap_model_path(monkeypatch):
    pytest.importorskip("sglang")
    from verl.workers.rollout.sglang_rollout import sglang_rollout

    captured = {}

    class FakeAsyncHttpServerAdapter:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    class FakeRemoteMethod:
        async def remote(self):
            return ("127.0.0.1", 8000)

    class FakeActor:
        get_server_address = FakeRemoteMethod()

    class FakeMesh:
        def __getitem__(self, key):
            return SimpleNamespace(get_local_rank=lambda: 0)

    adapter = object.__new__(sglang_rollout.ServerAdapter)
    adapter.config = SimpleNamespace(bootstrap_model_path="/tmp/quantized-model")
    adapter.model_config = SimpleNamespace(local_path="/tmp/bf16-model", use_shm=False, trust_remote_code=False)
    adapter.device_mesh = FakeMesh()
    adapter._engine = None
    adapter._has_server = True
    adapter._pd_role = None
    adapter.replica_rank = 0
    adapter.rollout_rank = 0
    adapter.node_rank = 0

    monkeypatch.setattr(sglang_rollout, "AsyncHttpServerAdapter", FakeAsyncHttpServerAdapter)
    monkeypatch.setattr(sglang_rollout.ray, "get_actor", lambda name: FakeActor())

    import asyncio

    asyncio.run(sglang_rollout.ServerAdapter._init_server_adapter(adapter))

    assert captured["kwargs"]["model_path"] == "/tmp/quantized-model"


@pytest.mark.parametrize("qat_mode", ["w8a16_mxfp8", "w8a8_mxfp8"])
def test_vllm_qat_mxfp8_routes_to_ascend_quantization(monkeypatch, qat_mode):
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout import vllm_async_server as vllm_server

    server = object.__new__(vllm_server.vLLMHttpServer)
    server.config = SimpleNamespace(
        quantization=None, quantization_config_file=None, qat={"enable": True, "mode": qat_mode}
    )
    server.model_config = SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=2))

    quant_config = {"quant_method": "ascend", "layer.weight": "W8A8_MXFP8"}
    monkeypatch.setattr(vllm_server, "is_torch_npu_available", lambda check_device=False: False)

    with patch("verl.utils.qat.load_quantization_config", return_value=quant_config), patch(
        "verl.workers.rollout.vllm_rollout.vllm_async_server.apply_vllm_fp8_patches"
    ) as mock_apply_patches:
        quantization, hf_overrides = vllm_server.vLLMHttpServer._apply_quantization(server)

    assert quantization == "ascend"
    assert hf_overrides["quantization_config"] == quant_config
    mock_apply_patches.assert_called_once()


def test_trtllm_launch_server_uses_bootstrap_model_path(monkeypatch):
    from verl.workers.rollout.trtllm_rollout import trtllm_async_server as trtllm_server

    captured = {}

    async def fake_async_llm(**kwargs):
        captured["llm_kwargs"] = kwargs
        return object()

    class FakeKvCacheConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCudaGraphConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSchedulerConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSleepConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCapacitySchedulerPolicy:
        MAX_UTILIZATION = "max-util"

    class FakeExecutorMemoryType:
        MODEL_WEIGHTS_MAIN = "weights"
        KV_CACHE = "kv"

    class FakeOpenAIServer:
        def __init__(self, **kwargs):
            captured["openai_model"] = kwargs["model"]
            self.app = object()

    fake_root = types.ModuleType("tensorrt_llm")
    fake_root.AsyncLLM = fake_async_llm

    fake_llmapi = types.ModuleType("tensorrt_llm.llmapi")
    fake_llmapi.CapacitySchedulerPolicy = FakeCapacitySchedulerPolicy
    fake_llmapi.CudaGraphConfig = FakeCudaGraphConfig
    fake_llmapi.KvCacheConfig = FakeKvCacheConfig
    fake_llmapi.SchedulerConfig = FakeSchedulerConfig

    fake_llm_args = types.ModuleType("tensorrt_llm.llmapi.llm_args")
    fake_llm_args.ExecutorMemoryType = FakeExecutorMemoryType
    fake_llm_args.SleepConfig = FakeSleepConfig

    fake_serve = types.ModuleType("tensorrt_llm.serve")
    fake_serve.OpenAIServer = FakeOpenAIServer

    monkeypatch.setitem(sys.modules, "tensorrt_llm", fake_root)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", fake_llmapi)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi.llm_args", fake_llm_args)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.serve", fake_serve)
    monkeypatch.setattr(trtllm_server, "run_uvicorn", AsyncMock(return_value=(8000, object())))

    server = object.__new__(trtllm_server.TRTLLMHttpServer.__ray_metadata__.modified_class)
    server.config = _make_rollout_config("trtllm")
    server.model_config = SimpleNamespace(trust_remote_code=False)
    server.rollout_bootstrap_model_path = "/tmp/quantized-model"
    server.max_colocate_count = 1
    server.pgs = []
    server.bundle_indices = []
    server.is_reward_model = False
    server.is_vlm_model = False
    server._use_torch_sampler = False
    server._server_address = "127.0.0.1"

    import asyncio

    asyncio.run(server.launch_server())

    assert captured["llm_kwargs"]["model"] == "/tmp/quantized-model"
    assert captured["openai_model"] == "/tmp/quantized-model"
