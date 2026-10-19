# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import pytest

from verl.workers.config.engine import FSDPEngineConfig, McoreEngineConfig, QATEngineConfig


class TestMcoreEngineConfig:
    def test_default_values(self):
        config = McoreEngineConfig()
        assert config.tensor_model_parallel_size == 1
        assert config.sequence_parallel is False  # Should be auto-corrected
        assert config.seed == 42

    def test_post_init_validation(self):
        # Test TP size 1 forces sequence_parallel=False
        config = McoreEngineConfig(tensor_model_parallel_size=1)
        assert config.sequence_parallel is False

        # Test TP >1 keeps sequence_parallel=True
        config = McoreEngineConfig(tensor_model_parallel_size=2)
        assert config.sequence_parallel is True

    def test_mutable_fields(self):
        config = McoreEngineConfig()
        config.sequence_parallel = True  # Should be mutable
        with pytest.raises(AttributeError):
            config.tensor_model_parallel_size = 2  # Frozen field

    @pytest.mark.parametrize("offload_field", ["param_offload", "grad_offload", "optimizer_offload"])
    def test_offload_flags(self, offload_field):
        config = McoreEngineConfig(**{offload_field: True})
        assert getattr(config, offload_field) is True


class TestFSDPEngineConfigCPU:
    def test_default_values(self):
        config = FSDPEngineConfig()
        assert config.param_offload is False
        assert config.optimizer_offload is False
        assert config.fsdp_size == -1

    @pytest.mark.parametrize(
        "offload_params",
        [{"param_offload": True}, {"optimizer_offload": True}, {"param_offload": True, "optimizer_offload": True}],
    )
    def test_offload_combinations(self, offload_params):
        config = FSDPEngineConfig(**offload_params)
        assert config.param_offload == offload_params.get("param_offload", False)
        assert config.optimizer_offload == offload_params.get("optimizer_offload", False)

    def test_wrap_policy_configuration(self):
        test_policy = {"layer_class": "TransformerBlock"}
        config = FSDPEngineConfig(wrap_policy=test_policy)
        assert config.wrap_policy == test_policy


class TestQATEngineConfig:
    @pytest.mark.parametrize("mode", ["w4a8_mxfp4", "w4a4_mxfp4"])
    def test_mxfp4_shared_options_and_backend_validation(self, mode):
        config = QATEngineConfig(
            enable=True,
            mode=mode,
            group_size=32,
            quant_backend="torch",
            rounding_mode="hash",
            rotation_enable=True,
            rotation_targets=["Fprop", "Dgrad", "Wgrad"],
            fake_quant_targets=["Dgrad", "Wgrad"],
            experts={"enable": True},
            fallback_layers="0-1,4",
        )
        assert config.fake_quant_targets == ["dgrad", "wgrad"]
        assert config.rotation_targets == ["fprop", "dgrad", "wgrad"]
        assert config.experts.enable and config.fallback_layers == [[0, 1], [4, 4]]
        with pytest.raises(ValueError, match="quant_backend='torch'"):
            QATEngineConfig(enable=True, mode=mode, group_size=32, quant_backend="npu")

    def test_mxfp8_rounding_mode_defaults_to_rint(self):
        config = QATEngineConfig()
        assert config.rounding_mode == "rint"

    def test_mxfp8_fake_quant_targets_default_and_normalization(self):
        assert QATEngineConfig().fake_quant_targets == ["fprop"]

        config = QATEngineConfig(fake_quant_targets=["Wgrad", "Dgrad"])
        assert config.fake_quant_targets == ["dgrad", "wgrad"]

    def test_mxfp8_backward_reduction_axis_fake_quant_default_and_override(self):
        assert QATEngineConfig().backward_reduction_axis_fake_quant is False

        config = QATEngineConfig(backward_reduction_axis_fake_quant=True)
        assert config.backward_reduction_axis_fake_quant is True

    def test_mxfp8_fallback_layers_normalize_to_inclusive_ranges(self):
        config = QATEngineConfig(fallback_layers="0-3,37-48")
        assert config.fallback_layers == [[0, 3], [37, 48]]

    def test_qat_experts_subconfig_defaults_to_disabled(self):
        config = QATEngineConfig()
        assert config.experts.enable is False

    def test_qat_experts_subconfig_accepts_dict(self):
        config = QATEngineConfig(experts={"enable": True})
        assert config.experts.enable is True

    def test_rejects_stochastic_rounding_with_npu_backend(self):
        with pytest.raises(ValueError, match="quant_backend='torch'"):
            QATEngineConfig(
                enable=True,
                mode="w8a8_mxfp8",
                group_size=32,
                quant_backend="npu",
                rounding_mode="hash",
            )
