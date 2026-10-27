import json

import pytest
from omegaconf import OmegaConf

from verl.utils.qat.vllm_ascend import mxfp_qat_config_to_vllm_ascend_env


@pytest.mark.parametrize(
    ("mode", "expected_mode"),
    [
        ("mxfp_w8a8", "w8a8_mxfp8"),
        ("mxfp_w4a8", "w4a8_mxfp4"),
        ("mxfp_w4a4", "w4a4_mxfp4"),
    ],
)
def test_mxfp_qat_config_maps_vllm_ascend_env(mode, expected_mode):
    env = mxfp_qat_config_to_vllm_ascend_env(
        {
            "mode": mode,
            "group_size": 32,
            "quant_backend": "torch",
            "rounding_mode": "hash",
            "fake_quant_targets": ["Fprop", "Dgrad", "Wgrad"],
            "rotation_enable": True,
            "rotation_targets": ["Fprop", "Wgrad"],
            "rotation_block_size": 32,
            "rotation_seed": 19,
            "ignore_patterns": ["lm_head"],
            "experts": {"enable": True},
            "fallback_layers": [[0, 2], [37, 40]],
            "mxfp4_scale_alg": "cx",
        }
    )

    assert env["VLLM_ASCEND_QAT_FAKE_QUANT"] == "1"
    assert env["VLLM_ASCEND_QAT_FAKE_QUANT_MODE"] == expected_mode
    assert env["VLLM_ASCEND_QAT_MXFP8_QUANT_BACKEND"] == "torch"
    assert env["VLLM_ASCEND_QAT_MXFP8_ROUNDING_MODE"] == "hash"
    assert env["VLLM_ASCEND_QAT_MXFP4_SCALE_ALG"] == "cx"
    assert json.loads(env["VLLM_ASCEND_QAT_MXFP8_FAKE_QUANT_TARGETS"]) == ["fprop", "dgrad", "wgrad"]
    assert env["VLLM_ASCEND_QAT_MXFP8_ROTATION_ENABLE"] == "true"
    assert json.loads(env["VLLM_ASCEND_QAT_MXFP8_ROTATION_TARGETS"]) == ["fprop", "wgrad"]
    assert json.loads(env["VLLM_ASCEND_QAT_MXFP8_IGNORE_PATTERNS"]) == ["lm_head"]
    assert json.loads(env["VLLM_ASCEND_QAT_MXFP8_FALLBACK_LAYERS"]) == [[0, 2], [37, 40]]


def test_mxfp_qat_config_keeps_disabled_experts_in_high_precision():
    env = mxfp_qat_config_to_vllm_ascend_env(
        {"mode": "mxfp_w8a8", "group_size": 32, "experts": {"enable": False}}
    )

    ignore_patterns = json.loads(env["VLLM_ASCEND_QAT_MXFP8_IGNORE_PATTERNS"])
    assert any("mlp\\.experts" in pattern for pattern in ignore_patterns)
    assert any("shared_expert" in pattern for pattern in ignore_patterns)


def test_mxfp_qat_config_accepts_hydra_config_objects():
    config = OmegaConf.create(
        {
            "mode": "mxfp_w8a8",
            "group_size": 32,
            "quant_backend": "npu",
            "fake_quant_targets": ["Fprop", "Dgrad"],
            "rotation_targets": ["Wgrad"],
            "fallback_layers": [[0, 2]],
            "experts": {"enable": True},
        }
    )

    env = mxfp_qat_config_to_vllm_ascend_env(config)

    assert json.loads(env["VLLM_ASCEND_QAT_MXFP8_FAKE_QUANT_TARGETS"]) == ["fprop", "dgrad"]
    assert json.loads(env["VLLM_ASCEND_QAT_MXFP8_ROTATION_TARGETS"]) == ["wgrad"]
    assert json.loads(env["VLLM_ASCEND_QAT_MXFP8_FALLBACK_LAYERS"]) == [[0, 2]]


def test_mxfp_qat_config_rejects_unsupported_group_size():
    with pytest.raises(ValueError, match="group_size=32"):
        mxfp_qat_config_to_vllm_ascend_env({"mode": "mxfp_w8a8", "group_size": 16})


def test_mxfp_qat_config_rejects_npu_stochastic_rounding():
    with pytest.raises(ValueError, match="requires quant_backend='torch'"):
        mxfp_qat_config_to_vllm_ascend_env(
            {"mode": "mxfp_w8a8", "group_size": 32, "quant_backend": "npu", "rounding_mode": "hash"}
        )
