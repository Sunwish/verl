# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import importlib.util
from pathlib import Path

import pytest

_GSM8K_REWARD_PATH = Path(__file__).resolve().parents[3] / "verl" / "utils" / "reward_score" / "gsm8k.py"
_SPEC = importlib.util.spec_from_file_location("gsm8k_reward_score", _GSM8K_REWARD_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
gsm8k = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gsm8k)


@pytest.mark.parametrize(
    ("solution_str", "expected"),
    [
        ("We solve it step by step.\n#### 1,234", "1234"),
        (r"The final answer is \boxed{1,234}.", "1234"),
        (r"The final answer is $\boxed{-3.5}$.", "-3.5"),
        (
            r"""First answer:
\boxed{7}
Final answer:
#### 8""",
            "8",
        ),
        (
            r"""First answer:
#### 7
Final answer:
\boxed{8}""",
            "8",
        ),
        ("No final answer here.", None),
    ],
)
def test_extract_solution_strict_accepts_hash_and_boxed_answers(solution_str, expected):
    assert gsm8k.extract_solution(solution_str, method="strict") == expected


def test_compute_score_accepts_boxed_answer():
    assert gsm8k.compute_score(solution_str=r"The final answer is \boxed{42}.", ground_truth="42") == 1.0
