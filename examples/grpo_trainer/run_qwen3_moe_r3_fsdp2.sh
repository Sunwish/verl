#!/usr/bin/env bash
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

# Wrap either archived BF16 or FDW W4A8 + rotFD script without changing its
# training, quantization, TP/EP, graph, or async-scheduling settings.
set -euo pipefail

smoke=false
if [[ ${1:-} == --smoke ]]; then
    smoke=true
    shift
fi
if [[ $# -lt 1 || ! -f $1 ]]; then
    echo "Usage: bash $0 [--smoke] /path/to/original_baseline_resume.sh [Hydra overrides...]" >&2
    exit 2
fi
baseline_script=$(realpath "$1")
shift
launcher_script=$(realpath "${BASH_SOURCE[0]}")
repo_root=$(cd -- "$(dirname -- "$launcher_script")/../.." && pwd)
cd "$repo_root"

variant=bf16
if [[ $baseline_script == *qatw4a8* ]]; then
    variant=fdw_w4a8_rotfd
fi
run_stamp=$(date +%Y%m%d_%H%M%S)
if $smoke; then
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_30b_a3b_${variant}_r3_smoke_${run_stamp}}
    export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
    export RESUME_MODE=${RESUME_MODE:-disable}
    export SAVE_FREQ=${SAVE_FREQ:--1}
    export TEST_FREQ=${TEST_FREQ:--1}
    export VAL_BEFORE_TRAIN=False
else
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_30b_a3b_${variant}_r3}
fi
export RUN_ROOT=${RUN_ROOT:-/home/s00885728/outputs/${EXPERIMENT_NAME}/resume_${EXPERIMENT_NAME}}
export LOG_DIR=${LOG_DIR:-${RUN_ROOT}/logs/${run_stamp}}
export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
mkdir -p "$LOG_DIR"
cp "$launcher_script" "$LOG_DIR/r3_launcher.sh"

# Record actual import locations and installed versions without importing NPU
# plugins before Ray assigns devices to its workers.
python3 - "$repo_root" <<'PY' 2>&1 | tee "$LOG_DIR/r3_environment.log"
import importlib.metadata
import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
for module, package in (("verl", "verl"), ("vllm", "vllm"), ("vllm_ascend", "vllm-ascend")):
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"[R3][ENV] missing package: {module}")
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "source-checkout"
    root = Path(spec.origin).resolve().parent.parent
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    logging.warning("[R3][ENV] module=%s version=%s file=%s git=%s", module, version, spec.origin, result.stdout.strip())
logging.warning("[R3][ENV] python=%s repo=%s", sys.executable, sys.argv[1])
PY

echo "[R3] baseline=${baseline_script} smoke=${smoke} RUN_ROOT=${RUN_ROOT} LOG_DIR=${LOG_DIR}" >&2
bash "$baseline_script" "$@" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.router_replay.mode=R3 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent \
    actor_rollout_ref.ref.fsdp_config.router_replay.mode=disabled \
    ++actor_rollout_ref.actor.use_rollout_log_probs=False \
    algorithm.rollout_correction.bypass_mode=False

python3 scripts/check_fsdp_r3_log.py "$LOG_DIR/train.log" --ranks 8 --min-batches 2 | tee "$LOG_DIR/r3_log_audit.json"
