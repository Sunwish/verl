# Qwen3 MoE 的 FSDP2 + vLLM-Ascend R3 路由回放

Last updated: 11/11/2026

适用场景：Qwen3-30B-A3B，单机 8 张 A2，actor FSDP2、SP=1，rollout vLLM-Ascend TP=8 / EP=8 / DP=1。同时支持当前脚本中的 BF16 和 MXFP W4A8 F/D/W 伪量化 + F/D 旋转。路由回放固定专家 ID，混合权重由当前 actor gate logits 重新计算并归一化，保留 gate 和专家参数的梯度。

## 实现边界

- 采集并回放整个 prompt 和已计算的 response 输入位置。vLLM 产生 R 个 token 时，只有 P+R-1 个输入位置有路由；最后一个生成 token 没有执行推理前向。该位置使用训练前向的原生路由，检查点反向重算继续使用相同的专家。
- old-log-prob 重算、actor 更新和梯度检查点重算均回放；reference policy 使用自身路由。
- 支持动态微批次、样本重排、left/right padding、remove-padding、用于分布式补齐的零 loss 合成样本。
- rollout 支持原脚本的 chunked prefill、async scheduling 和 FULL_DECODE_ONLY 图配置；这些组合仍需服务器 NPU 实跑验收。Ascend 采集补丁使用等长 all-gather 重组 TP 分片，再移除每个 rank 的 padding，避免 HCCL 不支持不等长 all-gather。
- 首版要求 single_turn_agent、prefix caching 关闭、无 LoRA、torch.compile 关闭、remove-padding 开启。FSDP1、SP>1、多轮、原生量化推理、rollout DP>1 不在这两个脚本的支持范围。检测到不支持的 actor 配置会直接报错。
- 非重入检查点为原脚本默认行为。训练按微批次持有回放状态直到 backward 完成，异常退出会释放状态。

## 需要同步的仓库

必须同时同步 **verl** 与 **vllm-ascend** 的本次修改，确保服务器实际 import 的代码来自更新后的仓库。vLLM 本体无需修改。启动包装脚本会输出三者的 import 路径、版本、Git HEAD 和 Python 路径到 `r3_environment.log`；可编辑安装目录或源码覆盖必须与这些路径一致。

已检查的本地基线：verl `248b1159`，vLLM-Ascend `35101f05b`，vLLM `0fc695fc6d`。这些是修改前的基线 commit，工作区补丁不会改变 HEAD。

## 在两个原始脚本上启动

在 verl 仓库根目录运行，BASELINE 指向服务器上的对应原脚本。原脚本内部的模型、数据、batch、量化、TP/EP、graph 和 async 参数继续生效。

```bash
# BF16：BASELINE 指向之前的 ...bf16_bf16...resume_2.sh
# QAT：BASELINE 指向之前的 ...fdw_qatw4a8_qatw4a8_rotFD...resume.sh
BASELINE=/absolute/path/to/original_resume_script.sh
bash examples/grpo_trainer/run_qwen3_moe_r3_fsdp2.sh --smoke "$BASELINE"
```

`--smoke` 默认运行 2 个训练 step，保留原始 64/32、rollout n=4、512/512 等场景参数；使用带时间戳的新实验目录，禁用自动续训、保存及评测。无需缩短序列或改成 eager 就可先尝试原场景。不要指向旧实验的 RUN_ROOT/CKPT_DIR，除非有意续训它。

若第一轮需要区分采集问题和图/异步问题，可单独运行：

```bash
bash examples/grpo_trainer/run_qwen3_moe_r3_fsdp2.sh --smoke "$BASELINE" \
  actor_rollout_ref.rollout.enforce_eager=True \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=False
```

该轮通过后必须重新运行默认图 + async 配置，才能认为原脚本场景通过。两种精度分别验收。正常训练去掉 `--smoke`，默认实验名带 `_r3`，保持自己的 checkpoint 目录；恢复 R3 训练沿用同一实验名与 RUN_ROOT。路由属于当前 rollout batch，不存进模型 checkpoint；恢复后会重新采集。

直接修改自己的脚本时，关键 Hydra 参数为：

```bash
actor_rollout_ref.actor.fsdp_config.router_replay.mode=R3
actor_rollout_ref.rollout.enable_rollout_routing_replay=True
actor_rollout_ref.rollout.enable_prefix_caching=False
actor_rollout_ref.ref.fsdp_config.router_replay.mode=disabled
algorithm.rollout_correction.bypass_mode=False
```

不能打开 bypass_mode，否则 old-log-prob 不再进入本次回放路径。不要使用旧的 `actor.router_replay.mode` 字段：本实现读取 `actor.fsdp_config.router_replay.mode`。

## WARNING 日志与验收

所有新增功能日志均为 WARNING，无需打开 INFO。保留 `RAY_DEDUP_LOGS=0`。启动包装脚本在训练成功退出后自动执行下面的检查；也可独立运行：

```bash
python3 scripts/check_fsdp_r3_log.py /path/to/logs/train.log --ranks 8 --min-batches 2
# 额外要求本次训练确实遇到了“原生路由与 rollout 不同”的位置：
python3 scripts/check_fsdp_r3_log.py /path/to/logs/train.log --ranks 8 --min-batches 2 --require-native-diff
```

| 日志 | 能说明什么 |
| --- | --- |
| `[R3][CONFIG]` / `[INSTALL]` | 每个 actor rank 实际安装的层数、专家数、top-k、QAT/旋转和 checkpoint 设置 |
| `[R3][TP_CAPTURE]` | 推理端分片通信类型、TP 大小、重组前后的 token 数；图模式通常只在 tracing 时出现 |
| `[R3][CAPTURE]` | 每个 request 的 P/R、严格 P+R-1 形状、合法且无重复的专家 ID、顺序敏感指纹 |
| `[R3][TRANSPORT]` | 每个微批次每个真实样本的长度和指纹；检查工具逐个匹配到 rollout CAPTURE，检查传输、padding 和重排 |
| `[R3][TARGET]` / `[LAYERS]` | 首微批次长度、捕获/未捕获 token 数、各层原生与目标的差异和前向/重算次数 |
| `[R3][VERIFY]` | 每个 rank/阶段/批次的覆盖、实际专家 ID 差异和调用次数；只在整个批次完成后输出 PASS |
| `[R3][FAILED]` / `[CAPTURE_INVALID]` | 错误阶段、rank、batch、microbatch、重算标记、已执行层和 traceback；异常不会静默退回原生路由 |

`coverage=1.000000` 的分母是有 vLLM 路由的所有真实输入位置×层×top-k，不含最后一个未计算 token 和合成样本。`replay_mismatch_slots=0` 表示执行分发前使用的专家 ID 与目标一致。`native_diff_slots>0` 则进一步表明这次运行实际覆盖了路由被改写的情况；BF16 两端也可能全一致，零差异本身不是故障。

指纹是包含 token/layer/top-k 顺序的 int64 诊断 checksum，不是密码学哈希。日志验证不能单独证明推理核数值精度或训练收敛。第一轮请同时检查 loss、grad_norm 有限，且训练连续推进至少 2 step；随后分别做恢复训练与较长稳定性运行。WARNING 的逐样本校验和日志会增加开销，本版优先提供可定位性。

## 本地测试与服务器反馈

```bash
python3 -m pytest tests/utils/test_fsdp_router_replay_on_cpu.py \
  tests/workers/test_fsdp_r3_engine_on_cpu.py tests/utils/test_fsdp_r3_log_on_cpu.py \
  tests/utils/test_mxfp_moe_cpu.py -q
# 在 vllm-ascend 仓库（无 torch_npu 的 CPU 环境也可执行）：
python3 -m pytest --noconftest tests/ut/patch/worker/test_r3_capture_layout_cpu.py -q
```

CPU 测试使用生产前向/driver 代码和真实 autograd/checkpoint/QAT，只替换 NPU 算子、collective 和分布式环境。覆盖 BF16/FP32、两种 Transformers gate API、专家和 gate 梯度对照、带/不带 checkpoint 数值梯度一致、W4A8 F/D/W + F/D 旋转、动态打包、无效输入拒绝、reference 隔离、TP=8 的 1/7/8/9/63/65 token 采集与日志损坏检测。

本地 Windows 无 Ascend，尚不能声明 8 卡 NPU 端到端通过。服务器首跑需保存完整 `train.log`、`r3_environment.log`、自动检查结果；失败时保留第一次 traceback 前后的日志及 Ray worker 对应日志。若缺失 CAPTURE/VERIFY，检查工具会失败，不会把训练进程正常退出当作功能验收。
