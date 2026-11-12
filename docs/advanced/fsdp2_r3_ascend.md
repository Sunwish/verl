# Qwen3 MoE 的 FSDP2 + vLLM-Ascend R3 路由回放

Last updated: 11/12/2026

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

### Pearson 下降时的同批次对照

`[VERIFY] PASS` 证明采集 ID 被用于 forward/backward，不证明这些 ID 对应正确 token，也不证明训推概率数值一致。BF16 关闭 R3 时 Pearson 约 0.98、开启后约 0.6，应视为待定位回归，不能验收。

```bash
bash examples/grpo_trainer/run_qwen3_moe_r3_fsdp2.sh --smoke --diagnose "$BASELINE"
```

`--diagnose` 仅在每个 rank、每轮 old-log-prob 的第一个微批次额外执行一次原生路由 forward；使用同一输入、同一权重、eval/no-grad。它不替换 PPO 使用的回放概率，不改采集、更新、graph、async 或量化配置。该诊断有额外时间开销，默认关闭。

- `[PROB_COMPARE]` 同批次的 `native_rollout_pearson`、`replay_rollout_pearson` 和 `replay_native_pearson`，以及概率/对数概率绝对差。如果原生高而回放低，可以排除不同 run 的采样差异，将问题收敛到回放路径。
- `[ROUTE_COMPARE]` 对每层比较专家集合，而非 top-k 槽位顺序。`native_hidden` 是独立原生 forward，`replay_hidden` 是回放 forward 进入各层时的原生 top-k。`token_delta_set_overlap[d]` 比较 capture[t] 与 native[t+d]，不跨样本边界；`layer_delta_set_overlap` 检查相邻层错位。相邻 token 本身可能相似，偏移重合度仅提供线索，不能单独判定错位或用于自动修正。

### 第 0 层 MoE 算子对照

2026-11-12 BF16 日志已定位到一个明确回归：Transformers 5.0/5.1 的 `Qwen3MoeTopKRouter.forward` 返回 `(probabilities, weights, indices)`，第一个张量虽然命名为 `router_logits`，实际已经过 softmax；较新版本返回原始 logits。之前 R3 统一对第一个张量再做 softmax，把权重压平到约 1/8。即使专家集合和顺序完全一致，也会改变 MoE 输出。这是接口兼容错误，不能归因于正常 BF16 误差。参见 [Transformers 5.1 源码](https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L243)。

修复在 R3 tuple gate 路径上显式执行 `F.linear(hidden_states, gate.weight, bias)` 得到原始 logits，再且仅做一次 softmax；不按版本号或数值阈值猜测 tuple 的含义。保留原生 gate 的专家选择，保留当前策略 gate 和专家梯度；old-log-prob、actor 更新、checkpoint 重算均生效。额外增加一次较小的 gate 投影。独立 native 对照继续使用原生权重，只为诊断补原始 logits。未启用 R3 的模型沿用原生路径。

安装时每个 rank 的 `[ROUTER_CONTRACT]` 日志确认采用明确的原始 logits 来源。`[MOE_WEIGHTS]` 现在增加 `router_logits_source`、`gate_class`、`gate_first_return_row_sum_min/max`，并比较 gate 第一返回值与 raw logits / raw softmax。对于返回概率的 gate，行和应接近 1，`gate_first_return_vs_raw_softmax` 误差很小；`native_weights_cpu_ref` 和相同集合上的 `same_set_weights` 应回到正常舍入误差水平。此前日志的 `native_weights_cpu_ref` 比较使用了错误的“logits”参考，不能据此认定 native gate 算错。

同一个 `--diagnose` 命令还会输出以下 WARNING，不需要新增参数。每个 rank、每轮 old-log-prob 的首微批次分别观察 replay/native 第 0 层，不改前向结果和 PPO 概率，也不在 actor 更新或 checkpoint 重算中执行。

| 日志 | 重点字段与定位方式 |
| --- | --- |
| `[MOE_WEIGHTS]` | 全部 token 的 gate logits 在 CPU FP32 上重新 softmax、按实际专家 ID gather、归一化。`selected_weights_cpu_ref` 检查回放权重；`native_weights_cpu_ref` 对照原生 gate 的精度。同时记录 cast 后误差、归一化行和、dtype、stride、128 专家/top-8，以及采样 token 行号。 |
| `[MOE_DISPATCH]` | 全部 token-slot 的行映射必须是置换，按映射后的专家分组必须与排序后的专家 ID 一致，histc 必须等于 CPU bincount。`integer_invariants=PASS` 只表示这三个整数条件；`sample_permuted_input_ref` 另行检查实际分发的 hidden state 是否对应原 token。 |
| `[MOE_MERGE]` | 将采样 token 的实际 down 输出按行映射搬到 CPU，乘实际 cast 后路由权重，独立求和。`unpermute_vs_cpu_fp32_merge` / `unpermute_vs_cpu_dtype_merge` 分别参考 FP32 累加和原 dtype 运算；另查 cast 及 SwiGLU。BF16 有舍入差异，不要求浮点误差逐位为零。 |
| `[MOE_COMPARE]` | 两次实际前向的 input/logits、按专家 ID 对齐的 permuted/up/gate/activation/down、相同专家集合上的 weights/output。`shared_experts` 也比较集合有变化的 token 的公共专家，权重因归一化分母变化可以不同，因此判断权重用 `same_set_weights`。 |

`tensor_difference` 字段含 `max_abs`、`mean_abs`、`relative_rms`、参考 RMS 和非有限值数。`relative_rms` 是误差 RMS / 参考 RMS；参考接近零时必须结合绝对误差判断。诊断只报告数值，不按任意容差输出数值验收 PASS。

定位顺序：先查 WEIGHTS 的 CPU 参考；再查 DISPATCH 的整数条件和采样输入；然后在 COMPARE 中沿 `permuted -> up/gate -> activation -> down -> output` 找第一个明显增大的误差。如果公共专家的 down 一致、同集合权重一致，而同集合输出不一致，结合 MERGE 可直接检查 unpermute 的加权合并。如果各阶段都接近，仍需结合实际改变专家集合的 token 和后续层传播继续分析，不能据此宣告整个模型通过。

路由权重和整数行映射检查覆盖第 0 层全部 token。激活只采样最多 32 个 token，覆盖相同集合、改变集合、未捕获位置和样本边界，并在各组内分散取点；两次前向使用相同行号。不会搬运专家权重或整个 top-k 展开的激活，但同步和 CPU 计算仍增加诊断耗时。检查按照 [Ascend permute API](https://github.com/Ascend/op-plugin/blob/master/docs/en/custom_APIs/torch_npu/torch_npu-npu_moe_token_permute.md) 的逆映射契约 `row_map[token*K+slot] -> expert_buffer_row`；无效映射会先输出日志再报错，避免错误索引。

下一轮优先继续 BF16 原配置，保持图模式和 async 不变，保留完整 `train.log`、`r3_environment.log`。至少要同时看到每个 rank/batch 的 WEIGHTS、DISPATCH、MERGE 各两条（replay/native）、一条 COMPARE，以及原有 PROB_COMPARE。可用下面的命令提取便于阅读的副本，完整日志仍需保留：

```bash
grep -E '\[R3\]\[(MOE_WEIGHTS|MOE_DISPATCH|MOE_MERGE|MOE_COMPARE|PROB_COMPARE)\]' "$LOG_DIR/train.log" > "$LOG_DIR/r3_moe_diagnostics.log"
```

保留完整 `train.log`（含 CAPTURE、TP_CAPTURE、rank 0），不能只截 VERIFY。Pearson 的实现比较有效 response token 上 `exp(old_log_probs)` 与 `exp(rollout_log_probs)`，在 actor 参数更新之前计算。R3 保留当前 router 的混合权重，不保证 BF16 概率逐位一致，但明显恶化需要查明原因。

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

CPU 测试使用生产前向/driver 代码和真实 autograd/checkpoint/QAT，只替换 NPU 算子、collective 和分布式环境。覆盖 BF16/FP32、三种 gate 返回接口（旧版 Linear、tuple raw logits、tuple probabilities）、专家和 gate 梯度对照、带/不带 checkpoint 数值梯度一致、norm_topk_prob 开/关、W4A8 F/D/W + F/D 旋转、动态打包、无效输入拒绝、reference 隔离、TP=8 的 1/7/8/9/63/65 token 采集与日志损坏检测。返回概率的 gate 在旧实现上有 6 项回归测试失败，修复后通过。算子诊断另外覆盖 128 专家/top-8 的槽位重排，并分别注入错误权重、分发激活、histogram、行映射、GMM 和合并输出，检查对应日志能定位错误；CPU permute 替身遵循 CANN 逆映射契约。

本地 Windows 无 Ascend，尚不能声明 8 卡 NPU 端到端通过。服务器首跑需保存完整 `train.log`、`r3_environment.log`、自动检查结果；失败时保留第一次 traceback 前后的日志及 Ray worker 对应日志。若缺失 CAPTURE/VERIFY，检查工具会失败，不会把训练进程正常退出当作功能验收。
