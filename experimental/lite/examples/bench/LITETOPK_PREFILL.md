# GLM-5.2 and DeepSeek-V4-Flash prefill

These experimental entry points load the complete checkpoint and measure token
IDs through the final next-token logits on eight B200 GPUs. GLM uses TP8/EP8/ETP1;
DeepSeek uses TP8/EP1/ETP8. Both use CP1, with every prompt query on each rank.
They provide forward-only prefill, without a persistent decode cache.

| Policy | GLM-5.2 | DeepSeek-V4-Flash |
| --- | --- | --- |
| Layers | 78 | 43 |
| Routed experts | FP8 weights/activations | FP4 weights, FP8 activations |
| Indexer | FP8 | FP4 |
| Sparse attention | unit-scale FP8 | BF16 |
| Dense score budget | 2 GiB | 1 GiB |
| LiteTopK startup | 188416 queries | 49152 queries |
| Candidate storage | 13 overflow pages/query budget | 60000 candidates/query |
| Final selected-ID sort | disabled | disabled |

GLM uses serial local Q1776 tiles and group8 HOT updates. Its physical candidate
arena averages 61440 records/query (720 MiB for the minimum 2048-row backing);
this is a shared storage budget, not a logical per-query selection limit.
Native startup and the initial 8192-score seed remain enabled. There is no
boundary rescoring. Encoded cutoff ties choose the larger original logical ID.

GLM's long-context numerical comparison has **not passed** the existing 0.05
relative-L2 threshold; this entry preserves the performance experiment accepted
by the user. The earlier DeepSeek implementation passed that comparison.
Neither statement substitutes for validating a new checkpoint or environment.
Both drivers save final logits outside the timed interval.

## Dependencies

Use the existing Megatron-Lite environment with CUDA 13, Transformer Engine,
DeepGEMM (FP8/FP4 MQA logits and grouped GEMM), FlashMLA, FlashInfer with TRTLLM-GEN,
and cuDNN frontend/CuTe DSL. These are additional experimental runtime requirements.

The companion source archive contains the exact external LiteTopK and Native
exact-tie sources. They are local extensions, not capabilities of stock SGLang.
GLM requires LiteTopK source ID 996e735c52df; DeepSeek uses ac1c7f51b362.
The three Native exact-tie Python files are shared. CUDA binaries and weights
are deliberately excluded; the external loader builds its extension as needed.
Set MEGATRON_LITETOPK_BUILD to a writable build directory if required.

After extracting that archive, set DEPS to its dependencies directory:

```bash
export PYTHONPATH="$PWD/experimental/lite:$PWD:$PYTHONPATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc-per-node=8 experimental/lite/examples/bench/glm_tp_e2e.py \
  --checkpoint /path/to/glm-checkpoint \
  --litetopk-source "$DEPS/glm-litetopk/litetopk.py" \
  --native-exact-tie-source "$DEPS/native-exact-tie" \
  --lengths 262144 524288 786432 1048576 --warmups 1 --repeats 3 \
  --output /path/to/new-results/glm.json
torchrun --standalone --nproc-per-node=8 experimental/lite/examples/bench/dsv4_tp_e2e.py \
  --checkpoint /path/to/deepseek-checkpoint \
  --litetopk-source "$DEPS/dsv4-litetopk/litetopk.py" \
  --native-exact-tie-source "$DEPS/native-exact-tie" \
  --lengths 262144 524288 786432 1048576 --warmups 1 --repeats 3 \
  --output /path/to/new-results/deepseek.json
```

Latency is the median of per-repeat maximum-rank CUDA intervals. Speedup is
Raw latency divided by LiteTopK latency. Loading, warmup and logit saving are
excluded. A zero-warmup, single-repeat run is only an execution smoke test.
