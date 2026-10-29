# WidowXAI FasterWAM policy server

This server is wire-compatible with FastWAM's binary WebSocket/MessagePack
server and `sbr-wm-deploy`'s `eval_policy_remote_inf.py` client. It returns only
action chunks; predicted-video streaming is not supported.

The default run is:

```text
/lustre/data/fasterwam/runs/widowxai_fasterwam_2wrist_vertical_384_1e-4/widowxai_2wrist_vertical_scratch_4443674
```

Without an explicit `CHECKPOINT`, the latest `step_*.pt` is selected. Open-loop
evaluation found `step_000960.pt` best among this run's checkpoints, so it can
be selected explicitly for real-robot comparison.

## Input and output

Every request needs finite 14-D `qpos` and RGB `uint8` images of shape
`[480,640,3]`. The camera requirement is resolved from the saved config:

- `drop_top_view: true`: require `cam_left_wrist` and `cam_right_wrist`; an
  extra `cam_high` is accepted but ignored.
- `drop_top_view` absent or `false`: treat the run as a legacy three-camera
  model and require `cam_high`, `cam_left_wrist`, and `cam_right_wrist`.

For `drop_top_view=true`, the server reproduces the two-wrist transform exactly:

1. Resize each wrist image to `240x320`.
2. Aspect-fit it to `192x256` and add 32 neutral-gray pixels on both sides.
3. Put the left wrist in the top half and the right wrist in the bottom half.
4. Normalize the resulting `384x320` mosaic to `[-1,1]`.
5. Normalize `qpos` with the run's dataset statistics.

For a missing/false flag, `concat_multi_camera` selects the three-camera layout.
The legacy `robotwin` composition is:
`cam_high` becomes the `256x320` top panel, both wrists become `128x160`, and
the wrists are concatenated horizontally below it. The result is also
`384x320`, so the model architecture and FasterWAM inference path are shared.

`left_high_right_horizontal` directly resizes each complete frame to one third
of the configured output width, then concatenates **left wrist, high, right
wrist**. For the 3-camera ratio-2 run below, each panel is `192x256`, producing
`192x768` RGB normalized to `[-1,1]`. Input/shape-meta camera order remains
`cam_high, cam_left_wrist, cam_right_wrist`.

FasterWAM defaults to `one_pass_future_cache`: the current image is encoded,
future-video KV features are prefetched once, and the 32-step normalized action
chunk is denoised for 10 steps. The server reverses z-score normalization, adds
the current joint positions to the 12 relative arm dimensions, keeps the two
gripper dimensions absolute, and returns physical absolute targets `[32,14]`.

Check the online transform against a real offline sample without loading the
5B checkpoint:

```bash
bash scripts/pixi_run.sh \
  python scripts/validate_widowxai_online_preprocessing.py
```

## Start with Slurm

```bash
cd /home/user128004/gitrepo/FasterWAM-rtc
sbatch \
  --export=ALL,CHECKPOINT=/lustre/data/fasterwam/runs/widowxai_fasterwam_2wrist_vertical_384_1e-4/widowxai_2wrist_vertical_scratch_4443674/checkpoints/weights/step_000960.pt \
  scripts/slurm/serve_widowxai_policy.sbatch
```

The launcher defaults `REPO_ROOT` to the Slurm submission directory. Submit from
this repository, or set `REPO_ROOT` explicitly. `scripts/pixi_run.sh` reuses the
installed training Pixi environment without solving or installing packages;
`FASTERWAM_PIXI_ENV` can override its environment directory.

For the left/high/right ratio-2 checkpoint:

```bash
cd /home/user128004/gitrepo/FasterWAM-rtc
export RUN_DIR=/lustre/data/fasterwam/runs/widowxai_fasterwam_3cam_384_1e-4/widowxai_3cam_vtoa_pick_cap_policy_random_policy_deploy_failed_success_stats_left_high_right_horizontal_action_video_freq_ratio2_4528728
export CHECKPOINT="$RUN_DIR/checkpoints/weights/step_001600.pt"
export TASK="pick up the brown cap and put it in the light blue bucket"
sbatch --export=ALL scripts/slurm/serve_widowxai_policy.sbatch
```

The task must be explicit because this dataset contains three tasks. Its cached
embedding is loaded from `data.val.text_embedding_cache_dir`; no text encoder is
loaded. The run's `dataset_stats.json` supplies both state and action statistics.
The saved ratio of 2 produces **32 action targets and 17 video frames**; physical
control stays at **30 Hz**. Normal serving uses 10 denoising steps and
`one_pass_future_cache`. `SERVING_CONFIG=configs/serve/rtc.yaml` or
`configs/serve/simple.yaml` optionally enables asynchronous guidance.

The inference config adapter accepts newer training-only options for this
V-to-A run while rejecting bidirectional attention, active A2V layers, and random
future blocks. Existing checkpoint weights are loaded directly.

To compare the training repository and this repository with real servers on one
Slurm GPU:

```bash
sbatch scripts/slurm/compare_widowxai_servers.sbatch
```

This starts both servers for each of `one_pass_future_cache` and `joint_rollout`,
using `step_001600.pt`, seed 42, and 10 denoising steps. It requires exact
elementwise equality for three reproducible synthetic observations plus a repeat
after reset. It also compares CPU images, proprioception, language embeddings,
and action denormalization. Logs, input arrays, action arrays and `results.json`
are saved under `/lustre/data/fasterwam/validations/rtc_server_parity/job_JOBID`.
The servers stop on completion or failure. `REFERENCE_REPO`, `RUN_DIR`,
`CHECKPOINT`, `TASK`, and `RESULT_DIR` may be overridden through Slurm exports.

Validated on 2026-10-22, Slurm job `4924588`, NVIDIA H100 80GB, using the existing
Pixi environment. Both modes passed all four requests: physical `[32,14]` action
chunks were exactly equal, with maximum absolute difference `0.0`. CPU image,
proprioception, language context/mask, and inverse-action arrays also matched
exactly. Results and logs are in
`/lustre/data/fasterwam/validations/rtc_server_parity/job_4924588/results.json`.

To repeat the comparison with RTC enabled on the candidate server:

```bash
sbatch --export=ALL,CANDIDATE_SERVING_CONFIG=configs/serve/rtc.yaml,EXERCISE_RTC=1 \
  scripts/slurm/compare_widowxai_servers.sbatch
```

The test requires `supports_rtc=true` in the candidate metadata. Ordinary
synchronous requests must still match the reference exactly. It then sends
`guidance_method=rtc` requests synchronously (one request followed by waiting
for its response), with `inference_delay_steps=0`. An empty prefix must preserve
the ordinary result. A nonempty 16-step tail from the previous chunk exercises
RTC guidance against a changed current observation; the output must be finite
and differ from the ordinary result. Repeat-request numerical differences are
reported separately in `repeat_comparison`; ordinary sync and empty-prefix
comparisons still require exact equality, without a tolerance.
These guided outputs are saved as `rtc_*.npy` and reported in `rtc_requests`.
The reference repository does not implement RTC guidance, so nonempty-prefix
outputs are compared for the effect and reproducibility of guidance rather
than requiring equality to its ordinary prediction. No asynchronous executor
is used in this test.

RTC-enabled server validation on 2026-10-22 passed in Slurm job `4924687`:
both modes advertised `supports_rtc=true`, all eight ordinary synchronous
requests matched the reference exactly, and both empty-prefix RTC requests
matched their ordinary outputs exactly (maximum difference `0.0`). Nonempty
RTC requests returned finite `[32,14]` targets and changed the predictions.
The report is
`/lustre/data/fasterwam/validations/rtc_server_parity/job_4924687/results.json`.

With the default GPU backend, identical nonempty RTC requests had repeat
differences of `0.0022865534` (future cache) and `0.0034939498` (joint rollout).
A separate SDPA test reproduced nondeterministic attention backward, while
forward remained exact. A controlled real-model test with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`torch.use_deterministic_algorithms(True)` gave exact RTC repeat equality for
both modes. This diagnostic does not change the server's production defaults.
Diagnostic scripts and results are under
`/lustre/data/fasterwam/validations/rtc_attention_diagnostic/`, including
`real_model_4924691/results.json`.

After model loading, create an SSH tunnel from the robot PC:

```bash
ssh -N -L 8000:<compute-node>:8000 <user>@<login-node>
```

The existing three-camera client configuration works for both modes. The server
uses its top image for a legacy config and ignores it when `drop_top_view=true`:

```bash
uv run scripts/eval_policy_remote_inf.py \
  -r asratec_xai_stationary_inference_3cam \
  --host 127.0.0.1 --port 8000 --no_predicted_video
```

Environment overrides include `SERVER_PORT`, `CHECKPOINT`, `NORM_STATS`,
`TASK`, `NUM_INFERENCE_STEPS`, and `ACTION_INFER_MODE`. The alternative
`joint_rollout` action mode is provided for comparison, but
`one_pass_future_cache` is the FasterWAM default.

## Dummy communication test

The CPU-only test starts both a two-wrist and a legacy three-camera server,
transmits raw and JPEG-tagged frames plus an ndarray `qpos`, and checks metadata,
ping, action, reset, and ndarray response serialization:

```bash
sbatch scripts/slurm/test_widowxai_server_dummy.sbatch
```

For a full one-GPU smoke test that loads the 5B checkpoint and sends a real
client request through preprocessing, FasterWAM inference, action
denormalization, and WebSocket response:

```bash
sbatch scripts/slurm/test_widowxai_server_gpu.sbatch
```

It defaults to the two-wrist `step_000960.pt`. A legacy three-camera run can be
tested with `RUN_DIR`, `CONFIG`, `CHECKPOINT`, and `NORM_STATS` environment
overrides.

## Asynchronous chunks and inference-only guidance

RTC and simple are disabled by default. No training config, training procedure, checkpoint,
robot YAML, or ndarray serializer changes are needed. The physical boundary is
absolute `[L,14]` joint targets. Both `one_pass_future_cache` and `joint_rollout`
support `none`, `rtc`, and `simple`.

```bash
# Sync or unguided async clients: ordinary server, RTC disabled.
PYTHONPATH=src python scripts/serve_widowxai_policy.py \
  --run-dir /path/to/local/run --model-base-path /path/to/local/base-weights

# Guided async clients: explicit serving config enables RTC.
PYTHONPATH=src python scripts/serve_widowxai_policy.py \
  --run-dir /path/to/local/run --model-base-path /path/to/local/base-weights \
  --serving-config configs/serve/rtc.yaml

# The same config, joint rollout, and an explicit CLI override.
PYTHONPATH=src python scripts/serve_widowxai_policy.py \
  --run-dir /path/to/local/run --model-base-path /path/to/local/base-weights \
  --serving-config configs/serve/rtc.yaml --action-infer-mode joint_rollout \
  --rtc-prefix-schedule exp --rtc-max-guidance-weight 5.0

# Gradient-free clean-action mixing, available in both paths.
PYTHONPATH=src python scripts/serve_widowxai_policy.py \
  --run-dir /path/to/local/run --model-base-path /path/to/local/base-weights \
  --serving-config configs/serve/simple.yaml
PYTHONPATH=src python scripts/serve_widowxai_policy.py \
  --run-dir /path/to/local/run --model-base-path /path/to/local/base-weights \
  --serving-config configs/serve/simple.yaml --action-infer-mode joint_rollout \
  --simple-prefix-schedule exp
```

`--config` remains the resolved training config. `--serving-config` is separate;
CLI overrides its fields. With no serving file, both methods stay disabled and ordinary
inference defaults remain unchanged. `--rtc`/`--no-rtc` override enablement.
`--simple`/`--no-simple` independently override simple enablement, and
`--simple-prefix-schedule` overrides `simple.prefix_attention_schedule`.
The server may enable both capabilities, e.g. `--rtc --simple`; each request
selects exactly one method. Clients use `--mode async --guidance-method simple`
to request simple. The first chunk still uses ordinary inference.
The server publishes `control_dt`; clients use it without action resampling.
No model downloads occur unless the existing `--allow-download` flag is given.

Protocol version 2 remains backward compatible. `select_action(obs, history)`
is unchanged; a client may add the following `chunk_context` mapping:

| Field | Contract |
|---|---|
| `episode_id` | Nonempty string identifying this episode |
| `request_id`, `observation_step` | Nonnegative integers; `action[0]` corresponds to `observation_step` |
| `control_dt` | Positive finite seconds per control step, matching server metadata |
| `previous_actions` | Finite physical `[L,14]` unexecuted targets, rebased by the server |
| `inference_delay_steps` | Nonnegative predicted delay; never interpreted as the actual response skip |
| `guidance_method` | `none`, `rtc`, or `simple` |

Responses echo the three identifiers in `chunk_context` and return the entire
chunk. Only the client skips elapsed *executed control steps*. Metadata advertises
`supports_timed_chunks`, `supports_rtc`, and `supports_simple`; each guidance
capability is true only when that method is enabled and the selected model path
accepts it. Untimed requests and legacy
policies retain their original call signature. Client-side single-flight
correlation permits unguided async against old servers. Requesting RTC/simple
against a server without that capability fails before robot initialization.
Reset and inference share
the same lock, including while a repeatedly cancelled request's worker finishes. A response
to a disconnected client is discarded, never carried into another connection.

For every RTC/simple request, physical previous targets pass through the training
`action_state_transform`, normalizer, and merger, using the *current observation*
qpos. Existing transform settings decide relative joints and absolute grippers.
Stepwise normalization is supported by padding before transformation and retaining
only the valid prefix. Existing normalizer clipping remains in effect; extreme
out-of-statistics actions therefore need not round-trip exactly.

The shared `wan22/rtc.py` ports `realtime_action` and `get_prefix_weights` from
`real-time-chunking/src/model.py`. `start = min(predicted_delay, L)`, `end = L`,
with zero weights outside `L`. Defaults are `exp` and maximum weight `5.0`.
Other schedules are `linear`, `ones`, `zeros` (the last still fixes positions
before `start`, as in the reference). The prediction is
`x_hat = x_sigma - sigma * u`. For reference time `t = 1-sigma` and velocity
`v = -u`, the corrected flow is
`u - guidance_weight * VJP(x_hat, (previous - x_hat) * prefix_weight)`.
The cotangent is weighted once. The scheduler already supplies negative sigma
deltas. Model parameters stay frozen, only action-input gradients are enabled,
and each denoise step detaches its graph. Video flow is unchanged; the future
cache is computed without gradients and reused. Disabling RTC preserves the
original no-grad entry points. Guidance arguments are never silently signature-filtered.

The shared `wan22/simple.py` implements clean-action mixing without VJP or
autograd. It reuses only RTC's prefix weights: `start = min(predicted_delay, L)`,
`end = L`, zero outside the valid prefix. The default is `exp`; `linear`, `ones`,
and `zeros` are also supported with the same semantics described above.
Simple has no maximum guidance weight; the mask alone controls the mixture.
In FasterWAM's decreasing-sigma convention (`tau = 1-sigma`, `v = -u`):

```python
x_pred = x - sigma * u
x_target = weights * previous_actions + (1 - weights) * x_pred
u_simple = (x - x_target) / sigma
x_next = x + delta_sigma * u_simple
```

Both paths use the shared helper after their ordinary scheduler update. The
implementation adds the equivalent correction
`(-delta_sigma / sigma) * weights * (previous_actions - x_pred)` to that update,
using float32 for the correction. This avoids constructing a large intermediate
velocity at small sigma and respects nonuniform, shifted schedules. The mixed
latent is the next denoise step's input. Terminal sigma zero is never evaluated.
Disabled, empty, or zero-weight guidance preserves the ordinary update exactly,
as do all positions outside the valid prefix. Model weights stay frozen; simple
uses one normal prediction per step and no input gradients. Video and future-cache
processing retain their existing paths. A model call cannot accept both
`rtc_guidance` and `simple_guidance`.

RTC and simple condition the planned targets, including known fixed-gripper constraints.
The robot's observation-dependent safety filter still runs immediately before
execution. Its changes are logged by the client; planned-prefix guidance does
not guarantee the executed trajectory matches that prefix.

## Reproducible validation without a robot

All commands below are local; no job submission is necessary. Set `SBR_CLIENT_SCRIPTS`
or pass `--client-scripts`; the dummy test no longer has a user-specific client path.

```bash
# Client repository, from trossen_ai_aloha/:
python -m unittest discover -s tests -p 'test_*.py' -v
# FasterWAM repository:
PYTHONPATH=src python scripts/test_widowxai_config.py -v
PYTHONPATH=src python scripts/test_rtc.py -v
PYTHONPATH=src python scripts/test_simple.py -v
PYTHONPATH=src python scripts/test_chunk_protocol.py -v \
  --client-scripts /path/to/sbr-wm-deploy/trossen_ai_aloha/scripts
PYTHONPATH=src python scripts/test_widowxai_server_dummy.py \
  --client-scripts /path/to/sbr-wm-deploy/trossen_ai_aloha/scripts
```

These cover fake-clock switches and holds, snapshot isolation, one outstanding
request, blocked-send/receive deadlines, disconnect/reset/termination, repeated
cancellation while holding the policy lock, robot cleanup ordering with mocked
drivers, old-server correlation, raw/JPEG and old sync API, prefix known values,
analytic VJP/sign/time conversion, global/stepwise normalization round trips,
frozen weights, and both production denoise loops with tiny analytic experts.
Simple tests compare the conceptual tau/velocity loop with the sigma implementation
for uniform and shifted schedules, verify next-step propagation, endpoint behavior,
zero/empty masks, and absence of gradients. Protocol tests check independent
RTC/simple capability combinations and reject unsupported methods.
Numeric tests use seed 42, synthetic inputs/statistics, and no dataset/checkpoint.

With a locally available GPU and artifacts, the following short smoke loads the
model once and runs both paths, each with `none`, `rtc`, and `simple`. It uses synthetic images,
never a robot, and refuses downloads or an existing result directory:

```bash
PYTHONPATH=src python scripts/test_widowxai_rtc_smoke.py \
  --run-dir /path/to/local/run --model-base-path /path/to/local/base-weights \
  --checkpoint /path/to/local/run/checkpoints/weights/step_XXXXXX.pt \
  --num-inference-steps 2 --seed 42 \
  --output-dir /groups/gcg51472/jin_tmp/rtc-smoke-UNIQUE-RUN-ID
```

The manifest records config/statistics SHA256, checkpoint path/size/mtime, model,
seed, device, effective RTC/simple settings, synthetic-data identity and timings.
CPU verification does not establish real-checkpoint GPU memory/latency, real-robot
performance improvements, or safe continuous operation. Validate those separately.
