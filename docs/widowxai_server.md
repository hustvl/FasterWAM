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

For a missing/false flag, it reproduces the legacy `robotwin` composition:
`cam_high` becomes the `256x320` top panel, both wrists become `128x160`, and
the wrists are concatenated horizontally below it. The result is also
`384x320`, so the model architecture and FasterWAM inference path are shared.

FasterWAM defaults to `one_pass_future_cache`: the current image is encoded,
future-video KV features are prefetched once, and the 32-step normalized action
chunk is denoised for 10 steps. The server reverses z-score normalization, adds
the current joint positions to the 12 relative arm dimensions, keeps the two
gripper dimensions absolute, and returns physical absolute targets `[32,14]`.

Check the online transform against a real offline sample without loading the
5B checkpoint:

```bash
PYTHONPATH=$PWD/src bash ../FastWAM/scripts/pixi_run.sh \
  python scripts/validate_widowxai_online_preprocessing.py
```

## Start with Slurm

```bash
cd /home/user128004/gitrepo/FasterWAM
sbatch \
  --export=ALL,CHECKPOINT=/lustre/data/fasterwam/runs/widowxai_fasterwam_2wrist_vertical_384_1e-4/widowxai_2wrist_vertical_scratch_4443674/checkpoints/weights/step_000960.pt \
  scripts/slurm/serve_widowxai_policy.sbatch
```

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
