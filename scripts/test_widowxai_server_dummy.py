#!/usr/bin/env python3
"""End-to-end CPU test using the real sbr-wm-deploy WebSocket client."""

from __future__ import annotations

import sys
import io
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from fasterwam.serve.websocket_server import WebsocketPolicyServer

CLIENT_SCRIPTS = Path(
    "/home/user128004/gitrepo/sbr-wm-deploy/trossen_ai_aloha/scripts"
)
if not CLIENT_SCRIPTS.is_dir():
    raise FileNotFoundError(f"sbr-wm-deploy client scripts not found: {CLIENT_SCRIPTS}")
sys.path.insert(0, str(CLIENT_SCRIPTS))
from policy.remote.websocket_client_policy import WebsocketClientPolicy  # noqa: E402
from policy.remote import serialization as client_serialization  # noqa: E402


def _pillow_encode_jpeg(image: np.ndarray, quality: int = 90) -> dict[str, bytes]:
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"JPEG encoding expects HxWx3 uint8 RGB, got {image.dtype} {image.shape}")
    stream = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(stream, format="JPEG", quality=int(quality))
    return {"__jpeg__": stream.getvalue()}


try:
    import cv2  # noqa: F401
except ImportError:
    # The real client uses the same MessagePack JPEG tag with OpenCV. The FastWAM
    # pixi environment doesn't include that optional robot-side encoder, so use
    # Pillow to produce an equivalent payload while retaining the real client.
    client_serialization.encode_jpeg = _pillow_encode_jpeg


class DummyTopViewFreePolicy:
    history_tail_len = 0
    action_dim = 14
    action_horizon = 32
    control_fps = 30.0
    control_dt = 1.0 / control_fps
    camera_names = ("cam_left_wrist", "cam_right_wrist")
    raw_image_shape = (480, 640, 3)
    model_image_shape = (384, 320, 3)
    num_video_frames = 9
    num_inference_steps = 10
    action_infer_mode = "one_pass_future_cache"
    drop_top_view = True
    checkpoint_path = "DUMMY_NO_MODEL"
    task = "dummy communication test"

    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def select_action(self, obs, history=None):
        del history
        qpos = np.asarray(obs["qpos"], dtype=np.float32)
        if qpos.shape != (self.action_dim,):
            raise ValueError(f"qpos shape mismatch: {qpos.shape}")
        aliases = {
            "cam_high": ("cam_high", "camera_high"),
            "cam_left_wrist": ("cam_left_wrist", "camera_wrist_left"),
            "cam_right_wrist": ("cam_right_wrist", "camera_wrist_right"),
        }
        for camera_name in self.camera_names:
            accepted_names = aliases[camera_name]
            selected = next((name for name in accepted_names if name in obs["images"]), None)
            if selected is None:
                raise KeyError(f"missing {camera_name}; accepted={accepted_names}")
            image = np.asarray(obs["images"][selected])
            if image.shape != self.raw_image_shape or image.dtype != np.uint8:
                raise ValueError(
                    f"{camera_name} must be uint8 {self.raw_image_shape}, "
                    f"got {image.dtype} {image.shape}"
                )
        action = np.repeat(qpos[None, :], self.action_horizon, axis=0)
        return action, "DUMMY", False


class DummyThreeCameraPolicy(DummyTopViewFreePolicy):
    camera_names = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    drop_top_view = False


def run_client(port: int, policy: DummyTopViewFreePolicy, codec: str) -> None:
    with WebsocketClientPolicy(
        host="127.0.0.1",
        port=port,
        image_codec=codec,
        jpeg_quality=90,
        echo_obs_summary=True,
        connect_timeout=10.0,
    ) as client:
        metadata = client.server_metadata
        assert metadata["type"] == "metadata", metadata
        assert metadata["protocol_version"] == 2, metadata
        assert metadata["supports_predicted_video"] is False, metadata
        assert metadata["camera_names"] == list(policy.camera_names), metadata
        assert metadata["drop_top_view"] is policy.drop_top_view, metadata
        assert metadata["action_infer_mode"] == "one_pass_future_cache", metadata

        pong = client.ping()
        assert pong["type"] == "pong" and pong["rtt_seconds"] >= 0.0, pong

        rows = np.arange(480, dtype=np.uint16)[:, None]
        cols = np.arange(640, dtype=np.uint16)[None, :]
        left = np.stack(
            [
                np.broadcast_to(rows % 256, (480, 640)),
                np.broadcast_to(cols % 256, (480, 640)),
                (rows + cols) % 256,
            ],
            axis=-1,
        ).astype(np.uint8)
        right = np.flip(left, axis=1).copy()
        # Include cam_high deliberately: drop_top_view servers must ignore it.
        images = {
            "cam_high": np.zeros_like(left),
            "camera_wrist_left": left,
            "camera_wrist_right": right,
        }
        qpos = np.linspace(-0.7, 0.7, 14, dtype=np.float32)
        action, phase, terminate = client.select_action(
            {"qpos": qpos, "images": images}, history=None
        )
        assert phase == "DUMMY" and terminate is False, (phase, terminate)
        assert action.shape == (32, 14), action.shape
        np.testing.assert_array_equal(action, np.repeat(qpos[None, :], 32, axis=0))
        assert client.last_obs_summary is not None
        for camera_name in images:
            summary = client.last_obs_summary["images"][camera_name]
            assert summary["shape"] == [480, 640, 3], summary
            assert summary["dtype"] == "uint8", summary

        if codec == "raw":
            client.reset()


def run_policy(policy: DummyTopViewFreePolicy) -> int:
    server = WebsocketPolicyServer(policy, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.wait_until_ready(timeout=10.0)
    try:
        run_client(port, policy, codec="raw")
        run_client(port, policy, codec="jpeg")
    finally:
        server.shutdown()
        thread.join(timeout=10.0)
    if thread.is_alive():
        raise RuntimeError("Dummy server thread did not stop")
    if policy.reset_count != 1:
        raise AssertionError(f"Expected one reset, got {policy.reset_count}")
    return port


def main() -> None:
    top_view_free_port = run_policy(DummyTopViewFreePolicy())
    three_camera_port = run_policy(DummyThreeCameraPolicy())
    print(
        "[dummy-communication] PASS "
        f"protocol=2 ports={top_view_free_port},{three_camera_port} "
        "clients=sbr-wm-deploy codecs=raw,jpeg action_shape=(32,14) "
        "visual_modes=2wrist_drop_top,3camera_legacy"
    )


if __name__ == "__main__":
    main()
