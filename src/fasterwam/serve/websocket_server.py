"""Expose a ``select_action(observation, history)`` policy over WebSocket."""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from websockets.asyncio.server import serve

from .serialization import packb, unpackb

PROTOCOL_VERSION = 2


def _metadata_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def build_policy_metadata(policy: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "type": "metadata",
        "protocol_version": PROTOCOL_VERSION,
        "policy_class": type(policy).__name__,
        "supports_predicted_video": False,
    }
    attributes = (
        "control_dt",
        "control_fps",
        "action_horizon",
        "action_dim",
        "history_tail_len",
        "task",
        "camera_names",
        "raw_image_shape",
        "model_image_shape",
        "num_video_frames",
        "num_inference_steps",
        "action_infer_mode",
        "drop_top_view",
        "checkpoint_path",
    )
    for name in attributes:
        value = getattr(policy, name, None)
        if value is not None:
            metadata[name] = _metadata_value(value)
    metadata.setdefault("history_tail_len", 0)
    return metadata


def _as_observation(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "qpos": np.asarray(payload["qpos"]),
        "qvel": np.asarray(payload.get("qvel", np.zeros(0, dtype=np.float32))),
        "effort": np.asarray(payload.get("effort", np.zeros(0, dtype=np.float32))),
        "images": dict(payload.get("images", {})),
        "depth_images": dict(payload.get("depth_images", {})),
    }


def _as_history(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        "qpos": list(payload.get("qpos", [])),
        "qvel": list(payload.get("qvel", [])),
        "effort": list(payload.get("effort", [])),
        "actions": list(payload.get("actions", [])),
        "images": {name: list(frames) for name, frames in payload.get("images", {}).items()},
        "depth_images": {
            name: list(frames) for name, frames in payload.get("depth_images", {}).items()
        },
    }


def _normalize_policy_outputs(outputs: Any) -> tuple[np.ndarray, str, bool]:
    if isinstance(outputs, tuple):
        if len(outputs) != 3:
            raise ValueError(f"Policy output tuple must have length 3, got {len(outputs)}")
        action, phase, terminate = outputs
        return np.asarray(action), str(phase), bool(terminate)
    return np.asarray(outputs), "", False


def _observation_summary(obs: dict[str, Any], history: dict[str, Any] | None) -> dict[str, Any]:
    images = {}
    for name, value in obs["images"].items():
        image = np.asarray(value)
        images[name] = {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "mean": float(image.mean()) if image.size else None,
        }
    summary: dict[str, Any] = {
        "qpos": np.asarray(obs["qpos"], dtype=np.float32),
        "images": images,
    }
    if history is not None:
        summary["history_len"] = {
            "qpos": len(history["qpos"]),
            "images": {name: len(frames) for name, frames in history["images"].items()},
        }
    return summary


class WebsocketPolicyServer:
    """Single-policy, serialized request/response WebSocket server."""

    def __init__(self, policy: Any, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._policy = policy
        self._host = host
        self._port = int(port)
        self._metadata = build_policy_metadata(policy)
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._infer_lock: asyncio.Lock | None = None
        self.bound_port: int | None = None

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    def wait_until_ready(self, timeout: float = 10.0) -> int:
        if not self._ready.wait(timeout):
            raise TimeoutError("WebsocketPolicyServer did not start in time")
        assert self.bound_port is not None
        return self.bound_port

    def shutdown(self) -> None:
        if self._loop is not None and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        self._infer_lock = asyncio.Lock()
        async with serve(
            self._handler,
            self._host,
            self._port,
            max_size=None,
            compression=None,
        ) as server:
            ports = sorted({socket.getsockname()[1] for socket in server.sockets})
            self.bound_port = int(ports[0])
            self._ready.set()
            print(
                f"[fasterwam-server] listening on {self._host}:{self.bound_port} "
                f"(policy={self._metadata['policy_class']})",
                flush=True,
            )
            await self._stop_event.wait()

    async def _handler(self, websocket) -> None:
        peer = websocket.remote_address
        print(f"[fasterwam-server] client connected: {peer}", flush=True)
        await websocket.send(packb(self._metadata))
        try:
            async for message in websocket:
                try:
                    if not isinstance(message, bytes):
                        raise TypeError("The FasterWAM policy protocol accepts binary messages only")
                    response = await self._dispatch(unpackb(message))
                except Exception:
                    response = {"type": "error", "message": traceback.format_exc()}
                    print(f"[fasterwam-server] request failed:\n{response['message']}", flush=True)
                await websocket.send(packb(response))
        finally:
            print(f"[fasterwam-server] client disconnected: {peer}", flush=True)

    async def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = request.get("type")
        if request_type == "select_action":
            obs = _as_observation(request["observation"])
            history = _as_history(request.get("history"))
            assert self._infer_lock is not None
            async with self._infer_lock:
                start = time.perf_counter()
                outputs = await asyncio.to_thread(self._policy.select_action, obs, history)
                infer_seconds = time.perf_counter() - start
            action, phase, terminate = _normalize_policy_outputs(outputs)
            response: dict[str, Any] = {
                "type": "action",
                "action": action,
                "phase": phase,
                "terminate": terminate,
                "infer_seconds": infer_seconds,
            }
            if request.get("echo_obs_summary"):
                response["obs_summary"] = _observation_summary(obs, history)
            return response
        if request_type == "ping":
            return {
                "type": "pong",
                "client_time": request.get("client_time"),
                "server_time": time.time(),
            }
        if request_type == "reset":
            reset = getattr(self._policy, "reset", None)
            if callable(reset):
                await asyncio.to_thread(reset)
            return {"type": "reset_ok"}
        raise ValueError(f"Unknown request type: {request_type!r}")
