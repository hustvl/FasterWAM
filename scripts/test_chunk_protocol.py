"""Local-only transport lifecycle tests. --client-scripts or SBR_CLIENT_SCRIPTS."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import os
import socket
import sys
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.frames import OP_BINARY, Frame

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--client-scripts", type=Path, default=os.environ.get("SBR_CLIENT_SCRIPTS")
)
args, remaining = parser.parse_known_args()
client_scripts = (
    args.client_scripts
    or Path(__file__).resolve().parents[2] / "sbr-wm-deploy/trossen_ai_aloha/scripts"
)
sys.path.insert(0, str(client_scripts))
from policy.remote.chunk_executor import (
    ChunkExecutor,
    InferenceConfig,
    WebsocketChunkAdapter,
)
from policy.remote.websocket_client_policy import (
    RemotePolicyError,
    WebsocketClientPolicy,
)

from fasterwam.serve.serialization import packb
from fasterwam.serve.websocket_server import (
    WebsocketPolicyServer,
    validate_chunk_context,
)


def context(episode="episode", request=0, step=0):
    return {
        "episode_id": episode,
        "request_id": request,
        "observation_step": step,
        "control_dt": 0.01,
        "previous_actions": np.zeros((3, 14), dtype=np.float32),
        "inference_delay_steps": 2,
        "guidance_method": "none",
    }


def observation():
    return {"qpos": np.zeros(14, dtype=np.float32), "images": {}}


class Policy:
    history_tail_len = 0
    control_dt = 0.01
    action_horizon = 20

    def __init__(self, delay=0):
        self.delay = delay
        self.started, self.release = threading.Event(), threading.Event()
        self.calls, self.running, self.maximum = 0, 0, 0
        self.events = []

    def select_action(self, obs, history=None):
        self.running += 1
        self.maximum = max(self.maximum, self.running)
        self.calls += 1
        self.events.append("infer_start")
        self.started.set()
        if self.delay is None:
            assert self.release.wait(3)
        else:
            time.sleep(self.delay)
        self.running -= 1
        self.events.append("infer_end")
        return np.zeros((20, 14), dtype=np.float32), "TEST", False

    def reset(self):
        assert self.running == 0
        self.events.append("reset")


@contextmanager
def running_server(policy, legacy=False, wrong_id=False):
    class Server(WebsocketPolicyServer):
        async def _dispatch(self, request):
            if legacy:
                assert "chunk_context" not in request
            response = await super()._dispatch(request)
            if wrong_id and response.get("chunk_context"):
                response["chunk_context"]["episode_id"] = "wrong"
            return response

    server = Server(policy, host="127.0.0.1", port=0)
    if legacy:
        server._metadata.pop("supports_timed_chunks")
        server._metadata.pop("supports_rtc")
        server._metadata.pop("supports_simple")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.wait_until_ready()
    try:
        yield port
    finally:
        policy.release.set()
        server.shutdown()
        thread.join(5)
        assert not thread.is_alive()


class ProtocolTests(unittest.TestCase):
    def test_send_deadline_when_peer_stops_reading(self):
        # A TCP peer completes the handshake/metadata exchange, then never reads
        # the image payload. A protocol close frame also blocks behind sendall.
        release = threading.Event()
        failures = []
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(5)

        def blackhole_peer():
            try:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(5)
                    connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
                    headers = b""
                    while b"\r\n\r\n" not in headers:
                        data = connection.recv(4096)
                        if not data:
                            raise RuntimeError("Client disconnected during handshake")
                        headers += data
                    fields = dict(
                        line.split(b":", 1) for line in headers.split(b"\r\n")[1:] if b":" in line
                    )
                    key = fields[b"Sec-WebSocket-Key"].strip()
                    accept = base64.b64encode(hashlib.sha1(
                        key + b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
                    ).digest())
                    connection.sendall(
                        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                        b"Connection: Upgrade\r\nSec-WebSocket-Accept: " + accept + b"\r\n\r\n"
                    )
                    connection.sendall(Frame(OP_BINARY, packb({
                        "type": "metadata", "history_tail_len": 0,
                    })).serialize(mask=False))
                    release.wait(3)
            except Exception as error:  # noqa: BLE001 - surface thread failures
                failures.append(error)

        peer = threading.Thread(target=blackhole_peer, daemon=True)
        peer.start()
        try:
            with WebsocketClientPolicy(
                port=listener.getsockname()[1], image_codec="raw", request_timeout=0.1,
            ) as client:
                client._ws.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
                obs = observation()
                obs["images"] = {"camera": np.zeros((2048, 2048, 3), dtype=np.uint8)}
                start = time.monotonic()
                with self.assertRaises((TimeoutError, ConnectionClosed)):
                    client.select_action(obs)
                self.assertLess(time.monotonic() - start, 1.0)
                with self.assertRaises((TimeoutError, ConnectionClosed)):
                    client.reset()
        finally:
            release.set()
            listener.close()
            peer.join(5)
        self.assertFalse(peer.is_alive())
        self.assertEqual(failures, [])

    def test_old_server_single_flight_and_context_correlation(self):
        policy = Policy(0.02)
        with (
            running_server(policy, legacy=True) as port,
            WebsocketClientPolicy(port=port) as client,
        ):
            adapter = WebsocketChunkAdapter(client)
            try:
                adapter.submit(observation(), None, context())
                with self.assertRaises(RuntimeError):
                    adapter.submit(observation(), None, context(request=1))
                deadline = time.monotonic() + 3
                result = None
                while result is None:
                    self.assertLess(time.monotonic(), deadline)
                    result = adapter.poll()
                    time.sleep(0.001)
                self.assertEqual(result.context["episode_id"], "episode")
                self.assertEqual(result.context["observation_step"], 0)
                with self.assertRaises(RemotePolicyError):
                    client.require_guidance("rtc")
                with self.assertRaises(RemotePolicyError):
                    client.require_guidance("simple")
                adapter.reset()
                self.assertIsNone(adapter.poll())
            finally:
                adapter.close()
        self.assertEqual(policy.maximum, 1)

    def test_actual_worker_progresses_old_chunk_during_inference(self):
        policy = Policy()
        with running_server(policy) as port, WebsocketClientPolicy(port=port) as client:
            adapter = WebsocketChunkAdapter(client)
            executor = ChunkExecutor(
                adapter, InferenceConfig(mode="async", execute_horizon=1), 0.01
            )
            try:
                executor.start(observation())
                deadline = time.monotonic() + 3
                while not executor.poll_initial():
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.001)
                policy.started.clear()
                policy.delay = None
                for _ in range(6):
                    action = executor.action(observation())
                    executor.advance(action, action)
                self.assertTrue(policy.started.wait(2))
                self.assertEqual(executor.step, 6)
                self.assertEqual(policy.calls, 2)
                policy.release.set()
                executor.reset()
                self.assertEqual(executor.step, 0)
                self.assertIsNone(adapter.poll())
                self.assertEqual(policy.maximum, 1)
            finally:
                policy.release.set()
                executor.close()

    def test_timeout_poisoned_connection_and_wrong_identity(self):
        policy = Policy(0.15)
        with (
            running_server(policy) as port,
            WebsocketClientPolicy(port=port, request_timeout=0.04) as client,
        ):
            with self.assertRaises((TimeoutError, ConnectionClosed)):
                client.select_action(observation(), chunk_context=context())
            with self.assertRaises((TimeoutError, ConnectionClosed)):
                client.reset()
        policy = Policy()
        with (
            running_server(policy, wrong_id=True) as port,
            WebsocketClientPolicy(port=port) as client,
            self.assertRaises(RemotePolicyError),
        ):
            client.select_action(observation(), chunk_context=context())

    def test_reset_serialized_across_connections(self):
        policy = Policy(None)
        with (
            running_server(policy) as port,
            WebsocketClientPolicy(port=port) as first,
            WebsocketClientPolicy(port=port) as second,
        ):
            errors = []

            def infer():
                try:
                    first.select_action(observation())
                except Exception as error:  # noqa: BLE001 - report worker failures to the test thread
                    errors.append(error)

            thread = threading.Thread(target=infer)
            thread.start()
            self.assertTrue(policy.started.wait(2))
            reset_done = threading.Event()

            def reset():
                try:
                    second.reset()
                except Exception as error:  # noqa: BLE001 - report worker failures to the test thread
                    errors.append(error)
                finally:
                    reset_done.set()

            reset_thread = threading.Thread(target=reset)
            reset_thread.start()
            self.assertFalse(reset_done.wait(0.03))
            policy.release.set()
            thread.join(3)
            reset_thread.join(3)
            self.assertFalse(errors)
            self.assertEqual(policy.events, ["infer_start", "infer_end", "reset"])

    def test_rtc_forwarding_and_validation(self):
        class GuidedPolicy(Policy):
            supports_chunk_context = supports_rtc = True

            def select_action(self, obs, history=None, *, chunk_context=None):
                self.context = chunk_context
                return super().select_action(obs, history)

        policy = GuidedPolicy()
        with running_server(policy) as port, WebsocketClientPolicy(port=port) as client:
            request = context()
            request["guidance_method"] = "rtc"
            client.select_action(observation(), chunk_context=request)
            self.assertEqual(policy.context["guidance_method"], "rtc")
            np.testing.assert_array_equal(
                policy.context["previous_actions"], request["previous_actions"]
            )
        for key, value in [
            ("control_dt", 0),
            ("control_dt", float("nan")),
            ("observation_step", -1),
            ("request_id", True),
            ("episode_id", ""),
            ("inference_delay_steps", 1.2),
            ("previous_actions", np.full((2, 14), np.inf)),
            ("previous_actions", np.zeros((3, 13))),
            ("guidance_method", "other"),
        ]:
            invalid = context()
            invalid[key] = value
            with self.assertRaises(ValueError, msg=key):
                validate_chunk_context(invalid)

    def test_simple_and_rtc_capabilities_are_independent(self):
        class GuidedPolicy(Policy):
            supports_chunk_context = True

            def select_action(self, obs, history=None, *, chunk_context=None):
                self.context = chunk_context
                return super().select_action(obs, history)

        for rtc, simple in ((False, False), (True, False), (False, True), (True, True)):
            policy = GuidedPolicy()
            policy.supports_rtc, policy.supports_simple = rtc, simple
            with (
                self.subTest(rtc=rtc, simple=simple),
                running_server(policy) as port,
                WebsocketClientPolicy(port=port) as client,
            ):
                self.assertEqual(client.supports_simple, simple)
                self.assertEqual(client.supports_rtc, rtc)
                for method, enabled in (("rtc", rtc), ("simple", simple)):
                    request = context(step=7)
                    request["guidance_method"] = method
                    if not enabled:
                        with self.assertRaises(RemotePolicyError):
                            client.require_guidance(method)
                        # Verify the server also rejects clients bypassing capability checks.
                        with self.assertRaises(RemotePolicyError):
                            client._request({
                                "type": "select_action", "observation": observation(),
                                "chunk_context": request,
                            })
                        continue
                    action, _, _ = client.select_action(observation(), chunk_context=request)
                    self.assertEqual(policy.context["guidance_method"], method)
                    self.assertEqual(client.last_chunk_context["observation_step"], 7)
                    self.assertEqual(action.shape, (20, 14))  # Full chunk, no server-side skip.
                    np.testing.assert_array_equal(
                        policy.context["previous_actions"], request["previous_actions"]
                    )

    def test_cancellation_keeps_lock_until_inference_finishes(self):
        async def check():
            policy = Policy(None)
            server = WebsocketPolicyServer(policy)
            server._infer_lock = asyncio.Lock()
            request = {"type": "select_action", "observation": observation()}
            infer = asyncio.create_task(server._dispatch(request))
            self.assertTrue(await asyncio.to_thread(policy.started.wait, 2))
            infer.cancel()
            reset = asyncio.create_task(server._dispatch({"type": "reset"}))
            await asyncio.sleep(0.01)
            self.assertFalse(reset.done())
            # Repeated cancellation must not cancel the executor's future and
            # release the lock while its Python thread is still in the model.
            infer.cancel()
            await asyncio.sleep(0.01)
            self.assertFalse(reset.done())
            policy.release.set()
            with self.assertRaises(asyncio.CancelledError):
                await infer
            await reset
            self.assertEqual(policy.events, ["infer_start", "infer_end", "reset"])

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]] + remaining)
