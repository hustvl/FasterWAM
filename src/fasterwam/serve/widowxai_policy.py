"""Online WidowXAI FasterWAM policy used by the real-robot server."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as transforms_f
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fasterwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from fasterwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fasterwam.models.wan22.rtc import RTCConfig, RTCGuidance
from fasterwam.models.wan22.simple import SimpleConfig, SimpleGuidance
from fasterwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision

DEFAULT_RUN_DIR = Path(
    "/lustre/data/fasterwam/runs/widowxai_fasterwam_2wrist_vertical_384_1e-4/"
    "widowxai_2wrist_vertical_scratch_4443674"
)

CAMERA_ALIASES = {
    "cam_high": ("cam_high", "camera_high"),
    "cam_left_wrist": ("cam_left_wrist", "camera_wrist_left"),
    "cam_right_wrist": ("cam_right_wrist", "camera_wrist_right"),
}
ACTION_INFER_MODES = ("one_pass_future_cache", "joint_rollout")


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    config_path: Path
    checkpoint_path: Path
    norm_stats_path: Path


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"step_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def resolve_run_artifacts(
    run_dir: str | Path,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    norm_stats_path: str | Path | None = None,
) -> RunArtifacts:
    run = Path(run_dir).expanduser().resolve()
    config = Path(config_path).expanduser().resolve() if config_path else run / "config.yaml"
    stats = Path(norm_stats_path).expanduser().resolve() if norm_stats_path else run / "dataset_stats.json"
    if checkpoint_path:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
    else:
        candidates = sorted((run / "checkpoints" / "weights").glob("step_*.pt"), key=_checkpoint_step)
        if not candidates:
            raise FileNotFoundError(f"No step_*.pt checkpoint found under {run / 'checkpoints/weights'}")
        checkpoint = candidates[-1]
    for label, path in (
        ("run config", config),
        ("checkpoint", checkpoint),
        ("normalization stats", stats),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    return RunArtifacts(run, config, checkpoint, stats)


def _load_single_dataset_task(cfg: DictConfig) -> str:
    dataset_dirs = list(cfg.data.train.dataset_dirs)
    if not dataset_dirs:
        raise ValueError("The resolved training config has no data.train.dataset_dirs")
    tasks_path = Path(str(dataset_dirs[0])) / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Training task metadata not found: {tasks_path}")
    tasks = []
    with tasks_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                tasks.append(str(json.loads(line)["task"]))
    if len(tasks) != 1:
        raise ValueError(f"Expected one WidowXAI task in {tasks_path}, got {tasks}")
    return tasks[0]


def build_inference_model_config(cfg: DictConfig) -> DictConfig:
    """Return a checkpoint-only, inference-safe copy of the resolved model config."""

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.skip_dit_load_from_pretrain = True
    model_cfg.action_dit_pretrained_path = None
    model_cfg.mot_checkpoint_mixed_attn = False
    model_cfg.video_dit_config.use_gradient_checkpointing = False
    model_cfg.action_dit_config.use_gradient_checkpointing = False
    return model_cfg


class WidowXAIOnlinePreprocessor:
    """Reproduce a two-wrist or legacy three-camera run for one observation."""

    def __init__(self, cfg: DictConfig, norm_stats_path: str | Path) -> None:
        self.cfg = cfg
        self.data_cfg = cfg.data.val
        self.processor = instantiate(self.data_cfg.processor)
        self.processor.eval()
        stats = load_dataset_stats_from_json(str(norm_stats_path))
        self.processor.set_normalizer_from_stats(stats)

        shape_meta = OmegaConf.to_container(self.data_cfg.shape_meta, resolve=True)
        if not isinstance(shape_meta, dict):
            raise TypeError("data.val.shape_meta must resolve to a dictionary")
        self.shape_meta = shape_meta
        self.camera_meta = list(shape_meta["images"])
        self.camera_names = tuple(str(meta["key"]) for meta in self.camera_meta)
        # Legacy configs predate this flag and are three-camera runs. Absence
        # must therefore mean False rather than guessing from the checkpoint.
        self.drop_top_view = bool(self.data_cfg.get("drop_top_view", False))
        expected_cameras = (
            ("cam_left_wrist", "cam_right_wrist")
            if self.drop_top_view
            else ("cam_high", "cam_left_wrist", "cam_right_wrist")
        )
        if self.camera_names != expected_cameras:
            raise ValueError(
                f"drop_top_view={self.drop_top_view} requires the trained camera order "
                f"{expected_cameras}, got {self.camera_names}"
            )
        expected_concat = "vertical" if self.drop_top_view else "robotwin"
        if str(self.data_cfg.concat_multi_camera) != expected_concat:
            raise ValueError(
                f"drop_top_view={self.drop_top_view} requires "
                f"concat_multi_camera={expected_concat!r}, got "
                f"{self.data_cfg.concat_multi_camera!r}"
            )
        action_meta = list(shape_meta["action"])
        state_meta = list(shape_meta["state"])
        if action_meta != [{"key": "default", "raw_shape": 14, "shape": 14}]:
            raise ValueError(f"Expected one 14-D default action field, got {action_meta}")
        if state_meta != [{"key": "default", "raw_shape": 14, "shape": 14}]:
            raise ValueError(f"Expected one 14-D default state field, got {state_meta}")

        self.action_dim = 14
        self.action_horizon = int(self.data_cfg.num_frames) - 1
        ratio = int(self.data_cfg.action_video_freq_ratio)
        self.num_video_frames = len(range(0, int(self.data_cfg.num_frames), ratio))
        self.video_size = tuple(int(v) for v in self.data_cfg.video_size)
        if self.drop_top_view and self.video_size[0] % 2:
            raise ValueError(f"drop_top_view requires an even model height, got {self.video_size}")
        self.raw_image_shape = tuple(int(v) for v in self.camera_meta[0]["raw_shape"][1:]) + (3,)
        self.model_image_shape = (self.video_size[0], self.video_size[1], 3)
        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}
        )
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

    @staticmethod
    def _lookup_image(images: dict[str, Any], canonical_name: str) -> np.ndarray:
        aliases = CAMERA_ALIASES.get(canonical_name, (canonical_name,))
        selected = next((name for name in aliases if name in images), None)
        if selected is None:
            raise KeyError(
                f"Missing camera {canonical_name!r}; accepted aliases={aliases}, "
                f"available={sorted(images)}"
            )
        return np.asarray(images[selected])

    def _prepare_camera(self, images: dict[str, Any], meta: dict[str, Any]) -> torch.Tensor:
        name = str(meta["key"])
        image = self._lookup_image(images, name)
        raw_c, raw_h, raw_w = (int(v) for v in meta["raw_shape"])
        expected = (raw_h, raw_w, raw_c)
        if image.shape != expected or image.dtype != np.uint8:
            raise ValueError(
                f"Camera {name!r} must be RGB uint8 with shape {expected}, "
                f"got dtype={image.dtype} shape={image.shape}"
            )
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        transforms = self.processor.val_transforms
        current_transforms = transforms[name] if isinstance(transforms, dict) else transforms
        for transform in current_transforms:
            tensor = transform(tensor)
        expected_processed = (1, *(int(v) for v in meta["shape"]))
        if tuple(tensor.shape) != expected_processed:
            raise ValueError(
                f"Camera {name!r} preprocessing produced {tuple(tensor.shape)}, "
                f"expected {expected_processed}"
            )
        return tensor[0]

    def _fit_wrist_panel(self, camera: torch.Tensor) -> torch.Tensor:
        panel_h = self.video_size[0] // 2
        panel_w = self.video_size[1]
        source_h, source_w = camera.shape[-2:]
        scale = min(panel_h / source_h, panel_w / source_w)
        resized_h = min(panel_h, max(1, int(round(source_h * scale))))
        resized_w = min(panel_w, max(1, int(round(source_w * scale))))
        camera = transforms_f.resize(
            camera,
            size=[resized_h, resized_w],
            interpolation=transforms_f.InterpolationMode.BILINEAR,
            antialias=True,
        )
        pad_h = panel_h - resized_h
        pad_w = panel_w - resized_w
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        panel = transforms_f.pad(
            camera,
            padding=[pad_left, pad_top, pad_right, pad_bottom],
            fill=0.5,
        )
        expected = (3, panel_h, panel_w)
        if tuple(panel.shape) != expected:
            raise ValueError(f"Wrist panel has shape {tuple(panel.shape)}, expected {expected}")
        return panel

    def prepare_image(self, images: dict[str, Any]) -> torch.Tensor:
        cameras = [self._prepare_camera(images, meta) for meta in self.camera_meta]
        if self.drop_top_view:
            panels = [self._fit_wrist_panel(camera) for camera in cameras]
            mosaic = torch.cat(panels, dim=-2)
        else:
            cam_high = transforms_f.resize(
                cameras[0],
                size=[256, 320],
                interpolation=transforms_f.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_left = transforms_f.resize(
                cameras[1],
                size=[128, 160],
                interpolation=transforms_f.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = transforms_f.resize(
                cameras[2],
                size=[128, 160],
                interpolation=transforms_f.InterpolationMode.BILINEAR,
                antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            mosaic = torch.cat([cam_high, bottom], dim=-2)
        mosaic = self.resize_transform(mosaic)
        mosaic = self.crop_transform(mosaic)
        mosaic = self.normalize_transform(mosaic).contiguous()
        expected = (3, self.video_size[0], self.video_size[1])
        if tuple(mosaic.shape) != expected:
            raise ValueError(f"Model image has shape {tuple(mosaic.shape)}, expected {expected}")
        return mosaic

    def prepare_proprio(self, qpos: Any) -> torch.Tensor:
        state = np.asarray(qpos, dtype=np.float32)
        if state.shape != (self.action_dim,):
            raise ValueError(f"WidowXAI qpos must have shape (14,), got {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("WidowXAI qpos contains NaN or Inf")
        batch: dict[str, Any] = {
            "state": {"default": torch.from_numpy(state.copy()).unsqueeze(0)}
        }
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        batch = self.processor.action_state_merger.forward(batch)
        proprio = batch["state"][0].to(dtype=torch.float32).contiguous()
        if tuple(proprio.shape) != (self.action_dim,):
            raise ValueError(f"Normalized proprio has shape {tuple(proprio.shape)}, expected (14,)")
        return proprio

    def prepare_observation(self, obs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.prepare_image(dict(obs.get("images", {}))), self.prepare_proprio(obs["qpos"])

    def normalize_previous_actions(self, actions: Any, qpos: Any) -> torch.Tensor:
        """Rebase physical targets on *this* observation with the training pipeline.

        Pad before normalization to support stepwise statistics. The returned
        prefix retains its valid length; padding never participates in guidance.
        """
        actions = np.asarray(actions, dtype=np.float32)
        state = np.asarray(qpos, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim or len(actions) > self.action_horizon:
            raise ValueError("previous_actions must have shape [L,14], L <= action_horizon")
        if state.shape != (self.action_dim,) or not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError("Invalid physical prefix or current qpos")
        length = len(actions)
        if not length:
            return torch.empty((0, self.action_dim), dtype=torch.float32)
        padded = np.repeat(actions[-1:], self.action_horizon, axis=0)
        padded[:length] = actions
        batch = {
            "action": {"default": torch.from_numpy(padded)},
            "state": {"default": torch.from_numpy(state.copy()).reshape(1, self.action_dim)},
        }
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        batch = self.processor.action_state_merger.forward(batch)
        result = batch["action"][:length].to(dtype=torch.float32).contiguous()
        if tuple(result.shape) != (length, self.action_dim) or not torch.isfinite(result).all():
            raise ValueError("Training transforms produced invalid normalized prefix")
        return result

    def denormalize_action(
        self,
        normalized_action: torch.Tensor,
        normalized_proprio: torch.Tensor,
    ) -> np.ndarray:
        action = normalized_action.detach().to(device="cpu", dtype=torch.float32)
        proprio = normalized_proprio.detach().to(device="cpu", dtype=torch.float32)
        if tuple(action.shape) != (self.action_horizon, self.action_dim):
            raise ValueError(
                f"Normalized action must have shape ({self.action_horizon}, {self.action_dim}), "
                f"got {tuple(action.shape)}"
            )
        if tuple(proprio.shape) != (self.action_dim,):
            raise ValueError(f"Normalized proprio must have shape (14,), got {tuple(proprio.shape)}")
        batch: dict[str, Any] = {
            "action": action.unsqueeze(0),
            "state": proprio.reshape(1, 1, self.action_dim),
        }
        batch = self.processor.action_state_merger.backward(batch)
        batch = self.processor.normalizer.backward(batch)
        if self.processor.action_state_transforms is not None:
            for transform in reversed(self.processor.action_state_transforms):
                batch = transform.backward(batch)
        result = batch["action"]["default"][0].numpy().astype(np.float32, copy=False)
        if result.shape != (self.action_horizon, self.action_dim):
            raise ValueError(f"Physical action has unexpected shape {result.shape}")
        if not np.isfinite(result).all():
            raise ValueError("Physical action contains NaN or Inf")
        return np.ascontiguousarray(result)


class WidowXAIFasterWAMPolicy:
    """Load a resolved WidowXAI run and return absolute 14-DoF action chunks."""

    history_tail_len = 0
    action_dim = 14
    supports_chunk_context = True
    def __init__(
        self,
        run_dir: str | Path = DEFAULT_RUN_DIR,
        config_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        norm_stats_path: str | Path | None = None,
        task: str | None = None,
        text_embedding_cache_dir: str | Path | None = None,
        device: str = "cuda:0",
        num_inference_steps: int | None = None,
        action_infer_mode: str | None = None,
        seed: int = 42,
        control_fps: float | None = None,
        rtc_config: RTCConfig | None = None,
        simple_config: SimpleConfig | None = None,
    ) -> None:
        self.rtc_config = rtc_config or RTCConfig()
        self.supports_rtc = self.rtc_config.enabled
        self.simple_config = simple_config or SimpleConfig()
        self.supports_simple = self.simple_config.enabled
        self.artifacts = resolve_run_artifacts(
            run_dir=run_dir,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            norm_stats_path=norm_stats_path,
        )
        self.checkpoint_path = self.artifacts.checkpoint_path
        self.cfg = OmegaConf.load(self.artifacts.config_path)
        self.preprocessor = WidowXAIOnlinePreprocessor(self.cfg, self.artifacts.norm_stats_path)
        self.camera_names = self.preprocessor.camera_names
        self.drop_top_view = self.preprocessor.drop_top_view
        self.raw_image_shape = self.preprocessor.raw_image_shape
        self.model_image_shape = self.preprocessor.model_image_shape
        self.action_horizon = self.preprocessor.action_horizon
        self.num_video_frames = self.preprocessor.num_video_frames

        self.task = str(task) if task is not None else _load_single_dataset_task(self.cfg)
        if not self.task.strip():
            raise ValueError("Task must be non-empty")
        self.context, self.context_mask = self._load_text_context(text_embedding_cache_dir)

        self.num_inference_steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else self.cfg.get("eval_num_inference_steps", 10)
        )
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        self.action_infer_mode = str(
            action_infer_mode
            if action_infer_mode is not None
            else self.cfg.get("default_action_infer_mode", "one_pass_future_cache")
        )
        if self.action_infer_mode not in ACTION_INFER_MODES:
            raise ValueError(
                f"action_infer_mode must be one of {ACTION_INFER_MODES}, got {self.action_infer_mode!r}"
            )
        self.seed = int(seed)
        self.control_fps = float(control_fps) if control_fps is not None else self._training_fps()
        if self.control_fps <= 0:
            raise ValueError("control_fps must be positive")
        self.control_dt = 1.0 / self.control_fps

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to load this FasterWAM checkpoint")
        precision = _normalize_mixed_precision(str(self.cfg.get("mixed_precision", "bf16")))
        model_dtype = _mixed_precision_to_model_dtype(precision)
        model_cfg = build_inference_model_config(self.cfg)
        print(
            "[WidowXAIFasterWAMPolicy] loading "
            f"config={self.artifacts.config_path} checkpoint={self.checkpoint_path} "
            f"stats={self.artifacts.norm_stats_path} device={device} dtype={model_dtype}",
            flush=True,
        )
        self.model = instantiate(model_cfg, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(str(self.checkpoint_path))
        self.model.eval()
        self.model.requires_grad_(False)
        self._infer_fn = self._resolve_infer_fn()
        print(
            "[WidowXAIFasterWAMPolicy] ready "
            f"task={self.task!r} cameras={self.camera_names} "
            f"drop_top_view={str(self.drop_top_view).lower()} "
            f"raw_image={self.raw_image_shape} model_image={self.model_image_shape} "
            f"action=({self.action_horizon},14) video_frames={self.num_video_frames} "
            f"mode={self.action_infer_mode} steps={self.num_inference_steps} "
            f"fps={self.control_fps:g}",
            flush=True,
        )

    def _resolve_infer_fn(self):
        if self.action_infer_mode == "one_pass_future_cache":
            infer_fn = getattr(self.model, "infer_action_one_pass_future_cache", None)
        else:
            infer_fn = getattr(self.model, "infer_action", None)
        if not callable(infer_fn):
            raise TypeError(
                f"Configured model {type(self.model).__name__} does not support "
                f"action mode {self.action_infer_mode!r}"
            )
        if self.supports_rtc and "rtc_guidance" not in inspect.signature(infer_fn).parameters:
            raise TypeError(f"{self.action_infer_mode} does not accept rtc_guidance")
        if self.supports_simple and "simple_guidance" not in inspect.signature(infer_fn).parameters:
            raise TypeError(f"{self.action_infer_mode} does not accept simple_guidance")
        return infer_fn

    def _training_fps(self) -> float:
        dataset_dirs = list(self.cfg.data.train.dataset_dirs)
        if dataset_dirs:
            info_path = Path(str(dataset_dirs[0])) / "meta" / "info.json"
            if info_path.is_file():
                with info_path.open(encoding="utf-8") as stream:
                    return float(json.load(stream)["fps"])
        return 30.0

    def _load_text_context(
        self, cache_dir_override: str | Path | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_dir = (
            Path(cache_dir_override).expanduser().resolve()
            if cache_dir_override is not None
            else Path(str(self.cfg.data.val.text_embedding_cache_dir)).expanduser().resolve()
        )
        prompt = DEFAULT_PROMPT.format(task=self.task)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        context_len = int(self.cfg.data.val.context_len)
        cache_path = cache_dir / f"{digest}.t5_len{context_len}.wan22ti2v5b.pt"
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Text embedding cache for task {self.task!r} not found: {cache_path}. "
                "Run scripts/precompute_text_embeds.py for this exact task first."
            )
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        context = payload["context"].detach().to(device="cpu").clone()
        mask = payload["mask"].detach().to(device="cpu", dtype=torch.bool).clone()
        if context.ndim != 2 or tuple(context.shape[:1]) != (context_len,):
            raise ValueError(f"Invalid cached context shape in {cache_path}: {tuple(context.shape)}")
        if not torch.is_floating_point(context) or tuple(mask.shape) != (context_len,):
            raise ValueError(
                f"Invalid cached context dtype/mask in {cache_path}: {context.dtype}, {tuple(mask.shape)}"
            )
        context[~mask] = 0.0
        mask = torch.ones_like(mask)
        return context, mask

    def reset(self) -> None:
        return None

    def select_action(
        self,
        obs: dict[str, Any],
        history: dict[str, Any] | None = None,
        *, chunk_context: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, str, bool]:
        del history
        input_image, proprio = self.preprocessor.prepare_observation(obs)
        kwargs: dict[str, Any] = {
            "prompt": None,
            "input_image": input_image.unsqueeze(0),
            "action_horizon": self.action_horizon,
            "num_video_frames": self.num_video_frames,
            "proprio": proprio,
            "context": self.context,
            "context_mask": self.context_mask,
            "negative_prompt": None,
            "text_cfg_scale": 1.0,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": None,
            "seed": self.seed,
            "rand_device": "cpu",
            "tiled": False,
        }
        signature = inspect.signature(self._infer_fn)
        kwargs = {name: value for name, value in kwargs.items() if name in signature.parameters}
        method = chunk_context["guidance_method"] if chunk_context is not None else "none"
        if method not in ("none", "rtc", "simple"):
            raise ValueError(f"Unknown guidance_method: {method}")
        if method != "none":
            if not getattr(self, f"supports_{method}", False):
                raise ValueError(f"{method} is disabled by serving configuration")
            # Add guidance after the legacy filter; never silently drop it.
            keyword = f"{method}_guidance"
            if keyword not in signature.parameters:
                raise TypeError(f"Inference path cannot accept {method} guidance")
            previous = self.preprocessor.normalize_previous_actions(
                chunk_context["previous_actions"], obs["qpos"],
            )
            guidance_type = RTCGuidance if method == "rtc" else SimpleGuidance
            kwargs[keyword] = guidance_type(
                previous.unsqueeze(0), chunk_context["inference_delay_steps"],
                getattr(self, f"{method}_config"),
            )
        prediction = self._infer_fn(**kwargs)
        action = self.preprocessor.denormalize_action(prediction["action"], proprio)
        return action, "FASTERWAM", False
