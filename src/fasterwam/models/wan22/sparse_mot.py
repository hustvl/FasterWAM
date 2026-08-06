from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from fasterwam.utils.logging_config import get_logger

from .sparse_action_dit import DEFAULT_SPARSE_MOT_CONDITION_LAYERS
from .mot import MoT
from .wan_video_dit import flash_attention

logger = get_logger(__name__)


class SparseMoT(MoT):
    """Sparse mixed-attention MoT with lightweight action-only refinement layers."""

    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        condition_layers: Optional[Sequence[int]] = None,
        mot_checkpoint_mixed_attn: bool = True,
        video_kv_fusion: Optional[str] = None,
        video_kv_fusion_init: str = "current_layer",
    ):
        nn.Module.__init__(self)
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        if mot_checkpoint_mixed_attn:
            logger.info(
                "Using gradient checkpointing for SparseMoT attention. "
                "This will save memory but use more computation."
            )

        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        self.num_layers = len(video_expert.blocks)
        self.action_num_layers = len(action_expert.blocks)
        self.num_heads = int(video_expert.num_heads)
        self.attn_head_dim = int(video_expert.attn_head_dim)

        if self.action_num_layers != self.num_layers:
            raise ValueError(
                "SparseMoT requires equal video/action depth: "
                f"{self.num_layers} vs {self.action_num_layers}."
            )
        if int(action_expert.attn_head_dim) != self.attn_head_dim:
            raise ValueError(
                "SparseActionDiT `attn_head_dim` must match video expert: "
                f"{action_expert.attn_head_dim} vs {self.attn_head_dim}."
            )

        self.condition_layers = self._normalize_condition_layers(condition_layers, self.num_layers)
        self.condition_layer_set = frozenset(self.condition_layers)
        self.video_kv_fusion = self._normalize_video_kv_fusion(video_kv_fusion)
        self.video_kv_fusion_init = self._normalize_video_kv_fusion_init(video_kv_fusion_init)
        self.condition_intervals = self._build_condition_intervals(self.condition_layers)
        self._condition_layer_to_interval_idx = {
            layer_idx: interval_idx for interval_idx, layer_idx in enumerate(self.condition_layers)
        }
        if hasattr(action_expert, "condition_layers") and tuple(action_expert.condition_layers) != self.condition_layers:
            raise ValueError(
                "SparseMoT `condition_layers` must match SparseActionDiT condition layers: "
                f"{list(self.condition_layers)} vs {list(action_expert.condition_layers)}."
            )

        for layer_idx, block in enumerate(action_expert.blocks):
            expected_heads = (
                int(action_expert.num_heads)
                if layer_idx in self.condition_layer_set
                else int(action_expert.noncondition_num_heads)
            )
            if int(block.num_heads) != expected_heads:
                raise ValueError(
                    f"Action block {layer_idx} has num_heads={block.num_heads}, expected {expected_heads}."
                )
            if layer_idx in self.condition_layer_set:
                if int(block.num_heads) != self.num_heads:
                    raise ValueError(
                        f"Condition action block {layer_idx} must match video heads: "
                        f"{block.num_heads} vs {self.num_heads}."
                    )
                if not hasattr(block, "cross_attn") or not hasattr(block, "norm3"):
                    raise ValueError(f"Condition action block {layer_idx} must include cross_attn and norm3.")
            else:
                if hasattr(block, "cross_attn") or hasattr(block, "norm3"):
                    raise ValueError(f"Non-condition action block {layer_idx} must not include cross_attn/norm3.")

        self.video_kv_fusion_logits = nn.ParameterList()
        if self.video_kv_fusion == "interval_weighted_sum":
            ref_param = next(video_expert.parameters(), None)
            logit_device = ref_param.device if ref_param is not None else torch.device("cpu")
            for interval in self.condition_intervals:
                logits = torch.full((len(interval),), -6.0, dtype=torch.float32, device=logit_device)
                logits[-1] = 0.0
                self.video_kv_fusion_logits.append(nn.Parameter(logits))

        logger.info(
            "Initialized SparseMoT with layers=%d, condition_layers=%s, condition_heads=%d, "
            "noncondition_heads=%d, video_kv_fusion=%s",
            self.num_layers,
            list(self.condition_layers),
            int(action_expert.num_heads),
            int(action_expert.noncondition_num_heads),
            self.video_kv_fusion,
        )
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(
                "  Expert '%s': num_params=%.2f B",
                name,
                sum(p.numel() for p in expert.parameters()) / 1e9,
            )

    @staticmethod
    def _normalize_condition_layers(
        condition_layers: Optional[Sequence[int]],
        num_layers: int,
    ) -> tuple[int, ...]:
        if condition_layers is None:
            condition_layers = DEFAULT_SPARSE_MOT_CONDITION_LAYERS
        if isinstance(condition_layers, (str, bytes)):
            raise ValueError("`condition_layers` must be a sequence of layer indices, not a string.")
        normalized = []
        seen = set()
        for raw_idx in condition_layers:
            if isinstance(raw_idx, bool) or not isinstance(raw_idx, int):
                raise ValueError(
                    "`condition_layers` must contain integers in "
                    f"[0, {num_layers - 1}], got {raw_idx!r}."
                )
            if raw_idx < 0 or raw_idx >= num_layers:
                raise ValueError(
                    "`condition_layers` indices out of range: "
                    f"got {raw_idx}, expected [0, {num_layers - 1}]."
                )
            if raw_idx in seen:
                raise ValueError(f"`condition_layers` contains duplicate layer index {raw_idx}.")
            seen.add(raw_idx)
            normalized.append(raw_idx)
        if normalized != sorted(normalized):
            raise ValueError("`condition_layers` must be sorted in ascending order.")
        return tuple(normalized)

    @staticmethod
    def _normalize_video_kv_fusion(video_kv_fusion: Optional[str]) -> Optional[str]:
        if video_kv_fusion is None:
            return None
        if video_kv_fusion != "interval_weighted_sum":
            raise ValueError(
                "`video_kv_fusion` must be None or 'interval_weighted_sum', "
                f"got {video_kv_fusion!r}."
            )
        return video_kv_fusion

    @staticmethod
    def _normalize_video_kv_fusion_init(video_kv_fusion_init: str) -> str:
        if video_kv_fusion_init != "current_layer":
            raise ValueError(
                "`video_kv_fusion_init` must be 'current_layer', "
                f"got {video_kv_fusion_init!r}."
            )
        return video_kv_fusion_init

    @staticmethod
    def _build_condition_intervals(condition_layers: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        intervals = []
        prev_layer = -1
        for condition_layer in condition_layers:
            intervals.append(tuple(range(prev_layer + 1, condition_layer + 1)))
            prev_layer = condition_layer
        return tuple(intervals)

    def _fuse_video_kv(
        self,
        condition_layer: int,
        interval_kv: Sequence[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        interval_idx = self._condition_layer_to_interval_idx[condition_layer]
        expected_len = len(self.condition_intervals[interval_idx])
        if len(interval_kv) != expected_len:
            raise ValueError(
                f"Condition layer {condition_layer} expected {expected_len} interval K/V entries, "
                f"got {len(interval_kv)}."
            )

        logits = self.video_kv_fusion_logits[interval_idx]
        weights = torch.softmax(logits, dim=0)
        fused_k = None
        fused_v = None
        for weight, layer_kv in zip(weights, interval_kv):
            if "k" not in layer_kv or "v" not in layer_kv:
                raise ValueError("Each interval K/V entry must contain `k` and `v`.")
            k = layer_kv["k"]
            v = layer_kv["v"]
            layer_weight = weight.to(device=k.device, dtype=k.dtype)
            weighted_k = k * layer_weight
            weighted_v = v * layer_weight.to(device=v.device, dtype=v.dtype)
            fused_k = weighted_k if fused_k is None else fused_k + weighted_k
            fused_v = weighted_v if fused_v is None else fused_v + weighted_v
        if fused_k is None or fused_v is None:
            raise ValueError(f"Condition layer {condition_layer} has an empty K/V interval.")
        return {"k": fused_k, "v": fused_v}

    def _mixed_attention_with_num_heads(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        num_heads: int,
    ) -> torch.Tensor:
        attn_mask = None if attention_mask is None else attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(q=q, k=k, v=v, num_heads=num_heads, ctx_mask=attn_mask)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    def _build_video_attention_io(self, tokens_all, freqs_all, t_mod_all, context_all, layer_idx: int):
        video_expert = self.mixtures["video"]
        video_block = video_expert.blocks[layer_idx]
        (
            q_video,
            k_video,
            v_video,
            residual_video,
            gate_video,
            shift_video,
            scale_video,
            gate_mlp_video,
            checkpoint_video,
        ) = self._build_expert_attention_io(
            expert=video_expert,
            block=video_block,
            x=tokens_all["video"],
            freqs=freqs_all["video"],
            t_mod=t_mod_all["video"],
        )
        return {
            "q": q_video,
            "k": k_video,
            "v": v_video,
            "block": video_block,
            "residual_x": residual_video,
            "gate_msa": gate_video,
            "shift_mlp": shift_video,
            "scale_mlp": scale_video,
            "gate_mlp": gate_mlp_video,
            "use_gradient_checkpointing": checkpoint_video,
            "context_payload": context_all.get("video"),
        }

    def _build_action_attention_io(self, tokens_all, freqs_all, t_mod_all, context_all, layer_idx: int):
        action_expert = self.mixtures["action"]
        action_block = action_expert.blocks[layer_idx]
        (
            q_action,
            k_action,
            v_action,
            residual_action,
            gate_action,
            shift_action,
            scale_action,
            gate_mlp_action,
            checkpoint_action,
        ) = self._build_expert_attention_io(
            expert=action_expert,
            block=action_block,
            x=tokens_all["action"],
            freqs=freqs_all["action"],
            t_mod=t_mod_all["action"],
        )
        return {
            "q": q_action,
            "k": k_action,
            "v": v_action,
            "block": action_block,
            "residual_x": residual_action,
            "gate_msa": gate_action,
            "shift_mlp": shift_action,
            "scale_mlp": scale_action,
            "gate_mlp": gate_mlp_action,
            "use_gradient_checkpointing": checkpoint_action,
            "context_payload": context_all.get("action") if layer_idx in self.condition_layer_set else None,
        }

    def _forward_with_video_kv_fusion(
        self,
        tokens_all: Dict[str, torch.Tensor],
        attention_masks: Sequence[torch.Tensor],
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        video_seq_len: int,
        action_seq_len: int,
    ) -> Dict[str, torch.Tensor]:
        total_seq_len = video_seq_len + action_seq_len
        interval_kv: list[dict[str, torch.Tensor]] = []

        for layer_idx in range(self.num_layers):
            layer_attention_mask = attention_masks[layer_idx]
            video_io = self._build_video_attention_io(
                tokens_all=tokens_all,
                freqs_all=freqs_all,
                t_mod_all=t_mod_all,
                context_all=context_all,
                layer_idx=layer_idx,
            )
            action_io = self._build_action_attention_io(
                tokens_all=tokens_all,
                freqs_all=freqs_all,
                t_mod_all=t_mod_all,
                context_all=context_all,
                layer_idx=layer_idx,
            )

            interval_kv.append({"k": video_io["k"], "v": video_io["v"]})
            video_mask = layer_attention_mask[:video_seq_len, :video_seq_len]
            mixed_video = self._mixed_attention_with_num_heads(
                q_cat=video_io["q"],
                k_cat=video_io["k"],
                v_cat=video_io["v"],
                attention_mask=video_mask,
                num_heads=self.num_heads,
            )

            if layer_idx in self.condition_layer_set:
                action_attention_mask = layer_attention_mask[video_seq_len:total_seq_len, :total_seq_len]
                fused_video_kv = self._fuse_video_kv(layer_idx, interval_kv)
                mixed_action = self._mixed_attention_with_num_heads(
                    q_cat=action_io["q"],
                    k_cat=torch.cat([fused_video_kv["k"], action_io["k"]], dim=1),
                    v_cat=torch.cat([fused_video_kv["v"], action_io["v"]], dim=1),
                    attention_mask=action_attention_mask,
                    num_heads=self.num_heads,
                )
                interval_kv = []
            else:
                mixed_action = self._mixed_attention_with_num_heads(
                    q_cat=action_io["q"],
                    k_cat=action_io["k"],
                    v_cat=action_io["v"],
                    attention_mask=None,
                    num_heads=int(action_io["block"].num_heads),
                )

            tokens_all["video"] = self._apply_post_with_optional_checkpoint(
                block=video_io["block"],
                residual_x=video_io["residual_x"],
                gate_msa=video_io["gate_msa"],
                shift_mlp=video_io["shift_mlp"],
                scale_mlp=video_io["scale_mlp"],
                gate_mlp=video_io["gate_mlp"],
                use_gradient_checkpointing=video_io["use_gradient_checkpointing"],
                mixed_slice=mixed_video,
                context_payload=video_io["context_payload"],
            )
            tokens_all["action"] = self._apply_post_with_optional_checkpoint(
                block=action_io["block"],
                residual_x=action_io["residual_x"],
                gate_msa=action_io["gate_msa"],
                shift_mlp=action_io["shift_mlp"],
                scale_mlp=action_io["scale_mlp"],
                gate_mlp=action_io["gate_mlp"],
                use_gradient_checkpointing=action_io["use_gradient_checkpointing"],
                mixed_slice=mixed_action,
                context_payload=action_io["context_payload"],
            )

        return tokens_all

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor | Sequence[torch.Tensor],
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ):
        for name in ("video", "action"):
            if name not in embeds_all:
                raise ValueError(f"Missing expert tokens for {name!r}")
            if name not in freqs_all:
                raise ValueError(f"Missing expert freqs for {name!r}")
            if name not in t_mod_all:
                raise ValueError(f"Missing expert t_mod for {name!r}")

        tokens_all = {k: v for k, v in embeds_all.items()}
        video_seq_len = int(tokens_all["video"].shape[1])
        action_seq_len = int(tokens_all["action"].shape[1])
        total_seq_len = video_seq_len + action_seq_len
        attention_masks = self._resolve_attention_masks(attention_mask, expected_seq_len=total_seq_len)

        if self.video_kv_fusion == "interval_weighted_sum":
            return self._forward_with_video_kv_fusion(
                tokens_all=tokens_all,
                attention_masks=attention_masks,
                freqs_all=freqs_all,
                context_all=context_all,
                t_mod_all=t_mod_all,
                video_seq_len=video_seq_len,
                action_seq_len=action_seq_len,
            )

        for layer_idx in range(self.num_layers):
            layer_attention_mask = attention_masks[layer_idx]
            video_io = self._build_video_attention_io(
                tokens_all=tokens_all,
                freqs_all=freqs_all,
                t_mod_all=t_mod_all,
                context_all=context_all,
                layer_idx=layer_idx,
            )
            action_io = self._build_action_attention_io(
                tokens_all=tokens_all,
                freqs_all=freqs_all,
                t_mod_all=t_mod_all,
                context_all=context_all,
                layer_idx=layer_idx,
            )

            if layer_idx in self.condition_layer_set:
                q_cat = torch.cat([video_io["q"], action_io["q"]], dim=1)
                k_cat = torch.cat([video_io["k"], action_io["k"]], dim=1)
                v_cat = torch.cat([video_io["v"], action_io["v"]], dim=1)
                mixed = self._mixed_attention_with_num_heads(
                    q_cat=q_cat,
                    k_cat=k_cat,
                    v_cat=v_cat,
                    attention_mask=layer_attention_mask,
                    num_heads=self.num_heads,
                )
                mixed_video = mixed[:, :video_seq_len, :]
                mixed_action = mixed[:, video_seq_len:, :]
            else:
                video_mask = layer_attention_mask[:video_seq_len, :video_seq_len]
                mixed_video = self._mixed_attention_with_num_heads(
                    q_cat=video_io["q"],
                    k_cat=video_io["k"],
                    v_cat=video_io["v"],
                    attention_mask=video_mask,
                    num_heads=self.num_heads,
                )
                mixed_action = self._mixed_attention_with_num_heads(
                    q_cat=action_io["q"],
                    k_cat=action_io["k"],
                    v_cat=action_io["v"],
                    attention_mask=None,
                    num_heads=int(action_io["block"].num_heads),
                )

            tokens_all["video"] = self._apply_post_with_optional_checkpoint(
                block=video_io["block"],
                residual_x=video_io["residual_x"],
                gate_msa=video_io["gate_msa"],
                shift_mlp=video_io["shift_mlp"],
                scale_mlp=video_io["scale_mlp"],
                gate_mlp=video_io["gate_mlp"],
                use_gradient_checkpointing=video_io["use_gradient_checkpointing"],
                mixed_slice=mixed_video,
                context_payload=video_io["context_payload"],
            )
            tokens_all["action"] = self._apply_post_with_optional_checkpoint(
                block=action_io["block"],
                residual_x=action_io["residual_x"],
                gate_msa=action_io["gate_msa"],
                shift_mlp=action_io["shift_mlp"],
                scale_mlp=action_io["scale_mlp"],
                gate_mlp=action_io["gate_mlp"],
                use_gradient_checkpointing=action_io["use_gradient_checkpointing"],
                mixed_slice=mixed_action,
                context_payload=action_io["context_payload"],
            )

        return tokens_all

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[Optional[dict[str, torch.Tensor]]]:
        if self.video_kv_fusion is None:
            return super().prefill_video_cache(
                video_tokens=video_tokens,
                video_freqs=video_freqs,
                video_t_mod=video_t_mod,
                video_context_payload=video_context_payload,
                video_attention_mask=video_attention_mask,
            )
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        video_expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[Optional[dict[str, torch.Tensor]]] = [None] * self.num_layers
        interval_kv: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = video_expert.blocks[layer_idx]
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=video_expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            interval_kv.append({"k": k, "v": v})
            mixed = self._mixed_attention_with_num_heads(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
                num_heads=self.num_heads,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            if layer_idx in self.condition_layer_set:
                kv_cache[layer_idx] = self._fuse_video_kv(layer_idx, interval_kv)
                interval_kv = []
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[Optional[dict[str, torch.Tensor]]],
        attention_mask: torch.Tensor | Sequence[torch.Tensor],
        video_seq_len: int,
    ) -> torch.Tensor:
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} video layers, got {len(video_kv_cache)}."
            )

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        attention_masks = self._resolve_attention_masks(attention_mask, expected_seq_len=total_seq_len)

        action_expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = action_expert.blocks[layer_idx]
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=action_expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )

            if layer_idx in self.condition_layer_set:
                layer_attention_mask = attention_masks[layer_idx]
                action_attention_mask = layer_attention_mask[video_seq_len:total_seq_len, :total_seq_len]
                layer_cache = video_kv_cache[layer_idx]
                if layer_cache is None:
                    raise ValueError(f"`video_kv_cache[{layer_idx}]` must contain fused video K/V.")
                if "k" not in layer_cache or "v" not in layer_cache:
                    raise ValueError(f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`.")
                k_video = layer_cache["k"]
                v_video = layer_cache["v"]
                if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                    raise ValueError(
                        f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                    )
                mixed = self._mixed_attention_with_num_heads(
                    q_cat=q_action,
                    k_cat=torch.cat([k_video, k_action], dim=1),
                    v_cat=torch.cat([v_video, v_action], dim=1),
                    attention_mask=action_attention_mask,
                    num_heads=self.num_heads,
                )
                context_payload = action_context_payload
            else:
                mixed = self._mixed_attention_with_num_heads(
                    q_cat=q_action,
                    k_cat=k_action,
                    v_cat=v_action,
                    attention_mask=None,
                    num_heads=int(block.num_heads),
                )
                context_payload = None

            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=context_payload,
            )
        return x
