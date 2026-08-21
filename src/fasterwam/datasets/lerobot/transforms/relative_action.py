from typing import List, Dict

import torch
from .rotation import quaternion_to_matrix, matrix_to_quaternion

# pose: position and quaternion in x, y, z, i, j, k, r
# mat: homogeneous transformation matrix in 4×4
# pos: position in x, y, z
# quat: quaternion in i, j, k, r


class RelativePoseTransform:
    def __init__(self, keys: List[str]):
        self.keys = keys
    
    def forward(self, batch: Dict):
        # for close-loop eval, "action" may not be in batch
        if "action" not in batch:
            return batch
        
        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k]
            batch["action"][k] = self._forward(action, state[..., -1:, :])
        
        return batch
        
    def backward(self, batch: Dict):
        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k]
            batch["action"][k] = self._backward(action, state[..., -1:, :])

        return batch

    def _forward(self, pose: torch.Tensor, base_pose: torch.Tensor):
        # pose & base_pose: position and quaternion in x, y, z, i, j, k, r
        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._absolute_to_relative(pose_matrix, base_pose_matrix)
        pose = self._matrix_to_pose(pose_matrix)
        return pose
    
    def _backward(self, pose: torch.Tensor, base_pose: torch.Tensor):
        # pose & base_pose: position and quaternion in x, y, z, i, j, k, r
        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._relative_to_absolute(pose_matrix, base_pose_matrix)
        pose = self._matrix_to_pose(pose_matrix)
        return pose
    
    @staticmethod
    def _pose_to_matrix(pose: torch.Tensor):
        position = pose[..., 0: 3]
        quaternion = pose[..., [6, 3, 4, 5]] # (i j k r) to (r i j k)
        rotation = quaternion_to_matrix(quaternion)
        matrix = torch.zeros(pose.shape[:-1] + (4, 4), dtype=pose.dtype, device=pose.device)
        matrix[..., 0: 3, 0: 3] = rotation
        matrix[..., 0: 3, 3] = position
        matrix[..., 3, 3] = 1
        return matrix

    @staticmethod
    def _matrix_to_pose(matrix: torch.Tensor):
        position = matrix[..., 0: 3, 3] / matrix[..., 3, 3][..., None]
        rotation = matrix[..., 0: 3, 0: 3]
        quaternion = matrix_to_quaternion(rotation)
        quaternion = quaternion[..., [1, 2, 3, 0]] # (r i j k) to (i j k r)
        pose = torch.cat([position, quaternion], dim=-1)
        return pose
    @staticmethod
    def _absolute_to_relative(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
        return torch.linalg.inv(base_pose_matrix) @ pose_matrix

    @staticmethod
    def _relative_to_absolute(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
        return base_pose_matrix @ pose_matrix
    

class RelativeJointTransform:
    def __init__(self, keys: List[str]):
        self.keys = keys

    def forward(self, batch: Dict):
        # for close-loop eval, "action" may not be in batch
        if "action" not in batch:
            return batch
        
        for k in self.keys:
            # NOTE: fixed to the first frame
            batch["action"][k] = batch["action"][k] - batch["state"][k][..., :1, :]

        return batch

    def backward(self, batch: Dict):
        for k in self.keys:
            # NOTE: fixed to the first frame
            batch["action"][k] = batch["action"][k] + batch["state"][k][..., :1, :]
        
        return batch


class MaskedRelativeJointTransform:
    """Rebase selected action dimensions while leaving the rest absolute.

    ``dim_masks`` maps each action key to one boolean per action dimension.
    True dimensions are made relative to the state at the first observation
    frame; False dimensions (WidowXAI grippers) remain absolute.

    This transform runs before normalization. Do not combine it with
    ``delta_action_dim_mask`` for absolute joint targets: end-of-episode
    padding must replicate the last target before it is rebased.
    """

    def __init__(self, dim_masks: Dict[str, List[bool]]):
        if not dim_masks:
            raise ValueError("`dim_masks` must not be empty.")
        self.dim_masks = {
            key: torch.as_tensor(list(mask), dtype=torch.bool)
            for key, mask in dim_masks.items()
        }
        for key, mask in self.dim_masks.items():
            if mask.ndim != 1:
                raise ValueError(
                    f"`dim_masks[{key}]` must be 1-D, got shape {tuple(mask.shape)}."
                )

    def _base(self, batch: Dict, key: str, mask: torch.Tensor) -> torch.Tensor:
        if key not in batch["state"]:
            raise KeyError(
                f"State key '{key}' is required to rebase actions; "
                f"available keys: {sorted(batch['state'].keys())}"
            )
        state = batch["state"][key]
        action = batch["action"][key]
        if action.shape[-1] != mask.shape[0]:
            raise ValueError(
                f"`dim_masks[{key}]` has length {mask.shape[0]} but action has "
                f"{action.shape[-1]} dimensions."
            )
        if state.shape[-1] != action.shape[-1]:
            raise ValueError(
                f"State '{key}' has {state.shape[-1]} dims but action has "
                f"{action.shape[-1]}."
            )
        return state[..., :1, :]

    def _assert_covers(self, batch: Dict) -> None:
        missing = sorted(set(batch["action"].keys()) - set(self.dim_masks.keys()))
        if missing:
            raise ValueError(
                f"`dim_masks` is missing action key(s) {missing}; "
                f"got {sorted(self.dim_masks.keys())}."
            )

    def forward(self, batch: Dict):
        if "action" not in batch:
            return batch
        self._assert_covers(batch)
        for key, mask in self.dim_masks.items():
            action = batch["action"][key]
            base = self._base(batch, key, mask)
            dim_mask = mask.to(device=action.device)
            batch["action"][key] = torch.where(dim_mask, action - base, action)
        return batch

    def backward(self, batch: Dict):
        self._assert_covers(batch)
        for key, mask in self.dim_masks.items():
            action = batch["action"][key]
            base = self._base(batch, key, mask)
            dim_mask = mask.to(device=action.device)
            batch["action"][key] = torch.where(dim_mask, action + base, action)
        return batch
