"""ACT inference adapter for RMB's pinned ACT submodule, including CPU inference.

Never modify RMB or third_party/act. Only load checkpoints you trust.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


class ActPolicyAdapter:
    def __init__(self, project_root: Path, checkpoint: Path, device: str = "cpu"):
        self.root = Path(project_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")

        repo = self.root / "app" / "RoboManipBaselines"
        act_dir = repo / "third_party" / "act"
        if not (act_dir / "policy.py").is_file():
            raise FileNotFoundError(f"ACT submodule missing: {act_dir}")
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(act_dir))

        from detr.models.detr_vae import DETRVAE
        from policy import ACTPolicy
        from robo_manip_baselines.common import normalize_data, denormalize_data

        self._normalize = normalize_data
        self._denormalize = denormalize_data
        meta_path = self.checkpoint.parent / "model_meta_info.pkl"
        if not self.checkpoint.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"Model files not found: {self.checkpoint}, {meta_path}")
        with meta_path.open("rb") as f:  # only open your own trusted pickle
            self.meta = pickle.load(f)

        for section, desired in (("state", ["measured_joint_pos"]),
                                 ("action", ["command_joint_pos"])):
            if self.meta[section]["keys"] != desired:
                raise ValueError(f"Unsupported {section} keys: {self.meta[section]['keys']}")
        if self.meta["image"]["camera_names"] != ["front"]:
            raise ValueError("Expected one camera named 'front'")
        self.state_dim = len(self.meta["state"]["example"])
        self.action_dim = len(self.meta["action"]["example"])
        self.chunk_size = int(self.meta["data"]["chunk_size"])
        self.skip = int(self.meta["data"]["skip"])
        if (self.state_dim, self.action_dim) != (7, 7) or self.skip <= 0:
            raise ValueError("Expected 7D state/action and positive skip")

        DETRVAE.set_state_dim(self.state_dim)
        DETRVAE.set_action_dim(self.action_dim)
        if self.device.type == "cpu":
            # Upstream builder calls nn.Module.cuda() unconditionally.
            # Disable it only during construction; RMB/ACT files remain intact.
            with patch.object(torch.nn.Module, "cuda", lambda self, *a, **kw: self):
                self.policy = ACTPolicy(self.meta["policy"]["args"])
        else:
            self.policy = ACTPolicy(self.meta["policy"]["args"])
        self.policy.load_state_dict(
            torch.load(self.checkpoint, map_location="cpu", weights_only=True),
            strict=True,
        )
        self.policy.to(self.device).eval()

    def predict_chunk(self, state: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """Return chunk (chunk_size,7), in original MuJoCo control units.

        State: arm angles (6, rad) + observed gripper angle (1, rad).
        Actions: arm position targets (rad) + gripper actuator command (0..255).
        """
        state = np.asarray(state, dtype=np.float64)
        rgb = np.asarray(rgb)
        if state.shape != (7,) or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Bad observation shapes: {state.shape}, {rgb.shape}")
        if rgb.dtype != np.uint8 or not np.isfinite(state).all():
            raise ValueError("Expected uint8 RGB image and finite state")
        state_norm = self._normalize(state, self.meta["state"]).astype(np.float32)
        state_t = torch.from_numpy(state_norm).unsqueeze(0).to(self.device)
        rgb_t = (torch.from_numpy(np.ascontiguousarray(rgb))
                 .permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
                 .float().div(255.0).to(self.device))
        with torch.inference_mode():
            pred = self.policy(state_t, rgb_t)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        chunk = pred.detach().cpu().numpy()
        expected_shape = (1, self.chunk_size, 7)
        if chunk.shape != expected_shape:
            raise ValueError(f"ACT output {chunk.shape}, expected {expected_shape}")
        chunk = self._denormalize(chunk[0], self.meta["action"])
        if not np.isfinite(chunk).all():
            raise ValueError("ACT returned non-finite commands")
        return np.asarray(chunk, dtype=np.float64)
