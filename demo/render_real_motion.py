"""Renders a REAL, stored HumanML3D motion (not a generated one) to an .mp4 -- used for the
nearest-neighbour retrieval pane, so the baseline is shown as an actual moving skeleton, not just
a caption. Reuses MDM's own recover_from_ric + plot_3d_motion (F1's corrected decode), the same
decode path validated throughout this project's forensics and every prior experiment.

The materialized new_joint_vecs/<id>.npy files are the raw (non-z-normalized) 263-d HumanML3D
vector -- recover_from_ric operates on this directly, no mean/std inverse-transform needed (that
step is only for undoing the DATASET LOADER's own training-time normalization, which never
touched these raw files).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _mpl_moviepy_compat  # noqa: E402,F401 -- applies the tostring_rgb shim on import

MDM_ROOT = os.path.join(os.path.dirname(__file__), "..", "third_party", "motion-diffusion-model")
MOTION_DIR = os.path.join(MDM_ROOT, "dataset", "HumanML3D", "new_joint_vecs")


def render_real_motion(motion_id: str, out_path: str) -> str:
    sys.path.insert(0, os.path.abspath(MDM_ROOT))
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    import data_loaders.humanml.utils.paramUtil as paramUtil
    from data_loaders.humanml.utils.plot_script import plot_3d_motion

    motion = np.load(os.path.join(MOTION_DIR, f"{motion_id}.npy"))  # [seq_len, 263]
    motion_t = torch.from_numpy(motion).float().unsqueeze(0)  # [1, seq_len, 263]
    joints = recover_from_ric(motion_t, 22)  # [1, seq_len, 22, 3]
    joints = joints.squeeze(0).numpy()

    skeleton = paramUtil.t2m_kinematic_chain
    # plot_3d_motion returns a moviepy VideoClip but does NOT write it to save_path itself
    # (a real quirk in MDM's own vendored code -- sample/generate.py only ever writes via its
    # own save_multiple_samples helper, never plot_3d_motion directly) -- write it explicitly.
    fps = 20
    clip = plot_3d_motion(out_path, skeleton, joints, dataset="humanml", title=motion_id, fps=fps)
    clip.duration = joints.shape[0] / fps
    clip.write_videofile(out_path, fps=fps, logger=None)
    clip.close()
    return out_path
