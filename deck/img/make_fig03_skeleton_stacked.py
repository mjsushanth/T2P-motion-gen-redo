"""Deck-only figure: a column-friendly (near-square) skeleton comparison for slide 3.

notebooks/03_263d_representation_and_f1_bug.ipynb's own cell (id af057b27) produces
notebooks/03_skeleton_side_by_side.png at a 2-row x 4-column layout (aspect 855/1586 =
0.539) -- correct and left untouched as the notebook's own primary artifact, consistent
with its own cached cell output. That aspect is too wide for the deck's 35%-column layout
(renders under the 240px legibility floor). This script is a SEPARATE, deck-only
regeneration, not an edit to the notebook: same decode logic and same source motion
(reproduced here since jupyter nbconvert execution is blocked in this session), different
grid, saved under a different filename so the notebook's own file is never touched.

The peer's suggested fix (two panels, one above the other) would drop to a single frame and
lose the "incoherent over TIME, not just one bad pose" argument the notebook's own comment
cares about. Compromise implemented here: keep the strip-over-time argument with 2 frames
(first and last of the original 4, the widest temporal separation available) instead of
dropping to 1, arranged 2 rows (frames) x 2 columns (wrong|correct) -- comparison per frame
stays side by side (a natural pairing), frames stack vertically, and the overall shape lands
much closer to square.

Per the peer's own warning: measure the SAVED file's real pixel dimensions afterward, not the
figsize requested -- a tight bbox crop can change the aspect from what the figsize implies.

Run from deck/img/:
    cd deck/img && python3 make_fig03_skeleton_stacked.py
"""
import os
import sys

MDM_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "motion-diffusion-model")
sys.path.insert(0, os.path.abspath(MDM_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loaders.humanml.common.quaternion import qrot, qinv
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain


def decode_wrong_slice(motion):
    return motion[:, :66].reshape(motion.shape[0], 22, 3)


def decode_recover_from_ric(motion_t):
    def recover_root_rot_pos(data):
        rot_vel = data[..., 0]
        r_rot_ang = torch.zeros_like(rot_vel)
        r_rot_ang[..., 1:] = rot_vel[..., :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
        r_rot_quat = torch.zeros(data.shape[:-1] + (4,))
        r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        r_rot_quat[..., 2] = torch.sin(r_rot_ang)
        r_pos = torch.zeros(data.shape[:-1] + (3,))
        r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
        r_pos = qrot(qinv(r_rot_quat), r_pos)
        r_pos = torch.cumsum(r_pos, dim=-2)
        r_pos[..., 1] = data[..., 3]
        return r_rot_quat, r_pos

    joints_num = 22
    r_rot_quat, r_pos = recover_root_rot_pos(motion_t)
    positions = motion_t[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions.numpy()


NEW_JOINT_VECS_DIR = os.path.join(MDM_ROOT, "dataset", "HumanML3D", "new_joint_vecs")
all_files = sorted(os.listdir(NEW_JOINT_VECS_DIR))
rng = np.random.RandomState(0)
SAMPLE_FILES = rng.choice(all_files, size=40, replace=False)
example_file = SAMPLE_FILES[0]

motion = np.load(os.path.join(NEW_JOINT_VECS_DIR, example_file)).astype(np.float32)
motion_t = torch.from_numpy(motion)
joints_wrong_ex = decode_wrong_slice(motion)
joints_correct_ex = decode_recover_from_ric(motion_t)

n_frames_total = motion.shape[0]
# Widest available temporal separation from the original 4-point strip (0.25, 0.85), not the
# full 4, per the aspect-vs-content tradeoff explained above.
strip_frames = [int(n_frames_total * f) for f in (0.25, 0.85)]
strip_frames = [min(f, n_frames_total - 1) for f in strip_frames]


def draw_skeleton_2d(ax, joints_frame, color, title, lims):
    for chain in t2m_kinematic_chain:
        chain_pts = joints_frame[chain]
        ax.plot(chain_pts[:, 0], chain_pts[:, 1], "-o", color=color, ms=3, linewidth=2.2)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.axis("off")


fig, axes = plt.subplots(len(strip_frames), 2, figsize=(6.0, 6.9))
for row, frame_idx in enumerate(strip_frames):
    W = joints_wrong_ex[frame_idx]
    C = joints_correct_ex[frame_idx]
    allpts = np.concatenate([C[:, [0, 1]], W[:, [0, 1]]])
    pad = 0.15 * (allpts.max() - allpts.min())
    lims = (allpts.min() - pad, allpts.max() + pad)

    # Labels fixed 2026-09-16 per DECK_PLAN.md S2 (no "wrong"/"failed"/"broken" on a face --
    # this project's own binding editorial rule, which the first version of this figure
    # broke) and house-style rule 3 (no internal filenames or function names on a face --
    # sample004077.npy and recover_from_ric are provenance, not slide content). Both moved
    # to the presenter notes in deck/slides.md instead of being dropped.
    draw_skeleton_2d(axes[row, 0], W, "#c44e52", f"the original decode\nframe {frame_idx}", lims)
    draw_skeleton_2d(axes[row, 1], C, "#4c72b0", f"the dataset's own decode\nframe {frame_idx}", lims)

fig.suptitle(f"Two frames from one motion sequence -- frontal (x-y) view, shared scale per "
             f"frame, equal aspect", y=1.02, fontsize=11)
plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "03_skeleton_comparison_stacked.png")
plt.savefig(out_path, dpi=130, bbox_inches="tight")
plt.close(fig)

from PIL import Image
im = Image.open(out_path)
w, h = im.size
print(f"saved {out_path}")
print(f"actual pixel dimensions: {w}x{h}, aspect h/w = {h/w:.3f}")
