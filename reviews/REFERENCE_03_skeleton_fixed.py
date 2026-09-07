import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain

f = "dataset/HumanML3D/new_joint_vecs/sample004077.npy"
m = np.load(f)
frame = 77
joints = recover_from_ric(torch.from_numpy(m).float().unsqueeze(0), 22)[0].numpy()
wrong = m[:, :66].reshape(len(m), 22, 3)
print("correct:", joints.shape, " wrong:", wrong.shape)

def draw(ax, J, color, title, lims):
    for chain in t2m_kinematic_chain:
        ax.plot(J[chain, 0], J[chain, 1], "-o", color=color, ms=3.5, lw=2.2)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal"); ax.axis("off")

C, W = joints[frame], wrong[frame]
allpts = np.concatenate([C[:, [0, 1]], W[:, [0, 1]]])
pad = 0.15 * (allpts.max() - allpts.min())
lims = (allpts.min() - pad, allpts.max() + pad)

fig, axes = plt.subplots(1, 2, figsize=(9, 5))
draw(axes[0], W, "#c44e52", "wrong slice [:66]", lims)
draw(axes[1], C, "#4c72b0", "recover_from_ric", lims)
fig.suptitle("Front view (x-y, y is up), shared scale, equal aspect", fontsize=11)
plt.tight_layout(); plt.savefig("/tmp/skel_fixed.png", dpi=130)
print("saved /tmp/skel_fixed.png")
