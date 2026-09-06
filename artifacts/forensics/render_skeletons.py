"""Render 3 example skeletons under each decode (raw [:66] vs recover_from_ric).

Archived as-run for provenance; moved here from primary_source/ (see run_f1_f3_analysis.py
docstring for the same run-context note: `import paramUtil` needs that module on the path).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paramUtil import t2m_kinematic_chain

BONES = []
for chain in t2m_kinematic_chain:
    for a, b in zip(chain[:-1], chain[1:]):
        BONES.append((a, b))

with open("f1_example_skeletons.json") as f:
    examples = json.load(f)

OUT_DIR = "."

for idx, ex in enumerate(examples):
    raw = np.array(ex["raw"])   # (22,3)
    ric = np.array(ex["ric"])  # (22,3)
    caption = ex["caption"].split("#")[0]

    fig = plt.figure(figsize=(11, 5.5))
    for col, (joints, title) in enumerate([(raw, "decode (a): raw motion[:66]"), (ric, "decode (b): recover_from_ric")]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        for a, b in BONES:
            xs = [joints[a, 0], joints[b, 0]]
            ys = [joints[a, 2], joints[b, 2]]
            zs = [joints[a, 1], joints[b, 1]]
            ax.plot(xs, ys, zs, c="tab:blue" if col == 1 else "tab:red", linewidth=2)
        ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], c="black", s=15)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y (up)")
    fig.suptitle(f"Sample {idx}: \"{caption}\"", fontsize=10, wrap=True)
    fig.tight_layout()
    out_path = f"{OUT_DIR}/f1_skeleton_sample{idx}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print("saved", out_path)
