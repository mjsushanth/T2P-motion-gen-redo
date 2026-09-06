"""F1/F3 empirical forensics: HumanML3D 66-dim slice decode test + frame-0 neutrality test.

Archived as-run for provenance. This script and its outputs now live in
artifacts/forensics/ (moved out of primary_source/ so that directory holds only
unmodified upstream files), but `import quaternion` / `import paramUtil` still need
those modules on the path. To re-run: copy this file into primary_source/ (where
quaternion.py and paramUtil.py live) and run it from there, or symlink; do not add
sys.path manipulation to this file itself.
"""
import json
import re

import numpy as np
import torch

from paramUtil import t2m_kinematic_chain
from quaternion import qinv, qrot

N_SAMPLES = 250
MIN_FRAMES = 12
JOINTS_NUM = 22

# 21 bones as consecutive pairs within each of the 5 kinematic chains in t2m_kinematic_chain
BONES = []
for chain in t2m_kinematic_chain:
    for a, b in zip(chain[:-1], chain[1:]):
        BONES.append((a, b))
assert len(BONES) == 21, f"expected 21 bones, got {len(BONES)}"


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


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions


def bone_lengths(joint_positions):
    # joint_positions: (T, 22, 3) numpy
    lens = np.zeros((joint_positions.shape[0], len(BONES)))
    for i, (a, b) in enumerate(BONES):
        lens[:, i] = np.linalg.norm(joint_positions[:, a] - joint_positions[:, b], axis=-1)
    return lens  # (T, 21)


def main():
    from datasets import load_dataset

    print("Streaming train split from TeoGchx/HumanML3D ...")
    ds = load_dataset("TeoGchx/HumanML3D", split="train", streaming=True)

    samples = []
    for ex in ds:
        motion = np.asarray(ex["motion"], dtype=np.float64)
        if motion.shape[0] < MIN_FRAMES or motion.shape[1] != 263:
            continue
        samples.append({"caption": ex["caption"], "motion": motion})
        if len(samples) >= N_SAMPLES:
            break
    print(f"Collected {len(samples)} samples with >= {MIN_FRAMES} frames.")

    # ---------- F1: bone length CV under decode (a) raw[:66] vs decode (b) recover_from_ric ----------
    raw_bone_lens = []   # list of (T,21) arrays
    ric_bone_lens = []
    for s in samples:
        motion = s["motion"]
        T = motion.shape[0]

        # decode (a): raw motion[t][:66] reshaped (22,3)
        raw_joints = motion[:, :66].reshape(T, 22, 3)
        raw_bone_lens.append(bone_lengths(raw_joints))

        # decode (b): recover_from_ric
        data_t = torch.from_numpy(motion).float().unsqueeze(0)  # (1, T, 263)
        rec = recover_from_ric(data_t, JOINTS_NUM)  # (1, T, 22, 3)
        ric_joints = rec.squeeze(0).numpy()
        ric_bone_lens.append(bone_lengths(ric_joints))

    raw_all = np.concatenate(raw_bone_lens, axis=0)  # (sum_T, 21)
    ric_all = np.concatenate(ric_bone_lens, axis=0)

    def summarize(all_lens):
        mean = all_lens.mean(axis=0)
        std = all_lens.std(axis=0)
        cv = std / np.where(mean == 0, np.nan, mean)
        return mean, std, cv

    raw_mean, raw_std, raw_cv = summarize(raw_all)
    ric_mean, ric_std, ric_cv = summarize(ric_all)

    print("\n=== F1 bone length summary (mean / std / CV) across", raw_all.shape[0], "frames ===")
    print(f"{'bone':>10s} {'raw_mean':>10s} {'raw_std':>10s} {'raw_CV%':>10s} | {'ric_mean':>10s} {'ric_std':>10s} {'ric_CV%':>10s}")
    for i, (a, b) in enumerate(BONES):
        print(f"{a:>4d}-{b:<4d} {raw_mean[i]:10.4f} {raw_std[i]:10.4f} {100*raw_cv[i]:9.2f}% | "
              f"{ric_mean[i]:10.4f} {ric_std[i]:10.4f} {100*ric_cv[i]:9.2f}%")
    print(f"\nMEAN CV across bones: raw={100*np.nanmean(raw_cv):.2f}%  ric={100*np.nanmean(ric_cv):.2f}%")
    print(f"MEDIAN CV across bones: raw={100*np.nanmedian(raw_cv):.2f}%  ric={100*np.nanmedian(ric_cv):.2f}%")

    np.savez(
        "f1_bone_length_results.npz",
        bones=np.array(BONES),
        raw_mean=raw_mean, raw_std=raw_std, raw_cv=raw_cv,
        ric_mean=ric_mean, ric_std=ric_std, ric_cv=ric_cv,
    )

    # save 3 example skeletons under each decode (frame 0 of first 3 samples)
    examples = []
    for s in samples[:3]:
        motion = s["motion"]
        raw_joints0 = motion[0, :66].reshape(22, 3)
        data_t = torch.from_numpy(motion).float().unsqueeze(0)
        rec = recover_from_ric(data_t, JOINTS_NUM)
        ric_joints0 = rec.squeeze(0).numpy()[0]
        examples.append({"caption": s["caption"], "raw": raw_joints0.tolist(), "ric": ric_joints0.tolist()})
    with open("f1_example_skeletons.json", "w") as f:
        json.dump(examples, f, indent=2)

    # ---------- F3: frame-0 near-neutral-pose test ----------
    # ric_data (local, rotation-invariant joint positions relative to root) = motion[:, 4:67]
    # This is already root-centered and rotated to face +Z per-frame in the primary-source
    # construction, so comparing it across samples/frames removes translation & facing direction.
    frame0_vecs = []
    mid_vecs = []
    first_verbs = []
    for s in samples:
        motion = s["motion"]
        T = motion.shape[0]
        mid = T // 2
        frame0_vecs.append(motion[0, 4:67])
        mid_vecs.append(motion[mid, 4:67])
        cap = s["caption"]
        m = re.findall(r"([A-Za-z']+)/VERB", cap)
        first_verbs.append(m[0].lower() if m else None)

    frame0_vecs = np.stack(frame0_vecs)  # (N, 63)
    mid_vecs = np.stack(mid_vecs)

    def mean_pairwise_dist(X, max_pairs=20000):
        n = X.shape[0]
        diffs = X[:, None, :] - X[None, :, :]
        d = np.linalg.norm(diffs, axis=-1)
        iu = np.triu_indices(n, k=1)
        return d[iu].mean(), d[iu].std(), d[iu]

    d0_mean, d0_std, d0_all = mean_pairwise_dist(frame0_vecs)
    dmid_mean, dmid_std, dmid_all = mean_pairwise_dist(mid_vecs)

    n_distinct_verbs = len(set(v for v in first_verbs if v))
    print("\n=== F3 frame-0 neutrality test ===")
    print(f"N samples = {len(samples)}, distinct first-verbs in captions = {n_distinct_verbs}")
    print(f"Mean pairwise L2 distance between frame-0 local poses (ric_data, 63-d): {d0_mean:.4f} (std {d0_std:.4f})")
    print(f"Mean pairwise L2 distance between mid-frame local poses (ric_data, 63-d): {dmid_mean:.4f} (std {dmid_std:.4f})")
    print(f"Ratio (mid-frame dispersion / frame-0 dispersion): {dmid_mean / d0_mean:.3f}x")

    np.savez(
        "f3_neutrality_results.npz",
        frame0_vecs=frame0_vecs, mid_vecs=mid_vecs,
        d0_mean=d0_mean, d0_std=d0_std, dmid_mean=dmid_mean, dmid_std=dmid_std,
        n_distinct_verbs=n_distinct_verbs,
    )
    with open("f3_verbs.json", "w") as f:
        json.dump(first_verbs, f)
    with open("f3_captions_sample.json", "w") as f:
        json.dump([s["caption"] for s in samples[:20]], f, indent=2)

    print("\nDone. Results saved to f1_bone_length_results.npz, f1_example_skeletons.json, "
          "f3_neutrality_results.npz, f3_verbs.json, f3_captions_sample.json")


if __name__ == "__main__":
    main()
