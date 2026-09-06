# Patches to vendored upstream files

This directory holds files fetched verbatim from `github.com/EricGuo5513/HumanML3D`
(default branch, fetched 2026-09-05) for primary-source verification. One file was
patched for compatibility with a newer numpy on this host; everything else is pristine.

## quaternion.py

**Line 13**, one-line compatibility fix, no logic change:

```diff
- _FLOAT_EPS = np.finfo(np.float).eps
+ _FLOAT_EPS = np.finfo(float).eps
```

`np.float` was a deprecated alias for the builtin `float`, removed in numpy >= 1.24. The
repo predates that removal (numpy pinned much older in the original Windows/CUDA env).
This host runs numpy 1.26.4 (via the `mjs_mlcvdl_unified_m5` conda env), where the
original line raises `AttributeError: module 'numpy' has no attribute 'float'` at import
time. The fix is numpy's own documented migration (`float` and `np.float` were always
the same builtin), does not change `_FLOAT_EPS`'s value, and does not touch any function
used in the F1 empirical decode test (`qrot`, `qinv`, `recover_root_rot_pos`,
`recover_from_ric`), none of which reference `_FLOAT_EPS`.

No other files in this directory have been modified from what was fetched.
