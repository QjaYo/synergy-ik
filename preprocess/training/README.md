# Training-data preparation

Convert canonical BVHs to one NPZ per clip, fixed 80/10/10 recording-group splits,
and training-only normalization statistics. Original/mirror pairs stay together.
The default includes all catalog motions except `quality_exclude` recordings.

## Setup

Use the [project installation](../../README.md#installation) with
`requirements-training.txt`. Create the catalog with the
[review tool](../motion_review/README.md). The default requires no mesh or contact model.

## Run

From the repository root:

```sh
python preprocess/training/prepare.py --data-root data/canonical_bvh --review-db preprocess/motion_review/results/review.sqlite3 --output data/preprocessed --mirrors include --workers 2
```

The output contains `clips/`, `manifest.json`, `statistics.json` and `summary.json`.
Use a new output directory when changing a dataset's settings.

The default has 201 inputs, 24 EE task velocities and 72 target joint velocities.
Contacts are two ankle labels: height < 0.120 m AND speed < 0.4 m/s.
For each current frame t, contact uses t-1 to t; supervision uses t to t+1.
Task vectors use current pelvis axes; joint angular velocities remain joint-local.
History and joint-position arrays are cached for later comparisons but are not baseline inputs.

`--bvh-list` selects clips, `--manifest-only` checks selection, and `--task position`
removes angular constraints. See `--help` for other preparation options.
