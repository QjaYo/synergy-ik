# Motion review

Inspect canonical motions and save inclusion, exclusion, unsure and quality-exclusion
decisions in a local SQLite catalog. Review does not modify the BVH files.

## Setup

Follow the [project installation](../../README.md#installation), including viewer
dependencies. Supply canonical BVHs and the BONES-SEED metadata CSV.
Mesh display additionally requires `usd-core`, original Uniform BVHs and their USD rig.

## Run

From the repository root:

```sh
python preprocess/motion_review/motion_review_data.py build --data-root data/canonical_bvh --metadata path/to/seed_metadata_v004.csv
python preprocess/motion_review/review_motions.py --no-mesh --metadata path/to/seed_metadata_v004.csv
```

The viewer saves each decision and resumes at the next unresolved group.
Quality exclusion applies to the current recording and its mirror; other decisions
apply to the behavior group. Current training excludes only quality-defective recordings.

```sh
python preprocess/motion_review/review_motions.py --unit clip --queue quality-exclude --no-mesh
python preprocess/motion_review/motion_review_data.py export
```

Decisions and exported lists are stored in `preprocess/motion_review/results/`.
Use `--uniform-root` and `--uniform-rig` to supply mesh paths. Other options are in `--help`.
