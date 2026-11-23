# SynergyIK

Learn state-dependent joint costs for full-body inverse kinematics. The current
model predicts a positive-definite W and follows four end effectors' position
and orientation targets. Training uses motion velocities and two ankle-contact
values. Full-SPD training remains experimental and can become ill-conditioned.

## Installation

Use Python 3.10. Run commands from the repository root in your Python environment.
Choose the PyTorch CPU/CUDA build for your machine before installing dependencies.

```sh
git submodule update --init -- third_party/fairmotion
python -m pip install -r requirements-training.txt
```

For viewers, also install `requirements-viewer.txt`. They require an OpenGL/GLUT
runtime; mesh rendering additionally needs `usd-core` and the original SOMA
Uniform assets. Optional contact-model setup is in [third_party](third_party/README.md).
Motion data, prepared datasets, checkpoints and local experiments are not included.

## Run

Convert original BVHs, then follow [data preparation](preprocess/training/README.md)
to create `data/preprocessed/`. The [review tool](preprocess/motion_review/README.md)
creates the motion catalog used during preparation.

```sh
python preprocess/canonicalize.py --input-dir path/to/soma_uniform/bvh --output-dir data/canonical_bvh
python source/train.py --dataset data/preprocessed --output experiment/runs/spd_01 --epochs 30 --device cuda --patience 0
python source/evaluate.py --dataset data/preprocessed --checkpoint experiment/runs/spd_01/best.pt --split validation
python source/run.py --bvh-file data/canonical_bvh/DATE/clip.bvh --checkpoint experiment/runs/spd_01/best.pt
```

Replace example paths with your files. The IK viewer compares reference,
Moore-Penrose and weighted motion with matching EE constraints.

```sh
# Resume training; monitor its curves at http://127.0.0.1:6006.
python source/train.py --resume experiment/runs/spd_01/last.pt
python -m tensorboard.main --logdir experiment/runs --host 127.0.0.1 --port 6006

# Inspect a motion or its ankle-contact labels.
python source/visualize_canonicalized_bvh.py --bvh-file data/canonical_bvh/DATE/clip.bvh --no-mesh
python source/visualize_contacts.py --bvh-file data/canonical_bvh/DATE/clip.bvh --no-mesh
```

Additional options are listed by each command's `--help`.

## Tests

With training and viewer dependencies installed:

```sh
python -m unittest discover -s test -v
```

Asset-dependent checks are skipped when their optional data/model files are absent.
