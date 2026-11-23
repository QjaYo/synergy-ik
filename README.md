# SynergyIK

<img src="https://github.com/user-attachments/assets/261ca837-f33c-44bb-ab37-02ddf65bad6a" alt="SynergyIK demo (0.5x)">

Learn state-dependent joint costs from motion data to select human-like movement
in redundant full-body inverse kinematics.

**Redundant inverse kinematics.** The Jacobian relates joint velocities to
end-effector (EE) velocities:

$$
J(q)\dot q = \dot x.
$$

Here, $\dot q$ contains root linear and joint angular velocities, while $\dot x$
contains EE linear and angular velocities. With more joint degrees of freedom
than task constraints, this underdetermined system has multiple solutions for a
feasible task.

**Moore-Penrose baseline.** The pseudoinverse selects the minimum-norm solution:

$$
\begin{aligned}
\dot q_{\mathrm{MP}} &= J(q)^{+}\dot x
= \underset{\dot q}{\operatorname{argmin}}\; \frac{1}{2}\dot q^{\mathsf T}\dot q, \\
&\text{subject to}\quad J(q)\dot q = \dot x.
\end{aligned}
$$

More generally, it gives the minimum-norm least-squares solution when the task
cannot be satisfied exactly. Minimizing velocity norm alone does not encode
human joint coordination or anatomical limits, so the resulting motion can
include unnatural joint bending.

**Learned joint costs.** We learn a state-dependent cost matrix $W$ from motion
data, conditioned on the current pose, desired EE velocities and foot contact.
The aim is to resolve redundancy by selecting joint motion that better matches
human movement:

$$
\begin{aligned}
\dot q_{\mathrm{pred}} &= \underset{\dot q}{\operatorname{argmin}}\;
\frac{1}{2}\dot q^{\mathsf T}W\dot q, \\
&\text{subject to}\quad J(q)\dot q = \dot x.
\end{aligned}
$$

We require $W$ to be symmetric positive definite, giving a strictly positive
cost to every nonzero joint velocity. A Cholesky parameterization enforces this
by predicting a lower-triangular $L$ with positive diagonal entries:

$$
W = LL^{\mathsf T} \succ 0, \qquad L_{ii} > 0.
$$

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

**Status:** Research is ongoing; the current model uses four EE position/orientation
targets and two ankle-contact inputs, and full-SPD training can still become
ill-conditioned.
