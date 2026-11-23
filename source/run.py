"""Configure fixed or learned IK playback and launch the comparison viewer.

Usage: python source/run.py --bvh-file data/canonical_bvh/...bvh
"""

import argparse
from functools import partial
from pathlib import Path

from util import load_canonicalized_bvh
from solver import SolverMode, solve_ik_step
from playback import IKPlayback
from viewer import run_ik_viewer
from weights.fixed import create_fixed_weights
from weight_editor import DiagonalWeightEditor


# 1 = DIAGONAL, 2 = SPD. The comparison baseline always uses diagonal W = I.
MODE = SolverMode.DIAGONAL

WINDOW_TITLES = {
    SolverMode.DIAGONAL: "Diagonal Weighted Pseudoinverse IK",
    SolverMode.SPD: "SPD Weighted Pseudoinverse IK",
}


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the BVH path, length unit, and headless validation option."""
    parser = argparse.ArgumentParser(description="Reference / Identity / Fixed Weighted IK comparison")
    parser.add_argument("--bvh-file", type=Path, required=True)
    parser.add_argument("--unit", choices=("m", "cm"), default="m")
    parser.add_argument("--check", action="store_true", help="Check initial state without a window")
    parser.add_argument("--checkpoint", type=Path, help="Learned SPD checkpoint; omitted for fixed W")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--check-frames", type=int, default=0, help="Run this many physical frames without a window")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--uniform-root", type=Path, help="Original Uniform BVH directory for surface rendering")
    parser.add_argument("--uniform-rig", type=Path, help="Uniform USD rig for surface rendering")
    return parser.parse_args(argv)


def main(mode: SolverMode = MODE, argv: list[str] | None = None) -> None:
    mode = SolverMode(mode)
    args = parse_arguments(argv)
    import torch
    torch.set_num_threads(args.threads)
    motion = load_canonicalized_bvh(args.bvh_file, args.unit)
    W = create_fixed_weights(motion.skel)
    learned = None
    if args.checkpoint:
        from inference import LearnedWeights
        learned = LearnedWeights(args.checkpoint, motion.skel, args.device)
        mode = SolverMode.SPD
    solver = partial(solve_ik_step, mode=mode, W=learned or W,
                     pinv_rtol=learned.pinv_rtol if learned else 1e-12)
    playback = IKPlayback(motion, solver, panel_label="LEARNED SPD IK" if learned else f"FIXED {mode.name} IK",
                          prepare_solver=learned.begin_frame if learned else None,
                          reset_solver=learned.reset_history if learned else None,
                          fixed_ee_names=learned.layout.ee_names if learned else None,
                          constrain_orientation=learned.schema.angular_task if learned else False,
                          pinv_rtol=learned.pinv_rtol if learned else 1e-12)
    print(f"Frames: {motion.num_frames()} | FPS: {motion.fps} | dt: {playback.dt}", flush=True)
    print(f"Pose nodes: {len(motion.skel.joints)} | IK DoFs: {W.shape[0]}", flush=True)
    print(f"EE: {', '.join(playback.ee_names)} | starts paused", flush=True)
    for _ in range(min(args.check_frames, motion.num_frames() - 1)):
        before = playback.frame_index
        playback.step_frame()
        if playback.frame_index == before:
            raise RuntimeError(playback.status)
    if not args.check and not args.check_frames:
        editor = DiagonalWeightEditor(W, motion.poses[0]) if not learned and mode == SolverMode.DIAGONAL else None
        run_ik_viewer(playback, title=WINDOW_TITLES[mode], weight_editor=editor, bvh_path=args.bvh_file,
                      uniform_root=args.uniform_root, uniform_rig=args.uniform_rig,
                      surface_height_threshold=learned.saved["contract"]["contact"].get("height_threshold", .040)
                      if learned else .040)


if __name__ == "__main__":
    main()
