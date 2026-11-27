"""Independent GT-reset assessment of a checkpoint against Identity IK."""

import argparse
import json
from pathlib import Path

import torch

from learning import SolveConfig, default_device, evaluate, load_checkpoint
from features import FeatureSchema
from training_data import PreparedDataset, batch_loader
from velocity_solvers import METHODS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contact-mode", help="Defaults to the checkpoint's selected contact variant")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--solver-device", choices=("cpu", "cuda"), help="Defaults to the model device")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ik-solver", choices=METHODS, help="Override the checkpoint's solver for comparison")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, standardizer, saved = load_checkpoint(args.checkpoint, args.device)
    contact_mode = args.contact_mode if args.contact_mode is not None else saved["settings"].get("contact_mode")
    schema = FeatureSchema.from_dict(saved["contract"]["schema"])
    dataset = PreparedDataset(args.dataset, contact_mode, root_rotation=schema.root_rotation)
    if dataset.contract != saved["contract"] or dataset.statistics != saved["statistics"]:
        raise ValueError("Evaluation data must use the checkpoint's feature/contact/statistics reference")
    config = SolveConfig(**dict(saved["solve_config"], device=args.solver_device or args.device,
                               method=args.ik_solver or saved["solve_config"]["method"]))
    batches = batch_loader(dataset.batches(args.split, args.batch_size), workers=args.workers,
                           pin_memory="cuda" in (args.device, config.device), seed=42)
    result = evaluate(model, standardizer, batches, dataset, config)
    report = dict(checkpoint=str(args.checkpoint), dataset=str(args.dataset), split=args.split,
                  solve_config=config.to_dict(), results=result)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
