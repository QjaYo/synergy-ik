"""Independent GT-reset assessment of a checkpoint against Identity IK."""

import argparse
import json
from pathlib import Path

import torch

from learning import SolveConfig, evaluate, load_checkpoint
from training_data import PreparedDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contact-mode", help="Defaults to the checkpoint's selected contact variant")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--solver-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, standardizer, saved = load_checkpoint(args.checkpoint, args.device)
    contact_mode = args.contact_mode if args.contact_mode is not None else saved["settings"].get("contact_mode")
    dataset = PreparedDataset(args.dataset, contact_mode)
    if dataset.contract != saved["contract"] or dataset.statistics != saved["statistics"]:
        raise ValueError("Evaluation data must use the checkpoint's feature/contact/statistics reference")
    config = SolveConfig(**dict(saved["solve_config"], device=args.solver_device))
    result = evaluate(model, standardizer, dataset.batches(args.split, args.batch_size), dataset, config)
    report = dict(checkpoint=str(args.checkpoint), dataset=str(args.dataset), split=args.split, results=result)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
