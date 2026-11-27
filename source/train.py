"""Train state-conditioned SPD weights from a prepared dataset."""

import argparse
import hashlib
import json
import random
from pathlib import Path
import time

import numpy as np
import torch

from checkpoint import CHECKPOINT_VERSION, read_checkpoint, write_checkpoint, random_state, restore_random_state
from features import Standardizer, FeatureSchema, ROOT_ROTATIONS, DEFAULT_ROOT_ROTATION
from learning import (SolveConfig, VelocityMetrics, check_finite_gradients, default_device, evaluate,
                      predict_batch, scalar_values, velocity_errors)
from training_data import PreparedDataset, batch_loader
from training_log import TrainingLog
from weights.learned import ModelConfig, WPrediction
from velocity_solvers import DEFAULT_METHOD, METHODS


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = PreparedDataset(args.dataset, args.contact_mode, root_rotation=args.root_rotation)
    resume = read_checkpoint(args.resume) if args.resume else None
    same_run = resume is not None and args.output.resolve() == args.resume.parent.resolve()
    if not same_run and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Choose an empty run directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    config = ModelConfig(dataset.schema.width, dataset.layout.dofs, args.width, args.depth)
    solve_config = SolveConfig(args.solver_device, args.solver_dtype, args.pinv_rtol, args.ik_solver)
    model = WPrediction(config).to(args.device)
    standardizer = Standardizer(dataset.statistics["mean"], dataset.statistics["scale"]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    manifest_hash = hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest()
    best, reference, stale, step = float("inf"), float("inf"), 0, 0
    start_epoch = next_batch = 0
    elapsed = 0.0
    metrics = VelocityMetrics(dataset.schema.angular_task)
    if args.initialize_from:
        initial = read_checkpoint(args.initialize_from)
        if (initial["contract"] != dataset.contract or initial["statistics"] != dataset.statistics
                or initial["model_config"] != config.to_dict()):
            raise ValueError("Fine-tuning requires matching features, normalization and model dimensions")
        model.load_state_dict(initial["model_state"])
        print(f"Initialized model from {args.initialize_from}; new optimizer and run", flush=True)
    if resume:
        if "training_state" not in resume:
            raise ValueError("This older checkpoint has no batch/RNG/early-stopping state for exact resume")
        if (resume["manifest_sha256"] != manifest_hash or resume["contract"] != dataset.contract
                or resume["statistics"] != dataset.statistics or resume["model_config"] != config.to_dict()):
            raise ValueError("Resume requires the original dataset, statistics and model configuration")
        for name in ("lr", "batch_size", "seed", "workers", "patience", "solver_dtype", "pinv_rtol", "ik_solver"):
            if getattr(args, name) != resume["settings"][name]:
                raise ValueError(f"Exact resume cannot change {name}; start a separate experiment instead")
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        state = resume["training_state"]
        start_epoch, next_batch, step = state["next_epoch"], state["next_batch"], resume["step"]
        best, reference, stale = resume["best_validation"], state["reference"], state["stale"]
        elapsed = state["elapsed_seconds"]
        metrics.count, metrics.sums = state["metric_count"], state["metric_sums"]
        restore_random_state(state["random"])
        for filename in ("metrics.jsonl", "steps.jsonl"):
            path = args.output / filename
            if path.exists():
                reports = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
                if reports and reports[-1]["step"] > step:
                    raise ValueError("Output logs extend past this checkpoint; choose a new --output directory")
        print(f"Resuming step={step}, epoch={start_epoch+1}, next_batch={next_batch}", flush=True)
        if args.patience and stale >= args.patience:
            print("This run already reached its early-stopping criterion", flush=True)
            return
        if args.max_steps is not None and step >= args.max_steps:
            print("This checkpoint already reached the requested step budget", flush=True)
            return
    settings = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output / ("resume_config.json" if resume else "config.json")).write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    start = time.perf_counter()

    def save(
        next_epoch: int,
        batch_index: int,
        current_metrics: VelocityMetrics,
        completed_epoch: int,
    ) -> dict:
        saved = dict(version=CHECKPOINT_VERSION, model_config=config.to_dict(), model_state=model.state_dict(),
                     statistics=dataset.statistics, contract=dataset.contract, solve_config=solve_config.to_dict(),
                     settings=settings, manifest_sha256=manifest_hash, optimizer_state=optimizer.state_dict(),
                     epoch=completed_epoch, step=step, best_validation=best,
                     training_state=dict(next_epoch=next_epoch, next_batch=batch_index, reference=reference,
                         stale=stale, metric_count=current_metrics.count, metric_sums=scalar_values(current_metrics.sums),
                         elapsed_seconds=elapsed + time.perf_counter()-start, random=random_state()))
        write_checkpoint(args.output / "last.pt", saved)
        return saved

    metadata = dict(settings=settings, model_config=config.to_dict(), solve_config=solve_config.to_dict(),
                    statistics=dataset.statistics, contract=dataset.contract, manifest_sha256=manifest_hash)
    with TrainingLog(args.output, metadata, diagnostics_every=args.diagnostics_every,
                     purge_step=step + 1 if resume else None) as log:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            if epoch != start_epoch or not next_batch:
                metrics = VelocityMetrics(dataset.schema.angular_task)
            batches = batch_loader(dataset.batches("train", args.batch_size, shuffle=True, epoch=epoch, seed=args.seed),
                                   workers=args.workers, pin_memory="cuda" in (args.device, solve_config.device),
                                   seed=args.seed + epoch)
            for batch_index, batch in enumerate(batches):
                if batch_index < next_batch:
                    continue
                with log.batch(model, optimizer, batch, step=step + 1, epoch=epoch + 1, batch_index=batch_index + 1):
                    optimizer.zero_grad(set_to_none=True)
                    target = solve_config.move(batch["target"])
                    prediction, J, x_dot = predict_batch(model, standardizer, batch, dataset.layout, solve_config,
                                                        observe_solve=log.observe_solve)
                    errors = velocity_errors(prediction, target, dataset.statistics)
                    loss = errors["loss"]
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite velocity loss")
                    loss.backward()
                    check_finite_gradients(model)
                    optimizer.step()
                    values = metrics.update(prediction, target, J, x_dot, dataset.statistics, errors)
                    step += 1
                    if step == 1 or step % args.log_every == 0:
                        report = metrics.report(len(prediction), values)
                        log.training_step(model, optimizer, report, epoch + 1)
                if step % args.log_every == 0:
                    progress = dict(epoch=epoch+1, batch=batch_index+1, batches=len(batches), step=step,
                                    loss=report["loss"], elapsed_seconds=elapsed + time.perf_counter()-start)
                    with (args.output / "steps.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(progress) + "\n")
                    print(f"epoch={epoch+1} batch={batch_index+1}/{len(batches)} step={step} loss={report['loss']:.6g}", flush=True)
                if step % args.checkpoint_every == 0:
                    save(epoch, batch_index + 1, metrics, epoch)
                if args.max_steps is not None and step >= args.max_steps:
                    save(epoch, batch_index + 1, metrics, epoch)
                    print("Step budget reached; saved last.pt for continuation", flush=True)
                    return
            next_batch = 0
            validation_batches = batch_loader(dataset.batches("validation", args.batch_size), workers=args.workers,
                                              pin_memory="cuda" in (args.device, solve_config.device), seed=args.seed)
            validation = evaluate(model, standardizer, validation_batches, dataset, solve_config)
            value = validation["learned"]["loss"]
            if not np.isfinite(value):
                raise FloatingPointError("Non-finite validation loss")
            if value <= reference * (1 - 0.001):
                reference, stale = value, 0
            else:
                stale += 1
            improved = value < best
            best = min(best, value)
            report = dict(epoch=epoch+1, step=step, elapsed_seconds=elapsed+time.perf_counter()-start,
                          train=metrics.result(), validation=validation, best_validation=best)
            log.epoch(step, report["train"], validation)
            with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report) + "\n")
            saved = save(epoch + 1, 0, VelocityMetrics(dataset.schema.angular_task), epoch + 1)
            if improved:
                write_checkpoint(args.output / "best.pt", saved)
            if (epoch + 1) % args.save_every_epochs == 0:
                write_checkpoint(args.output / f"epoch_{epoch+1:04d}.pt", saved)
            print(json.dumps(report), flush=True)
            if (args.patience and stale >= args.patience) or (args.max_steps is not None and step >= args.max_steps):
                break


def parser(*, resuming: bool = False) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--resume", type=Path, help="Continue from a checkpoint including training state")
    result.add_argument("--initialize-from", type=Path, help="Start a new experiment from model weights only")
    result.add_argument("--dataset", type=Path, required=not resuming)
    result.add_argument("--contact-mode", help="Named contact variant in a shared NPZ dataset")
    result.add_argument("--root-rotation", choices=ROOT_ROTATIONS, default=DEFAULT_ROOT_ROTATION,
                        help="Root input encoding; existing NPZ pose matrices are reused (default: %(default)s)")
    result.add_argument("--output", type=Path, required=not resuming)
    result.add_argument("--epochs", type=int, default=30, help="Total epoch budget, including completed epochs")
    result.add_argument("--max-steps", type=int)
    result.add_argument("--device", default=default_device(), choices=("cpu", "cuda"))
    result.add_argument("--solver-device", choices=("cpu", "cuda"), help="Defaults to the model device")
    result.add_argument("--solver-dtype", default="float64", choices=("float32", "float64"))
    result.add_argument("--ik-solver", choices=METHODS, default=DEFAULT_METHOD,
                        help="Weighted velocity solve (default: %(default)s)")
    result.add_argument("--pinv-rtol", type=float, default=1e-12)
    result.add_argument("--width", type=int, default=ModelConfig.hidden_width)
    result.add_argument("--depth", type=int, default=ModelConfig.depth,
                        help="Total Linear layers, including the output head (default: %(default)s)")
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--patience", type=int, default=0, help="Early-stopping epochs; 0 disables early stopping")
    result.add_argument("--save-every-epochs", type=int, default=10, help="Keep an epoch snapshot at this interval")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--workers", type=int, default=2)
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--log-every", type=int, default=20)
    result.add_argument("--diagnostics-every", type=int, default=100, help="Record L condition and solve operands every N steps")
    result.add_argument("--checkpoint-every", type=int, default=1000, help="Save continuation state every N optimizer steps")
    return result


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--resume", type=Path)
    initial, _ = bootstrap.parse_known_args(argv)
    arguments = parser(resuming=initial.resume is not None)
    if initial.resume:
        saved = read_checkpoint(initial.resume)
        defaults = dict(saved["settings"], output=initial.resume.parent, max_steps=None, initialize_from=None)
        defaults["dataset"] = Path(defaults["dataset"])
        defaults["root_rotation"] = FeatureSchema.from_dict(saved["contract"]["schema"]).root_rotation
        arguments.set_defaults(**defaults)
    options = arguments.parse_args(argv)
    options.solver_device = options.solver_device or options.device
    if options.workers < 0:
        arguments.error("--workers must be nonnegative")
    if options.resume and options.initialize_from:
        arguments.error("--resume and --initialize-from select different training modes")
    if min(options.epochs, options.batch_size, options.threads, options.log_every,
           options.diagnostics_every, options.checkpoint_every, options.save_every_epochs) < 1 or options.patience < 0:
        raise ValueError("Epochs, batch size, threads and intervals must be positive; patience must be nonnegative")
    if options.max_steps is not None and options.max_steps < 1:
        raise ValueError("max-steps must be positive")
    return options


if __name__ == "__main__":
    train(parse_arguments())
