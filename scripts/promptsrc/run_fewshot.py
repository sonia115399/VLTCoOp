#!/usr/bin/env python3
"""Run/resume the PromptSRC few-shot experiments and aggregate results."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["allamanda_cathartica", "corn", "datepalm", "duranta_erecta_gold",
            "murraya_exotica", "ruellia_simplex", "tea"]
MODELS = ["PromptSRC"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, fields, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def make_jobs(args):
    jobs = []
    tag = "vit_b16" if args.backbone == "ViT-B/16" else "vit_b32"
    # Include training sources in the resume identity so changed implementations
    # never silently reuse older results.
    source_hash = hashlib.sha256()
    for filename in ("trainers/PromptSRC/promptsrc_clip.py",
                     "trainers/promptsrc_templates.py", "clip/model.py"):
        source_hash.update((ROOT / filename).read_bytes())
    for model, dataset, shots, seed in itertools.product(args.models, args.datasets, args.shots, args.seeds):
        output = args.output / dataset / ("shots_{}".format(shots)) / (model + "_CLIP") / tag / ("seed{}".format(seed))
        config = ROOT / "configs" / "trainers" / model / "few_shot.yaml"
        command = [sys.executable, "-u", str(ROOT / "train.py"), "--root", str(args.data),
                   "--seed", str(seed), "--trainer", model, "--backbone", args.backbone,
                   "--dataset-config-file", str(ROOT / "configs" / "datasets" / (dataset + ".yaml")),
                   "--config-file", str(config), "--output-dir", str(output),
                   "DATASET.NUM_SHOTS", str(shots), "DATASET.SUBSAMPLE_CLASSES", "all"]
        if args.epochs is not None:
            command += ["OPTIM.MAX_EPOCH", str(args.epochs)]
        if args.workers is not None:
            command += ["DATALOADER.NUM_WORKERS", str(args.workers)]
        fingerprint = hashlib.sha256((source_hash.hexdigest() + config.read_text() + json.dumps(command)).encode()).hexdigest()
        jobs.append(dict(model=model, dataset=dataset, shots=shots, seed=seed, backbone=args.backbone,
                         output=str(output), command=command, fingerprint=fingerprint))
    return jobs


def completed(job):
    directory = Path(job["output"])
    try:
        result = json.loads((directory / "results.json").read_text())
        identity = json.loads((directory / "run.json").read_text())
        return (result["status"] == "complete" and identity["fingerprint"] == job["fingerprint"]
                and all(result[key] == job[key] for key in ("model", "dataset", "shots", "seed", "backbone"))
                and (directory / "prompt_learner" / "model.pth.tar").is_file()
                and all(math.isfinite(result[key]) for key in ("accuracy", "macro_f1", "error_rate")))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def summarize(jobs, directory):
    rows = []
    for job in jobs:
        row = {key: job[key] for key in ("model", "dataset", "shots", "seed", "backbone")}
        row.update(status="pending", accuracy="", macro_f1="", error_rate="", epochs="", elapsed_seconds="", output=job["output"])
        if completed(job):
            result = json.loads((Path(job["output"]) / "results.json").read_text())
            row.update({key: result[key] for key in ("status", "accuracy", "macro_f1", "error_rate", "epochs", "elapsed_seconds")})
        elif (Path(job["output"]) / "failure.json").exists():
            row["status"] = "failed"
        rows.append(row)
    fields = list(rows[0])
    write_csv(directory / "runs.csv", fields, rows)
    for model in dict.fromkeys(job["model"] for job in jobs):
        model_rows = [row for row in rows if row["model"] == model]
        write_csv(directory / (model + "_runs.csv"), fields, model_rows)
        groups = {}
        for row in model_rows:
            groups.setdefault((row["dataset"], row["shots"], row["backbone"]), []).append(row)
        means = []
        for (dataset, shots, backbone), group in groups.items():
            done = [row for row in group if row["status"] == "complete"]
            aggregate = dict(model=model, dataset=dataset, shots=shots, backbone=backbone,
                             seeds=" ".join(str(row["seed"]) for row in group),
                             completed_seeds=len(done), expected_seeds=len(group),
                             status="complete" if len(done) == len(group) else "incomplete")
            for metric in ("accuracy", "macro_f1"):
                values = [row[metric] for row in done]
                # Only publish mean/std when ALL requested seeds have completed.
                aggregate[metric + "_mean"] = statistics.mean(values) if len(done) == len(group) else ""
                aggregate[metric + "_std"] = statistics.pstdev(values) if len(done) == len(group) else ""
            means.append(aggregate)
        write_csv(directory / (model + "_summary.csv"), list(means[0]), means)
    counts = {state: sum(row["status"] == state for row in rows) for state in ("complete", "pending", "failed")}
    write_json(directory / "status.json", dict(total=len(jobs), **counts, updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--shots", nargs="+", type=int, choices=[1, 2, 4, 8, 16], default=[1, 2, 4, 8, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--backbone", choices=["ViT-B/16", "ViT-B/32"], default="ViT-B/16")
    parser.add_argument("--epochs", type=int, help="Smoke tests/custom runs only; use a separate --output")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent training processes (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="List commands without creating results")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    if any(seed < 0 for seed in args.seeds) or (args.epochs is not None and args.epochs < 1):
        parser.error("Seeds must be nonnegative and epochs positive")
    if args.workers is not None and args.workers < 0:
        parser.error("Workers must be nonnegative")
    if args.jobs < 1:
        parser.error("Jobs must be positive")
    for field in ("models", "datasets", "shots", "seeds"):
        if len(set(getattr(args, field))) != len(getattr(args, field)):
            parser.error("Duplicate values in --{}".format(field))
    if args.epochs is not None and args.output == ROOT / "output":
        parser.error("Custom epoch counts require a separate --output to keep smoke tests out of full results")
    jobs = make_jobs(args)
    if args.dry_run:
        print("{} experiments".format(len(jobs)))
        for job in jobs:
            print(shlex.join(job["command"]))
        return 0
    tag = "vit_b16" if args.backbone == "ViT-B/16" else "vit_b32"
    summary_dir = args.output / "_summary" / ("promptsrc_" + tag)
    summary_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        print(summarize(jobs, summary_dir))
        return 0
    # Do not allow competing workers to train into the same results tree.
    with (summary_dir / "batch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("An experiment batch is already running in {}".format(summary_dir))
        write_json(summary_dir / "manifest.json", jobs)
        write_json(summary_dir / "process.json", dict(pid=os.getpid(), python=sys.executable, jobs=args.jobs,
                                                       started_at=time.strftime("%Y-%m-%d %H:%M:%S")))
        environment = os.environ.copy()
        environment.setdefault("OMP_NUM_THREADS", "4")
        environment.setdefault("MKL_NUM_THREADS", "4")
        environment.setdefault("PYTHONHASHSEED", "0")
        failures = []
        summarize(jobs, summary_dir)
        active = {}
        state_lock = threading.Lock()

        def run_job(index, job):
            if completed(job):
                print("[{}/{}] skip completed {} {} k={} seed={}".format(index, len(jobs), job["model"], job["dataset"], job["shots"], job["seed"]), flush=True)
                return None
            output = Path(job["output"])
            # Retain partial or differently configured runs for inspection.
            if output.exists() and any(output.iterdir()):
                archive = output.with_name(output.name + ".previous_" + str(time.time_ns()))
                output.rename(archive)
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "run.json", job)
            with state_lock:
                active[index] = dict(index=index, **job)
                write_json(summary_dir / "current.json", dict(total=len(jobs), active=list(active.values())))
            print("[{}/{}] {} {} k={} seed={}".format(index, len(jobs), job["model"], job["dataset"], job["shots"], job["seed"]), flush=True)
            with (output / "console.log").open("w") as log:
                process = subprocess.run(job["command"], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode != 0 or not completed(job):
                write_json(output / "failure.json", dict(returncode=process.returncode,
                           message="Training or final evaluation did not complete; inspect console.log"))
                print("FAILED: {}".format(output / "console.log"), flush=True)
                failure = job["output"]
            else:
                result = json.loads((output / "results.json").read_text())
                print("[{}/{}] accuracy={:.2f}% macro_f1={:.2f}%".format(index, len(jobs), result["accuracy"], result["macro_f1"]), flush=True)
                failure = None
            with state_lock:
                active.pop(index)
                write_json(summary_dir / "current.json", dict(total=len(jobs), active=list(active.values())))
            return failure

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_job, index, job): job for index, job in enumerate(jobs, 1)}
            for future in as_completed(futures):
                try:
                    failure = future.result()
                except Exception as error:
                    job = futures[future]
                    failure = job["output"]
                    write_json(Path(job["output"]) / "failure.json", dict(message=repr(error)))
                    print("FAILED: {}: {}".format(failure, error), flush=True)
                if failure:
                    failures.append(failure)
                counts = summarize(jobs, summary_dir)
                print(counts, flush=True)
        write_json(summary_dir / "current.json", dict(status="finished", failures=failures))
        print("Results: {}".format(summary_dir), flush=True)
        return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

