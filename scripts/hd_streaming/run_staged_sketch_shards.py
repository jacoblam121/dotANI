#!/usr/bin/env python3
"""Stage dotANI sketch shards to scratch and run manifest-mode sketching."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path


RUN_COLUMNS = [
    "shard_id",
    "manifest_path",
    "sketch_path",
    "ull_path",
    "start_index",
    "record_count",
    "status",
    "wall_secs",
    "sketch_checksum",
    "ull_checksum",
    "output_bytes",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(rows, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_by_shard(manifest_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    shards: dict[str, list[dict[str, str]]] = {}
    for row in manifest_rows:
        shards.setdefault(row["shard_id"], []).append(row)
    return shards


def parse_nonnegative_int(value: str, field: str, shard_id: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} for {shard_id} is not an integer: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{field} for {shard_id} must be nonnegative: {value!r}")
    return parsed


def validate_plan_row(plan_row: dict[str, str], shard_rows: list[dict[str, str]]) -> None:
    shard_id = plan_row["shard_id"]
    if not shard_rows:
        raise ValueError(f"shard_plan references {shard_id}, but manifest has no rows for it")

    expected_start_index = parse_nonnegative_int(shard_rows[0]["global_index"], "global_index", shard_id)
    expected_record_count = len(shard_rows)
    expected_input_bytes = sum(
        parse_nonnegative_int(row["size_bytes"], "size_bytes", shard_id) for row in shard_rows
    )

    start_index = parse_nonnegative_int(plan_row["start_index"], "start_index", shard_id)
    record_count = parse_nonnegative_int(plan_row["record_count"], "record_count", shard_id)
    input_bytes = parse_nonnegative_int(plan_row["input_bytes"], "input_bytes", shard_id)

    mismatches = []
    if start_index != expected_start_index:
        mismatches.append(f"start_index={start_index}, expected {expected_start_index}")
    if record_count != expected_record_count:
        mismatches.append(f"record_count={record_count}, expected {expected_record_count}")
    if input_bytes != expected_input_bytes:
        mismatches.append(f"input_bytes={input_bytes}, expected {expected_input_bytes}")

    if mismatches:
        raise ValueError(f"shard_plan row for {shard_id} does not match manifest: {', '.join(mismatches)}")


def load_ok_runs(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {
        row["shard_id"]: row
        for row in read_tsv(path)
        if row.get("status") == "ok" and row.get("shard_id")
    }


def copy_shard_inputs(shard_rows: list[dict[str, str]], scratch_dir: Path) -> None:
    for row in shard_rows:
        rel_path = Path(row["rel_path"])
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"unsafe rel_path in manifest: {row['rel_path']!r}")
        dest = scratch_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["read_path"], dest)


def write_staged_manifest(path: Path, shard_rows: list[dict[str, str]], scratch_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        handle.write("# version=1\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["global_index", "shard_id", "read_path", "file_id", "rel_path", "size_bytes"])
        for row in shard_rows:
            staged_path = scratch_dir / row["rel_path"]
            writer.writerow(
                [
                    row["global_index"],
                    row["shard_id"],
                    str(staged_path.resolve()),
                    row["file_id"],
                    row["rel_path"],
                    row["size_bytes"],
                ]
            )


def run_dotani(args: argparse.Namespace, staged_manifest: Path, shard_out_dir: Path, shard_id: str) -> int:
    sketch_path = shard_out_dir / f"{shard_id}.sketch"
    metrics_prefix = shard_out_dir / f"{shard_id}.metrics"
    cmd = [
        str(args.dotani),
        "sketch",
        "--manifest",
        str(staged_manifest),
        "--device",
        "cuda",
        "-T",
        str(args.threads),
        "--cuda-dedup",
        args.cuda_dedup,
        "-o",
        str(sketch_path),
        "--metrics-out",
        str(metrics_prefix),
    ]
    if args.max_readers is not None:
        cmd.extend(["--max-readers", str(args.max_readers)])

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    with (shard_out_dir / "dotani.command.txt").open("w") as handle:
        handle.write(" ".join(cmd))
        handle.write("\n")

    return subprocess.run(cmd, env=env, cwd=shard_out_dir).returncode


def shard_status(shard_out_dir: Path, shard_id: str, expected_records: int) -> tuple[str, str, str, int]:
    sketch_path = shard_out_dir / f"{shard_id}.sketch"
    ull_path = shard_out_dir / f"{shard_id}.sketch.ull"
    metrics_files_path = shard_out_dir / f"{shard_id}.metrics.files.tsv"
    metrics_summary_path = shard_out_dir / f"{shard_id}.metrics.summary.tsv"

    if (
        not sketch_path.exists()
        or not ull_path.exists()
        or not metrics_files_path.exists()
        or not metrics_summary_path.exists()
    ):
        output_bytes = sum(p.stat().st_size for p in [sketch_path, ull_path] if p.exists())
        return "failed", "", "", output_bytes

    with metrics_files_path.open() as handle:
        metric_rows = max(sum(1 for _ in handle) - 1, 0)
    if metric_rows != expected_records:
        output_bytes = sketch_path.stat().st_size + ull_path.stat().st_size
        return "partial", sha256_file(sketch_path), sha256_file(ull_path), output_bytes

    output_bytes = sketch_path.stat().st_size + ull_path.stat().st_size
    return "ok", sha256_file(sketch_path), sha256_file(ull_path), output_bytes


def update_run_row(run_path: Path, new_row: dict[str, str]) -> None:
    rows = read_tsv(run_path) if run_path.exists() else []
    rows = [row for row in rows if row.get("shard_id") != new_row["shard_id"]]
    rows.append(new_row)
    rows.sort(key=lambda row: row["shard_id"])
    write_tsv(run_path, rows, RUN_COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--shard-plan", required=True, type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dotani", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--max-readers", type=int, default=8)
    parser.add_argument("--cuda-dedup", default="sort_unstable", choices=["hashset", "sort_unstable"])
    parser.add_argument("--shard-id", action="append", help="Run only this shard_id; may be repeated")
    parser.add_argument("--cleanup-staged-inputs", action="store_true")
    args = parser.parse_args()

    args.manifest = args.manifest.resolve()
    args.shard_plan = args.shard_plan.resolve()
    args.scratch_root = args.scratch_root.resolve()
    args.output_root = args.output_root.resolve()
    args.dotani = args.dotani.resolve()

    manifest_rows = read_tsv(args.manifest)
    plan_rows = read_tsv(args.shard_plan)
    shards = rows_by_shard(manifest_rows)
    selected = set(args.shard_id or [])
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_path = args.output_root / "shard_runs.tsv"
    ok_runs = load_ok_runs(run_path)

    for plan_row in plan_rows:
        shard_id = plan_row["shard_id"]
        if selected and shard_id not in selected:
            continue
        if shard_id in ok_runs:
            print(f"skip {shard_id}: already ok")
            continue

        shard_rows = shards.get(shard_id, [])
        try:
            validate_plan_row(plan_row, shard_rows)
        except ValueError as exc:
            print(f"error: {exc}")
            return 2

        shard_out_dir = args.output_root / shard_id
        scratch_dir = args.scratch_root / shard_id
        shard_out_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir.mkdir(parents=True, exist_ok=True)

        start = time.monotonic()
        copy_shard_inputs(shard_rows, scratch_dir)
        staged_manifest = shard_out_dir / "manifest.staged.tsv"
        write_staged_manifest(staged_manifest, shard_rows, scratch_dir)
        exit_status = run_dotani(args, staged_manifest, shard_out_dir, shard_id)
        wall_secs = time.monotonic() - start

        status, sketch_checksum, ull_checksum, output_bytes = shard_status(
            shard_out_dir, shard_id, len(shard_rows)
        )
        if exit_status != 0 and status == "ok":
            status = "failed"

        update_run_row(
            run_path,
            {
                "shard_id": shard_id,
                "manifest_path": str(staged_manifest.resolve()),
                "sketch_path": str((shard_out_dir / f"{shard_id}.sketch").resolve()),
                "ull_path": str((shard_out_dir / f"{shard_id}.sketch.ull").resolve()),
                "start_index": plan_row["start_index"],
                "record_count": str(len(shard_rows)),
                "status": status,
                "wall_secs": f"{wall_secs:.3f}",
                "sketch_checksum": sketch_checksum,
                "ull_checksum": ull_checksum,
                "output_bytes": str(output_bytes),
            },
        )

        print(f"{shard_id}: {status} in {wall_secs:.1f}s")
        if status == "ok" and args.cleanup_staged_inputs:
            shutil.rmtree(scratch_dir)

        if status != "ok":
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
