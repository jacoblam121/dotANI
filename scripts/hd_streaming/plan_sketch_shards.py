#!/usr/bin/env python3
"""Plan dotANI sketch shards and manifests for large FASTA trees."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path


FASTA_SUFFIXES = (
    ".fna",
    ".fa",
    ".fasta",
    ".fna.gz",
    ".fa.gz",
    ".fasta.gz",
    ".fna.bz2",
    ".fa.bz2",
    ".fasta.bz2",
    ".fna.xz",
    ".fa.xz",
    ".fasta.xz",
    ".fna.zst",
    ".fa.zst",
    ".fasta.zst",
)

SIZE_UNITS = {
    "": 1,
    "b": 1,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
}


@dataclass(frozen=True)
class ManifestRow:
    global_index: int
    shard_id: str
    read_path: Path
    file_id: str
    rel_path: str
    size_bytes: int


def parse_size(value: str) -> int:
    text = value.strip()
    number = []
    suffix = []
    for ch in text:
        if ch.isdigit() or ch == ".":
            if suffix:
                raise ValueError(f"invalid size {value!r}")
            number.append(ch)
        elif not ch.isspace():
            suffix.append(ch.lower())

    if not number:
        raise ValueError(f"invalid size {value!r}")

    unit = "".join(suffix)
    if unit not in SIZE_UNITS:
        raise ValueError(f"unsupported size suffix {unit!r}")

    return int(float("".join(number)) * SIZE_UNITS[unit])


def is_fasta_path(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in FASTA_SUFFIXES)


def scan_fasta_files(input_root: Path) -> list[Path]:
    paths: set[Path] = set()
    for root, dirnames, filenames in os.walk(input_root):
        dirnames.sort()
        for filename in sorted(filenames):
            path = Path(root) / filename
            if path.is_file() and is_fasta_path(path):
                paths.add(path.resolve())
    return sorted(paths)


def checked_rel_path(path: Path, input_root: Path) -> str:
    rel = path.relative_to(input_root).as_posix()
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError(f"unsafe relative path for {path}: {rel}")
    return rel


def assign_shards(files: list[Path], input_root: Path, target_shard_bytes: int) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    shard_number = 0
    shard_bytes = 0

    for global_index, path in enumerate(files):
        size_bytes = path.stat().st_size
        if rows and shard_bytes > 0 and shard_bytes + size_bytes > target_shard_bytes:
            shard_number += 1
            shard_bytes = 0

        rel_path = checked_rel_path(path, input_root)
        shard_id = f"shard_{shard_number:06d}"
        rows.append(
            ManifestRow(
                global_index=global_index,
                shard_id=shard_id,
                read_path=path,
                file_id=rel_path,
                rel_path=rel_path,
                size_bytes=size_bytes,
            )
        )
        shard_bytes += size_bytes

    return rows


def write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        handle.write("# version=1\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["global_index", "shard_id", "read_path", "file_id", "rel_path", "size_bytes"])
        for row in rows:
            writer.writerow(
                [
                    row.global_index,
                    row.shard_id,
                    str(row.read_path),
                    row.file_id,
                    row.rel_path,
                    row.size_bytes,
                ]
            )


def write_outputs(out_dir: Path, rows: list[ManifestRow]) -> None:
    manifest_path = out_dir / "manifest.tsv"
    shard_plan_path = out_dir / "shard_plan.tsv"
    shard_manifest_root = out_dir / "shard_manifests"

    write_manifest(manifest_path, rows)

    shards: dict[str, list[ManifestRow]] = {}
    for row in rows:
        shards.setdefault(row.shard_id, []).append(row)

    for shard_id, shard_rows in shards.items():
        write_manifest(shard_manifest_root / f"{shard_id}.tsv", shard_rows)

    with shard_plan_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["shard_id", "start_index", "record_count", "input_bytes", "manifest_path"])
        for shard_id in sorted(shards):
            shard_rows = shards[shard_id]
            writer.writerow(
                [
                    shard_id,
                    shard_rows[0].global_index,
                    len(shard_rows),
                    sum(row.size_bytes for row in shard_rows),
                    str((shard_manifest_root / f"{shard_id}.tsv").resolve()),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target-shard-bytes", default="100GiB")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        parser.error(f"--input-root is not a directory: {input_root}")

    target_shard_bytes = parse_size(args.target_shard_bytes)
    if target_shard_bytes <= 0:
        parser.error("--target-shard-bytes must be greater than zero")

    files = scan_fasta_files(input_root)
    rows = assign_shards(files, input_root, target_shard_bytes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(args.out_dir.resolve(), rows)

    total_bytes = sum(row.size_bytes for row in rows)
    shard_count = len({row.shard_id for row in rows})
    print(f"planned {len(rows)} files, {total_bytes} bytes, {shard_count} shards")
    print(f"manifest: {(args.out_dir / 'manifest.tsv').resolve()}")
    print(f"shard plan: {(args.out_dir / 'shard_plan.tsv').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
