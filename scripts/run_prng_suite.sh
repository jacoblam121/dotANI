#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CRATE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
WORKSPACE_DIR=$(cd -- "$CRATE_DIR/.." && pwd)

OUT_ROOT="dotani_outputs/philox_tests"
THREADS=16
DEDUP="sort_unstable"
SMALL_INPUT="testing_genomes/testing_genomes"
TRUTH="accuracy_tests/GCF_bin_dist"
GCA946_INPUT="gtdb_genomes/gtdb_genomes_reps_r220/database/GCA/946"
SKIP_BUILD=0
ONLY_SMALL=0
SKIP_GCA946=0

usage() {
  cat <<'USAGE'
Usage: dotANI_jacob/scripts/run_prng_suite.sh [options]

Options:
  --out-root PATH       Output root directory (default: dotani_outputs/philox_tests)
  --threads N          Thread count passed to dotani (default: 16)
  --dedup STRATEGY     CUDA dedup strategy (default: sort_unstable)
  --small-input PATH   Small accuracy input directory
  --truth PATH         Small accuracy truth file
  --gca946-input PATH  GCA/946 speed input directory
  --skip-build         Do not build the CUDA release binary first
  --only-small         Run only the small accuracy suite
  --skip-gca946        Skip GCA/946 speed suite
  -h, --help           Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --dedup)
      DEDUP="$2"
      shift 2
      ;;
    --small-input)
      SMALL_INPUT="$2"
      shift 2
      ;;
    --truth)
      TRUTH="$2"
      shift 2
      ;;
    --gca946-input)
      GCA946_INPUT="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --only-small)
      ONLY_SMALL=1
      SKIP_GCA946=1
      shift
      ;;
    --skip-gca946)
      SKIP_GCA946=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$WORKSPACE_DIR"

TS=$(date +%Y%m%d_%H%M%S)
OUT="$OUT_ROOT/$TS"
BIN="$CRATE_DIR/target/release/dotani"
PRNGS=(wyrng curand_philox10 direct_philox10 direct_philox7)

mkdir -p "$OUT/small"
if [[ "$SKIP_GCA946" -eq 0 ]]; then
  mkdir -p "$OUT/gca946"
fi

echo "Output dir: $OUT"
echo "CUDA dedup: $DEDUP"
echo "PRNGs: ${PRNGS[*]}"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  cargo build --manifest-path "$CRATE_DIR/Cargo.toml" --release --features cuda \
    > "$OUT/build.log" 2>&1
fi

run_small() {
  local prng="$1"
  local dir="$OUT/small"

  "$BIN" sketch \
    -p "$SMALL_INPUT" \
    -o "$dir/${prng}.sketch" \
    --device cuda \
    --cuda-dedup "$DEDUP" \
    --cuda-hd-prng "$prng" \
    -T "$THREADS" \
    --metrics-out "$dir/${prng}_metrics" \
    > "$dir/${prng}_sketch.log" 2>&1

  "$BIN" dist \
    -r "$dir/${prng}.sketch" \
    -q "$dir/${prng}.sketch" \
    -o "$dir/${prng}.ani" \
    -T "$THREADS" \
    --ani-th 0 \
    > "$dir/${prng}_dist.log" 2>&1

  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-dotani}" \
    python3 accuracy_tests/compare_dotani_accuracy.py \
    --dotani "$dir/${prng}.ani" \
    --truth "$TRUTH" \
    --out "$dir/${prng}_accuracy.tsv" \
    --plot "$dir/${prng}_accuracy.png" \
    > "$dir/${prng}_accuracy.log" 2>&1
}

run_gca946() {
  local prng="$1"
  local dir="$OUT/gca946"

  "$BIN" sketch \
    -p "$GCA946_INPUT" \
    -o "$dir/${prng}.sketch" \
    --device cuda \
    --cuda-dedup "$DEDUP" \
    --cuda-hd-prng "$prng" \
    -T "$THREADS" \
    --metrics-out "$dir/${prng}_metrics" \
    > "$dir/${prng}_sketch.log" 2>&1
}

for prng in "${PRNGS[@]}"; do
  echo "Running small suite: $prng"
  run_small "$prng"
done

if [[ "$SKIP_GCA946" -eq 0 ]]; then
  for prng in "${PRNGS[@]}"; do
    echo "Running GCA/946 suite: $prng"
    run_gca946 "$prng"
  done
fi

python3 - "$OUT" <<'PY'
import csv
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[1])
summary_path = out / "summary.tsv"
prngs = ["wyrng", "curand_philox10", "direct_philox10", "direct_philox7"]
datasets = ["small", "gca946"]
fields = [
    "dataset",
    "prng",
    "sketch_wall_s",
    "files_per_s",
    "hash_and_dedup_s",
    "hd_encode_s",
    "cuda_hd_hash_h2d_s",
    "cuda_hd_hv_h2d_s",
    "cuda_hd_alloc_s",
    "cuda_hd_kernel_launch_s",
    "cuda_hd_d2h_s",
    "pair_count",
    "missing_pairs",
    "extra_pairs",
    "mean_absolute_error",
    "median_absolute_error",
    "max_absolute_error",
    "rmse",
]

def ns_to_s(value):
    if value in ("", "NA", None):
        return ""
    return f"{int(value) / 1_000_000_000:.6f}"

def read_total_metrics(path):
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        if row.get("file") == "TOTAL":
            return row
    return {}

def read_log_timing(path):
    if not path.exists():
        return "", ""
    text = path.read_text(errors="replace")
    match = re.search(r"Sketching .* took ([0-9.]+)s - Speed: ([0-9.]+) files/s", text)
    if not match:
        return "", ""
    return match.group(1), match.group(2)

def read_accuracy(path):
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 2:
            values[parts[0]] = parts[1]
    return values

with summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    for dataset in datasets:
        dataset_dir = out / dataset
        if not dataset_dir.exists():
            continue
        for prng in prngs:
            metrics = read_total_metrics(dataset_dir / f"{prng}_metrics.summary.tsv")
            accuracy = read_accuracy(dataset_dir / f"{prng}_accuracy.log")
            wall_s, files_per_s = read_log_timing(dataset_dir / f"{prng}_sketch.log")
            row = {
                "dataset": dataset,
                "prng": prng,
                "sketch_wall_s": wall_s,
                "files_per_s": files_per_s,
                "hash_and_dedup_s": ns_to_s(metrics.get("hash_and_dedup_ns")),
                "hd_encode_s": ns_to_s(metrics.get("hd_encode_ns")),
                "cuda_hd_hash_h2d_s": ns_to_s(metrics.get("cuda_hd_hash_h2d_ns")),
                "cuda_hd_hv_h2d_s": ns_to_s(metrics.get("cuda_hd_hv_h2d_ns")),
                "cuda_hd_alloc_s": ns_to_s(metrics.get("cuda_hd_alloc_ns")),
                "cuda_hd_kernel_launch_s": ns_to_s(metrics.get("cuda_hd_kernel_launch_ns")),
                "cuda_hd_d2h_s": ns_to_s(metrics.get("cuda_hd_d2h_ns")),
                "pair_count": accuracy.get("pair_count", ""),
                "missing_pairs": accuracy.get("missing_pairs", ""),
                "extra_pairs": accuracy.get("extra_pairs", ""),
                "mean_absolute_error": accuracy.get("mean_absolute_error", ""),
                "median_absolute_error": accuracy.get("median_absolute_error", ""),
                "max_absolute_error": accuracy.get("max_absolute_error", ""),
                "rmse": accuracy.get("rmse", ""),
            }
            writer.writerow(row)

print(summary_path)
print(summary_path.read_text(), end="")
PY
