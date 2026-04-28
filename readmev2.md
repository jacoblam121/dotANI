## DotANI: Ultra-fast and Memory Efficient ANI Computation with GPU Acceleration

_dotANI_ samples k-mers, estimates intersections with dotHash, encodes k-mer
hashes into hyperdimensional vectors (HVs), and stores genome cardinality with
UltraLogLog (ULL). A sketch contains two outputs:

- the DotHash sketch: `<output>.sketch`
- the ULL sketch: `<output>.sketch.ull`

This `dotANI_jacob` branch adds explicit CPU/CUDA sketch selection and optional
Sprint 0 metrics output while preserving CPU as the correctness comparator.

## Quickstart

### Build

_dotANI_ requires Rust and Cargo. CUDA support requires an NVIDIA driver and a
working CUDA installation. Check GPU availability with:

```sh
nvidia-smi
```

CPU-only build:

```sh
cargo build --release
```

CUDA-capable build:

```sh
cargo build --release --features cuda
```

`--features cuda` is a build-time flag. After building with CUDA support, choose
the sketch execution device at runtime with `--device cpu|cuda`.

## Sketching

Basic CPU sketch:

```sh
target/release/dotani sketch --device cpu \
  -p ./data \
  -o ./fna.sketch
```

Basic CUDA sketch:

```sh
target/release/dotani sketch --device cuda \
  -p ./data \
  -o ./fna.sketch
```

CUDA-enabled builds default to CUDA, so this:

```sh
target/release/dotani sketch \
  -p ./data \
  -o ./fna.sketch
```

is equivalent to:

```sh
target/release/dotani sketch --device cuda \
  -p ./data \
  -o ./fna.sketch
```

CPU-only builds default to CPU. In a CPU-only build, `--device cuda` exits with a
clear error.

### Sketch Options

```text
Usage: dotani sketch [OPTIONS] --path <path> --out <out>

Options:
  -p, --path <path>                Input folder path containing .fna/.fa/.fasta files
                                   gzip/bzip2/xz/zstd compressed files are supported
  -o, --out <out>                  Output DotHash sketch file
  -T, --threads <threads>          Number of threads, default all logical cores
  -C, --canonical <canonical>      Whether to use canonical k-mers [default: true]
  -k, --ksize <ksize>              k-mer size for sketching [default: 16]
  -S, --seed <seed>                Hash seed [default: 1447]
      --ull-p <ull_p>              UltraLogLog precision parameter [default: 14]
  -d, --hv-d <hv_d>                Dimension for hypervector [default: 4096]
  -Q, --quant-scale <scale>        Scaling factor for HV quantization [default: 1.0]
      --device <cpu|cuda>          Sketch execution device
      --metrics-out <prefix>       Write metrics TSV files using this prefix
  -h, --help                       Print help
  -V, --version                    Print version
```

## Metrics

Metrics are disabled by default. Enable them with:

```sh
target/release/dotani sketch --device cuda \
  -p ./data \
  -o ./fna.sketch \
  --metrics-out ./fna_metrics
```

This writes:

```text
./fna_metrics.summary.tsv
./fna_metrics.files.tsv
```

CPU and CUDA runs use the same TSV schema. CPU runs emit `NA` for CUDA-specific
columns.

Important columns:

- `input_bases`
- `hashes_seen`
- `unique_hashes`
- `fasta_ns`
- `hash_and_dedup_ns`
- `hd_encode_ns`
- `hv_norm_ns`
- `hd_compress_ns`
- `total_worker_ns`
- `sketch_wall_ns`
- `cuda_h2d_ns`
- `cuda_alloc_ns`
- `cuda_launch_ns`
- `cuda_d2h_ns`
- `cuda_zero_filter_ns`
- `cuda_filter_ns`

`cuda_launch_ns` is host enqueue time, not true kernel duration. Use Nsight
Systems or Nsight Compute for true CUDA kernel timing.

`sketch_wall_ns` is populated on the summary row and is `NA` for individual file
rows. `cuda_zero_filter_ns` is the host-side removal of zero padding from the
CUDA output buffer. `cuda_filter_ns` is the CUDA path's host-side ULL and
HashSet construction time, and is also included in `hash_and_dedup_ns` for
CPU/CUDA comparison.

## ANI Estimation

```sh
target/release/dotani dist \
  -r fna1.sketch \
  -q fna2.sketch \
  -o output.ani
```

The corresponding ULL files are expected next to each sketch as:

```text
fna1.sketch.ull
fna2.sketch.ull
```

## Testing This Branch

Run from:

```sh
cd /home/jacob/HyperGen/dotANI_jacob
```

### Build and Unit Tests

```sh
cargo test --features cuda --lib --no-run
cargo test --features cuda --lib
cargo build --release --features cuda

env CARGO_TARGET_DIR=/tmp/dotani_cpu_target cargo build --release
```

The CPU-only binary is then available at:

```text
/tmp/dotani_cpu_target/release/dotani
```

The CUDA-capable binary is:

```text
target/release/dotani
```

### Tiny CPU/CUDA Correctness Test

```sh
INPUT=../gtdb_genomes/gtdb_genomes_reps_r220/database/GCA/946/626

/usr/bin/time -v target/release/dotani sketch --device cpu \
  -p "$INPUT" \
  -o /tmp/test_cpu_hvd4096.sketch \
  -T 16 -d 4096 \
  --metrics-out /tmp/test_cpu_hvd4096

/usr/bin/time -v target/release/dotani sketch --device cuda \
  -p "$INPUT" \
  -o /tmp/test_cuda_hvd4096.sketch \
  -T 16 -d 4096 \
  --metrics-out /tmp/test_cuda_hvd4096

sha256sum \
  /tmp/test_cpu_hvd4096.sketch \
  /tmp/test_cuda_hvd4096.sketch \
  /tmp/test_cpu_hvd4096.sketch.ull \
  /tmp/test_cuda_hvd4096.sketch.ull
```

The CPU and CUDA `.sketch` hashes should match. The CPU and CUDA `.ull` hashes
should also match.

Inspect metrics:

```sh
cat /tmp/test_cpu_hvd4096.summary.tsv
cat /tmp/test_cuda_hvd4096.summary.tsv

head -5 /tmp/test_cpu_hvd4096.files.tsv
head -5 /tmp/test_cuda_hvd4096.files.tsv
```

### Test Both HV Dimensions

```sh
INPUT=../gtdb_genomes/gtdb_genomes_reps_r220/database/GCA/946/626

for D in 1024 4096; do
  /usr/bin/time -v target/release/dotani sketch --device cpu \
    -p "$INPUT" \
    -o /tmp/test_cpu_hvd${D}.sketch \
    -T 16 -d "$D" \
    --metrics-out /tmp/test_cpu_hvd${D}

  /usr/bin/time -v target/release/dotani sketch --device cuda \
    -p "$INPUT" \
    -o /tmp/test_cuda_hvd${D}.sketch \
    -T 16 -d "$D" \
    --metrics-out /tmp/test_cuda_hvd${D}

  sha256sum \
    /tmp/test_cpu_hvd${D}.sketch \
    /tmp/test_cuda_hvd${D}.sketch \
    /tmp/test_cpu_hvd${D}.sketch.ull \
    /tmp/test_cuda_hvd${D}.sketch.ull
done
```

### Medium Local Testing Genomes

```sh
INPUT=/home/jacob/HyperGen/testing_genomes/testing_genomes

/usr/bin/time -v target/release/dotani sketch --device cpu \
  -p "$INPUT" \
  -o /tmp/testing_genomes_cpu.sketch \
  -T 16 -d 4096 \
  --metrics-out /tmp/testing_genomes_cpu

/usr/bin/time -v target/release/dotani sketch --device cuda \
  -p "$INPUT" \
  -o /tmp/testing_genomes_cuda.sketch \
  -T 16 -d 4096 \
  --metrics-out /tmp/testing_genomes_cuda

sha256sum \
  /tmp/testing_genomes_cpu.sketch \
  /tmp/testing_genomes_cuda.sketch \
  /tmp/testing_genomes_cpu.sketch.ull \
  /tmp/testing_genomes_cuda.sketch.ull
```

## Full GTDB-Scale Runs

The full local GTDB database root is:

```sh
INPUT=/home/jacob/HyperGen/gtdb_genomes/gtdb_genomes_reps_r220/database
```

Create an output directory:

```sh
mkdir -p /tmp/dotani_scale
```

Run full-scale CUDA sketching:

```sh
/usr/bin/time -v target/release/dotani sketch --device cuda \
  -p "$INPUT" \
  -o /tmp/dotani_scale/gtdb_cuda_hvd4096.sketch \
  -T 16 -d 4096 \
  --metrics-out /tmp/dotani_scale/gtdb_cuda_hvd4096 \
  > /tmp/dotani_scale/gtdb_cuda_hvd4096.log 2>&1
```

Run full-scale CPU sketching as a correctness and performance comparator:

```sh
/usr/bin/time -v target/release/dotani sketch --device cpu \
  -p "$INPUT" \
  -o /tmp/dotani_scale/gtdb_cpu_hvd4096.sketch \
  -T 16 -d 4096 \
  --metrics-out /tmp/dotani_scale/gtdb_cpu_hvd4096 \
  > /tmp/dotani_scale/gtdb_cpu_hvd4096.log 2>&1
```

For a background CUDA run:

```sh
nohup /usr/bin/time -v target/release/dotani sketch --device cuda \
  -p /home/jacob/HyperGen/gtdb_genomes/gtdb_genomes_reps_r220/database \
  -o /tmp/dotani_scale/gtdb_cuda_hvd4096.sketch \
  -T 16 -d 4096 \
  --metrics-out /tmp/dotani_scale/gtdb_cuda_hvd4096 \
  > /tmp/dotani_scale/gtdb_cuda_hvd4096.log 2>&1 &
```

Monitor progress:

```sh
tail -f /tmp/dotani_scale/gtdb_cuda_hvd4096.log
nvidia-smi dmon
```

Compare final outputs:

```sh
sha256sum \
  /tmp/dotani_scale/gtdb_cpu_hvd4096.sketch \
  /tmp/dotani_scale/gtdb_cuda_hvd4096.sketch \
  /tmp/dotani_scale/gtdb_cpu_hvd4096.sketch.ull \
  /tmp/dotani_scale/gtdb_cuda_hvd4096.sketch.ull
```

## Backward Compatibility Notes

Older commands may have used names like `dotani-cuda` or `dotani_gpu`. This
branch currently builds a single binary:

```text
target/release/dotani
```

The recommended CUDA command is:

```sh
target/release/dotani sketch --device cuda \
  -p ./data \
  -o ./fna.sketch
```

In CUDA-enabled builds, this older single-binary command remains compatible and
defaults to CUDA:

```sh
target/release/dotani sketch \
  -p ./data \
  -o ./fna.sketch
```

If an old script requires `dotani_gpu`, create a local compatibility symlink
after building:

```sh
cd target/release
ln -sf dotani dotani_gpu
```

Then old commands like this will resolve to the CUDA-capable `dotani` binary:

```sh
./dotANI_jacob/target/release/dotani_gpu sketch \
  -p ./gtdb_genomes/gtdb_genomes_reps_r220/database \
  -o ./gtdb_gpu.sketch \
  -T 16
```

Make sure the binary was built with CUDA support first:

```sh
cargo build --release --features cuda
```
