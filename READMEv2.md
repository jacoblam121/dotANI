# Changes vs upstream (dotANI) and findings

## Changes

High Level:
- Added mode selection (cpu vs cuda) to the CLI for sketching
- Stage timing metrics
- Verification to make sure CPU and CUDA runs have the same output
- CUDA now performs both kmer hashing and HD encode on GPU (existing `.sketch` and `.ull` format preserved) for ~5x speedup
- Full at scale database tested on local machine and Russell

## Metrics
In ns by default, `sketch_wall_ns` is wall clock (end to end time)

- `input_bases`
  Total nucleotide bases processed across all input files.
- `hashes_seen`
  Total hashed k-mers seen (before deduplication)
- `unique_hashes`
  Total unique hashes kept
- `fasta_ns`
  Input side prep time
  For CUDA this is `read_merge_seq`, which reads/decompresses and merges
  each FASTA into a buffer
- `hash_and_dedup_ns`
  CPU-side k-mer hash stream handling after input prep
  For CUDA this is the host side ULL update and `HashSet` construction after the
  GPU returns raw hashes
- `hd_encode_ns`
  Time spent encoding the deduplicated hash set into the HD vector.
  On CUDA builds this is the host-observed GPU HD encode stage.
- `hv_norm_ns`
  Time spent computing the HV L2 norm
- `hd_compress_ns`
  Time spent compressing the HV sketch (compress for output)
- `total_worker_ns`
  Total worker time (Rayon threads/workers)
- `sketch_wall_ns`
  Wall clock time
- `cuda_h2d_ns`
  PCIe time, or time copying FASTA sequence buffers from host memory to GPU memory
- `cuda_alloc_ns`
  Time to allocate GPU memory for hash output
- `cuda_launch_ns`
  Time for the CPU to start CUDA kernel execution (not kernel runtime)
- `cuda_d2h_ns`
  Time to copy raw hash buffer from GPU to CPU
- `cuda_zero_filter_ns`
  Time to remove padding from returned hash buffer (zeros)
- `cuda_filter_ns`
  CPU time to build ULL and HashSet from CUDA output
  `hash_and_dedup_ns` for comparison
- `cuda_hd_hash_h2d_ns`
  Time to copy sampled hashes to GPU for HD encoding
- `cuda_hd_hv_h2d_ns`
  Time to copy HD vector to GPU
- `cuda_hd_alloc_ns`
  Time to allocate GPU buffers for HD encoding
- `cuda_hd_kernel_launch_ns`
  Time for the CPU to launch the CUDA HD encode kernel
- `cuda_hd_d2h_ns`
  Time to copy the encoded HD vector back to CPU

## Findings

Local machine: Ryzen 9 8945HS, 32GB DDR5, RTX 4070 mobile

Lab server (Russell): Ryzen Threadripper 9985WX, 768GB DDR5, RTX PRO 6000 (only used 1x GPU)

GPU now hashes kmers and does HD encoding, for about a ~5x speedup

On original subset `GCA/946`
(~1120 genomes) with `-T 16 -d 4096`:

*Cuda HD metrics unavailable as were not yet implemented

- `sketch_wall_s`: `230.671`
- `fasta_s`: `21.365`
- `hash_dedup_s`: `266.229`
- `hd_encode_s`: `3331.676`
- `hd_compress_s`: `0.025`
- `cuda_h2d_s`: `2.404`
- `cuda_alloc_s`: `5.007`
- `cuda_launch_s`: `0.387`
- `cuda_d2h_s`: `27.944`
- `cuda_zero_filter_s`: `2.911`
- `cuda_filter_s`: `266.229`
- `cuda_hd_hash_h2d_s`: `NA`
- `cuda_hd_hv_h2d_s`: `NA`
- `cuda_hd_alloc_s`: `NA`
- `cuda_hd_launch_s`: `NA`
- `cuda_hd_d2h_s`: `NA`


After moving HD encoding to GPU (3 run median): 

- `sketch_wall_s`: `44.666`
- `fasta_s`: `18.858`
- `hash_dedup_s`: `592.706`
- `hd_encode_s`: `39.985`
- `hd_compress_s`: `0.026`
- `cuda_h2d_s`: `3.060`
- `cuda_alloc_s`: `11.016`
- `cuda_launch_s`: `1.167`
- `cuda_d2h_s`: `10.012`
- `cuda_zero_filter_s`: `2.706`
- `cuda_filter_s`: `592.706`
- `cuda_hd_hash_h2d_s`: `5.389`
- `cuda_hd_hv_h2d_s`: `1.613`
- `cuda_hd_alloc_s`: `9.887`
- `cuda_hd_launch_s`: `0.986`
- `cuda_hd_d2h_s`: `16.023`

HD encode worker time dropped from `3331.676s` to `39.985s`, ~83x faster.

Wall clock time dropped from `230.671s` to `44.666`, ~5.2x faster. 

### Testing at scale (full GTDB database):

Local: 

- `sketch_wall_s`: `5711.572`
- `fasta_s`: `2476.974`
- `hash_dedup_s`: `79545.811`
- `hd_encode_s`: `4101.001`
- `hd_compress_s`: `2.284`
- `cuda_h2d_s`: `281.053`
- `cuda_alloc_s`: `946.676`
- `cuda_launch_s`: `101.854`
- `cuda_d2h_s`: `950.724`
- `cuda_zero_filter_s`: `348.173`
- `cuda_filter_s`: `79545.811`
- `cuda_hd_hash_h2d_s`: `593.772`
- `cuda_hd_hv_h2d_s`: `148.233`
- `cuda_hd_alloc_s`: `858.284`
- `cuda_hd_launch_s`: `83.699`
- `cuda_hd_d2h_s`: `1893.399`

Russell:

- `sketch_wall_s`: `1347.999`
- `fasta_s`: `1152.078`
- `hash_dedup_s`: `21402.781`
- `hd_encode_s`: `66854.554`
- `hd_compress_s`: `1.141`
- `cuda_h2d_s`: `12062.720`
- `cuda_alloc_s`: `22211.992`
- `cuda_launch_s`: `1683.784`
- `cuda_d2h_s`: `8131.035`
- `cuda_zero_filter_s`: `179.630`
- `cuda_filter_s`: `21402.781`
- `cuda_hd_hash_h2d_s`: `12941.544`
- `cuda_hd_hv_h2d_s`: `2661.834`
- `cuda_hd_alloc_s`: `22478.886`
- `cuda_hd_launch_s`: `3283.399`
- `cuda_hd_d2h_s`: `3515.591`

Local finished in `5711.572s` or ~95 minutes, previous ETA was ~8 hours, which checks out for a ~5.05x speedup

Russell finished in `1347.999s` or ~22.5 minutes, previous ETA was ~110 minutes (if I remember correctly), but would be a ~4.89x speedup

Interestingly, on local machine CPU was constantly hammered at full CPU util, while GPU util fluctuated from around 50-80%. Memory was maxed out as well. 

However, on Russell, neither CPU or GPU was fully being utilized. Load average was ~15 even though running with -T 128. GPU util mirrored local, fluctuating from 50-80% util. 

Not sure what the bottleneck is here, entire dataset was loaded into ram so we can rule out the hdd being the bottleneck. 

## Current Update:

- HD encoding is no longer the main bottleneck
- CPU decompression is not as much of an issue as we thought earlier
- CPU still spends non insignificant amount of time on hash deduplication / ULL work
- There is still GPU overhead
- Next: multi file GPU batching, GPU ULL, GPU dedup (or reduce CPU time on it), GPU compression, less CPU<->GPU transfer?

## How HD encoding was moved to GPU

- Replaced CPU HD encode step with CUDA kernel that computes the same signed HD count vector in parallel
- After kmer hashing, CPU still updates ULL and dedup/sample hashes, then sends these samples back to GPU
- GPU then computes the pseudorandom word as CPU WyRng (hash, 64 coordinate chunk), and returns same expected HD vector Vec<i32>

## HyperSpec

Implementation inspired by HyperSpec's GPU method, but not a direct port. 
HyperSpec's mass spectra clustering has different inputs, HD vector representation, distance math, and file output format. 


# (Everything below this line generated by AI)

## DotANI CUDA Sketch Branch

This branch adds explicit CPU/CUDA sketch device selection, stage timing
metrics, recursive compressed FASTA input discovery, and CUDA acceleration for
both k-mer hashing and HD encoding. CPU sketching remains the correctness
baseline.

A sketch command writes two files:

- DotHash sketch: `<output>.sketch`
- UltraLogLog sketch: `<output>.sketch.ull`

The existing `.sketch` and `.ull` formats are preserved. This branch does not
add binary HVs, GPU distance, GPU ULL, GPU dedup, GPU norm, GPU compression, or
multi-file GPU batching.

## CUDA HD Encoding

### Method

The CUDA sketch path now replaces the previous CPU HD encode step with a CUDA
kernel that computes the same signed HD count vector in parallel. The rest of
the sketch contract is intentionally unchanged: the output is still the same
`Vec<i32>` hypervector expected by the existing norm, compression, and
serialization code.

The CPU HD encoder builds the signed count vector from deduplicated sampled
hashes:

- initialize every coordinate to `-N`, where `N` is the number of unique sampled
  hashes
- for each hash, generate deterministic random bits from a WyRng stream seeded
  by that hash
- for each random `1` bit, add `2` to that coordinate

Starting at `-N` accounts for every hash contributing `-1` to every coordinate.
Each random `1` bit changes that hash's contribution for the coordinate from
`-1` to `+1`, which is why the update is `+2`.

The current CUDA sketch flow is:

1. GPU k-mer hashing runs first and returns raw hashes.
2. Host code still performs ULL updates and `HashSet` dedup/filtering.
3. The unique sampled hashes are collected into a `Vec<u64>`.
4. `src/sketch_cuda.rs` calls `hd_cuda::encode_hash_hd_cuda(&sampled_hashes, sketch.hv_d, ...)`.
5. `src/hd_cuda.rs` copies the sampled hashes and initialized count vector to
   GPU, launches `cuda_hd_encode_counts_direct`, and copies the completed
   `Vec<i32>` back to host.
6. Existing host-side norm calculation, compression, and serialization run
   unchanged.

The key implementation detail is direct random access to the random word for a
given `(hash, 64-coordinate chunk)`. The CPU path advances a serial `WyRng`
stream for each hash. That serial stream does not map cleanly to thousands of
independent GPU threads, so the CUDA path computes the WyRng-equivalent random
word directly from the hash and chunk index. Tests verify that the direct-seek
logic matches Rust `WyRng`, including the CUDA implementation.

The HD kernel is tiled across both coordinates and hashes:

- `blockIdx.x` selects one 64-coordinate HD chunk
- `blockIdx.y` selects one tile of sampled hashes
- each block uses 256 threads, or 8 warps
- each active thread loads one hash and computes its 64-bit random contribution
  for the selected coordinate chunk
- warp ballot/popcount-style logic counts which hashes have each bit set
- the first 64 threads reduce per-warp bit counts into coordinate counts
- `atomicAdd(&hv[d], ones * 2)` applies the count update to the signed vector

The atomic add is needed because multiple hash tiles update the same HD
coordinates. This is still the same exact count math as the CPU path, not a new
approximation.

Scope was kept deliberately narrow. The CUDA HD encoder does not change ULL,
deduplication, HV norm calculation, compression, distance estimation, or the
serialized `.sketch` / `.ull` formats. That is why CPU, older CUDA, and current
CUDA outputs can be compared byte-for-byte.

### HyperSpec Comparison

The design was inspired by HyperSpec's GPU execution pattern, not directly
ported from HyperSpec. The useful idea was that HD encoding can be expressed as
many coordinate-level reductions over input features, which maps naturally to
GPU parallelism.

The analogy is:

- HyperSpec feature: spectrum peak
- dotANI feature: unique sampled k-mer hash
- HyperSpec output: packed binary hypervector bit
- dotANI output: signed `i32` count hypervector coordinate
- HyperSpec reduction: combine feature contributions, then majority/sign
- dotANI reduction: count deterministic random `1` bits, starting from `-N` and
  adding `2` for each `1`

What we borrowed:

- the coordinate-parallel HD encoding shape
- the idea that many input features can be reduced into HD coordinates on GPU
- bit-level GPU counting/reduction patterns
- the performance lesson that batching, transfer cost, and compact
  representations matter

What we did not borrow:

- HyperSpec encodes mass spectra peaks with m/z and intensity features; dotANI
  encodes sampled genome k-mer hashes.
- HyperSpec uses ID and level hypervectors for spectrum features; dotANI uses
  hash-seeded pseudorandom contributions.
- HyperSpec produces packed binary hypervectors; dotANI uses signed `i32` count
  hypervectors before compression.
- HyperSpec can use XOR/popcount Hamming distance; dotANI's ANI estimator uses
  dot products over the HD sketch plus ULL cardinality information.
- A direct HyperSpec-style port would require new sketch formats, new distance
  code, and new accuracy validation.

In short, HyperSpec supported the argument that HD encoding is GPU-friendly. The
implementation here applies that GPU execution shape while preserving dotANI's
existing estimator and file formats.

### Evidence

Implementation evidence:

- `src/sketch_cuda.rs` collects `sampled_hashes: Vec<u64>` after host
  dedup/filtering, then calls `hd_cuda::encode_hash_hd_cuda`.
- `src/hd_cuda.rs` initializes the host count vector to `-N`, copies hashes and
  the initialized vector to GPU, launches `cuda_hd_encode_counts_direct`, and
  returns a `Vec<i32>`.
- `src/cuda_kernel.cu` implements the coordinate/hash-tiled HD encode kernel
  using warp ballot/popcount-style counting and `atomicAdd`.
- Metrics record the GPU HD stage as separate transfer/allocation/launch/D2H
  fields: `cuda_hd_hash_h2d_ns`, `cuda_hd_hv_h2d_ns`, `cuda_hd_alloc_ns`,
  `cuda_hd_kernel_launch_ns`, and `cuda_hd_d2h_ns`.

Correctness evidence:

- CUDA HD encode is tested against CPU HD encode for empty inputs, short
  `hv_d`, representative hashes, and larger random hash sets.
- Direct-seek WyRng logic is tested against sequential Rust `WyRng`, including
  the CUDA `wyrng_at_chunk` implementation.
- CPU/CUDA `.sketch` and `.ull` byte equality was verified on tiny inputs at
  `hv_d=1024` and `hv_d=4096`.
- CPU/CUDA `.sketch` and `.ull` byte equality was verified on `GCA/946/626` at
  `hv_d=1024` and `hv_d=4096`.
- CPU/CUDA `.sketch` and `.ull` byte equality was verified on full `GCA/946` at
  `hv_d=4096`.
- The older CUDA path and current CUDA path produced identical full `GCA/946`
  outputs:
  - `.sketch`:
    `f602adddef6674ad80c9f923d2c32f891573a32648376b710f8c49e4b9bb0465`
  - `.ull`:
    `cff6bdb6577ee4658bf58de5ad71d7f1bf48ac96c853ad1a02821dd79365c713`

Performance evidence:

- On `GCA/946` with `-T 16 -d 4096`, the older CUDA path had
  `sketch_wall_s=230.671` and `hd_encode_s=3331.676`.
- With GPU HD encoding, the 3-run median on the same input was
  `sketch_wall_s=44.666` and `hd_encode_s=39.985`.
- That is about `5.2x` faster end-to-end and about `83x` lower HD encode worker
  time for the reported baseline.
- On the full GTDB R220 representative database, the lab server run completed
  `113104` files in `22:29.07` at `84.4 files/s`.
- The local full GTDB R220 run completed in about `95.2 minutes`; the lab
  server completed in about `22.5 minutes`.

Recent targeted verification run locally:

```sh
cargo test --features cuda --lib cuda_hd -- --nocapture
cargo test --features cuda --lib wyrng -- --nocapture
```

Result:

- CUDA HD parity/metrics tests: `4 passed`
- direct WyRng parity tests: `2 passed`

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

Input discovery is recursive under `--path`, so nested GTDB-style directory
trees such as `database/GCA/946/.../*.fna.gz` are supported.

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

CPU and CUDA runs use the same TSV schema. CPU rows emit `NA` for CUDA-specific
columns. CUDA rows emit `NA` for `cuda_hd_*` only when no GPU HD work applies,
such as empty sampled hashes or `hv_d < 64`.

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
- `cuda_hd_hash_h2d_ns`
- `cuda_hd_hv_h2d_ns`
- `cuda_hd_alloc_ns`
- `cuda_hd_kernel_launch_ns`
- `cuda_hd_d2h_ns`

`cuda_launch_ns` is host enqueue time, not true kernel duration. Use Nsight
Systems or Nsight Compute for true CUDA kernel timing.

`sketch_wall_ns` is populated on the summary row and is `NA` for individual file
rows. `cuda_zero_filter_ns` is the host-side removal of zero padding from the
CUDA output buffer. `cuda_filter_ns` is the CUDA path's host-side ULL and
HashSet construction time, and is also included in `hash_and_dedup_ns` for
CPU/CUDA comparison.

CUDA HD encode columns:

- `cuda_hd_hash_h2d_ns`
  Time to copy the unique sampled hash vector to GPU for HD encoding.
- `cuda_hd_hv_h2d_ns`
  Time to copy the initialized HD vector to GPU.
- `cuda_hd_alloc_ns`
  Time to allocate GPU buffers for HD encoding.
- `cuda_hd_kernel_launch_ns`
  Host-side CUDA HD kernel enqueue time.
- `cuda_hd_d2h_ns`
  Time to copy the encoded HD vector back to host.

Readable seconds view:

```sh
awk -F'\t' '
NR==1 { for (i=1;i<=NF;i++) h[$i]=i; next }
{
  printf "sketch_wall_s=%.3f\n", $h["sketch_wall_ns"]/1e9
  printf "fasta_s=%.3f\n", $h["fasta_ns"]/1e9
  printf "hash_dedup_s=%.3f\n", $h["hash_and_dedup_ns"]/1e9
  printf "hd_encode_s=%.3f\n", $h["hd_encode_ns"]/1e9
  printf "hd_compress_s=%.3f\n", $h["hd_compress_ns"]/1e9
  printf "cuda_h2d_s=%.3f\n", $h["cuda_h2d_ns"]/1e9
  printf "cuda_alloc_s=%.3f\n", $h["cuda_alloc_ns"]/1e9
  printf "cuda_launch_s=%.3f\n", $h["cuda_launch_ns"]/1e9
  printf "cuda_d2h_s=%.3f\n", $h["cuda_d2h_ns"]/1e9
  printf "cuda_zero_filter_s=%.3f\n", $h["cuda_zero_filter_ns"]/1e9
  printf "cuda_filter_s=%.3f\n", $h["cuda_filter_ns"]/1e9
  printf "cuda_hd_hash_h2d_s=%.3f\n", $h["cuda_hd_hash_h2d_ns"]/1e9
  printf "cuda_hd_hv_h2d_s=%.3f\n", $h["cuda_hd_hv_h2d_ns"]/1e9
  printf "cuda_hd_alloc_s=%.3f\n", $h["cuda_hd_alloc_ns"]/1e9
  printf "cuda_hd_launch_s=%.3f\n", $h["cuda_hd_kernel_launch_ns"]/1e9
  printf "cuda_hd_d2h_s=%.3f\n", $h["cuda_hd_d2h_ns"]/1e9
}' /tmp/gca946_cuda_4096.summary.tsv
```

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

## Full GCA/946 Scale Run

The main local benchmark input is:

```sh
INPUT=/home/jacob/HyperGen/gtdb_genomes/gtdb_genomes_reps_r220/database/GCA/946
```

This subtree has `1124` genome files and is about `728 MB`.

Run CUDA:

```sh
target/release/dotani sketch --device cuda \
  -p "$INPUT" \
  -o /tmp/gca946_cuda_4096.sketch \
  -T 16 -d 4096 \
  --metrics-out /tmp/gca946_cuda_4096
```

Run CPU for parity:

```sh
target/release/dotani sketch --device cpu \
  -p "$INPUT" \
  -o /tmp/gca946_cpu_4096.sketch \
  -T 16 -d 4096 \
  --metrics-out /tmp/gca946_cpu_4096
```

Compare outputs:

```sh
cmp /tmp/gca946_cpu_4096.sketch /tmp/gca946_cuda_4096.sketch
cmp /tmp/gca946_cpu_4096.sketch.ull /tmp/gca946_cuda_4096.sketch.ull

sha256sum \
  /tmp/gca946_cpu_4096.sketch \
  /tmp/gca946_cuda_4096.sketch \
  /tmp/gca946_cpu_4096.sketch.ull \
  /tmp/gca946_cuda_4096.sketch.ull
```

`cmp` prints nothing when files are identical.

Full `GCA/946` verified hashes:

```text
.sketch  f602adddef6674ad80c9f923d2c32f891573a32648376b710f8c49e4b9bb0465
.ull     cff6bdb6577ee4658bf58de5ad71d7f1bf48ac96c853ad1a02821dd79365c713
```

## CUDA HD Encode Benchmark Result

Old reported CUDA path from `update/update_4_28.md`:

```text
sketch_wall_s=230.671
hd_encode_s=3331.676
cuda_filter_s=266.229
```

Recreated older CUDA path from commit `0ef3495`:

```text
sketch_wall_s=264.909
hd_encode_s=3808.702
cuda_filter_s=329.913
```

Current CUDA repeated runs:

| Metric | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| `sketch_wall_s` | 44.666 | 45.924 | 42.973 |
| `fasta_s` | 18.858 | 19.328 | 18.350 |
| `hash_dedup_s` | 592.706 | 608.040 | 569.639 |
| `hd_encode_s` | 39.985 | 41.847 | 37.590 |
| `hd_compress_s` | 0.026 | 0.025 | 0.026 |
| `cuda_h2d_s` | 3.060 | 3.827 | 3.025 |
| `cuda_alloc_s` | 10.615 | 11.426 | 11.016 |
| `cuda_launch_s` | 1.008 | 1.171 | 1.167 |
| `cuda_d2h_s` | 10.012 | 10.282 | 9.683 |
| `cuda_zero_filter_s` | 2.706 | 2.714 | 2.659 |
| `cuda_filter_s` | 592.706 | 608.040 | 569.639 |
| `cuda_hd_hash_h2d_s` | 5.389 | 6.154 | 5.300 |
| `cuda_hd_hv_h2d_s` | 1.613 | 1.799 | 1.160 |
| `cuda_hd_alloc_s` | 9.995 | 9.887 | 9.287 |
| `cuda_hd_launch_s` | 0.986 | 1.130 | 0.731 |
| `cuda_hd_d2h_s` | 16.023 | 16.160 | 15.405 |

Median current CUDA result:

```text
sketch_wall_s=44.666
hd_encode_s=39.985
hash_dedup_s=592.706
```

Compared with the old reported CUDA path, the current CUDA path is about `5.2x`
faster end-to-end and about `83x` faster in `hd_encode_s`. Compared with the
recreated older CUDA baseline, the current CUDA path is about `5.9x` faster
end-to-end and about `95x` faster in `hd_encode_s`.

## Full GTDB R220 Scale Result

The full GTDB R220 representative database was tested with `hv_d=4096`.

Lab server command:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/bin/time -v target/release/dotani sketch --device cuda \
  -p "$INPUT" \
  -o /tmp/gtdb_r220_cuda_4096.sketch \
  -T 128 -d 4096 \
  --metrics-out /tmp/gtdb_r220_cuda_4096
```

Lab server result:

- input files: `113104`
- elapsed wall time: `22:29.07`
- summary `sketch_wall_s`: `1347.999`
- throughput: `84.4 files/s`
- max RSS: `22687804 KB`, about `21.6 GiB`
- `.sketch` size: `791.83 MB`
- compressed `.ull` size: `983.37 MB`
- raw `.ull` size: `1780.63 MB`
- exit status: `0`

Lab server summary metrics:

| Metric | Seconds |
|---|---:|
| `sketch_wall_s` | 1347.999 |
| `fasta_s` | 1152.078 |
| `hash_dedup_s` | 21402.781 |
| `hd_encode_s` | 66854.554 |
| `hd_compress_s` | 1.141 |
| `cuda_h2d_s` | 12062.720 |
| `cuda_alloc_s` | 22211.992 |
| `cuda_launch_s` | 1683.784 |
| `cuda_d2h_s` | 8131.035 |
| `cuda_zero_filter_s` | 179.630 |
| `cuda_filter_s` | 21402.781 |
| `cuda_hd_hash_h2d_s` | 12941.544 |
| `cuda_hd_hv_h2d_s` | 2661.834 |
| `cuda_hd_alloc_s` | 22478.886 |
| `cuda_hd_launch_s` | 3283.399 |
| `cuda_hd_d2h_s` | 3515.591 |

Local machine summary metrics on the same full GTDB R220 input:

| Metric | Seconds |
|---|---:|
| `sketch_wall_s` | 5711.572 |
| `fasta_s` | 2476.974 |
| `hash_dedup_s` | 79545.811 |
| `hd_encode_s` | 4101.001 |
| `hd_compress_s` | 2.284 |
| `cuda_h2d_s` | 281.053 |
| `cuda_alloc_s` | 946.676 |
| `cuda_launch_s` | 101.854 |
| `cuda_d2h_s` | 950.724 |
| `cuda_zero_filter_s` | 348.173 |
| `cuda_filter_s` | 79545.811 |
| `cuda_hd_hash_h2d_s` | 593.772 |
| `cuda_hd_hv_h2d_s` | 148.233 |
| `cuda_hd_alloc_s` | 858.284 |
| `cuda_hd_launch_s` | 83.699 |
| `cuda_hd_d2h_s` | 1893.399 |

The previous lab-server ETA was about `2 hours`; the measured run finished in
about `22.5 minutes`. The previous local ETA was about `8 hours`; the measured
local run finished in about `95.2 minutes`.

## Current Bottleneck

GPU HD encoding removed HD encoding as the dominant CUDA-path cost. The largest
remaining measured worker cost is now host-side dedup/filtering:

```text
median cuda_filter_s=592.706
```

This is summed worker time and can exceed wall time. It still identifies the
main remaining worker-side cost. The next performance work should target
batching, buffer reuse, and/or reducing the host hash filtering/dedup round
trip. GPU distance, binary HV formats, GPU ULL, GPU norm, and GPU compression
should remain separate follow-up experiments unless explicitly approved.

## Backward Compatibility Notes

Older commands may have used names like `dotani-cuda` or `dotani_gpu`. This
branch builds a single binary at `target/release/dotani`. If an old script
requires `dotani_gpu`, create a local symlink after building.
