# GPU HD Encode Implementation

## High Level
- WyRng is standard and already used by CPU encoder
- GPU HD encoding itsellf was previously implemented in HyperSpec and is not new
- Existing HD encoder in dotANI was moved onto the GPU while preserving existing behavior, distance path, and .sketch/.ull format
- GPU HD encode is able to be done by reproducing the same pseudorandom WyRng word that the CPU would output for each `(hash, 64-coordinate chunk)` without actually calling the Rust WyRng function (reimplement WyRng on GPU). This lets independent CUDA threads compute chunk contributions without going through a serial WyRng stream. 

## dotANI pipeline

Genome is sketched in dotANI by:

1. Decompress and read fasta input
2. Hash kmer
3. Deduplicate sampled hashes
4. Run ULL to estimate genome cardinality
5. Encode sampled hash set into HD vector
6. Compute HD vector norm
7. Compress HD vector for output
8. Writing .sketch (dothash) and .ull as output

## Original dotANI implementation

- GPU: fasta sequence buffer to raw kmer hash
- CPU: filtering, ULL update, HashSet deduplication, HD encoding, norm,
  compression, and output
- HD encode was CPU bound, using AVX-512, this fork moves this to GPU

```rust
let hv = if is_x86_feature_detected!("avx512f") {
    unsafe { hd::encode_hash_hd_avx512(&sampled_hash_set, &sketch) }
} else if is_x86_feature_detected!("avx2") {
    unsafe { hd::encode_hash_hd_avx2(&sampled_hash_set, &sketch) }
} else {
    hd::encode_hash_hd(&sampled_hash_set, &sketch)
};
```

## CPU HD Encode

Baseline implementation is as follows:

```rust
pub fn encode_hash_hd(kmer_hash_set: &HashSet<u64>, sketch: &FileSketch) -> Vec<i32> {
    let hv_d = sketch.hv_d;
    let seed_vec = Vec::from_iter(kmer_hash_set.clone());
    let mut hv = vec![-(kmer_hash_set.len() as i32); hv_d];

    for hash in seed_vec {
        let mut rng = WyRng::seed_from_u64(hash);

        for i in 0..(hv_d / 64) {
            let rnd_bits = rng.next_u64();

            for j in 0..64 {
                hv[i * 64 + j] += (((rnd_bits >> j) & 1) << 1) as i32;
            }
        }
    }

    hv
}
```

Let:

- `hv` be the hd sketch vector
- `H` be the unique sampled hash set (dedup HashSet<u64> of sampled hashes; this is the input to HD encode)
- `N = |H|` (cardinality of H)
- `D = hd vector dimension`
- `chunk = d / 64` (which 64bit pseudorandom output controls coordinate d)
- `bit = d % 64` (bit position in 64 bit word; LSB)
- `word(h, chunk)` is the `next_u64()` output at position `chunk` in the WyRng stream created by `WyRng::seed_from_u64(h)`, where position `0` is the first `next_u64()` call.

For each coordinate `d`, CPU definition is as follows:

$$
hv[d] = \sum_{h \in H} \text{contribution}(h, d)
$$

$$
\text{contribution}(h, d) =
\begin{cases}
    +1 & \text{if } \text{bit}(\text{word}(h, \lfloor d / 64 \rfloor), d \bmod 64) = 1 \\
    -1 & \text{otherwise}
\end{cases}
$$

And is computed by:

```text
hv[d] = -N + 2 * ones(d)

ones(d) = count of hashes h where bit(word(h, d / 64), d % 64) == 1
```

For one coordinate d, every hash in H contributes either:

+1 if its random bit for coordinate d is 1
-1 if its random bit for coordinate d is 0

So if there are N hashes, hv[d] is the sum of N contributions.

hv[d] starts at -N
```text
for each hash:
    if bit is 1:
        hv[d] += 2
    else:
        do nothing
```
So if N = 5 and three hashes have a 1 bit at coordinate d, then\
hv[d] = -5 + 2 + 2 + 2 = 1

### WyRng Chunking

For each sampled hash h, CPU starts a fresh WyRng stream:
```text
rng = WyRng::seed_from_u64(hash)
```
Because each u64 has 64 bits, one next_u64() output control 64 hd vector coordinates. 
```text
coordinates 0..63     use chunk 0, the first  next_u64()
coordinates 64..127   use chunk 1, the second next_u64()
coordinates 128..191  use chunk 2, the third  next_u64()
```

Within a word, coordinate `chunk * 64 + bit` reads bit position `bit` from the LSB numbering by:

```rust
(rnd_bits >> j) & 1
```

Cuda Implementation has to reproduce the same psuedorandom WyRng word for the same `(hash, chunk)` pair and must use the same bit numbering. 

## GPU HD Encode (dotANI_jacob)

`src/sketch_cuda.rs` still performs GPU k-mer hashing first, and CPU still does ULL update and HashSet dedup (unchanged). \
However, deduplicated sampled hashes are put into a `Vec<u64>` and sent to GPU for HD encoding:

```rust
let sampled_hashes: Vec<u64> = sampled_hash_set.iter().copied().collect();
let start = Instant::now();
let (hv, hd_metrics) =
    hd_cuda::encode_hash_hd_cuda(&sampled_hashes, sketch.hv_d, &ctx, &module).unwrap();
metrics.hd_encode_ns = start.elapsed().as_nanos();
```

So new CPU/GPU contribution:

- GPU: fasta sequence buffer to kmer hashes
- CPU: zero filtering, ULL update, and `HashSet` deduplication
- GPU: sampled hashes to signed HD count vector
- CPU: norm, compression, output

NOT a full GPU rewrite, most aspects (ULL, deduplication, compression, dist, etc.) unchanged and still compatible with existing pipeline

## Cuda Host Wrapper

in `dotANI_jacob/src/hd_cuda.rs`:

```rust
pub fn encode_hash_hd_cuda(
    hashes: &[u64],
    hv_d: usize,
    ctx: &Arc<CudaContext>,
    module: &Arc<CudaModule>,
) -> Result<(Vec<i32>, GpuHdEncodeMetrics)> {
    if hv_d == 0 {
        bail!("hv_d must be greater than zero");
    }
    if hashes.len() > i32::MAX as usize {
        bail!("too many hashes for i32 HD count vector: {}", hashes.len());
    }
    if hashes.is_empty() {
        return Ok((vec![0; hv_d], GpuHdEncodeMetrics::default()));
    }

    let num_chunks = hv_d / 64;
    let mut metrics = GpuHdEncodeMetrics::default();
    let mut hv_host = vec![-(hashes.len() as i32); hv_d];

    if num_chunks == 0 {
        return Ok((hv_host, metrics));
    }

    let stream = ctx.default_stream();

    let mut d_hv = stream.alloc_zeros::<i32>(hv_d)?;
    let d_hashes = stream.clone_htod(hashes)?;
    stream.memcpy_htod(&hv_host, &mut d_hv)?;

    let function = module.load_function("cuda_hd_encode_counts_direct")?;
    let num_hashes = hashes.len() as i32;
    let hv_d_i32 = hv_d as i32;
    let cfg = LaunchConfig {
        grid_dim: (
            num_chunks as u32,
            hashes.len().div_ceil(HASH_TILE) as u32,
            1,
        ),
        block_dim: (HASH_TILE as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut launch = stream.launch_builder(&function);
    launch.arg(&d_hashes);
    launch.arg(&num_hashes);
    launch.arg(&hv_d_i32);
    launch.arg(&mut d_hv);

    unsafe {
        launch.launch(cfg)?;
    }

    hv_host = stream.clone_dtoh(&d_hv)?;
    Ok((hv_host, metrics))
}
```

And stage timing:

```rust
pub struct GpuHdEncodeMetrics {
    pub cuda_hd_alloc_ns: u128,
    pub cuda_hd_hash_h2d_ns: u128,
    pub cuda_hd_hv_h2d_ns: u128,
    pub cuda_hd_kernel_launch_ns: u128,
    pub cuda_hd_d2h_ns: u128,
}
```

## Direct WyRng Computation

CPU HD encoding uses `WyRng::seed_from_u64(hash)` and then calls `next_u64()` once per chunk (sequentially). A direct GPU port of this serial implementation wouldn't exactly be parallel and wouldn't be very useful, so the GPU code instead computes RNG output for specific chunks directly.

```cuda
static const uint64_t WY_P0 = UINT64_C(0xa0761d6478bd642f);
static const uint64_t WY_P1 = UINT64_C(0xe7037ed1a0b428db);

extern "C" __device__ __forceinline__ uint64_t wymum_u64(uint64_t a,
                                                          uint64_t b) {
  uint64_t high = __umul64hi(a, b);
  uint64_t low = a * b;
  return high ^ low;
}

extern "C" __device__ __forceinline__ uint64_t
wyrng_at_chunk(uint64_t hash, int chunk) {
  uint64_t state = hash + (((uint64_t)chunk + 1) * WY_P0);
  return wymum_u64(state ^ WY_P1, state);
}
```

We do not actually call the Rust WyRng function, rather just implement the same formula, producing the same result. This lets a CUDA thread compute 

```text
rnd = WyRng(seed = hash).nth_u64(chunk)
``` 

without actually walking through every chunk sequentially.

## Verification

CPU and GPU HD encoders compute the same equation, only difference is execution order. 

```text
initial hv[d] = -N
for each hash h:
    rnd = WyRng(seed = h).next_u64_for_chunk(d / 64)
    if bit(d % 64) in rnd is 1:
        hv[d] += 2
```

- CPU iterates through hashes, then chunks, then bits
- CUDA processes hashes and chunks in parallel across threads and blocks, then adds partial counts within each block

Reordering doesn't change anything because it's repeated integer additions to the same initialized vector.

### Unit Tests

The direct-seek WyRng logic is tested against sequential Rust `WyRng`:

```rust
fn direct_seek_wyrng(hash: u64, chunk: usize) -> u64 {
    let chunk_offset = (chunk as u64).wrapping_add(1).wrapping_mul(WY_P0);
    let state = hash.wrapping_add(chunk_offset);
    test_wymum(state ^ WY_P1, state)
}
```

The CUDA implementation is tested through `cuda_test_wyrng_at_chunk`:

```rust
let f = module.load_function("cuda_test_wyrng_at_chunk").unwrap();
let mut builder = stream.launch_builder(&f);
builder.arg(&gpu_hashes);
builder.arg(&gpu_chunks);
builder.arg(&mut gpu_out);
builder.arg(&n_outputs);
```

The CUDA HD output is tested against the CPU encoder:

```rust
let cpu_hv = hd::encode_hash_hd(&hash_set, &sketch);
let (gpu_hv, _metrics) =
    hd_cuda::encode_hash_hd_cuda(input_hashes, hv_d, &ctx, &module).unwrap();

assert_eq!(cpu_hv, gpu_hv);
assert_eq!(
    dist::compute_hv_l2_norm(&cpu_hv),
    dist::compute_hv_l2_norm(&gpu_hv)
);
```

Tested cases include:

- Empty hash inputs.
- `hv_d < 64`.
- Single representative hashes.
- Hash value `0`.
- Mixed representative hashes including `u64::MAX`.
- `hv_d = 1024`.
- `hv_d = 4096`.

### End to End Test

The end-to-end test used the full local GTDB `GCA/946` subset:

```text
gtdb_genomes/gtdb_genomes_reps_r220/database/GCA/946
1124 .fna.gz files
728M on disk
```

The CPU and CUDA commands used the same input and sketch parameters:

```sh
dotANI_jacob/target/release/dotani sketch --device cpu \
  -p gtdb_genomes/gtdb_genomes_reps_r220/database/GCA/946 \
  -o dotani_outputs/gpu_hd_correctness_2026-05-09/gtdb_gca946_full_cpu_hvd4096.sketch \
  -T 16 -d 4096 \
  --metrics-out dotani_outputs/gpu_hd_correctness_2026-05-09/gtdb_gca946_full_cpu_hvd4096

dotANI_jacob/target/release/dotani sketch --device cuda \
  -p gtdb_genomes/gtdb_genomes_reps_r220/database/GCA/946 \
  -o dotani_outputs/gpu_hd_correctness_2026-05-09/gtdb_gca946_full_cuda_hvd4096.sketch \
  -T 16 -d 4096 \
  --metrics-out dotani_outputs/gpu_hd_correctness_2026-05-09/gtdb_gca946_full_cuda_hvd4096
```

The resulting `.sketch` and `.ull` files were identical:

```text
Files gtdb_gca946_full_cpu_hvd4096.sketch and gtdb_gca946_full_cuda_hvd4096.sketch are identical
Files gtdb_gca946_full_cpu_hvd4096.sketch.ull and gtdb_gca946_full_cuda_hvd4096.sketch.ull are identical
```

SHA256 hash outputs match:

```text
bb3be0177f58c733c9ab13181ac646d97ec633211c37c21749dc0a47c2b44e48  gtdb_gca946_full_cpu_hvd4096.sketch
bb3be0177f58c733c9ab13181ac646d97ec633211c37c21749dc0a47c2b44e48  gtdb_gca946_full_cuda_hvd4096.sketch
43c3d96dedfa31befd90b3b7cd0f75122ffa0d3b441d8b5743f89223b3bae1e8  gtdb_gca946_full_cpu_hvd4096.sketch.ull
43c3d96dedfa31befd90b3b7cd0f75122ffa0d3b441d8b5743f89223b3bae1e8  gtdb_gca946_full_cuda_hvd4096.sketch.ull
```

Run logs reported:

```text
CPU:  Sketching 1124 files took 255.44s - Speed: 4.4 files/s
CUDA: Sketching 1124 files took 39.33s - Speed: 28.6 files/s
```

## HyperSpec

Inspired by HyperSpec's GPU strategy (parallel coordinate computations), but not a direct port.  

HyperSpec:

- Input is mass spectra peaks (m/z)
- Encoding uses level and ID hd vector
- Output is a packed binary hd vector

dotANI:

- Input is unique sampled genome kmer hashes
- Encoding uses each hash as the seed for deterministic pseudorandom
  coordinate contributions
- Output before compression is a signed `i32` count vector

