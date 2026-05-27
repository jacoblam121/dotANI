# Dist Optimization Experiments

## High Level

- Dist speedup of ~4.4x on Russell (single GPU, other GPUs were busy) and ~3.8x on local
- .ani outputs are preserved as an unordered row set; raw row order can change (will explain, but at a high level we are able to implement pipelining/scheduling on GPU which gave us a bulk of our speedup)
- GTDB ~113k validated with identical sorted SHA 256
- Accuracy test retained byte identical result (Claire's notebook)

| Machine | Baseline | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| Russell single GPU run 1 | `98.663s` | `22.434s` | `4.40x` |
| Russell single GPU run 2 | `101.666s` | `22.531s` | `4.51x` |
| Local (median of 3 runs) | `209.170s` | `54.913s` | `3.81x` |

## Timing

### Russell (single GPU)

Run on -T 128

Run 1:

| Phase | Wall (s) | `dist_total_s` | `stream_s` |
| --- | ---: | ---: | ---: |
| baseline | `1:44.58` | NA | NA |
| progress timing | `1:39.43` | `98.663` | `93.235` |
| stream breakdown | `1:39.93` | `99.170` | `93.764` |
| ref H2D cache | `1:34.36` | `93.622` | `88.040` |
| resident symmetric | `1:24.44` | `83.355` | `77.855` |
| pipeline postprocess | `0:23.50` | `22.434` | `16.964` |

Run 2:

| Phase | Wall (s) | `dist_total_s` | `stream_s` |
| --- | ---: | ---: | ---: |
| baseline | `1:43.78` | NA | NA |
| progress timing | `1:42.20` | `101.666` | `96.379` |
| stream breakdown | `1:40.72` | `100.208` | `94.871` |
| ref H2D cache | `1:34.58` | `93.858` | `88.261` |
| resident symmetric | `1:25.10` | `84.110` | `78.768` |
| pipeline postprocess | `0:23.11` | `22.531` | `16.859` |

vs baseline:

Run 1:
dist total speedup = 98.663 / 22.434 = 4.40x
stream speedup     = 93.235 / 16.964 = 5.50x
Run 2:
dist total speedup = 101.666 / 22.531 = 4.51x
stream speedup     = 96.379 / 16.859 = 5.72x

Stream here is the main ANI computation phase, where dist goes through comparison tiles, computes ANI, and writes output

### Russell (2x GPU)

Run on -T 128

| `dist_total_s` | `stream_s` |
| --- | ---: |
| `15.702s` | `10.122s` |

vs single gpu run:
stream speedup = 16.964 / 10.122 = 1.68x
total speedup  = 22.434 / 15.702 = 1.43x

### Local

Median of 3 runs:
| Label | Elapsed | `dist_total_s` | `stream_s` |
| --- | ---: | ---: | ---: |
| baseline | `203.000` | NA | NA |
| progress timing | `192.670` | `209.170` | `190.333` |
| stream breakdown | `192.640` | `208.646` | `190.907` |
| ref H2D cache | `175.790` | `189.076` | `170.642` |
| resident symmetric | `146.810` | `155.692` | `138.918` |
| pipeline postprocess | `51.160` | `54.913` | `37.184` |

vs baseline:
dist total speedup = 209.170 / 54.913 = 3.81x
stream speedup     = 190.333 / 37.184 = 5.12x


## Commits

```text
ddc26fa pre-dist-progress baseline/documentation organization
c548555 dist progress bar fix + timing
d044612 dist stream breakdown
1371c1d ref H2D cache
2ff3694 resident symmetric GPU matrix
51d08d6 pipeline GPU tile compute and CPU postprocess
(on exp branch) CUDA event timing split for GPU dist
```
- Note: CUDA event timing split for GPU dist currently part of the branch dist_experiments, not on main
- Explanations of each commit at the bottom, after timing

## Correctness

Optimized path should be judged by row identity and formatted ANI values, not by row order. .ani file is a set of pairwise results:

```text
reference_id<TAB>query_id<TAB>ani
```

We have to make sure 

```text
sort(old.ani) == sort(new.ani)
```

Pipelining changes when a tile's rows are written as output rows. It does not change the following core parts:

- dot product calculation
- ULL cardinality estimate
- ANI formula
- threshold behavior
- row format
- final set of rows that pass threshold in optimized path

Before postprocess pipelining, each tile (matrix tile is a block of pairwise comparisons) had a serial path: GPU dot product->CPU ANI/filter/format->output. \
After pipelining, this is separated: GPU workers can start later tiles while CPU workers format earlier ones.\
Because tiles now finish independently though, output order of rows can be different, but row set and the actual ANI values are unchanged.\
This is why the raw output is not byte identical anymore, but row order isn't actually important for the .ani result; when we sort the rows we get the same SHA256 hash.

### 100k GTDB Row-Set Validation

Rows were sorted and hashed with:
```text
LC_ALL=C sort -T "$OUT/sort_tmp" "$ani_file" | sha256sum
```

| Check | Baseline | Patch 5 |
| --- | ---: | ---: |
| Lines | `17429080` | `17429080` |
| Bytes | `3433528760` | `3433528760` |
| Raw byte-identical | yes | no |
| Sorted row-set match | yes | yes |
| Sorted SHA-256 | `97650baa2fcbfd3fbf5a1662e9cc6e64425fe6eef8183d1214f00ba2923bc1d0` | `97650baa2fcbfd3fbf5a1662e9cc6e64425fe6eef8183d1214f00ba2923bc1d0` |

### Accuracy Validation

We also need to make sure that the optimized output gets the same accuracy against our truth from BinDash\

```text
baseline lines=120
optimized lines=120
baseline bytes=18195
optimized bytes=18195
byte_identical=0
sorted baseline sha256=bf03596a21564e53d672bae40a79e880607c99b64fa4e58f943af0a4a64ad2ea
sorted optimized sha256=bf03596a21564e53d672bae40a79e880607c99b64fa4e58f943af0a4a64ad2ea
```

Recomputed accuracy was byte identical to the baseline

```text
pair_count=120
missing_pairs=0
extra_pairs=0
mean_absolute_error=0.0007131666667
max_absolute_error=0.00171
```

So our optimized dist is performance uplift only; output order changed but pair results and accuracy did not.



## Optimizations by Commit

Did work in phases; commit 6 is currently on experimental branch `dist_experiments`, not on main

| Commit | Purpose | Notes |
| --- | --- | --- |
| Baseline `ddc26fa` | Reference | Used as baseline |
| Commit 1 `c548555` | Progress and timing | Preserved math and output behavior |
| Commit 2 `d044612` | Stream breakdown | Attempted prefilter but was removed |
| Commit 3 `1371c1d` | Ref H2D cache | Byte identical to commit 2 fixed output |
| Commit 4 `2ff3694` | Resident symmetric matrix | Sorted row-set identical |
| Commit 5 `51d08d6` | Pipeline postprocess | Sorted row-set identical |
| Experimental Branch `d005f64` | CUDA event timing | Experimental branch only; not merged to `main` |
|

### Progress and Timing

- Needed to add stage timing to see where to optimize; this commit also fixes the problem of the progress bar not appearing until dist was nearly complete
- Overall, this first patch fixed progress/output handling so completed tile results were received, written, and counted while GPU workers were still running rather than after

Metrics:

- ULL load
- sketch load
- validation
- decompression
- compute/write
- total time
- `compute_hv_ani` stream time

Baselines:

| Env | `dist_total` | `stream` |
| --- | ---: | ---: |
| Russell single GPU | `98.663s` | `93.235s` |
| Local median | `209.170s` | `190.333s` |

### Stream Breakdown and Prefilter (reverted)

- Added GPU stream breakdown with more detailed stage metrics, including flattening, transfer, GPU tile work, postprocess, write, and wall time.
- Also attempted was a threshold prefilter, where the idea was to skip pairs that did not meet the threshold from ANI calculation. This didn't work though, and ended up dropping 90 rows on the default ANI threshold of 85.

After removal:

```text
candidates = processed pairs
prefilter_skipped = 0
final filter = existing ani >= ani_threshold
```

### Reference Host to Device (CPU -> GPU) Cache

- Reduced repeated host to deice uploads of reference tiles
- Reference tiles in this context: for pairwise comparison; sketch 1 x sketch 2 -> reference x query
- For a fixed reference tile, we compare to many query tiles, so we can use the same reference tile over again
- Previously, we transferred the same tile to the GPU repeatedly, now reference tile is only uploaded to GPU on cache miss

Added more stage timing metrics:

```text
query_h2d_ns
ref_h2d_ns
compute_d2h_ns
total_ns
query_h2d_bytes
ref_h2d_bytes
out_d2h_bytes
ref_upload_performed
```

Slight performance uplift, but is mainly useful for later optimizations

New bottleneck picture:

| Metric | Time (s) |
| --- | ---: |
| `ref_h2d` | `0.175s` |
| `query_h2d` | `20.5s` |
| `flatten_query` | `18-21s` |
| `compute_d2h` | `52s` |
| `postprocess` | `76-77s` |

### (GPU) Resident Symmetric Matrix

"Resident" here means data stays in GPU memory; no transfers
- Comparison in dist is self comparison, coming from the sketch file (reference sketch x query sketch)
- Thus the reference and query sketches are the same matrix, and dist is just filling in a large pairwise ANI matrix when comparing the two
- Because the matrix is symmetric, ANI(A, B) == ANI(B, A), so from matrix symmetry we know that we only have to compute the upper triangle of the matrix
- The lower triangle is duplicate work; diagonal is our self comparison

Previously, each tile prepared and transferred query blocks ("block" is a group of sketch rows/genomes), even though these query blocks come from the same sketch matrix as the reference

Optimization is as follows:

```text
Before:
  for each tile:
    prep reference block
    prep query block
    transfer blocks to GPU
    compute tile
```
```text
After:
  transfer full sketch matrix to GPU one time

  for each tile in upper triangle:
    use relevant row range from GPU matrix for the reference block
    use relevant row range from GPU matrix for the query block
    compute tile
```

Thus, we reduce repeated sketch query prep and CPU->GPU transfer time

| Metric | Before (s) | After (s) |
| --- | ---: | ---: |
| query flattening | 18-21s | 0s |
| query H2D upload | 20.5s | 0s |
| ref H2D upload | 0.175s | 0s |

This is still relatively conservative though, this path only runs on symmetric self comparisons, checks that there is enough GPU memory, and falls back to the previous tiled transfer path if this new resident transfer path can't be used

### Pipeline GPU and CPU Postprocess

Now, the visible bottlenecks are as follows: 

```text
compute_d2h ~53-55s
postprocess ~80-82s
write ~4-5s
```
Or by what's actually happening:
```text
GPU dot-product / GPU-to-CPU output copy: ~53-55s
CPU ANI/filter/format work: ~80-82s
Output writing: ~4-5s
```

No single stage is slow, rather we're wasting time since the stages are serial:
```text
for each tile:
    compute GPU dot products for tile
    copy results back to CPU
    convert dot products to ANI values
    filter and format output rows
    write results
```

This can be pipelined:

```text
Before:
    GPU work -> CPU formatting -> write -> next tile
```
```text
After:
    GPU work for tile N+1 can happen independently while CPU does work on tile N
```
The performance benefit comes from overlap:
- GPU computes dot products
- CPU workers convert completed tiles into ANI rows
- Completed output is written

This lets the GPU start computing later tiles while the CPU is working on earlier tiles. The ANI formula, threshold check, and output format did not change; only scheduling work changed. 

However, this does change raw output order. Since tiles now complete independently, rows can be written in a different order than the serial traversal. We can verify it's still correct by sorting the .ani rows and confirming the row sets are identical

Also addded backpressure metrics (see if one stage of the pipeline is slowing down an earlier stage)

| Metric | Means |
| --- | --- |
| `postprocess_workers` | # of CPU workers formatting GPU tile results |
| `gpu_send_blocked` | How long GPU workers waited because the CPU postprocess queue was full |
| `postprocess_worker_sum` | Total CPU postprocess time across workers |
| `postprocess_result_send_blocked` | How long CPU workers waited because the output writer queue was full |

Testing showed low `gpu_send_blocked`, meaning backpressure was not a main bottleneck: 

| Run | `gpu_send_blocked` |
| --- | ---: |
| Server single GPU | `0.042s` |
| Server two GPU | `0.989s` |
| Local median | `0.649s` |

In this pipeline commit, the `-T` flag in dist now sizes the CPU postprocess thread pool (can further explain this if needed)

```text
postprocess_workers = clamp(threads / 8, 2, 8), capped by total_jobs
```

So -T to postprocess workers are as follows: 

| `-T` | CUDA `dist` postprocess workers |
| ---: | ---: |
| `16` | `2` |
| `24` | `3` |
| `32` | `4` |
| `48` | `6` |
| `64` | `8` |
| `128` | `8` |

So even though local machine CPU only has 16 threads, we find the best peformance at `-T 48`. The GPU path is not using 48 CPU workers, rather it's using one GPU worker while 6 CPU workers deal with ANI/filter/format postprocess, plus the writer and setup work.

Note: if `gpu_send_blocked` is low, increasing `-T` or postprocess workers probably is not going to give much further speedup

## Notes

- No full dist run on 900k GTDB (~63.4x pair count of small GTDB)
- No 4x GPU run after pipelining, but single gpu is now ~1.8x faster than old 4x GPU run
- Row order is no longer identical, if this matters (shouldn't unless I am overlooking something)
