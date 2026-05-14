# CUDA Multi-GPU Lanes, Scratch Reuse, And Sort-Path Copy Removal

## High Level

- Multi GPU implementation for sketching (CPU workers to feed CUDA)
- Each lane processes one file, no batching
- Each lane has reusable Cuda and scratch across files
- `sort_unstable` faster dedup path (will explain), `hashset` still default
- `sort_unstable` + ULL avoids extra vector copy by sorting
  and deduplicating `full_hashes` after ULL consumes hash stream

Sketch output was not changed (test to make sure I didn't miss anything)

## Multi GPU; CPU Workers

First attempt at multi GPU scheduling tied each CPU worker to a single GPU. This was a large regression and had gpu util at about ~4-10%.\
Fix was to make scheduler unit a cpu worker itself:

```text
worker_count = min(-T, number_of_files), at least 1
```

So each CPU worker takes the next file index (dynamic):

```text
next_file.fetch_add(1, Ordering::Relaxed)
```

Metrics use `cuda_stream_lane` for worker id ("lane" and "worker" are the same thing; CPU side CUDA worker)\
With multiple GPU, each CPU worker is assigned to device evenly in repeating order. Essentially, many CPU workers are distributed across all visible GPUs

```text
device = visible_devices[worker_id % visible_devices.len()]
```

Output order is unchanged. Each worker returns the original input index, and results are stored by input index so they can be put in order after all workers finish. 

## Sort Unstable

In last meeting it was mentioned that `HashSet` could potentially be replaced; this is done so here through `sort_unstable`. I know this looks suspicious since it changes how unique hashes are contructed before HD encoding, so verify outputs and read this explanation. 

Previously the dedup was done with `HashSet<u64>`:

```text
for each hash:
    add hash to ULL
    insert hash into HashSet

sampled_hashes = unique values from HashSet
HD(sampled_hashes)
```

After kmer hashing, CPU gets a list of hashes:

```text
[hash1, hash2, hash3, hash4, ...]
```

That list is already stored in a contiguous `Vec<u64>` (hashes next to each other in memory)\

So how do we get the unique hashes for HD encoding? `HashSet` is the standard way: 

```text
for hash in hashes:
    set.insert(hash)
```

But this is expensive on the full workload because each file can have many hash table insertions, table growth, non contiguous memory access, etc. The `sort_unstable` option instead deduplicates the existing vector instead of wasting time building a separate `HashSet`. 

```rust
hashes.sort_unstable();
hashes.dedup();
```

`sort_unstable` (Rust function) sorts the vector in place; not preserving original order. This doesn't matter though since equivalent hashes are the same, and the next step is removing the duplicates. We don't need the original order of deduplicated hashes anwyay.\

Sorting groups equal hashes next toe ach other, and then `dedup()` removes the duplictes. \
So for example: 

```text
before sort:
[9, 2, 9, 5, 2, 7]

after sort:
[2, 2, 5, 7, 9, 9]

after dedup:
[2, 5, 7, 9]
```

This gives the same unique hash set as `HashSet`, but avoids building this separate hash table. This is all CPU side, not GPU, and does not change anything around the kmer hash algorithm. So this only changes the way the CPU makes unique hash list that is send to GPU for HD encoded.\

`sort_unstable` was added during the multi GPU implementation because using HashSet did not improve performance as much as I was hoping; CPU dedup was a major bottleneck. With more CPU workers feeding GPUs, the CPU has to prepare deduplicated hash lists quickly.
If dedup is slow, the GPUs can end up waiting for the CPU again.

Runs with `sort_unstable` matched older `HashSet` runs from my testing, but this could be reverified (in case I missed something).

`hashset` is still the default though: 

```sh
--cuda-dedup hashset
```

Must flag `sort_unstable` to use

```sh
--cuda-dedup sort_unstable
```

### Use of `sort_unstable`

In GPU path, ULL needs full hash stream and HD needs unique deduped hashes.\
`sort_unstable` is only used for the HD dedup input; it does not prevent ULL from seeing the full stream.

With ULL enabled, the order is as follows:

```text
full_hashes = CUDA k-mer hash output

for h in full_hashes:
    ull.add(h)

sort_unstable(full_hashes)
dedup(full_hashes)

HD(full_hashes as unique hashes)
```

# Will Update Later

## Scratch Reuse (per CPU Worker)
Extra overhead:
- Wasted allocations
- Build new dedup

CPU worker processes many files, so we can save information at the worker level instead of per file.\
Now each CPU worker has its own:
- CUDA context/module/stream
- Cached function handles
- Reusable vectors
- Reusable hashset if applicable
- Reusable buffers

## Copy Removal
- 