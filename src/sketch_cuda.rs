use crate::types::*;

#[cfg(feature = "cuda")]
use {
    crate::{dist, fastx_reader, hd, hd_cuda, utils},
    anyhow::{Result, anyhow},
    cudarc::{
        driver::{
            CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig,
            PushKernelArg,
        },
        nvrtc::Ptx,
    },
    glob::glob,
    log::{info, warn},
    rayon::prelude::*,
    std::collections::HashSet,
    std::path::{Path, PathBuf},
    std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    std::time::Instant,
    ultraloglog::UltraLogLog,
};

#[cfg(feature = "cuda")]
const CUDA_KERNEL_MY_STRUCT: &str = include_str!(concat!(env!("OUT_DIR"), "/cuda_kmer_hash.ptx"));

#[cfg(feature = "cuda")]
const CUDA_DEVICE_PROBE_LIMIT: usize = 32;

#[cfg(feature = "cuda")]
const SEQ_NT4_TABLE: [u8; 256] = [
    0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
];

#[cfg(feature = "cuda")]
struct IndexedSketchResult {
    index: usize,
    sketch: FileSketch,
    ull_record: Option<FileUllSketch>,
    metrics: FileSketchMetrics,
}

#[cfg(feature = "cuda")]
struct CudaSketchLaneScratch {
    lane_id: usize,
    dev_id: usize,
    _ctx: Arc<CudaContext>,
    _module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    kmer_fn: CudaFunction,
    hd_fn: CudaFunction,
    full_hashes: Vec<u64>,
    sampled_hashes: Vec<u64>,
    sampled_hash_set: HashSet<u64>,
    host_kmer_hash: Vec<u64>,
    hv_host: Vec<i32>,
    d_seq: Option<CudaSlice<u8>>,
    d_kmer_hash: Option<CudaSlice<u64>>,
    d_hd_hashes: Option<CudaSlice<u64>>,
    d_hv: Option<CudaSlice<i32>>,
}

#[cfg(feature = "cuda")]
impl CudaSketchLaneScratch {
    fn new(lane_id: usize, dev_id: usize) -> Result<Self> {
        let ctx = CudaContext::new(dev_id)?;
        let module = ctx.load_module(Ptx::from_src(CUDA_KERNEL_MY_STRUCT))?;
        let stream = ctx.default_stream();
        let kmer_fn = module.load_function("cuda_kmer_t1ha2")?;
        let hd_fn = module.load_function("cuda_hd_encode_counts_direct")?;

        Ok(Self {
            lane_id,
            dev_id,
            _ctx: ctx,
            _module: module,
            stream,
            kmer_fn,
            hd_fn,
            full_hashes: Vec::new(),
            sampled_hashes: Vec::new(),
            sampled_hash_set: HashSet::new(),
            host_kmer_hash: Vec::new(),
            hv_host: Vec::new(),
            d_seq: None,
            d_kmer_hash: None,
            d_hd_hashes: None,
            d_hv: None,
        })
    }

    fn ensure_seq_capacity(&mut self, needed: usize) -> Result<u128> {
        let start = Instant::now();
        if self.d_seq.as_ref().map_or(0, |buf| buf.len()) < needed {
            let capacity = grow_capacity(self.d_seq.as_ref().map_or(0, |buf| buf.len()), needed);
            self.d_seq = Some(unsafe { self.stream.alloc::<u8>(capacity)? });
        }
        Ok(start.elapsed().as_nanos())
    }

    fn ensure_kmer_hash_capacity(&mut self, needed: usize) -> Result<u128> {
        let start = Instant::now();
        if self.d_kmer_hash.as_ref().map_or(0, |buf| buf.len()) < needed {
            let capacity =
                grow_capacity(self.d_kmer_hash.as_ref().map_or(0, |buf| buf.len()), needed);
            self.d_kmer_hash = Some(unsafe { self.stream.alloc::<u64>(capacity)? });
        }
        Ok(start.elapsed().as_nanos())
    }

    fn ensure_hd_hash_capacity(&mut self, needed: usize) -> Result<u128> {
        let start = Instant::now();
        if self.d_hd_hashes.as_ref().map_or(0, |buf| buf.len()) < needed {
            let capacity =
                grow_capacity(self.d_hd_hashes.as_ref().map_or(0, |buf| buf.len()), needed);
            self.d_hd_hashes = Some(unsafe { self.stream.alloc::<u64>(capacity)? });
        }
        Ok(start.elapsed().as_nanos())
    }

    fn ensure_hv_capacity(&mut self, needed: usize) -> Result<u128> {
        let start = Instant::now();
        if self.d_hv.as_ref().map_or(0, |buf| buf.len()) < needed {
            let capacity = grow_capacity(self.d_hv.as_ref().map_or(0, |buf| buf.len()), needed);
            self.d_hv = Some(unsafe { self.stream.alloc::<i32>(capacity)? });
        }
        Ok(start.elapsed().as_nanos())
    }
}

#[cfg(feature = "cuda")]
fn grow_capacity(current: usize, needed: usize) -> usize {
    if needed == 0 {
        return current;
    }

    let mut capacity = current.max(1);
    while capacity < needed {
        capacity = capacity.saturating_mul(2);
        if capacity == usize::MAX {
            break;
        }
    }
    capacity
}

#[allow(unused_variables)]
#[cfg(not(feature = "cuda"))]
pub fn sketch_cuda(params: SketchParams) {
    use log::error;

    error!(
        "Cuda sketching is not supported. Please add `--features cuda-sketch` for installation to enable it."
    );
}

#[cfg(all(target_arch = "x86_64", feature = "cuda"))]
pub fn sketch_cuda(params: SketchParams) {
    let sketch_wall_start = Instant::now();
    let files = utils::get_fasta_files(&params.path);
    let n_file = files.len();

    info!("Start GPU sketching...");
    let pb = utils::get_progress_bar(n_file);

    if n_file == 0 {
        pb.finish_and_clear();
        let all_filesketch = Vec::<FileSketch>::new();
        let all_ullsketch = Vec::<FileUllSketch>::new();
        let all_metrics = Vec::<FileSketchMetrics>::new();

        info!("Sketching 0 files took {:.2}s - Speed: 0.0 files/s", 0.0);
        utils::dump_sketch(&all_filesketch, &params.out_file);

        if params.if_ull {
            utils::dump_ull_sketch(&all_ullsketch, &params.ull_out_file);
        }

        if let Some(prefix) = &params.metrics_out {
            utils::dump_sketch_metrics(
                &all_metrics,
                prefix,
                sketch_wall_start.elapsed().as_nanos(),
            );
        }

        return;
    }

    let device_ids =
        visible_cuda_device_ids().expect("Failed to find visible CUDA devices for GPU sketching");
    let lane_count = (params.threads as usize).max(1).min(n_file);
    info!(
        "Using {} GPU worker host lane(s) for sketching across {} usable CUDA device(s)",
        lane_count,
        device_ids.len()
    );
    info!(
        "Using CUDA dedup strategy: {}",
        params.cuda_dedup_strategy.as_str()
    );

    let next_file = Arc::new(AtomicUsize::new(0));
    let stop_workers = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<Result<IndexedSketchResult>>();
    let mut result_slots: Vec<Option<IndexedSketchResult>> = (0..n_file).map(|_| None).collect();
    let mut worker_error = None;

    std::thread::scope(|scope| {
        for lane_id in 0..lane_count {
            let dev_id = device_ids[lane_id % device_ids.len()];
            let files = &files;
            let params = &params;
            let next_file = Arc::clone(&next_file);
            let stop_workers = Arc::clone(&stop_workers);
            let tx = tx.clone();

            scope.spawn(move || {
                let worker = || -> Result<()> {
                    let mut scratch = CudaSketchLaneScratch::new(lane_id, dev_id)?;

                    loop {
                        if stop_workers.load(Ordering::Relaxed) {
                            break;
                        }

                        let index = next_file.fetch_add(1, Ordering::Relaxed);
                        if index >= files.len() {
                            break;
                        }

                        let result =
                            sketch_one_file_cuda(index, &files[index], params, &mut scratch)?;
                        if tx.send(Ok(result)).is_err() {
                            break;
                        }
                    }

                    Ok(())
                };

                if let Err(e) = worker() {
                    stop_workers.store(true, Ordering::Relaxed);
                    let _ = tx.send(Err(e));
                }
            });
        }

        drop(tx);

        let mut received = 0usize;
        while received < n_file {
            match rx.recv() {
                Ok(Ok(result)) => {
                    if let Err(e) = store_indexed_sketch_result(&mut result_slots, result) {
                        worker_error = Some(e);
                        stop_workers.store(true, Ordering::Relaxed);
                        break;
                    }

                    received += 1;
                    pb.inc(1);
                    pb.eta();
                }
                Ok(Err(e)) => {
                    worker_error = Some(e);
                    stop_workers.store(true, Ordering::Relaxed);
                    break;
                }
                Err(e) => {
                    worker_error = Some(anyhow!(
                        "CUDA sketch worker channel closed before all files finished: {}",
                        e
                    ));
                    stop_workers.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }
    });

    if let Some(e) = worker_error {
        panic!("Multi-GPU CUDA sketching failed: {e:?}");
    }

    pb.finish_and_clear();

    let results = ordered_indexed_sketch_results(result_slots);

    let all_filesketch: Vec<FileSketch> = results.iter().map(|r| r.sketch.clone()).collect();
    let all_ullsketch: Vec<FileUllSketch> = results
        .iter()
        .filter_map(|r| r.ull_record.clone())
        .collect();
    let all_metrics: Vec<FileSketchMetrics> = results.into_iter().map(|r| r.metrics).collect();

    info!(
        "Sketching {} files took {:.2}s - Speed: {:.1} files/s",
        files.len(),
        pb.elapsed().as_secs_f32(),
        pb.per_sec()
    );

    utils::dump_sketch(&all_filesketch, &params.out_file);

    if params.if_ull {
        utils::dump_ull_sketch(&all_ullsketch, &params.ull_out_file);
    }

    if let Some(prefix) = &params.metrics_out {
        utils::dump_sketch_metrics(&all_metrics, prefix, sketch_wall_start.elapsed().as_nanos());
    }
}

#[cfg(feature = "cuda")]
fn visible_cuda_device_ids() -> Result<Vec<usize>> {
    let mut device_ids = Vec::new();

    for dev_id in 0..CUDA_DEVICE_PROBE_LIMIT {
        match CudaContext::new(dev_id) {
            Ok(_) => device_ids.push(dev_id),
            Err(_) if dev_id > 0 => break,
            Err(first_err) => {
                return Err(match CudaContext::device_count() {
                    Ok(n) => anyhow!(
                        "Failed to open CUDA device 0 ({first_err:?}); CUDA device_count() reported {n} visible device(s)"
                    ),
                    Err(count_err) => anyhow!(
                        "Failed to open CUDA device 0 ({first_err:?}); CUDA device_count() also failed ({count_err:?})"
                    ),
                });
            }
        }
    }

    if device_ids.is_empty() {
        return Err(anyhow!("No CUDA devices are visible for GPU sketching"));
    }

    match CudaContext::device_count() {
        Ok(n) if n as usize != device_ids.len() => warn!(
            "CUDA ordinal probing found {} usable device(s), but device_count() reported {}",
            device_ids.len(),
            n
        ),
        Ok(_) => {}
        Err(e) => warn!(
            "CUDA device_count() failed ({e:?}); using {} device(s) found by ordinal probing",
            device_ids.len()
        ),
    }

    Ok(device_ids)
}

#[cfg(feature = "cuda")]
fn store_indexed_sketch_result(
    result_slots: &mut [Option<IndexedSketchResult>],
    result: IndexedSketchResult,
) -> Result<()> {
    let index = result.index;
    if index >= result_slots.len() {
        return Err(anyhow!(
            "CUDA sketch worker returned out-of-range file index {}",
            index
        ));
    }
    if result_slots[index].is_some() {
        return Err(anyhow!(
            "CUDA sketch worker returned duplicate file index {}",
            index
        ));
    }

    result_slots[index] = Some(result);
    Ok(())
}

#[cfg(feature = "cuda")]
fn ordered_indexed_sketch_results(
    result_slots: Vec<Option<IndexedSketchResult>>,
) -> Vec<IndexedSketchResult> {
    result_slots
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            result.unwrap_or_else(|| panic!("Missing CUDA sketch result for file index {index}"))
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn sketch_one_file_cuda(
    index: usize,
    file: &PathBuf,
    params: &SketchParams,
    scratch: &mut CudaSketchLaneScratch,
) -> Result<IndexedSketchResult> {
    let worker_start = Instant::now();
    let mut sketch = FileSketch {
        ksize: params.ksize,
        scaled: params.scaled,
        seed: params.seed,
        canonical: params.canonical,
        hv_d: params.hv_d,
        hv_quant_bits: 16u8,
        hv_norm_2: 0,
        file_str: file.display().to_string(),
        hv: Vec::<i32>::new(),
    };

    let mut metrics = extract_kmer_t1ha2_cuda_full_hashes_into(&sketch, scratch)?;
    metrics.file = sketch.file_str.clone();
    metrics.hashes_seen = scratch.full_hashes.len();

    let threshold = u64::MAX / sketch.scaled;

    let hash_and_dedup_start = Instant::now();
    scratch.sampled_hashes.clear();
    let ull_record = match params.cuda_dedup_strategy {
        CudaDedupStrategy::HashSet => {
            scratch.sampled_hash_set.clear();
            let ull_record = if params.if_ull {
                let mut ull =
                    UltraLogLog::new(params.ull_p).expect("Invalid UltraLogLog precision");

                for &h in &scratch.full_hashes {
                    ull.add(h);
                    scratch.sampled_hash_set.insert(h);
                }

                Some(FileUllSketch {
                    ksize: params.ksize,
                    canonical: params.canonical,
                    seed: params.seed,
                    ull_p: params.ull_p,
                    file_str: sketch.file_str.clone(),
                    ull_state: ull.get_state().to_vec(),
                })
            } else {
                for &h in &scratch.full_hashes {
                    if h < threshold {
                        scratch.sampled_hash_set.insert(h);
                    }
                }
                None
            };

            scratch
                .sampled_hashes
                .extend(scratch.sampled_hash_set.iter().copied());
            ull_record
        }
        CudaDedupStrategy::SortUnstable => {
            let ull_record = if params.if_ull {
                let mut ull =
                    UltraLogLog::new(params.ull_p).expect("Invalid UltraLogLog precision");

                for &h in &scratch.full_hashes {
                    ull.add(h);
                }

                scratch.sampled_hashes.extend(&scratch.full_hashes);
                Some(FileUllSketch {
                    ksize: params.ksize,
                    canonical: params.canonical,
                    seed: params.seed,
                    ull_p: params.ull_p,
                    file_str: sketch.file_str.clone(),
                    ull_state: ull.get_state().to_vec(),
                })
            } else {
                scratch.sampled_hashes.extend(
                    scratch
                        .full_hashes
                        .iter()
                        .copied()
                        .filter(|&h| h < threshold),
                );
                None
            };
            scratch.sampled_hashes.sort_unstable();
            scratch.sampled_hashes.dedup();

            ull_record
        }
    };
    let hash_and_dedup_ns = hash_and_dedup_start.elapsed().as_nanos();
    metrics.hash_and_dedup_ns = hash_and_dedup_ns;
    metrics.cuda_filter_ns = Some(hash_and_dedup_ns);
    metrics.unique_hashes = scratch.sampled_hashes.len();

    let start = Instant::now();
    let hd_metrics = encode_hash_hd_cuda_into(scratch, sketch.hv_d)?;
    metrics.hd_encode_ns = start.elapsed().as_nanos();
    if !scratch.sampled_hashes.is_empty() && sketch.hv_d >= 64 {
        metrics.cuda_hd_hash_h2d_ns = Some(hd_metrics.cuda_hd_hash_h2d_ns);
        metrics.cuda_hd_hv_h2d_ns = Some(hd_metrics.cuda_hd_hv_h2d_ns);
        metrics.cuda_hd_alloc_ns = Some(hd_metrics.cuda_hd_alloc_ns);
        metrics.cuda_hd_kernel_launch_ns = Some(hd_metrics.cuda_hd_kernel_launch_ns);
        metrics.cuda_hd_d2h_ns = Some(hd_metrics.cuda_hd_d2h_ns);
    }

    let start = Instant::now();
    let hv = &scratch.hv_host[..sketch.hv_d];
    sketch.hv_norm_2 = dist::compute_hv_l2_norm(hv);
    metrics.hv_norm_ns = start.elapsed().as_nanos();

    let start = Instant::now();
    if params.if_compressed {
        sketch.hv_quant_bits = unsafe { hd::compress_hd_sketch(&mut sketch, hv) };
    } else {
        sketch.hv = hv.to_vec();
    }
    metrics.hd_compress_ns = start.elapsed().as_nanos();
    metrics.total_worker_ns = worker_start.elapsed().as_nanos();

    Ok(IndexedSketchResult {
        index,
        sketch,
        ull_record,
        metrics,
    })
}

#[cfg(feature = "cuda")]
fn extract_kmer_t1ha2_cuda_full_hashes_into(
    sketch: &FileSketch,
    scratch: &mut CudaSketchLaneScratch,
) -> Result<FileSketchMetrics> {
    scratch.full_hashes.clear();

    let fna_file = PathBuf::from(sketch.file_str.clone());
    let fasta_start = Instant::now();
    let fna_seqs = fastx_reader::read_merge_seq(&fna_file);
    let fasta_ns = fasta_start.elapsed().as_nanos();

    let n_bps = fna_seqs.len();
    let mut metrics = FileSketchMetrics {
        input_bases: n_bps,
        fasta_ns,
        cuda_stream_lane: Some(scratch.lane_id),
        cuda_device_id: Some(scratch.dev_id),
        ..FileSketchMetrics::default()
    };
    let ksize = sketch.ksize as usize;
    let canonical = sketch.canonical;
    let seed = sketch.seed;

    if n_bps < ksize {
        return Ok(metrics);
    }

    let n_kmers = n_bps - ksize + 1;
    let kmer_per_thread = 512usize;
    let n_threads = n_kmers.div_ceil(kmer_per_thread);

    let n_hash_per_thread = kmer_per_thread;
    let n_hash_array = n_hash_per_thread * n_threads;
    let seq_alloc_ns = scratch.ensure_seq_capacity(n_bps)?;
    let hash_alloc_ns = scratch.ensure_kmer_hash_capacity(n_hash_array)?;
    metrics.cuda_alloc_ns = Some(seq_alloc_ns + hash_alloc_ns);

    let gpu_seq = scratch
        .d_seq
        .as_mut()
        .expect("sequence device buffer should be allocated");
    let gpu_kmer_hash = scratch
        .d_kmer_hash
        .as_mut()
        .expect("k-mer hash device buffer should be allocated");

    let h2d_start = Instant::now();
    scratch
        .stream
        .memcpy_htod(&fna_seqs, &mut gpu_seq.slice_mut(0..n_bps))?;
    metrics.cuda_h2d_ns = Some(h2d_start.elapsed().as_nanos());

    let zero_start = Instant::now();
    scratch
        .stream
        .memset_zeros(&mut gpu_kmer_hash.slice_mut(0..n_hash_array))?;
    let zero_ns = zero_start.elapsed().as_nanos();
    metrics.cuda_alloc_ns = Some(metrics.cuda_alloc_ns.unwrap_or(0) + zero_ns);

    let mut builder = scratch.stream.launch_builder(&scratch.kmer_fn);
    builder.arg(&*gpu_seq);
    builder.arg(&n_bps);
    builder.arg(&kmer_per_thread);
    builder.arg(&n_hash_per_thread);
    builder.arg(&ksize);

    let full_threshold = u64::MAX;
    builder.arg(&full_threshold);

    builder.arg(&seed);
    builder.arg(&canonical);
    builder.arg(&mut *gpu_kmer_hash);

    let launch_start = Instant::now();
    unsafe {
        builder.launch(LaunchConfig::for_num_elems(n_threads as u32))?;
    }
    metrics.cuda_launch_ns = Some(launch_start.elapsed().as_nanos());

    scratch.host_kmer_hash.resize(n_hash_array, 0);
    let d2h_start = Instant::now();
    scratch.stream.memcpy_dtoh(
        &gpu_kmer_hash.slice(0..n_hash_array),
        &mut scratch.host_kmer_hash[..n_hash_array],
    )?;
    metrics.cuda_d2h_ns = Some(d2h_start.elapsed().as_nanos());

    let filter_start = Instant::now();
    scratch.full_hashes.extend(
        scratch.host_kmer_hash[..n_hash_array]
            .iter()
            .copied()
            .filter(|&h| h != 0),
    );
    metrics.cuda_zero_filter_ns = Some(filter_start.elapsed().as_nanos());

    Ok(metrics)
}

#[cfg(feature = "cuda")]
fn encode_hash_hd_cuda_into(
    scratch: &mut CudaSketchLaneScratch,
    hv_d: usize,
) -> Result<hd_cuda::GpuHdEncodeMetrics> {
    if hv_d == 0 {
        return Err(anyhow!("hv_d must be greater than zero"));
    }
    if scratch.sampled_hashes.len() > i32::MAX as usize {
        return Err(anyhow!(
            "too many hashes for i32 HD count vector: {}",
            scratch.sampled_hashes.len()
        ));
    }

    assert_eq!(
        hd_cuda::HASH_TILE % 32,
        0,
        "HASH_TILE must be a whole number of warps"
    );
    assert!(
        hd_cuda::HASH_TILE / 32 <= hd_cuda::MAX_KERNEL_WARPS,
        "HASH_TILE exceeds cuda_hd_encode_counts_direct shared-memory layout"
    );

    let mut metrics = hd_cuda::GpuHdEncodeMetrics::default();
    scratch.hv_host.resize(hv_d, 0);

    if scratch.sampled_hashes.is_empty() {
        scratch.hv_host[..hv_d].fill(0);
        return Ok(metrics);
    }

    scratch.hv_host[..hv_d].fill(-(scratch.sampled_hashes.len() as i32));
    let num_chunks = hv_d / 64;
    if num_chunks == 0 {
        return Ok(metrics);
    }

    let hash_alloc_ns = scratch.ensure_hd_hash_capacity(scratch.sampled_hashes.len())?;
    let hv_alloc_ns = scratch.ensure_hv_capacity(hv_d)?;
    metrics.cuda_hd_alloc_ns = hash_alloc_ns + hv_alloc_ns;

    let d_hashes = scratch
        .d_hd_hashes
        .as_mut()
        .expect("HD hash device buffer should be allocated");
    let d_hv = scratch
        .d_hv
        .as_mut()
        .expect("HD vector device buffer should be allocated");

    let hash_h2d_start = Instant::now();
    scratch.stream.memcpy_htod(
        &scratch.sampled_hashes,
        &mut d_hashes.slice_mut(0..scratch.sampled_hashes.len()),
    )?;
    metrics.cuda_hd_hash_h2d_ns = hash_h2d_start.elapsed().as_nanos();

    let hv_h2d_start = Instant::now();
    scratch
        .stream
        .memcpy_htod(&scratch.hv_host[..hv_d], &mut d_hv.slice_mut(0..hv_d))?;
    metrics.cuda_hd_hv_h2d_ns = hv_h2d_start.elapsed().as_nanos();

    let num_hashes = scratch.sampled_hashes.len() as i32;
    let hv_d_i32 = hv_d as i32;
    let cfg = LaunchConfig {
        grid_dim: (
            num_chunks as u32,
            scratch.sampled_hashes.len().div_ceil(hd_cuda::HASH_TILE) as u32,
            1,
        ),
        block_dim: (hd_cuda::HASH_TILE as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut launch = scratch.stream.launch_builder(&scratch.hd_fn);
    launch.arg(&*d_hashes);
    launch.arg(&num_hashes);
    launch.arg(&hv_d_i32);
    launch.arg(&mut *d_hv);

    let kernel_launch_start = Instant::now();
    unsafe {
        launch.launch(cfg)?;
    }
    metrics.cuda_hd_kernel_launch_ns = kernel_launch_start.elapsed().as_nanos();

    let d2h_start = Instant::now();
    scratch
        .stream
        .memcpy_dtoh(&d_hv.slice(0..hv_d), &mut scratch.hv_host[..hv_d])?;
    metrics.cuda_hd_d2h_ns = d2h_start.elapsed().as_nanos();

    Ok(metrics)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    fn indexed_result(index: usize, file: &str) -> IndexedSketchResult {
        IndexedSketchResult {
            index,
            sketch: FileSketch {
                ksize: 21,
                scaled: 1,
                seed: 123,
                canonical: true,
                hv_d: 64,
                hv_quant_bits: 16,
                hv_norm_2: 0,
                file_str: file.to_string(),
                hv: Vec::new(),
            },
            ull_record: None,
            metrics: FileSketchMetrics {
                file: file.to_string(),
                cuda_device_id: Some(index),
                ..FileSketchMetrics::default()
            },
        }
    }

    #[test]
    fn indexed_results_assemble_in_input_order() {
        let mut slots: Vec<Option<IndexedSketchResult>> = (0..3).map(|_| None).collect();

        store_indexed_sketch_result(&mut slots, indexed_result(2, "c.fna")).unwrap();
        store_indexed_sketch_result(&mut slots, indexed_result(0, "a.fna")).unwrap();
        store_indexed_sketch_result(&mut slots, indexed_result(1, "b.fna")).unwrap();

        let ordered = ordered_indexed_sketch_results(slots);
        let files: Vec<&str> = ordered.iter().map(|r| r.sketch.file_str.as_str()).collect();
        assert_eq!(files, vec!["a.fna", "b.fna", "c.fna"]);
    }

    #[test]
    fn indexed_result_store_rejects_duplicates() {
        let mut slots: Vec<Option<IndexedSketchResult>> = (0..1).map(|_| None).collect();

        store_indexed_sketch_result(&mut slots, indexed_result(0, "a.fna")).unwrap();
        let err = store_indexed_sketch_result(&mut slots, indexed_result(0, "again.fna"))
            .expect_err("duplicate index should be rejected");

        assert!(err.to_string().contains("duplicate file index 0"));
    }
}

#[cfg(feature = "cuda")]
pub fn cuda_mmhash_bitpack_parallel(
    path_fna: &String,
    ksize: usize,
    canonical: bool,
    scaled: u64,
) -> Vec<HashSet<u64>> {
    let files = utils::get_fasta_files(&PathBuf::from(path_fna));
    let n_file = files.len();
    let pb = utils::get_progress_bar(n_file);

    let ctx = Arc::new(CudaContext::new(0).unwrap());
    let module = Arc::new(
        ctx.load_module(Ptx::from_src(CUDA_KERNEL_MY_STRUCT))
            .unwrap(),
    );

    let index_vec: Vec<usize> = (0..files.len()).collect();
    let sketch_kmer_sets: Vec<HashSet<u64>> = index_vec
        .par_iter()
        .map(|&i| {
            let fna_seqs = fastx_reader::read_merge_seq(&files[i]);

            let n_bps = fna_seqs.len();
            if n_bps < ksize {
                pb.inc(1);
                return HashSet::new();
            }

            let n_kmers = n_bps - ksize + 1;
            let bp_per_thread = 512usize;
            let n_threads = n_kmers.div_ceil(bp_per_thread);

            let stream = ctx.default_stream();
            let gpu_seq = stream.clone_htod(&fna_seqs).unwrap();
            let gpu_seq_nt4_table = stream.clone_htod(&SEQ_NT4_TABLE).unwrap();

            let n_hash_per_thread = bp_per_thread;
            let n_hash_array = n_hash_per_thread * n_threads;
            let mut gpu_kmer_bit_hash = stream.alloc_zeros::<u64>(n_hash_array).unwrap();

            let f = module.load_function("cuda_kmer_bit_pack_mmhash").unwrap();
            let mut builder = stream.launch_builder(&f);
            builder.arg(&gpu_seq);
            builder.arg(&n_bps);
            builder.arg(&bp_per_thread);
            builder.arg(&n_hash_per_thread);
            builder.arg(&ksize);

            let full_threshold = u64::MAX;
            builder.arg(&full_threshold);

            builder.arg(&canonical);
            builder.arg(&gpu_seq_nt4_table);
            builder.arg(&mut gpu_kmer_bit_hash);

            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n_threads as u32))
                    .unwrap();
            }

            let host_kmer_bit_hash = stream.clone_dtoh(&gpu_kmer_bit_hash).unwrap();
            let threshold = u64::MAX / scaled;

            pb.inc(1);
            host_kmer_bit_hash
                .into_iter()
                .filter(|&h| h != 0 && h < threshold)
                .collect()
        })
        .collect();

    pb.finish_and_clear();
    sketch_kmer_sets
}

#[cfg(feature = "cuda")]
pub fn cuda_t1ha2_hash_parallel(
    path_fna: &String,
    ksize: usize,
    canonical: bool,
    scaled: u64,
    seed: u64,
) -> Vec<HashSet<u64>> {
    let files: Vec<_> = glob(Path::new(path_fna).join("*.fna").to_str().unwrap())
        .expect("Failed to read glob pattern")
        .collect();

    let n_file = files.len();
    let pb = utils::get_progress_bar(n_file);

    let ctx = Arc::new(CudaContext::new(0).unwrap());
    let module = Arc::new(
        ctx.load_module(Ptx::from_src(CUDA_KERNEL_MY_STRUCT))
            .unwrap(),
    );

    let index_vec: Vec<usize> = (0..files.len()).collect();
    let sketch_kmer_sets: Vec<HashSet<u64>> = index_vec
        .par_iter()
        .map(|i| {
            let fna_seqs = fastx_reader::read_merge_seq(files[*i].as_ref().unwrap());

            let n_bps = fna_seqs.len();
            if n_bps < ksize {
                pb.inc(1);
                return HashSet::new();
            }

            let n_kmers = n_bps - ksize + 1;
            let kmer_per_thread = 512usize;
            let n_threads = n_kmers.div_ceil(kmer_per_thread);

            let stream = ctx.default_stream();
            let gpu_seq = stream.clone_htod(&fna_seqs).unwrap();

            let n_hash_per_thread = kmer_per_thread;
            let n_hash_array = n_hash_per_thread * n_threads;
            let mut gpu_kmer_hash = stream.alloc_zeros::<u64>(n_hash_array).unwrap();

            let f = module.load_function("cuda_kmer_t1ha2").unwrap();
            let mut builder = stream.launch_builder(&f);
            builder.arg(&gpu_seq);
            builder.arg(&n_bps);
            builder.arg(&kmer_per_thread);
            builder.arg(&n_hash_per_thread);
            builder.arg(&ksize);

            let full_threshold = u64::MAX;
            builder.arg(&full_threshold);

            builder.arg(&seed);
            builder.arg(&canonical);
            builder.arg(&mut gpu_kmer_hash);

            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n_threads as u32))
                    .unwrap();
            }

            let host_kmer_hash = stream.clone_dtoh(&gpu_kmer_hash).unwrap();
            let threshold = u64::MAX / scaled;

            pb.inc(1);
            host_kmer_hash
                .into_iter()
                .filter(|&h| h != 0 && h < threshold)
                .collect()
        })
        .collect();

    pb.finish_and_clear();
    sketch_kmer_sets
}
