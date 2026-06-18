use ultraloglog::UltraLogLog;

use crate::hd;
use crate::types::*;
use crate::utils;

use log::{info, warn};
use rayon::prelude::*;

#[cfg(feature = "cuda")]
use crossbeam_channel as channel;
use std::fs::File;
use std::io::{BufWriter, Write};
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(feature = "cuda")]
use crate::cuda_dot::{GpuDotExecutor, device_count};

pub fn dist(sketch_dist: &mut SketchDist) {
    let tstart = Instant::now();
    let if_sym = sketch_dist.path_ref_sketch == sketch_dist.path_query_sketch;

    let ull_load_start = Instant::now();
    let ref_ull_sketch = utils::load_ull_sketch(sketch_dist.path_ref_ull.as_path());
    let query_ull_sketch_storage = if if_sym {
        None
    } else {
        Some(utils::load_ull_sketch(sketch_dist.path_query_ull.as_path()))
    };
    let ull_load_secs = ull_load_start.elapsed().as_secs_f32();

    let sketch_load_start = Instant::now();
    let mut ref_file_sketch = utils::load_sketch(sketch_dist.path_ref_sketch.as_path());
    let mut query_file_sketch_storage = if if_sym {
        None
    } else {
        Some(utils::load_sketch(sketch_dist.path_query_sketch.as_path()))
    };
    let sketch_load_secs = sketch_load_start.elapsed().as_secs_f32();

    let validation_start = Instant::now();
    let ksize_ref;
    let hv_d_ref;
    {
        let query_ull_sketch: &[FileUllSketch] = query_ull_sketch_storage
            .as_deref()
            .unwrap_or(&ref_ull_sketch);
        let query_file_sketch: &[FileSketch] = query_file_sketch_storage
            .as_deref()
            .unwrap_or(&ref_file_sketch);

        assert_eq!(
            ref_file_sketch.len(),
            ref_ull_sketch.len(),
            "Ref HD and ULL sketch counts differ"
        );
        assert_eq!(
            query_file_sketch.len(),
            query_ull_sketch.len(),
            "Query HD and ULL sketch counts differ"
        );

        for i in 0..ref_file_sketch.len() {
            assert_eq!(
                ref_file_sketch[i].file_str, ref_ull_sketch[i].file_str,
                "Ref HD/ULL file order mismatch"
            );
        }
        for i in 0..query_file_sketch.len() {
            assert_eq!(
                query_file_sketch[i].file_str, query_ull_sketch[i].file_str,
                "Query HD/ULL file order mismatch"
            );
        }

        ksize_ref = ref_file_sketch[0].ksize;
        let ksize_query = query_file_sketch[0].ksize;
        assert_eq!(
            ksize_ref, ksize_query,
            "Ref and query sketches use different kmer sizes!"
        );

        hv_d_ref = ref_file_sketch[0].hv_d;
        let hv_d_query = query_file_sketch[0].hv_d;
        assert_eq!(
            hv_d_ref, hv_d_query,
            "Ref and query sketches use different HV dimensions!"
        );
    }
    let validation_secs = validation_start.elapsed().as_secs_f32();

    let decompress_start = Instant::now();
    hd::decompress_file_sketch(&mut ref_file_sketch);
    if let Some(query_file_sketch) = query_file_sketch_storage.as_mut() {
        hd::decompress_file_sketch(query_file_sketch);
    }
    let decompress_secs = decompress_start.elapsed().as_secs_f32();

    let query_ull_sketch: &[FileUllSketch] = query_ull_sketch_storage
        .as_deref()
        .unwrap_or(&ref_ull_sketch);
    let query_file_sketch: &[FileSketch] = query_file_sketch_storage
        .as_deref()
        .unwrap_or(&ref_file_sketch);

    let compute_start = Instant::now();
    compute_hv_ani(
        sketch_dist,
        &ref_file_sketch,
        query_file_sketch,
        &ref_ull_sketch,
        query_ull_sketch,
        ksize_ref,
        if_sym,
    );
    let compute_secs = compute_start.elapsed().as_secs_f32();

    info!(
        "dist timings: ull_load={:.3}s sketch_load={:.3}s validation={:.3}s decompress={:.3}s compute_write={:.3}s total={:.3}s refs={} queries={}",
        ull_load_secs,
        sketch_load_secs,
        validation_secs,
        decompress_secs,
        compute_secs,
        tstart.elapsed().as_secs_f32(),
        ref_file_sketch.len(),
        query_file_sketch.len()
    );
}

pub fn compute_hv_l2_norm(hv: &[i32]) -> i64 {
    hv.iter()
        .map(|&num| {
            let x = num as i64;
            x * x
        })
        .sum()
}

#[inline]
pub fn ull_cardinality_from_state(state: &[u8]) -> f64 {
    let ull = UltraLogLog::wrap(state.to_vec()).expect("Invalid UltraLogLog state");
    ull.get_distinct_count_estimate()
}

#[inline]
pub fn compute_pairwise_dot(r: &[i32], q: &[i32]) -> i64 {
    r.iter()
        .zip(q.iter())
        .map(|(x, y)| (*x as i64) * (*y as i64))
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn compute_pairwise_dot_avx2(r: &[i32], q: &[i32]) -> i64 {
    assert_eq!(r.len(), q.len());

    let len = r.len();
    let n8 = len / 8;

    let mut acc_even = _mm256_setzero_si256();
    let mut acc_odd = _mm256_setzero_si256();

    for i in 0..n8 {
        let base = i * 8;

        let vr = unsafe { _mm256_loadu_si256(r.as_ptr().add(base) as *const __m256i) };
        let vq = unsafe { _mm256_loadu_si256(q.as_ptr().add(base) as *const __m256i) };

        let prod_even = _mm256_mul_epi32(vr, vq);

        let vr_shift = _mm256_srli_epi64(vr, 32);
        let vq_shift = _mm256_srli_epi64(vq, 32);
        let prod_odd = _mm256_mul_epi32(vr_shift, vq_shift);

        acc_even = _mm256_add_epi64(acc_even, prod_even);
        acc_odd = _mm256_add_epi64(acc_odd, prod_odd);
    }

    let acc = _mm256_add_epi64(acc_even, acc_odd);
    let mut tmp = [0i64; 4];
    unsafe { _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc) };

    let mut sum = tmp.iter().sum::<i64>();

    for i in (n8 * 8)..len {
        sum += (r[i] as i64) * (q[i] as i64);
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn compute_pairwise_dot_avx512(r: &[i32], q: &[i32]) -> i64 {
    assert_eq!(r.len(), q.len());

    let len = r.len();
    let n16 = len / 16;

    let mut acc_even = _mm512_setzero_si512();
    let mut acc_odd = _mm512_setzero_si512();

    for i in 0..n16 {
        let base = i * 16;

        let vr = unsafe { _mm512_loadu_si512(r.as_ptr().add(base) as *const __m512i) };
        let vq = unsafe { _mm512_loadu_si512(q.as_ptr().add(base) as *const __m512i) };

        let prod_even = _mm512_mul_epi32(vr, vq);

        let vr_shift = _mm512_srli_epi64(vr, 32);
        let vq_shift = _mm512_srli_epi64(vq, 32);
        let prod_odd = _mm512_mul_epi32(vr_shift, vq_shift);

        acc_even = _mm512_add_epi64(acc_even, prod_even);
        acc_odd = _mm512_add_epi64(acc_odd, prod_odd);
    }

    let acc = _mm512_add_epi64(acc_even, acc_odd);
    let mut tmp = [0i64; 8];
    unsafe { _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, acc) };

    let mut sum = tmp.iter().sum::<i64>();

    for i in (n16 * 16)..len {
        sum += (r[i] as i64) * (q[i] as i64);
    }

    sum
}

#[inline]
pub fn ani_from_intersection_and_cardinalities(
    inter_hat: f64,
    card_r: f64,
    card_q: f64,
    ksize: u8,
) -> f32 {
    if inter_hat <= 0.0 {
        return 0.0;
    }

    let union_hat = card_r + card_q - inter_hat;
    if union_hat <= 0.0 {
        return 0.0;
    }

    let jaccard = inter_hat / union_hat;
    if !jaccard.is_finite() || jaccard <= 0.0 {
        return 0.0;
    }

    if jaccard > 1.0 {
        return 100.0;
    }

    let ani = (2.0 * jaccard as f32 / (1.0 + jaccard as f32)).powf(1.0 / ksize as f32);

    if ani.is_nan() {
        0.0
    } else {
        ani.clamp(0.0, 1.0) * 100.0
    }
}

#[inline]
fn compute_pairwise_dot_best(r: &[i32], q: &[i32]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { compute_pairwise_dot_avx512(r, q) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { compute_pairwise_dot_avx2(r, q) };
        }
    }

    compute_pairwise_dot(r, q)
}

#[cfg(feature = "cuda")]
fn flatten_hv_matrix(filesketch: &[FileSketch]) -> Vec<i32> {
    if filesketch.is_empty() {
        return Vec::new();
    }

    let hv_d = filesketch[0].hv_d;
    let mut out = Vec::with_capacity(filesketch.len() * hv_d);
    for fs in filesketch {
        out.extend_from_slice(&fs.hv);
    }
    out
}

fn stream_hv_ani_cpu(
    out_path: &std::path::Path,
    pb: &indicatif::ProgressBar,
    ref_filesketch: &[FileSketch],
    query_filesketch: &[FileSketch],
    ref_cards: &[f64],
    query_cards: &[f64],
    ksize: u8,
    if_symmetric: bool,
    ani_threshold: f32,
    output_mode: DistOutputMode,
) -> usize {
    const ROW_BLOCK: usize = 32;
    const FLUSH_BYTES: usize = 8 * 1024 * 1024;

    let writer = if output_mode == DistOutputMode::Rows {
        Some(Arc::new(Mutex::new(BufWriter::new(
            File::create(out_path).expect("Failed to create ANI output file"),
        ))))
    } else {
        None
    };
    let total_hits = AtomicUsize::new(0);
    let total_ani_evals = AtomicUsize::new(0);
    let total_nonpositive_skipped = AtomicUsize::new(0);

    let row_starts: Vec<usize> = (0..ref_filesketch.len()).step_by(ROW_BLOCK).collect();

    row_starts.into_par_iter().for_each(|i0| {
        let i1 = (i0 + ROW_BLOCK).min(ref_filesketch.len());

        let mut local_text = if output_mode == DistOutputMode::Rows {
            String::with_capacity(1 << 20)
        } else {
            String::new()
        };
        let mut local_hits = 0usize;
        let mut local_pairs_done = 0usize;
        let mut local_ani_evals = 0usize;
        let mut local_nonpositive_skipped = 0usize;

        for i in i0..i1 {
            let j_start = if if_symmetric { i + 1 } else { 0 };

            for j in j_start..query_filesketch.len() {
                let r = &ref_filesketch[i];
                let q = &query_filesketch[j];

                let dot = compute_pairwise_dot_best(&r.hv, &q.hv) as f64;
                let inter_hat = dot / r.hv_d as f64;
                if inter_hat <= 0.0 && ani_threshold > 0.0 {
                    local_nonpositive_skipped += 1;
                    local_pairs_done += 1;
                    continue;
                }

                local_ani_evals += 1;
                let ani = ani_from_intersection_and_cardinalities(
                    inter_hat,
                    ref_cards[i],
                    query_cards[j],
                    ksize,
                );

                if ani >= ani_threshold {
                    if output_mode == DistOutputMode::Rows {
                        use std::fmt::Write as _;
                        let _ = writeln!(
                            &mut local_text,
                            "{}\t{}\t{:.3}",
                            r.file_str, q.file_str, ani
                        );
                    }
                    local_hits += 1;
                }

                local_pairs_done += 1;

                if local_text.len() >= FLUSH_BYTES {
                    let mut guard = writer
                        .as_ref()
                        .expect("rows mode writer missing")
                        .lock()
                        .expect("ANI writer mutex poisoned");
                    guard
                        .write_all(local_text.as_bytes())
                        .expect("Failed to write ANI batch");
                    local_text.clear();
                }
            }
        }

        if !local_text.is_empty() {
            let mut guard = writer
                .as_ref()
                .expect("rows mode writer missing")
                .lock()
                .expect("ANI writer mutex poisoned");
            guard
                .write_all(local_text.as_bytes())
                .expect("Failed to write ANI batch");
        }

        total_hits.fetch_add(local_hits, Ordering::Relaxed);
        total_ani_evals.fetch_add(local_ani_evals, Ordering::Relaxed);
        total_nonpositive_skipped.fetch_add(local_nonpositive_skipped, Ordering::Relaxed);
        pb.inc(local_pairs_done as u64);
    });

    if let Some(writer) = writer {
        writer
            .lock()
            .expect("ANI writer mutex poisoned")
            .flush()
            .expect("Failed to flush ANI output");
    }

    info!(
        "cpu stream breakdown: hits={} ani_evals={} nonpositive_skipped={}",
        total_hits.load(Ordering::Relaxed),
        total_ani_evals.load(Ordering::Relaxed),
        total_nonpositive_skipped.load(Ordering::Relaxed)
    );

    total_hits.load(Ordering::Relaxed)
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct GpuTileJob {
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
}

#[cfg(feature = "cuda")]
struct DotTileBatch {
    job: GpuTileJob,
    nq: usize,
    nr: usize,
    tile_dots: Vec<i64>,
    num_pairs_done: usize,
    candidate_pairs: usize,
    prefilter_skipped: usize,
    ref_flatten_events: usize,
    flatten_ref_ns: u128,
    flatten_query_ns: u128,
    query_h2d_ns: u128,
    ref_h2d_ns: u128,
    compute_d2h_ns: u128,
    gpu_tile_total_ns: u128,
    query_h2d_bytes: usize,
    ref_h2d_bytes: usize,
    out_d2h_bytes: usize,
    ref_uploads: usize,
    resident_tiles: usize,
    resident_fallback_tiles: usize,
    resident_upload_ns: u128,
    resident_upload_bytes: usize,
}

#[cfg(feature = "cuda")]
struct TileBatchResult {
    text: String,
    num_hits: usize,
    num_pairs_done: usize,
    candidate_pairs: usize,
    prefilter_skipped: usize,
    ani_evals: usize,
    nonpositive_skipped: usize,
    text_bytes: usize,
    ref_flatten_events: usize,
    flatten_ref_ns: u128,
    flatten_query_ns: u128,
    query_h2d_ns: u128,
    ref_h2d_ns: u128,
    compute_d2h_ns: u128,
    gpu_tile_total_ns: u128,
    postprocess_ns: u128,
    query_h2d_bytes: usize,
    ref_h2d_bytes: usize,
    out_d2h_bytes: usize,
    ref_uploads: usize,
    resident_tiles: usize,
    resident_fallback_tiles: usize,
    resident_upload_ns: u128,
    resident_upload_bytes: usize,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct GpuStreamBreakdown {
    jobs: usize,
    pairs: usize,
    hits: usize,
    candidates: usize,
    prefilter_skipped: usize,
    ani_evals: usize,
    nonpositive_skipped: usize,
    output_bytes: usize,
    ref_flatten_events: usize,
    flatten_ref_ns: u128,
    flatten_query_ns: u128,
    query_h2d_ns: u128,
    ref_h2d_ns: u128,
    compute_d2h_ns: u128,
    gpu_tile_total_ns: u128,
    postprocess_ns: u128,
    gpu_send_blocked_ns: u128,
    postprocess_result_send_blocked_ns: u128,
    write_ns: u128,
    query_h2d_bytes: usize,
    ref_h2d_bytes: usize,
    out_d2h_bytes: usize,
    ref_uploads: usize,
    resident_tiles: usize,
    resident_fallback_tiles: usize,
    resident_flatten_ns: u128,
    resident_upload_ns: u128,
    resident_upload_bytes: usize,
}

#[cfg(feature = "cuda")]
impl GpuStreamBreakdown {
    fn add_batch(&mut self, batch: &TileBatchResult) {
        self.jobs += 1;
        self.pairs += batch.num_pairs_done;
        self.hits += batch.num_hits;
        self.candidates += batch.candidate_pairs;
        self.prefilter_skipped += batch.prefilter_skipped;
        self.ani_evals += batch.ani_evals;
        self.nonpositive_skipped += batch.nonpositive_skipped;
        self.output_bytes += batch.text_bytes;
        self.ref_flatten_events += batch.ref_flatten_events;
        self.flatten_ref_ns += batch.flatten_ref_ns;
        self.flatten_query_ns += batch.flatten_query_ns;
        self.query_h2d_ns += batch.query_h2d_ns;
        self.ref_h2d_ns += batch.ref_h2d_ns;
        self.compute_d2h_ns += batch.compute_d2h_ns;
        self.gpu_tile_total_ns += batch.gpu_tile_total_ns;
        self.postprocess_ns += batch.postprocess_ns;
        self.query_h2d_bytes += batch.query_h2d_bytes;
        self.ref_h2d_bytes += batch.ref_h2d_bytes;
        self.out_d2h_bytes += batch.out_d2h_bytes;
        self.ref_uploads += batch.ref_uploads;
        self.resident_tiles += batch.resident_tiles;
        self.resident_fallback_tiles += batch.resident_fallback_tiles;
        self.resident_upload_ns += batch.resident_upload_ns;
        self.resident_upload_bytes += batch.resident_upload_bytes;
    }
}

#[cfg(feature = "cuda")]
#[inline]
fn ns_to_secs(ns: u128) -> f64 {
    ns as f64 / 1_000_000_000.0
}

#[cfg(feature = "cuda")]
enum GpuPipelineMessage {
    Batch(anyhow::Result<TileBatchResult>),
}

#[cfg(feature = "cuda")]
fn postprocess_dot_tile_batch(
    batch: DotTileBatch,
    ref_filesketch: &[FileSketch],
    query_filesketch: &[FileSketch],
    ref_cards: &[f64],
    query_cards: &[f64],
    ksize: u8,
    if_symmetric: bool,
    ani_threshold: f32,
    output_mode: DistOutputMode,
) -> TileBatchResult {
    let postprocess_start = Instant::now();
    let mut text = String::new();
    let mut num_hits = 0usize;
    let mut num_pairs_done = 0usize;
    let mut ani_evals = 0usize;
    let mut nonpositive_skipped = 0usize;

    for q_local in 0..batch.nq {
        for r_local in 0..batch.nr {
            let i = batch.job.i0 + r_local;
            let j = batch.job.j0 + q_local;

            if if_symmetric && i >= j {
                continue;
            }

            num_pairs_done += 1;

            let dot = batch.tile_dots[q_local * batch.nr + r_local] as f64;
            let inter_hat = dot / ref_filesketch[i].hv_d as f64;
            if inter_hat <= 0.0 && ani_threshold > 0.0 {
                nonpositive_skipped += 1;
                continue;
            }

            ani_evals += 1;
            let ani = ani_from_intersection_and_cardinalities(
                inter_hat,
                ref_cards[i],
                query_cards[j],
                ksize,
            );

            if ani >= ani_threshold {
                if output_mode == DistOutputMode::Rows {
                    use std::fmt::Write as _;
                    let _ = writeln!(
                        &mut text,
                        "{}\t{}\t{:.3}",
                        ref_filesketch[i].file_str, query_filesketch[j].file_str, ani
                    );
                }
                num_hits += 1;
            }
        }
    }

    debug_assert!(batch.num_pairs_done >= num_pairs_done);
    debug_assert_eq!(batch.candidate_pairs, batch.num_pairs_done);
    let postprocess_ns = postprocess_start.elapsed().as_nanos();
    let text_bytes = text.len();

    TileBatchResult {
        text,
        num_hits,
        num_pairs_done,
        candidate_pairs: num_pairs_done,
        prefilter_skipped: batch.prefilter_skipped,
        ani_evals,
        nonpositive_skipped,
        text_bytes,
        ref_flatten_events: batch.ref_flatten_events,
        flatten_ref_ns: batch.flatten_ref_ns,
        flatten_query_ns: batch.flatten_query_ns,
        query_h2d_ns: batch.query_h2d_ns,
        ref_h2d_ns: batch.ref_h2d_ns,
        compute_d2h_ns: batch.compute_d2h_ns,
        gpu_tile_total_ns: batch.gpu_tile_total_ns,
        postprocess_ns,
        query_h2d_bytes: batch.query_h2d_bytes,
        ref_h2d_bytes: batch.ref_h2d_bytes,
        out_d2h_bytes: batch.out_d2h_bytes,
        ref_uploads: batch.ref_uploads,
        resident_tiles: batch.resident_tiles,
        resident_fallback_tiles: batch.resident_fallback_tiles,
        resident_upload_ns: batch.resident_upload_ns,
        resident_upload_bytes: batch.resident_upload_bytes,
    }
}

#[cfg(feature = "cuda")]
fn stream_hv_ani_gpu_multi(
    mut writer: Option<&mut BufWriter<File>>,
    pb: &indicatif::ProgressBar,
    ref_filesketch: &[FileSketch],
    query_filesketch: &[FileSketch],
    ref_cards: &[f64],
    query_cards: &[f64],
    ksize: u8,
    if_symmetric: bool,
    ani_threshold: f32,
    threads: usize,
    output_mode: DistOutputMode,
    resident_matrix_mode: ResidentMatrixMode,
) -> anyhow::Result<usize> {
    if ref_filesketch.is_empty() || query_filesketch.is_empty() {
        return Ok(0);
    }

    let hv_d = ref_filesketch[0].hv_d;
    let tile_ref = 512usize;
    let tile_query = 512usize;

    let ng = device_count()?.max(1);
    info!("Using {} GPU worker(s) for tiled dot-product", ng);
    let use_resident_matrix = if_symmetric && resident_matrix_mode == ResidentMatrixMode::Auto;
    let (resident_flat_host, resident_flatten_ns) = if use_resident_matrix {
        let flatten_start = Instant::now();
        let flat = flatten_hv_matrix(ref_filesketch);
        (Some(flat), flatten_start.elapsed().as_nanos())
    } else {
        (None, 0)
    };
    let resident_flat_host = resident_flat_host.as_deref();

    let mut jobs = Vec::<GpuTileJob>::new();
    for i0 in (0..ref_filesketch.len()).step_by(tile_ref) {
        let i1 = (i0 + tile_ref).min(ref_filesketch.len());
        let j0_start = if if_symmetric { i0 } else { 0 };

        for j0 in (j0_start..query_filesketch.len()).step_by(tile_query) {
            let j1 = (j0 + tile_query).min(query_filesketch.len());
            jobs.push(GpuTileJob { i0, i1, j0, j1 });
        }
    }

    let total_jobs = jobs.len();
    let postprocess_workers = threads.clamp(2, 128).min(total_jobs);
    let work_queue_capacity = postprocess_workers * 2;
    let result_queue_capacity = 64usize;
    info!(
        "Using {} postprocess worker(s) for tiled ANI formatting",
        postprocess_workers
    );

    let jobs = Arc::new(jobs);
    let next = Arc::new(AtomicUsize::new(0));
    let cancel = Arc::new(AtomicBool::new(false));
    let gpu_send_blocked_ns = Arc::new(AtomicU64::new(0));
    let postprocess_result_send_blocked_ns = Arc::new(AtomicU64::new(0));
    let (work_tx, work_rx) = channel::bounded::<DotTileBatch>(work_queue_capacity);
    let (result_tx, result_rx) = channel::bounded::<GpuPipelineMessage>(result_queue_capacity);

    std::thread::scope(|scope| -> anyhow::Result<usize> {
        for _ in 0..postprocess_workers {
            let work_rx = work_rx.clone();
            let result_tx = result_tx.clone();
            let cancel = Arc::clone(&cancel);
            let postprocess_result_send_blocked_ns =
                Arc::clone(&postprocess_result_send_blocked_ns);

            scope.spawn(move || {
                while let Ok(batch) = work_rx.recv() {
                    if cancel.load(Ordering::Relaxed) {
                        break;
                    }

                    let result = postprocess_dot_tile_batch(
                        batch,
                        ref_filesketch,
                        query_filesketch,
                        ref_cards,
                        query_cards,
                        ksize,
                        if_symmetric,
                        ani_threshold,
                        output_mode,
                    );

                    if cancel.load(Ordering::Relaxed) {
                        break;
                    }

                    let send_start = Instant::now();
                    if result_tx
                        .send(GpuPipelineMessage::Batch(Ok(result)))
                        .is_err()
                    {
                        break;
                    }
                    let blocked_ns = send_start.elapsed().as_nanos();
                    postprocess_result_send_blocked_ns
                        .fetch_add(blocked_ns.min(u64::MAX as u128) as u64, Ordering::Relaxed);
                }
            });
        }

        for dev_id in 0..ng {
            let work_tx = work_tx.clone();
            let result_tx = result_tx.clone();
            let jobs = Arc::clone(&jobs);
            let next = Arc::clone(&next);
            let cancel = Arc::clone(&cancel);
            let gpu_send_blocked_ns = Arc::clone(&gpu_send_blocked_ns);
            let resident_flat_host = resident_flat_host;

            scope.spawn(move || {
                let worker = || -> anyhow::Result<()> {
                    let mut gpu = GpuDotExecutor::new(dev_id)?;
                    let matrix_bytes = resident_flat_host
                        .map(std::mem::size_of_val)
                        .unwrap_or(0);
                    let max_tile_out_bytes =
                        tile_query * tile_ref * std::mem::size_of::<i64>();
                    let safety_bytes = 128usize * 1024 * 1024;
                    let mut resident_upload_ns_pending = 0u128;
                    let mut resident_upload_bytes_pending = 0usize;
                    let resident_matrix = if let Some(flat) = resident_flat_host {
                        match gpu.free_memory_bytes() {
                            Ok(free_vram)
                                if free_vram
                                    > matrix_bytes + max_tile_out_bytes + safety_bytes =>
                            {
                                let upload_start = Instant::now();
                                match gpu.upload_resident_matrix(flat, ref_filesketch.len(), hv_d) {
                                    Ok(matrix) => {
                                        resident_upload_ns_pending =
                                            upload_start.elapsed().as_nanos();
                                        resident_upload_bytes_pending = matrix_bytes;
                                        Some(matrix)
                                    }
                                    Err(e) => {
                                        warn!(
                                            "GPU worker {} resident matrix upload failed, falling back to tiled path: {e:?}",
                                            dev_id
                                        );
                                        None
                                    }
                                }
                            }
                            Ok(free_vram) => {
                                warn!(
                                    "GPU worker {} free VRAM insufficient for resident symmetric matrix, falling back to tiled path: free_mb={:.1} required_mb={:.1}",
                                    dev_id,
                                    free_vram as f64 / (1024.0 * 1024.0),
                                    (matrix_bytes + max_tile_out_bytes + safety_bytes) as f64
                                        / (1024.0 * 1024.0)
                                );
                                None
                            }
                            Err(e) => {
                                warn!(
                                    "GPU worker {} failed to query free VRAM, falling back to tiled path: {e:?}",
                                    dev_id
                                );
                                None
                            }
                        }
                    } else {
                        None
                    };

                    let mut cached_i0 = usize::MAX;
                    let mut cached_i1 = usize::MAX;
                    let mut cached_ref_flat = Vec::<i32>::new();
                    let mut cached_nr = 0usize;

                    loop {
                        if cancel.load(Ordering::Relaxed) {
                            break;
                        }

                        let job_idx = next.fetch_add(1, Ordering::Relaxed);
                        if job_idx >= jobs.len() {
                            break;
                        }

                        let job = jobs[job_idx];

                        let query_block = &query_filesketch[job.j0..job.j1];
                        let nq = query_block.len();
                        let nr = job.i1 - job.i0;
                        let mut flatten_ref_ns = 0u128;
                        let mut flatten_query_ns = 0u128;
                        let mut ref_flatten_events = 0usize;
                        let mut resident_tiles = 0usize;
                        let mut resident_fallback_tiles = 0usize;

                        let mut tile_dots = vec![0i64; nq * nr];
                        let gpu_timings = if let Some(resident) = resident_matrix.as_ref() {
                            resident_tiles = 1;
                            gpu.compute_tile_resident(
                                resident,
                                job.j0,
                                nq,
                                resident,
                                job.i0,
                                nr,
                                &mut tile_dots,
                            )?
                        } else {
                            resident_fallback_tiles = usize::from(if_symmetric);
                            if job.i0 != cached_i0 || job.i1 != cached_i1 {
                                cached_i0 = job.i0;
                                cached_i1 = job.i1;

                                let ref_block = &ref_filesketch[job.i0..job.i1];
                                cached_nr = ref_block.len();
                                let flatten_ref_start = Instant::now();
                                cached_ref_flat = flatten_hv_matrix(ref_block);
                                flatten_ref_ns = flatten_ref_start.elapsed().as_nanos();
                                ref_flatten_events = 1;
                            }

                            let flatten_query_start = Instant::now();
                            let query_flat = flatten_hv_matrix(query_block);
                            flatten_query_ns = flatten_query_start.elapsed().as_nanos();

                            gpu.compute_tile(
                                &query_flat,
                                nq,
                                &cached_ref_flat,
                                cached_nr,
                                hv_d,
                                &mut tile_dots,
                                ref_flatten_events > 0,
                            )?
                        };

                        if cancel.load(Ordering::Relaxed) {
                            break;
                        }

                        let num_pairs_done = if if_symmetric {
                            let mut count = 0usize;
                            for q_local in 0..nq {
                                for r_local in 0..nr {
                                    if job.i0 + r_local < job.j0 + q_local {
                                        count += 1;
                                    }
                                }
                            }
                            count
                        } else {
                            nq * nr
                        };

                        let batch = DotTileBatch {
                            job,
                            nq,
                            nr,
                            tile_dots,
                            num_pairs_done,
                            candidate_pairs: num_pairs_done,
                            prefilter_skipped: 0,
                            ref_flatten_events,
                            flatten_ref_ns,
                            flatten_query_ns,
                            query_h2d_ns: gpu_timings.query_h2d_ns,
                            ref_h2d_ns: gpu_timings.ref_h2d_ns,
                            compute_d2h_ns: gpu_timings.compute_d2h_ns,
                            gpu_tile_total_ns: gpu_timings.total_ns,
                            query_h2d_bytes: gpu_timings.query_h2d_bytes,
                            ref_h2d_bytes: gpu_timings.ref_h2d_bytes,
                            out_d2h_bytes: gpu_timings.out_d2h_bytes,
                            ref_uploads: usize::from(gpu_timings.ref_upload_performed),
                            resident_tiles,
                            resident_fallback_tiles,
                            resident_upload_ns: std::mem::take(&mut resident_upload_ns_pending),
                            resident_upload_bytes: std::mem::take(
                                &mut resident_upload_bytes_pending,
                            ),
                        };

                        let send_start = Instant::now();
                        match work_tx.send(batch) {
                            Ok(()) => {
                                let blocked_ns = send_start.elapsed().as_nanos();
                                gpu_send_blocked_ns.fetch_add(
                                    blocked_ns.min(u64::MAX as u128) as u64,
                                    Ordering::Relaxed,
                                );
                            }
                            Err(_) if cancel.load(Ordering::Relaxed) => break,
                            Err(e) => {
                                return Err(anyhow::anyhow!(
                                    "postprocess work queue closed unexpectedly: {e}"
                                ));
                            }
                        }
                    }

                    Ok(())
                };

                if let Err(e) = worker() {
                    cancel.store(true, Ordering::Relaxed);
                    let _ = result_tx.send(GpuPipelineMessage::Batch(Err(e)));
                }
            });
        }

        drop(work_tx);
        drop(result_tx);

        let mut total_hits = 0usize;
        let mut received_jobs = 0usize;
        let mut first_error = None;
        let stream_wall_start = Instant::now();
        let mut breakdown = GpuStreamBreakdown::default();
        breakdown.resident_flatten_ns = resident_flatten_ns;

        while let Ok(message) = result_rx.recv() {
            match message {
                GpuPipelineMessage::Batch(Ok(batch)) => {
                    received_jobs += 1;
                    let write_start = Instant::now();
                    if let Some(writer) = writer.as_deref_mut() {
                        writer
                            .write_all(batch.text.as_bytes())
                            .expect("Failed to write ANI batch");
                    }
                    breakdown.write_ns += write_start.elapsed().as_nanos();
                    total_hits += batch.num_hits;
                    pb.inc(batch.num_pairs_done as u64);
                    breakdown.add_batch(&batch);
                }
                GpuPipelineMessage::Batch(Err(e)) => {
                    cancel.store(true, Ordering::Relaxed);
                    first_error.get_or_insert(e);
                }
            }
        }
        breakdown.gpu_send_blocked_ns = gpu_send_blocked_ns.load(Ordering::Relaxed) as u128;
        breakdown.postprocess_result_send_blocked_ns =
            postprocess_result_send_blocked_ns.load(Ordering::Relaxed) as u128;

        if received_jobs < total_jobs && first_error.is_none() {
            first_error = Some(anyhow::anyhow!(
                "GPU pipeline closed before all tile results were written: received_jobs={} total_jobs={}",
                received_jobs,
                total_jobs
            ));
        }

        if first_error.is_some() {
            cancel.store(true, Ordering::Relaxed);
        }

        if let Some(e) = first_error {
            Err(e)
        } else {
            // Worker timings are aggregate worker-sums; postprocess can exceed wall after pipelining.
            let resident_mode = if !if_symmetric {
                "disabled"
            } else if resident_matrix_mode == ResidentMatrixMode::Off {
                "off"
            } else if breakdown.resident_tiles > 0 && breakdown.resident_fallback_tiles == 0 {
                "symmetric"
            } else {
                "fallback"
            };
            info!(
                "gpu stream breakdown: jobs={} pairs={} hits={} candidates={} prefilter_skipped={} ani_evals={} nonpositive_skipped={} resident_mode={} postprocess_workers={} output_mb={:.3} ref_flatten_events={} ref_uploads={} resident_upload_mb={:.3} query_h2d_mb={:.3} ref_h2d_mb={:.3} out_d2h_mb={:.3} resident_flatten={:.3}s resident_upload={:.3}s flatten_ref_cache_miss={:.3}s flatten_query={:.3}s query_h2d={:.3}s ref_h2d={:.3}s compute_d2h={:.3}s gpu_tile_total={:.3}s gpu_send_blocked={:.3}s postprocess_worker_sum={:.3}s postprocess_result_send_blocked={:.3}s write={:.3}s wall={:.3}s",
                breakdown.jobs,
                breakdown.pairs,
                breakdown.hits,
                breakdown.candidates,
                breakdown.prefilter_skipped,
                breakdown.ani_evals,
                breakdown.nonpositive_skipped,
                resident_mode,
                postprocess_workers,
                breakdown.output_bytes as f64 / (1024.0 * 1024.0),
                breakdown.ref_flatten_events,
                breakdown.ref_uploads,
                breakdown.resident_upload_bytes as f64 / (1024.0 * 1024.0),
                breakdown.query_h2d_bytes as f64 / (1024.0 * 1024.0),
                breakdown.ref_h2d_bytes as f64 / (1024.0 * 1024.0),
                breakdown.out_d2h_bytes as f64 / (1024.0 * 1024.0),
                ns_to_secs(breakdown.resident_flatten_ns),
                ns_to_secs(breakdown.resident_upload_ns),
                ns_to_secs(breakdown.flatten_ref_ns),
                ns_to_secs(breakdown.flatten_query_ns),
                ns_to_secs(breakdown.query_h2d_ns),
                ns_to_secs(breakdown.ref_h2d_ns),
                ns_to_secs(breakdown.compute_d2h_ns),
                ns_to_secs(breakdown.gpu_tile_total_ns),
                ns_to_secs(breakdown.gpu_send_blocked_ns),
                ns_to_secs(breakdown.postprocess_ns),
                ns_to_secs(breakdown.postprocess_result_send_blocked_ns),
                ns_to_secs(breakdown.write_ns),
                stream_wall_start.elapsed().as_secs_f64(),
            );
            Ok(total_hits)
        }
    })
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use super::stream_hv_ani_gpu_multi;
    use super::{ani_from_intersection_and_cardinalities, stream_hv_ani_cpu};
    #[cfg(feature = "cuda")]
    use crate::types::ResidentMatrixMode;
    use crate::types::{DistOutputMode, FileSketch};
    use std::collections::HashSet;
    #[cfg(feature = "cuda")]
    use std::fs::File;
    #[cfg(feature = "cuda")]
    use std::io::{BufWriter, Write};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn ani_clamps_estimated_jaccard_overshoot_to_100() {
        let ani = ani_from_intersection_and_cardinalities(120.0, 100.0, 100.0, 16);
        assert!(
            (ani - 100.0).abs() < f32::EPSILON,
            "expected overshot Jaccard estimate to produce 100 ANI, got {ani}"
        );
    }

    fn temp_ani_path(label: &str) -> PathBuf {
        let id = TEST_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("dotani_{label}_{}_{}.ani", std::process::id(), id))
    }

    fn test_filesketch(file_str: &str, hv: Vec<i32>) -> FileSketch {
        FileSketch {
            ksize: 1,
            scaled: 1,
            canonical: true,
            seed: 123,
            hv_d: hv.len(),
            hv_quant_bits: 0,
            hv_norm_2: 0,
            file_str: file_str.to_string(),
            hv,
        }
    }

    fn read_rows_as_set(path: &std::path::Path) -> HashSet<String> {
        std::fs::read_to_string(path)
            .expect("failed to read ANI output")
            .lines()
            .map(str::to_owned)
            .collect()
    }

    #[cfg(feature = "cuda")]
    fn test_dot_tile_batch(
        job: super::GpuTileJob,
        nq: usize,
        nr: usize,
        dots: Vec<i64>,
    ) -> super::DotTileBatch {
        super::DotTileBatch {
            job,
            nq,
            nr,
            tile_dots: dots,
            num_pairs_done: nq * nr,
            candidate_pairs: nq * nr,
            prefilter_skipped: 0,
            ref_flatten_events: 0,
            flatten_ref_ns: 0,
            flatten_query_ns: 0,
            query_h2d_ns: 0,
            ref_h2d_ns: 0,
            compute_d2h_ns: 0,
            gpu_tile_total_ns: 0,
            query_h2d_bytes: 0,
            ref_h2d_bytes: 0,
            out_d2h_bytes: 0,
            ref_uploads: 0,
            resident_tiles: 0,
            resident_fallback_tiles: 0,
            resident_upload_ns: 0,
            resident_upload_bytes: 0,
        }
    }

    #[test]
    fn cpu_stream_ani_threshold_zero_keeps_zero_ani_rows() {
        let refs = vec![
            test_filesketch("r0", vec![4, 0, 0, 0]),
            test_filesketch("r1", vec![0, 4, 0, 0]),
        ];
        let queries = vec![
            test_filesketch("q0", vec![90, 0, 0, 0]),
            test_filesketch("q1", vec![0, 90, 0, 0]),
        ];
        let cards = vec![100.0, 100.0];
        let out = temp_ani_path("threshold_zero");
        let pb = indicatif::ProgressBar::hidden();

        let hits = stream_hv_ani_cpu(
            &out,
            &pb,
            &refs,
            &queries,
            &cards,
            &cards,
            1,
            false,
            0.0,
            DistOutputMode::Rows,
        );
        assert_eq!(hits, 4);
        assert_eq!(
            read_rows_as_set(&out),
            HashSet::from([
                "r0\tq0\t90.000".to_string(),
                "r0\tq1\t0.000".to_string(),
                "r1\tq0\t0.000".to_string(),
                "r1\tq1\t90.000".to_string(),
            ])
        );

        let _ = std::fs::remove_file(out);
    }

    #[test]
    fn cpu_stream_normal_threshold_matches_expected_rows_as_set() {
        let refs = vec![
            test_filesketch("r0", vec![4, 0, 0, 0]),
            test_filesketch("r1", vec![0, 4, 0, 0]),
        ];
        let queries = vec![
            test_filesketch("q0", vec![90, 0, 0, 0]),
            test_filesketch("q1", vec![0, 90, 0, 0]),
        ];
        let cards = vec![100.0, 100.0];
        let out = temp_ani_path("threshold_85");
        let pb = indicatif::ProgressBar::hidden();

        let hits = stream_hv_ani_cpu(
            &out,
            &pb,
            &refs,
            &queries,
            &cards,
            &cards,
            1,
            false,
            85.0,
            DistOutputMode::Rows,
        );
        assert_eq!(hits, 2);
        assert_eq!(
            read_rows_as_set(&out),
            HashSet::from(["r0\tq0\t90.000".to_string(), "r1\tq1\t90.000".to_string(),])
        );

        let _ = std::fs::remove_file(out);
    }

    #[test]
    fn cpu_stream_count_mode_matches_rows_hit_count_without_rows_file() {
        let refs = vec![
            test_filesketch("r0", vec![4, 0, 0, 0]),
            test_filesketch("r1", vec![0, 4, 0, 0]),
        ];
        let queries = vec![
            test_filesketch("q0", vec![90, 0, 0, 0]),
            test_filesketch("q1", vec![0, 90, 0, 0]),
        ];
        let cards = vec![100.0, 100.0];
        let rows_out = temp_ani_path("rows_count_compare");
        let count_out = temp_ani_path("count_no_rows");
        let _ = std::fs::remove_file(&count_out);
        let pb = indicatif::ProgressBar::hidden();

        let row_hits = stream_hv_ani_cpu(
            &rows_out,
            &pb,
            &refs,
            &queries,
            &cards,
            &cards,
            1,
            false,
            85.0,
            DistOutputMode::Rows,
        );
        let row_count = std::fs::read_to_string(&rows_out)
            .expect("failed to read rows output")
            .lines()
            .count();
        let count_hits = stream_hv_ani_cpu(
            &count_out,
            &pb,
            &refs,
            &queries,
            &cards,
            &cards,
            1,
            false,
            85.0,
            DistOutputMode::Count,
        );

        assert_eq!(row_hits, 2);
        assert_eq!(row_count, row_hits);
        assert_eq!(count_hits, row_hits);
        assert!(!count_out.exists());

        let _ = std::fs::remove_file(rows_out);
    }

    #[test]
    fn cpu_stream_symmetric_rows_preserve_upper_triangle_row_set() {
        let sketches = vec![
            test_filesketch("s0", vec![4, 0, 0, 0]),
            test_filesketch("s1", vec![0, 4, 0, 0]),
            test_filesketch("s2", vec![4, 0, 0, 0]),
        ];
        let cards = vec![100.0, 100.0, 100.0];
        let out = temp_ani_path("cpu_symmetric_rows");
        let pb = indicatif::ProgressBar::hidden();

        let hits = stream_hv_ani_cpu(
            &out,
            &pb,
            &sketches,
            &sketches,
            &cards,
            &cards,
            1,
            true,
            0.0,
            DistOutputMode::Rows,
        );

        assert_eq!(hits, 3);
        assert_eq!(
            read_rows_as_set(&out),
            HashSet::from([
                "s0\ts1\t0.000".to_string(),
                "s0\ts2\t4.000".to_string(),
                "s1\ts2\t0.000".to_string(),
            ])
        );

        let _ = std::fs::remove_file(out);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_postprocess_helper_symmetric_skips_diagonal_and_lower_triangle() {
        let sketches = vec![
            test_filesketch("s0", vec![1, 0, 0, 0]),
            test_filesketch("s1", vec![0, 1, 0, 0]),
            test_filesketch("s2", vec![1, 0, 0, 0]),
        ];
        let cards = vec![1.0, 1.0, 1.0];
        let batch = test_dot_tile_batch(
            super::GpuTileJob {
                i0: 0,
                i1: 3,
                j0: 0,
                j1: 3,
            },
            3,
            3,
            vec![4, 0, 4, 0, 4, 0, 4, 0, 4],
        );

        let result = super::postprocess_dot_tile_batch(
            batch,
            &sketches,
            &sketches,
            &cards,
            &cards,
            1,
            true,
            0.0,
            DistOutputMode::Rows,
        );

        assert_eq!(result.num_pairs_done, 3);
        assert_eq!(result.ani_evals, 3);
        assert_eq!(
            result.text.lines().collect::<HashSet<_>>(),
            HashSet::from(["s0\ts1\t0.000", "s0\ts2\t100.000", "s1\ts2\t0.000"])
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_postprocess_helper_threshold_zero_keeps_zero_ani_rows() {
        let refs = vec![test_filesketch("r0", vec![4, 0, 0, 0])];
        let queries = vec![test_filesketch("q0", vec![0, 4, 0, 0])];
        let cards = vec![100.0];
        let batch = test_dot_tile_batch(
            super::GpuTileJob {
                i0: 0,
                i1: 1,
                j0: 0,
                j1: 1,
            },
            1,
            1,
            vec![0],
        );

        let result = super::postprocess_dot_tile_batch(
            batch,
            &refs,
            &queries,
            &cards,
            &cards,
            1,
            false,
            0.0,
            DistOutputMode::Rows,
        );

        assert_eq!(result.num_hits, 1);
        assert_eq!(result.ani_evals, 1);
        assert_eq!(result.nonpositive_skipped, 0);
        assert_eq!(result.text, "r0\tq0\t0.000\n");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_postprocess_helper_positive_threshold_filters_and_counts() {
        let refs = vec![
            test_filesketch("r0", vec![4, 0, 0, 0]),
            test_filesketch("r1", vec![0, 4, 0, 0]),
        ];
        let queries = vec![
            test_filesketch("q0", vec![90, 0, 0, 0]),
            test_filesketch("q1", vec![0, 0, 0, 0]),
        ];
        let cards = vec![100.0, 100.0];
        let batch = test_dot_tile_batch(
            super::GpuTileJob {
                i0: 0,
                i1: 2,
                j0: 0,
                j1: 2,
            },
            2,
            2,
            vec![360, 0, 0, 0],
        );

        let result = super::postprocess_dot_tile_batch(
            batch,
            &refs,
            &queries,
            &cards,
            &cards,
            1,
            false,
            85.0,
            DistOutputMode::Rows,
        );

        assert_eq!(result.num_hits, 1);
        assert_eq!(result.num_pairs_done, 4);
        assert_eq!(result.ani_evals, 1);
        assert_eq!(result.nonpositive_skipped, 3);
        assert_eq!(result.text, "r0\tq0\t90.000\n");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_postprocess_helper_count_mode_matches_rows_hit_count_without_text() {
        let refs = vec![
            test_filesketch("r0", vec![4, 0, 0, 0]),
            test_filesketch("r1", vec![0, 4, 0, 0]),
        ];
        let queries = vec![
            test_filesketch("q0", vec![90, 0, 0, 0]),
            test_filesketch("q1", vec![0, 0, 0, 0]),
        ];
        let cards = vec![100.0, 100.0];
        let batch = test_dot_tile_batch(
            super::GpuTileJob {
                i0: 0,
                i1: 2,
                j0: 0,
                j1: 2,
            },
            2,
            2,
            vec![360, 0, 0, 0],
        );

        let result = super::postprocess_dot_tile_batch(
            batch,
            &refs,
            &queries,
            &cards,
            &cards,
            1,
            false,
            85.0,
            DistOutputMode::Count,
        );

        assert_eq!(result.num_hits, 1);
        assert_eq!(result.ani_evals, 1);
        assert_eq!(result.nonpositive_skipped, 3);
        assert!(result.text.is_empty());
        assert_eq!(result.text_bytes, 0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_stream_ani_threshold_zero_keeps_zero_ani_rows() {
        let refs = vec![
            test_filesketch("r0", vec![4, 0, 0, 0]),
            test_filesketch("r1", vec![0, 4, 0, 0]),
        ];
        let queries = vec![
            test_filesketch("q0", vec![90, 0, 0, 0]),
            test_filesketch("q1", vec![0, 90, 0, 0]),
        ];
        let cards = vec![100.0, 100.0];
        let out = temp_ani_path("gpu_threshold_zero");
        let pb = indicatif::ProgressBar::hidden();
        let mut writer = BufWriter::new(File::create(&out).expect("failed to create output"));

        let hits = stream_hv_ani_gpu_multi(
            Some(&mut writer),
            &pb,
            &refs,
            &queries,
            &cards,
            &cards,
            1,
            false,
            0.0,
            1,
            DistOutputMode::Rows,
            ResidentMatrixMode::Auto,
        )
        .expect("GPU stream should compute");
        writer.flush().expect("failed to flush output");

        assert_eq!(hits, 4);
        assert_eq!(
            read_rows_as_set(&out),
            HashSet::from([
                "r0\tq0\t90.000".to_string(),
                "r0\tq1\t0.000".to_string(),
                "r1\tq0\t0.000".to_string(),
                "r1\tq1\t90.000".to_string(),
            ])
        );

        let _ = std::fs::remove_file(out);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_stream_normal_threshold_matches_expected_rows_as_set() {
        let refs = vec![
            test_filesketch("r0", vec![4, 0, 0, 0]),
            test_filesketch("r1", vec![0, 4, 0, 0]),
        ];
        let queries = vec![
            test_filesketch("q0", vec![90, 0, 0, 0]),
            test_filesketch("q1", vec![0, 90, 0, 0]),
        ];
        let cards = vec![100.0, 100.0];
        let out = temp_ani_path("gpu_threshold_85");
        let pb = indicatif::ProgressBar::hidden();
        let mut writer = BufWriter::new(File::create(&out).expect("failed to create output"));

        let hits = stream_hv_ani_gpu_multi(
            Some(&mut writer),
            &pb,
            &refs,
            &queries,
            &cards,
            &cards,
            1,
            false,
            85.0,
            1,
            DistOutputMode::Rows,
            ResidentMatrixMode::Auto,
        )
        .expect("GPU stream should compute");
        writer.flush().expect("failed to flush output");

        assert_eq!(hits, 2);
        assert_eq!(
            read_rows_as_set(&out),
            HashSet::from(["r0\tq0\t90.000".to_string(), "r1\tq1\t90.000".to_string(),])
        );

        let _ = std::fs::remove_file(out);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_stream_count_mode_matches_rows_hit_count() {
        let refs = vec![
            test_filesketch("r0", vec![4, 0, 0, 0]),
            test_filesketch("r1", vec![0, 4, 0, 0]),
        ];
        let queries = vec![
            test_filesketch("q0", vec![90, 0, 0, 0]),
            test_filesketch("q1", vec![0, 90, 0, 0]),
        ];
        let cards = vec![100.0, 100.0];
        let out = temp_ani_path("gpu_rows_count_compare");
        let pb = indicatif::ProgressBar::hidden();
        let mut writer = BufWriter::new(File::create(&out).expect("failed to create output"));

        let row_hits = stream_hv_ani_gpu_multi(
            Some(&mut writer),
            &pb,
            &refs,
            &queries,
            &cards,
            &cards,
            1,
            false,
            85.0,
            1,
            DistOutputMode::Rows,
            ResidentMatrixMode::Auto,
        )
        .expect("GPU stream should compute rows");
        writer.flush().expect("failed to flush output");
        let row_count = std::fs::read_to_string(&out)
            .expect("failed to read rows output")
            .lines()
            .count();

        let count_hits = stream_hv_ani_gpu_multi(
            None,
            &pb,
            &refs,
            &queries,
            &cards,
            &cards,
            1,
            false,
            85.0,
            1,
            DistOutputMode::Count,
            ResidentMatrixMode::Off,
        )
        .expect("GPU stream should compute count");

        assert_eq!(row_hits, 2);
        assert_eq!(row_count, row_hits);
        assert_eq!(count_hits, row_hits);

        let _ = std::fs::remove_file(out);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_stream_symmetric_resident_path_matches_expected_rows_as_set() {
        let sketches = vec![
            test_filesketch("s0", vec![4, 0, 0, 0]),
            test_filesketch("s1", vec![0, 4, 0, 0]),
            test_filesketch("s2", vec![4, 0, 0, 0]),
        ];
        let cards = vec![100.0, 100.0, 100.0];
        let out = temp_ani_path("gpu_symmetric_resident");
        let pb = indicatif::ProgressBar::hidden();
        let mut writer = BufWriter::new(File::create(&out).expect("failed to create output"));

        let hits = stream_hv_ani_gpu_multi(
            Some(&mut writer),
            &pb,
            &sketches,
            &sketches,
            &cards,
            &cards,
            1,
            true,
            0.0,
            1,
            DistOutputMode::Rows,
            ResidentMatrixMode::Auto,
        )
        .expect("GPU symmetric stream should compute");
        writer.flush().expect("failed to flush output");

        assert_eq!(hits, 3);
        assert_eq!(
            read_rows_as_set(&out),
            HashSet::from([
                "s0\ts1\t0.000".to_string(),
                "s0\ts2\t4.000".to_string(),
                "s1\ts2\t0.000".to_string(),
            ])
        );

        let _ = std::fs::remove_file(out);
    }
}

pub fn compute_hv_ani(
    sketch_dist: &mut SketchDist,
    ref_filesketch: &[FileSketch],
    query_filesketch: &[FileSketch],
    ref_ull_sketch: &[FileUllSketch],
    query_ull_sketch: &[FileUllSketch],
    ksize: u8,
    if_symmetric: bool,
) {
    let compute_start = Instant::now();
    info!("Computing ANI..");

    let num_ref_files = ref_filesketch.len();
    let num_query_files = query_filesketch.len();

    let num_dists = if if_symmetric {
        num_ref_files * (num_query_files - 1) / 2
    } else {
        num_ref_files * num_query_files
    };

    let pb = utils::get_progress_bar(num_dists);

    let cardinality_start = Instant::now();
    let ref_cards: Vec<f64> = ref_ull_sketch
        .par_iter()
        .map(|s| ull_cardinality_from_state(&s.ull_state))
        .collect();

    let query_cards: Vec<f64> = if if_symmetric {
        ref_cards.clone()
    } else {
        query_ull_sketch
            .par_iter()
            .map(|s| ull_cardinality_from_state(&s.ull_state))
            .collect()
    };
    let cardinality_secs = cardinality_start.elapsed().as_secs_f32();

    let stream_start = Instant::now();
    #[cfg(feature = "cuda")]
    let (num_hits, output_open_secs, flush_secs, stream_mode) = {
        let mut writer = if sketch_dist.output_mode == DistOutputMode::Rows {
            let output_open_start = Instant::now();
            let out_file = File::create(sketch_dist.out_file.as_path())
                .expect("Failed to create ANI output file");
            let writer = BufWriter::new(out_file);
            (Some(writer), output_open_start.elapsed().as_secs_f32())
        } else {
            (None, 0.0)
        };

        match stream_hv_ani_gpu_multi(
            writer.0.as_mut(),
            &pb,
            ref_filesketch,
            query_filesketch,
            &ref_cards,
            &query_cards,
            ksize,
            if_symmetric,
            sketch_dist.ani_threshold,
            sketch_dist.threads as usize,
            sketch_dist.output_mode,
            sketch_dist.resident_matrix_mode,
        ) {
            Ok(n) => {
                let flush_start = Instant::now();
                if let Some(writer) = writer.0.as_mut() {
                    writer.flush().expect("Failed to flush ANI output");
                }
                let flush_secs = flush_start.elapsed().as_secs_f32();
                info!("Multi-GPU tiled dot-product completed successfully");
                (n, writer.1, flush_secs, "gpu")
            }
            Err(e) => {
                warn!("Multi-GPU tiled dot-product failed, falling back to CPU: {e:?}");
                drop(writer);
                pb.set_position(0);
                let n = stream_hv_ani_cpu(
                    sketch_dist.out_file.as_path(),
                    &pb,
                    ref_filesketch,
                    query_filesketch,
                    &ref_cards,
                    &query_cards,
                    ksize,
                    if_symmetric,
                    sketch_dist.ani_threshold,
                    sketch_dist.output_mode,
                );
                (n, 0.0, 0.0, "gpu_fallback_cpu")
            }
        }
    };

    #[cfg(not(feature = "cuda"))]
    let (num_hits, output_open_secs, flush_secs, stream_mode) = {
        let n = stream_hv_ani_cpu(
            sketch_dist.out_file.as_path(),
            &pb,
            ref_filesketch,
            query_filesketch,
            &ref_cards,
            &query_cards,
            ksize,
            if_symmetric,
            sketch_dist.ani_threshold,
            sketch_dist.output_mode,
        );
        (n, 0.0, 0.0, "cpu")
    };
    let stream_secs = stream_start.elapsed().as_secs_f32();

    let summary_start = Instant::now();
    pb.finish_and_clear();

    let total_dist = num_dists as u64;
    let cnt = num_hits as u64;
    let perc = if total_dist > 0 {
        (cnt as f64) / (total_dist as f64) * 100.0
    } else {
        0.0
    };

    if perc < 5.0 {
        warn!(
            "Output ANIs with threshold {:.1} are too divergent: {} of {} ({:.2}%) ANIs are reported",
            sketch_dist.ani_threshold, cnt, total_dist, perc
        );
    } else {
        info!(
            "Output {} of {} ANIs above threshold {:.1} to file {}",
            cnt,
            total_dist,
            sketch_dist.ani_threshold,
            sketch_dist.out_file.to_string_lossy()
        );
    }
    if sketch_dist.output_mode == DistOutputMode::Count {
        let mut writer = BufWriter::new(
            File::create(sketch_dist.out_file.as_path()).expect("Failed to create count output"),
        );
        writeln!(writer, "metric\tvalue").expect("Failed to write count output");
        writeln!(writer, "mode\t{}", sketch_dist.output_mode.as_str())
            .expect("Failed to write count output");
        writeln!(writer, "refs\t{}", num_ref_files).expect("Failed to write count output");
        writeln!(writer, "queries\t{}", num_query_files).expect("Failed to write count output");
        writeln!(writer, "pairs\t{}", total_dist).expect("Failed to write count output");
        writeln!(writer, "hits\t{}", cnt).expect("Failed to write count output");
        writeln!(writer, "ani_threshold\t{}", sketch_dist.ani_threshold)
            .expect("Failed to write count output");
        writer.flush().expect("Failed to flush count output");
    }
    let summary_secs = summary_start.elapsed().as_secs_f32();

    info!(
        "compute_hv_ani timings: cardinality={:.3}s output_open={:.3}s stream_mode={} stream={:.3}s flush={:.3}s summary={:.3}s total={:.3}s",
        cardinality_secs,
        output_open_secs,
        stream_mode,
        stream_secs,
        flush_secs,
        summary_secs,
        compute_start.elapsed().as_secs_f32()
    );
}
