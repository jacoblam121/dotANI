use ultraloglog::UltraLogLog;

use crate::hd;
use crate::types::*;
use crate::utils;

use log::{info, warn};
use rayon::prelude::*;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(feature = "cuda")]
use crate::cuda_dot::{device_count, GpuDotExecutor};

pub fn dist(sketch_dist: &mut SketchDist) {
    let tstart = Instant::now();
    let if_sym = sketch_dist.path_ref_sketch == sketch_dist.path_query_sketch;

    let ref_ull_sketch = utils::load_ull_sketch(sketch_dist.path_ref_ull.as_path());
    let query_ull_sketch = if if_sym {
        ref_ull_sketch.clone()
    } else {
        utils::load_ull_sketch(sketch_dist.path_query_ull.as_path())
    };

    let mut ref_file_sketch = utils::load_sketch(sketch_dist.path_ref_sketch.as_path());
    let mut query_file_sketch = if if_sym {
        ref_file_sketch.clone()
    } else {
        utils::load_sketch(sketch_dist.path_query_sketch.as_path())
    };

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
            ref_file_sketch[i].file_str,
            ref_ull_sketch[i].file_str,
            "Ref HD/ULL file order mismatch"
        );
    }
    for i in 0..query_file_sketch.len() {
        assert_eq!(
            query_file_sketch[i].file_str,
            query_ull_sketch[i].file_str,
            "Query HD/ULL file order mismatch"
        );
    }

    let ksize_ref = ref_file_sketch[0].ksize;
    let ksize_query = query_file_sketch[0].ksize;
    assert_eq!(
        ksize_ref, ksize_query,
        "Ref and query sketches use different kmer sizes!"
    );

    let hv_d_ref = ref_file_sketch[0].hv_d;
    let hv_d_query = query_file_sketch[0].hv_d;
    assert_eq!(
        hv_d_ref, hv_d_query,
        "Ref and query sketches use different HV dimensions!"
    );

    hd::decompress_file_sketch(&mut ref_file_sketch);
    hd::decompress_file_sketch(&mut query_file_sketch);

    compute_hv_ani(
        sketch_dist,
        &ref_file_sketch,
        &query_file_sketch,
        &ref_ull_sketch,
        &query_ull_sketch,
        ksize_ref,
        if_sym,
    );

    info!(
        "Computed ANIs for {} ref files and {} query files took {:.3}s",
        ref_file_sketch.len(),
        query_file_sketch.len(),
        tstart.elapsed().as_secs_f32()
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

        let vr = _mm256_loadu_si256(r.as_ptr().add(base) as *const __m256i);
        let vq = _mm256_loadu_si256(q.as_ptr().add(base) as *const __m256i);

        let prod_even = _mm256_mul_epi32(vr, vq);

        let vr_shift = _mm256_srli_epi64(vr, 32);
        let vq_shift = _mm256_srli_epi64(vq, 32);
        let prod_odd = _mm256_mul_epi32(vr_shift, vq_shift);

        acc_even = _mm256_add_epi64(acc_even, prod_even);
        acc_odd = _mm256_add_epi64(acc_odd, prod_odd);
    }

    let acc = _mm256_add_epi64(acc_even, acc_odd);
    let mut tmp = [0i64; 4];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc);

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

        let vr = _mm512_loadu_si512(r.as_ptr().add(base) as *const __m512i);
        let vq = _mm512_loadu_si512(q.as_ptr().add(base) as *const __m512i);

        let prod_even = _mm512_mul_epi32(vr, vq);

        let vr_shift = _mm512_srli_epi64(vr, 32);
        let vq_shift = _mm512_srli_epi64(vq, 32);
        let prod_odd = _mm512_mul_epi32(vr_shift, vq_shift);

        acc_even = _mm512_add_epi64(acc_even, prod_even);
        acc_odd = _mm512_add_epi64(acc_odd, prod_odd);
    }

    let acc = _mm512_add_epi64(acc_even, acc_odd);
    let mut tmp = [0i64; 8];
    _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, acc);

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
    if !jaccard.is_finite() || jaccard <= 0.0 || jaccard > 1.0 {
        return 0.0;
    }

    let ani = 1.0 + (2.0 / (1.0 / jaccard as f32 + 1.0)).ln() / (ksize as f32);

    if ani.is_nan() {
        0.0
    } else {
        ani.clamp(0.0, 1.0) * 100.0
    }
}

#[inline]
fn write_ani_record<W: Write>(writer: &mut W, ref_name: &str, query_name: &str, ani: f32) {
    writeln!(writer, "{}\t{}\t{:.3}", ref_name, query_name, ani)
        .expect("Failed to write ANI record");
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
    writer: &mut BufWriter<File>,
    pb: &indicatif::ProgressBar,
    ref_filesketch: &[FileSketch],
    query_filesketch: &[FileSketch],
    ref_cards: &[f64],
    query_cards: &[f64],
    ksize: u8,
    if_symmetric: bool,
    ani_threshold: f32,
) -> usize {
    let mut num_hits = 0usize;
    let mut debug_seen = 0usize;

    for i in 0..ref_filesketch.len() {
        let j_start = if if_symmetric { i + 1 } else { 0 };

        for j in j_start..query_filesketch.len() {
            let r = &ref_filesketch[i];
            let q = &query_filesketch[j];

            let dot = compute_pairwise_dot_best(&r.hv, &q.hv) as f64;
            let inter_hat = dot / r.hv_d as f64;
            let ani = ani_from_intersection_and_cardinalities(
                inter_hat,
                ref_cards[i],
                query_cards[j],
                ksize,
            );

            if debug_seen < 8 {
                let union_hat = ref_cards[i] + query_cards[j] - inter_hat;
                let jaccard = if union_hat > 0.0 {
                    inter_hat / union_hat
                } else {
                    -1.0
                };

                info!(
                    "DEBUG pair {}: {} vs {} | card_r={:.3} card_q={:.3} dot={:.3} inter_hat={:.3} union_hat={:.3} jaccard={:.6} ani={:.3}",
                    debug_seen,
                    r.file_str,
                    q.file_str,
                    ref_cards[i],
                    query_cards[j],
                    dot,
                    inter_hat,
                    union_hat,
                    jaccard,
                    ani
                );
                debug_seen += 1;
            }

            if ani >= ani_threshold {
                write_ani_record(writer, &r.file_str, &q.file_str, ani);
                num_hits += 1;
            }

            pb.inc(1);
        }
    }

    num_hits
}

#[cfg(feature = "cuda")]
struct TileBatchResult {
    text: String,
    num_hits: usize,
    num_pairs_done: usize,
}

#[cfg(feature = "cuda")]
fn stream_hv_ani_gpu_multi(
    writer: &mut BufWriter<File>,
    pb: &indicatif::ProgressBar,
    ref_filesketch: &[FileSketch],
    query_filesketch: &[FileSketch],
    ref_cards: &[f64],
    query_cards: &[f64],
    ksize: u8,
    if_symmetric: bool,
    ani_threshold: f32,
) -> anyhow::Result<usize> {
    if ref_filesketch.is_empty() || query_filesketch.is_empty() {
        return Ok(0);
    }

    let hv_d = ref_filesketch[0].hv_d;
    let tile_ref = 256usize;
    let tile_query = 256usize;

    let ng = device_count()?.max(1);
    info!("Using {} GPU worker(s) for tiled dot-product", ng);

    let mut total_hits = 0usize;

    for i0 in (0..ref_filesketch.len()).step_by(tile_ref) {
        let i1 = (i0 + tile_ref).min(ref_filesketch.len());
        let ref_block = &ref_filesketch[i0..i1];
        let ref_flat = Arc::new(flatten_hv_matrix(ref_block));
        let nr = ref_block.len();

        let j0_start = if if_symmetric { i0 } else { 0 };

        let mut jobs = Vec::<(usize, usize)>::new();
        for j0 in (j0_start..query_filesketch.len()).step_by(tile_query) {
            let j1 = (j0 + tile_query).min(query_filesketch.len());
            jobs.push((j0, j1));
        }

        let jobs = Arc::new(jobs);
        let next = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = mpsc::channel::<anyhow::Result<TileBatchResult>>();

        std::thread::scope(|scope| {
            for dev_id in 0..ng {
                let tx = tx.clone();
                let jobs = Arc::clone(&jobs);
                let next = Arc::clone(&next);
                let ref_flat = Arc::clone(&ref_flat);

                scope.spawn(move || {
                    let worker = || -> anyhow::Result<()> {
                        let gpu = GpuDotExecutor::new(dev_id)?;

                        loop {
                            let job_idx = next.fetch_add(1, Ordering::Relaxed);
                            if job_idx >= jobs.len() {
                                break;
                            }

                            let (j0, j1) = jobs[job_idx];
                            let query_block = &query_filesketch[j0..j1];
                            let query_flat = flatten_hv_matrix(query_block);
                            let nq = query_block.len();

                            let mut tile_dots = vec![0i64; nq * nr];
                            gpu.compute_tile(&query_flat, nq, &ref_flat, nr, hv_d, &mut tile_dots)?;

                            let mut text = String::new();
                            let mut num_hits = 0usize;
                            let mut num_pairs_done = 0usize;

                            for q_local in 0..nq {
                                for r_local in 0..nr {
                                    let i = i0 + r_local;
                                    let j = j0 + q_local;

                                    if if_symmetric && i >= j {
                                        continue;
                                    }

                                    num_pairs_done += 1;

                                    let dot = tile_dots[q_local * nr + r_local] as f64;
                                    let inter_hat = dot / hv_d as f64;
                                    let ani = ani_from_intersection_and_cardinalities(
                                        inter_hat,
                                        ref_cards[i],
                                        query_cards[j],
                                        ksize,
                                    );

                                    if ani >= ani_threshold {
                                        use std::fmt::Write as _;
                                        let _ = writeln!(
                                            &mut text,
                                            "{}\t{}\t{:.3}",
                                            ref_filesketch[i].file_str,
                                            query_filesketch[j].file_str,
                                            ani
                                        );
                                        num_hits += 1;
                                    }
                                }
                            }

                            tx.send(Ok(TileBatchResult {
                                text,
                                num_hits,
                                num_pairs_done,
                            }))
                            .expect("Failed to send tile batch result");
                        }

                        Ok(())
                    };

                    if let Err(e) = worker() {
                        let _ = tx.send(Err(e));
                    }
                });
            }
        });

        drop(tx);

        for _ in 0..jobs.len() {
            match rx.recv().expect("GPU worker channel closed unexpectedly") {
                Ok(batch) => {
                    writer
                        .write_all(batch.text.as_bytes())
                        .expect("Failed to write ANI batch");
                    total_hits += batch.num_hits;
                    pb.inc(batch.num_pairs_done as u64);
                }
                Err(e) => return Err(e),
            }
        }
    }

    Ok(total_hits)
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
    info!("Computing ANI..");

    let num_ref_files = ref_filesketch.len();
    let num_query_files = query_filesketch.len();

    let num_dists = if if_symmetric {
        num_ref_files * (num_query_files - 1) / 2
    } else {
        num_ref_files * num_query_files
    };

    let pb = utils::get_progress_bar(num_dists);

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

    let out_file = File::create(sketch_dist.out_file.as_path())
        .expect("Failed to create ANI output file");
    let mut writer = BufWriter::new(out_file);

    let num_hits = {
        #[cfg(feature = "cuda")]
        {
            match stream_hv_ani_gpu_multi(
                &mut writer,
                &pb,
                ref_filesketch,
                query_filesketch,
                &ref_cards,
                &query_cards,
                ksize,
                if_symmetric,
                sketch_dist.ani_threshold,
            ) {
                Ok(n) => {
                    info!("Multi-GPU tiled dot-product completed successfully");
                    n
                }
                Err(e) => {
                    warn!("Multi-GPU tiled dot-product failed, falling back to CPU: {e:?}");
                    stream_hv_ani_cpu(
                        &mut writer,
                        &pb,
                        ref_filesketch,
                        query_filesketch,
                        &ref_cards,
                        &query_cards,
                        ksize,
                        if_symmetric,
                        sketch_dist.ani_threshold,
                    )
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            stream_hv_ani_cpu(
                &mut writer,
                &pb,
                ref_filesketch,
                query_filesketch,
                &ref_cards,
                &query_cards,
                ksize,
                if_symmetric,
                sketch_dist.ani_threshold,
            )
        }
    };

    writer.flush().expect("Failed to flush ANI output");
    pb.finish_and_clear();

    let total_dist = num_dists as f32;
    let cnt = num_hits as f32;
    let perc = if total_dist > 0.0 {
        cnt / total_dist * 100.0
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
}