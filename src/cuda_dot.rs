use anyhow::{bail, Context, Result};
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use log::{debug, info};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
#ifndef BLOCK_M
#define BLOCK_M 16
#endif

#ifndef BLOCK_N
#define BLOCK_N 16
#endif

#ifndef BLOCK_K
#define BLOCK_K 32
#endif

extern "C" __global__
void dot_rect_i32_i64_tiled(
    const int* __restrict__ query_hv,   // [nq * d]
    const int* __restrict__ ref_hv,     // [nr * d]
    int nq,
    int nr,
    int d,
    int i0,
    int j0,
    int bw,
    int bh,
    long long* __restrict__ out // [bw * bh], row-major
) {
    const int local_j = blockIdx.x * blockDim.x + threadIdx.x; // within tile
    const int local_i = blockIdx.y * blockDim.y + threadIdx.y; // within tile

    const int qi = i0 + local_i;
    const int rj = j0 + local_j;

    extern __shared__ int smem[];
    int* As = smem;                                      // BLOCK_M * BLOCK_K
    int* Bs = As + (BLOCK_M * BLOCK_K);                  // BLOCK_N * BLOCK_K

    long long acc = 0;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;

    for (int k0 = 0; k0 < d; k0 += BLOCK_K) {
        const int bk = min(BLOCK_K, d - k0);

        // Load A tile: [BLOCK_M x bk]
        const int total_a = blockDim.y * bk;
        for (int idx = tid; idx < total_a; idx += nthreads) {
            const int row = idx / bk;
            const int kk  = idx - row * bk;

            const int gqi = i0 + (int)blockIdx.y * (int)blockDim.y + row;
            int v = 0;
            if (gqi < nq) {
                v = query_hv[(size_t)gqi * (size_t)d + (size_t)(k0 + kk)];
            }
            As[row * BLOCK_K + kk] = v;
        }

        // Load B tile: [BLOCK_N x bk]
        const int total_b = blockDim.x * bk;
        for (int idx = tid; idx < total_b; idx += nthreads) {
            const int col = idx / bk;
            const int kk  = idx - col * bk;

            const int ഗ്രj = j0 + (int)blockIdx.x * (int)blockDim.x + col;
            int v = 0;
            if (ഗ്രj < nr) {
                v = ref_hv[(size_t)ഗ്രj * (size_t)d + (size_t)(k0 + kk)];
            }
            Bs[col * BLOCK_K + kk] = v;
        }

        __syncthreads();

        if (local_i < bw && local_j < bh && qi < nq && rj < nr) {
            const int arow = threadIdx.y * BLOCK_K;
            const int brow = threadIdx.x * BLOCK_K;

            int kk = 0;
            #pragma unroll
            for (; kk + 3 < bk; kk += 4) {
                acc += (long long)As[arow + kk + 0] * (long long)Bs[brow + kk + 0];
                acc += (long long)As[arow + kk + 1] * (long long)Bs[brow + kk + 1];
                acc += (long long)As[arow + kk + 2] * (long long)Bs[brow + kk + 2];
                acc += (long long)As[arow + kk + 3] * (long long)Bs[brow + kk + 3];
            }
            for (; kk < bk; ++kk) {
                acc += (long long)As[arow + kk] * (long long)Bs[brow + kk];
            }
        }

        __syncthreads();
    }

    if (local_i < bw && local_j < bh && qi < nq && rj < nr) {
        out[(size_t)local_i * (size_t)bh + (size_t)local_j] = acc;
    }
}
"#;

pub fn device_count() -> Result<usize> {
    Ok(CudaContext::device_count()? as usize)
}

#[inline]
fn mib(x: usize) -> f64 {
    x as f64 / (1024.0 * 1024.0)
}

#[inline]
fn gib(x: usize) -> f64 {
    x as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[inline]
fn shared_mem_bytes_i32(block_m: usize, block_n: usize, block_k: usize) -> u32 {
    ((block_m * block_k + block_n * block_k) * std::mem::size_of::<i32>()) as u32
}

fn pairwise_dot_rect_single_gpu_i32_impl(
    query_hv: &[i32],
    nq: usize,
    ref_hv: &[i32],
    nr: usize,
    d: usize,
    out: &mut [i64],
    block_rows: usize,
    block_cols: usize,
    gpu_id: usize,
) -> Result<()> {
    let total_t0 = Instant::now();

    if query_hv.len() != nq * d {
        bail!(
            "query_hv length mismatch: got {}, expected {}",
            query_hv.len(),
            nq * d
        );
    }
    if ref_hv.len() != nr * d {
        bail!(
            "ref_hv length mismatch: got {}, expected {}",
            ref_hv.len(),
            nr * d
        );
    }
    if out.len() != nq * nr {
        bail!(
            "out length mismatch: got {}, expected {}",
            out.len(),
            nq * nr
        );
    }
    if d == 0 {
        bail!("d must be > 0");
    }

    let block_rows = block_rows.max(1).min(nq.max(1));
    let block_cols = block_cols.max(1).min(nr.max(1));

    info!(
        "single-GPU dot: gpu_id={} nq={} nr={} d={} block_rows={} block_cols={} query={:.2} GiB ref={:.2} GiB out={:.2} GiB",
        gpu_id,
        nq,
        nr,
        d,
        block_rows,
        block_cols,
        gib(query_hv.len() * std::mem::size_of::<i32>()),
        gib(ref_hv.len() * std::mem::size_of::<i32>()),
        gib(out.len() * std::mem::size_of::<i64>())
    );

    let ctx = CudaContext::new(gpu_id)?;
    let stream = ctx.default_stream();

    let compile_t0 = Instant::now();
    let ptx = compile_ptx(KERNEL_SRC)?;
    let module = ctx.load_module(ptx)?;
    let func = module
        .load_function("dot_rect_i32_i64_tiled")
        .context("load function dot_rect_i32_i64_tiled")?;
    info!(
        "single-GPU dot: kernel compile/load done in {:.3}s",
        compile_t0.elapsed().as_secs_f64()
    );

    let upload_t0 = Instant::now();
    let d_query: CudaSlice<i32> = stream.clone_htod(query_hv)?;
    let d_ref: CudaSlice<i32> = stream.clone_htod(ref_hv)?;
    info!(
        "single-GPU dot: uploaded query {:.2} MiB, ref {:.2} MiB in {:.3}s",
        mib(query_hv.len() * std::mem::size_of::<i32>()),
        mib(ref_hv.len() * std::mem::size_of::<i32>()),
        upload_t0.elapsed().as_secs_f64()
    );

    let max_bw = block_rows.min(nq.max(1));
    let max_bh = block_cols.min(nr.max(1));
    let scratch_elems = max_bw * max_bh;

    let mut d_tile: CudaSlice<i64> = stream
        .alloc_zeros(scratch_elems)
        .with_context(|| format!("alloc d_tile: {:.2} MiB", mib(scratch_elems * 8)))?;
    let mut h_tile = vec![0i64; scratch_elems];

    let nq_i32 = nq as i32;
    let nr_i32 = nr as i32;
    let d_i32 = d as i32;

    let nbq = nq.div_ceil(block_rows);
    let nbr = nr.div_ceil(block_cols);
    let total_tiles = nbq * nbr;
    let mut done_tiles = 0usize;

    let kernel_total_t0 = Instant::now();

    let blk_x = 16usize;
    let blk_y = 16usize;
    let blk_k = 32usize;
    let smem_bytes = shared_mem_bytes_i32(blk_y, blk_x, blk_k);

    for bi in 0..nbq {
        let i0 = bi * block_rows;
        let bw = (nq - i0).min(block_rows);

        for bj in 0..nbr {
            let tile_t0 = Instant::now();

            let j0 = bj * block_cols;
            let bh = (nr - j0).min(block_cols);

            let cfg = LaunchConfig {
                grid_dim: (bh.div_ceil(blk_x) as u32, bw.div_ceil(blk_y) as u32, 1),
                block_dim: (blk_x as u32, blk_y as u32, 1),
                shared_mem_bytes: smem_bytes,
            };

            let i0_i32 = i0 as i32;
            let j0_i32 = j0 as i32;
            let bw_i32 = bw as i32;
            let bh_i32 = bh as i32;

            debug!(
                "single-GPU dot: tile bi={} bj={} i0={} j0={} bw={} bh={} smem={}",
                bi, bj, i0, j0, bw, bh, smem_bytes
            );

            let mut launch = stream.launch_builder(&func);
            launch.arg(&d_query);
            launch.arg(&d_ref);
            launch.arg(&nq_i32);
            launch.arg(&nr_i32);
            launch.arg(&d_i32);
            launch.arg(&i0_i32);
            launch.arg(&j0_i32);
            launch.arg(&bw_i32);
            launch.arg(&bh_i32);
            launch.arg(&mut d_tile);

            unsafe { launch.launch(cfg) }?;
            stream.memcpy_dtoh(&d_tile, &mut h_tile)?;

            for ii in 0..bw {
                let qi = i0 + ii;
                let row_src = &h_tile[ii * bh..ii * bh + bh];
                let row_dst = &mut out[qi * nr + j0..qi * nr + j0 + bh];
                row_dst.copy_from_slice(row_src);
            }

            done_tiles += 1;
            if done_tiles == 1 || done_tiles % 100 == 0 || done_tiles == total_tiles {
                info!(
                    "single-GPU dot: completed tile {}/{} in {:.3}s (bw={} bh={})",
                    done_tiles,
                    total_tiles,
                    tile_t0.elapsed().as_secs_f64(),
                    bw,
                    bh
                );
            }
        }
    }

    info!(
        "single-GPU dot: all {} tiles completed in {:.3}s",
        total_tiles,
        kernel_total_t0.elapsed().as_secs_f64()
    );
    info!(
        "single-GPU dot: total runtime {:.3}s",
        total_t0.elapsed().as_secs_f64()
    );

    Ok(())
}

fn pairwise_dot_rect_multi_gpu_i32_impl(
    query_hv: &[i32],
    nq: usize,
    ref_hv: &[i32],
    nr: usize,
    d: usize,
    out: &mut [i64],
    block_rows: usize,
    block_cols: usize,
) -> Result<()> {
    let total_t0 = Instant::now();

    if query_hv.len() != nq * d {
        bail!(
            "query_hv length mismatch: got {}, expected {}",
            query_hv.len(),
            nq * d
        );
    }
    if ref_hv.len() != nr * d {
        bail!(
            "ref_hv length mismatch: got {}, expected {}",
            ref_hv.len(),
            nr * d
        );
    }
    if out.len() != nq * nr {
        bail!(
            "out length mismatch: got {}, expected {}",
            out.len(),
            nq * nr
        );
    }
    if d == 0 {
        bail!("d must be > 0");
    }

    let ng = device_count()?;
    if ng == 0 {
        bail!("No CUDA devices available");
    }
    if ng == 1 {
        return pairwise_dot_rect_single_gpu_i32_impl(
            query_hv,
            nq,
            ref_hv,
            nr,
            d,
            out,
            block_rows,
            block_cols,
            0,
        );
    }

    let block_rows = block_rows.max(1).min(nq.max(1));
    let block_cols = block_cols.max(1).min(nr.max(1));

    let nbq = nq.div_ceil(block_rows);
    let nbr = nr.div_ceil(block_cols);

    let mut tiles = Vec::<(usize, usize)>::with_capacity(nbq * nbr);
    for bi in 0..nbq {
        for bj in 0..nbr {
            tiles.push((bi, bj));
        }
    }

    info!(
        "multi-GPU dot: ng={} nq={} nr={} d={} block_rows={} block_cols={} tiles={} query={:.2} GiB ref={:.2} GiB out={:.2} GiB",
        ng,
        nq,
        nr,
        d,
        block_rows,
        block_cols,
        tiles.len(),
        gib(query_hv.len() * std::mem::size_of::<i32>()),
        gib(ref_hv.len() * std::mem::size_of::<i32>()),
        gib(out.len() * std::mem::size_of::<i64>())
    );

    let compile_t0 = Instant::now();
    let ptx = Arc::new(compile_ptx(KERNEL_SRC)?);
    info!(
        "multi-GPU dot: PTX compiled in {:.3}s",
        compile_t0.elapsed().as_secs_f64()
    );

    let tiles = Arc::new(tiles);
    let next = Arc::new(AtomicUsize::new(0));

    let q_arc: Arc<Vec<i32>> = Arc::new(query_hv.to_vec());
    let r_arc: Arc<Vec<i32>> = Arc::new(ref_hv.to_vec());

    let out_addr = out.as_mut_ptr() as usize;
    let total_tiles = tiles.len();

    std::thread::scope(|scope| {
        for dev_id in 0..ng {
            let ptx = Arc::clone(&ptx);
            let tiles = Arc::clone(&tiles);
            let next = Arc::clone(&next);
            let q_arc = Arc::clone(&q_arc);
            let r_arc = Arc::clone(&r_arc);

            scope.spawn(move || {
                let worker_t0 = Instant::now();

                let inner = || -> Result<()> {
                    info!("multi-GPU dot: dev {} worker starting", dev_id);

                    let ctx = CudaContext::new(dev_id)?;
                    let stream = ctx.default_stream();

                    let module = ctx.load_module((*ptx).clone())?;
                    let func = module
                        .load_function("dot_rect_i32_i64_tiled")
                        .context("load function dot_rect_i32_i64_tiled")?;

                    let upload_t0 = Instant::now();
                    let d_query: CudaSlice<i32> = stream.clone_htod(&q_arc[..])?;
                    let d_ref: CudaSlice<i32> = stream.clone_htod(&r_arc[..])?;
                    info!(
                        "multi-GPU dot: dev {} uploaded query/ref in {:.3}s",
                        dev_id,
                        upload_t0.elapsed().as_secs_f64()
                    );

                    let max_bw = block_rows.min(nq.max(1));
                    let max_bh = block_cols.min(nr.max(1));

                    let mut d_tile: CudaSlice<i64> = stream.alloc_zeros(max_bw * max_bh)?;
                    let mut h_tile = vec![0i64; max_bw * max_bh];

                    let nq_i32 = nq as i32;
                    let nr_i32 = nr as i32;
                    let d_i32 = d as i32;

                    let blk_x = 16usize;
                    let blk_y = 16usize;
                    let blk_k = 32usize;
                    let smem_bytes = shared_mem_bytes_i32(blk_y, blk_x, blk_k);

                    let mut worker_tiles = 0usize;

                    loop {
                        let tix = next.fetch_add(1, Ordering::Relaxed);
                        if tix >= tiles.len() {
                            break;
                        }

                        let tile_t0 = Instant::now();
                        let (bi, bj) = tiles[tix];

                        let i0 = bi * block_rows;
                        let j0 = bj * block_cols;
                        let bw = (nq - i0).min(block_rows);
                        let bh = (nr - j0).min(block_cols);

                        let cfg = LaunchConfig {
                            grid_dim: (bh.div_ceil(blk_x) as u32, bw.div_ceil(blk_y) as u32, 1),
                            block_dim: (blk_x as u32, blk_y as u32, 1),
                            shared_mem_bytes: smem_bytes,
                        };

                        let i0_i32 = i0 as i32;
                        let j0_i32 = j0 as i32;
                        let bw_i32 = bw as i32;
                        let bh_i32 = bh as i32;

                        let mut launch = stream.launch_builder(&func);
                        launch.arg(&d_query);
                        launch.arg(&d_ref);
                        launch.arg(&nq_i32);
                        launch.arg(&nr_i32);
                        launch.arg(&d_i32);
                        launch.arg(&i0_i32);
                        launch.arg(&j0_i32);
                        launch.arg(&bw_i32);
                        launch.arg(&bh_i32);
                        launch.arg(&mut d_tile);

                        unsafe { launch.launch(cfg) }?;
                        stream.memcpy_dtoh(&d_tile, &mut h_tile)?;

                        let base_ptr = out_addr as *mut i64;
                        unsafe {
                            for ii in 0..bw {
                                let qi = i0 + ii;
                                for jj in 0..bh {
                                    let rj = j0 + jj;
                                    *base_ptr.add(qi * nr + rj) = h_tile[ii * bh + jj];
                                }
                            }
                        }

                        worker_tiles += 1;
                        if worker_tiles == 1 || worker_tiles % 100 == 0 || tix + 1 == total_tiles {
                            info!(
                                "multi-GPU dot: dev {} completed worker_tile={} global_tile={}/{} in {:.3}s (bw={} bh={})",
                                dev_id,
                                worker_tiles,
                                tix + 1,
                                total_tiles,
                                tile_t0.elapsed().as_secs_f64(),
                                bw,
                                bh
                            );
                        }
                    }

                    info!(
                        "multi-GPU dot: dev {} worker finished {} tiles in {:.3}s",
                        dev_id,
                        worker_tiles,
                        worker_t0.elapsed().as_secs_f64()
                    );

                    Ok(())
                };

                if let Err(e) = inner() {
                    panic!("GPU worker {} failed: {e:?}", dev_id);
                }
            });
        }
    });

    info!(
        "multi-GPU dot: all workers finished in {:.3}s",
        total_t0.elapsed().as_secs_f64()
    );

    Ok(())
}

pub fn pairwise_dot_rect_single_gpu_i32(
    query_hv: &[i32],
    nq: usize,
    ref_hv: &[i32],
    nr: usize,
    d: usize,
    out: &mut [i64],
    block_rows: usize,
    block_cols: usize,
    gpu_id: usize,
) -> Result<()> {
    pairwise_dot_rect_single_gpu_i32_impl(
        query_hv, nq, ref_hv, nr, d, out, block_rows, block_cols, gpu_id,
    )
}

pub fn pairwise_dot_rect_multi_gpu_i32(
    query_hv: &[i32],
    nq: usize,
    ref_hv: &[i32],
    nr: usize,
    d: usize,
    out: &mut [i64],
    block_rows: usize,
    block_cols: usize,
) -> Result<()> {
    pairwise_dot_rect_multi_gpu_i32_impl(
        query_hv, nq, ref_hv, nr, d, out, block_rows, block_cols,
    )
}