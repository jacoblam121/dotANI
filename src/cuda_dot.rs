use anyhow::{bail, Context, Result};
use cudarc::driver::{CudaContext, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use log::info;
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
#define BLOCK_K 64
#endif

extern "C" __global__
void dot_rect_i32_i64_tiled(
    const int* __restrict__ query_hv,   // [nq * d]
    const int* __restrict__ ref_hv,     // [nr * d]
    int nq,
    int nr,
    int d,
    long long* __restrict__ out // [nq * nr], row-major
) {
    const int local_j = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_i = blockIdx.y * blockDim.y + threadIdx.y;
    const bool valid = (local_i < nq) && (local_j < nr);

    extern __shared__ int smem[];
    int* As = smem;                     // BLOCK_M * BLOCK_K
    int* Bs = As + (BLOCK_M * BLOCK_K); // BLOCK_N * BLOCK_K

    long long acc = 0;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;

    for (int k0 = 0; k0 < d; k0 += BLOCK_K) {
        const int bk = min(BLOCK_K, d - k0);

        // Load A tile
        const int total_a = blockDim.y * bk;
        for (int idx = tid; idx < total_a; idx += nthreads) {
            const int row = idx / bk;
            const int kk  = idx - row * bk;

            const int gqi = (int)blockIdx.y * (int)blockDim.y + row;
            int v = 0;
            if (gqi < nq) {
                v = query_hv[(size_t)gqi * (size_t)d + (size_t)(k0 + kk)];
            }
            As[row * BLOCK_K + kk] = v;
        }

        // Load B tile
        const int total_b = blockDim.x * bk;
        for (int idx = tid; idx < total_b; idx += nthreads) {
            const int col = idx / bk;
            const int kk  = idx - col * bk;

            const int grj = (int)blockIdx.x * (int)blockDim.x + col;
            int v = 0;
            if (grj < nr) {
                v = ref_hv[(size_t)grj * (size_t)d + (size_t)(k0 + kk)];
            }
            Bs[col * BLOCK_K + kk] = v;
        }

        __syncthreads();

        if (valid) {
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

    if (valid) {
        out[(size_t)local_i * (size_t)nr + (size_t)local_j] = acc;
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
fn shared_mem_bytes_i32(block_m: usize, block_n: usize, block_k: usize) -> u32 {
    ((block_m * block_k + block_n * block_k) * std::mem::size_of::<i32>()) as u32
}

pub struct GpuDotExecutor {
    gpu_id: usize,
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
}

impl GpuDotExecutor {
    pub fn new(gpu_id: usize) -> Result<Self> {
        let t0 = Instant::now();

        let ctx = Arc::new(CudaContext::new(gpu_id)?);
        let ptx = compile_ptx(KERNEL_SRC)?;
        let module = Arc::new(ctx.load_module(ptx)?);

        info!(
            "Initialized GPU dot executor on gpu_id={} in {:.3}s",
            gpu_id,
            t0.elapsed().as_secs_f64()
        );

        Ok(Self {
            gpu_id,
            ctx,
            module,
        })
    }

    pub fn compute_tile(
        &self,
        query_hv: &[i32],
        nq: usize,
        ref_hv: &[i32],
        nr: usize,
        d: usize,
        out: &mut [i64],
    ) -> Result<()> {
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

        let t0 = Instant::now();

        let stream = self.ctx.default_stream();
        let func = self
            .module
            .load_function("dot_rect_i32_i64_tiled")
            .context("load function dot_rect_i32_i64_tiled")?;

        let d_query: CudaSlice<i32> = stream.clone_htod(query_hv)?;
        let d_ref: CudaSlice<i32> = stream.clone_htod(ref_hv)?;
        let mut d_out: CudaSlice<i64> = stream.alloc_zeros(nq * nr)?;

        let blk_x = 16usize;
        let blk_y = 16usize;
        let blk_k = 64usize;
        let smem_bytes = shared_mem_bytes_i32(blk_y, blk_x, blk_k);

        let cfg = LaunchConfig {
            grid_dim: (nr.div_ceil(blk_x) as u32, nq.div_ceil(blk_y) as u32, 1),
            block_dim: (blk_x as u32, blk_y as u32, 1),
            shared_mem_bytes: smem_bytes,
        };

        let nq_i32 = nq as i32;
        let nr_i32 = nr as i32;
        let d_i32 = d as i32;

        let mut launch = stream.launch_builder(&func);
        launch.arg(&d_query);
        launch.arg(&d_ref);
        launch.arg(&nq_i32);
        launch.arg(&nr_i32);
        launch.arg(&d_i32);
        launch.arg(&mut d_out);

        unsafe { launch.launch(cfg) }?;
        stream.memcpy_dtoh(&d_out, out)?;

        info!(
            "GPU dot tile done on gpu_id={} nq={} nr={} d={} query={:.2} MiB ref={:.2} MiB out={:.2} MiB in {:.3}s",
            self.gpu_id,
            nq,
            nr,
            d,
            mib(query_hv.len() * std::mem::size_of::<i32>()),
            mib(ref_hv.len() * std::mem::size_of::<i32>()),
            mib(out.len() * std::mem::size_of::<i64>()),
            t0.elapsed().as_secs_f64()
        );

        Ok(())
    }
}