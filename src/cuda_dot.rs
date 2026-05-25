use anyhow::{Context, Result, bail};
use cudarc::driver::{CudaContext, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
#ifndef BLOCK_M
#define BLOCK_M 64
#endif

#ifndef BLOCK_N
#define BLOCK_N 32
#endif

#ifndef BLOCK_K
#define BLOCK_K 32
#endif

#ifndef THREAD_M
#define THREAD_M 4
#endif

#ifndef THREAD_N
#define THREAD_N 2
#endif

#ifndef PAD_A
#define PAD_A 1
#endif

#ifndef PAD_B
#define PAD_B 1
#endif

struct __align__(16) Int4 {
    int x, y, z, w;
};

extern "C" __global__
void dot_rect_i32_i64_tiled_rb(
    const int* __restrict__ query_hv,   // [nq * d]
    const int* __restrict__ ref_hv,     // [nr * d]
    int nq,
    int nr,
    int d,
    long long* __restrict__ out         // [nq * nr], row-major
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;

    const int local_row0 = ty * THREAD_M;
    const int local_col0 = tx * THREAD_N;

    const int global_row0 = block_row + local_row0;
    const int global_col0 = block_col + local_col0;

    const int STRIDE_A = BLOCK_K + PAD_A;
    const int STRIDE_B = BLOCK_K + PAD_B;

    extern __shared__ int smem[];
    int* As = smem;
    int* Bs = As + (BLOCK_M * STRIDE_A);

    long long acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            acc[i][j] = 0;
        }
    }

    const int tid = ty * blockDim.x + tx;
    const int nthreads = blockDim.x * blockDim.y;

    for (int k0 = 0; k0 < d; k0 += BLOCK_K) {
        const int bk = ((BLOCK_K < (d - k0)) ? BLOCK_K : (d - k0));

        // Load A tile
        const int vec_bk = bk / 4;
        const int total_a_vec = BLOCK_M * vec_bk;

        for (int idx = tid; idx < total_a_vec; idx += nthreads) {
            const int row = idx / vec_bk;
            const int kk4 = (idx - row * vec_bk) * 4;

            const int g_row = block_row + row;

            Int4 v;
            v.x = 0; v.y = 0; v.z = 0; v.w = 0;

            if (g_row < nq) {
                const Int4* gptr = (const Int4*)(query_hv + (size_t)g_row * (size_t)d + (size_t)(k0 + kk4));
                v = *gptr;
            }

            int* sptr = As + row * STRIDE_A + kk4;
            sptr[0] = v.x;
            sptr[1] = v.y;
            sptr[2] = v.z;
            sptr[3] = v.w;
        }

        const int kk_tail_a = vec_bk * 4;
        if (bk > kk_tail_a) {
            const int tail_cols = bk - kk_tail_a;
            const int total_a_tail = BLOCK_M * tail_cols;
            for (int idx = tid; idx < total_a_tail; idx += nthreads) {
                const int row = idx / tail_cols;
                const int kk  = kk_tail_a + (idx - row * tail_cols);

                const int g_row = block_row + row;
                int v = 0;
                if (g_row < nq) {
                    v = query_hv[(size_t)g_row * (size_t)d + (size_t)(k0 + kk)];
                }
                As[row * STRIDE_A + kk] = v;
            }
        }

        // Load B tile
        const int total_b_vec = BLOCK_N * vec_bk;

        for (int idx = tid; idx < total_b_vec; idx += nthreads) {
            const int col = idx / vec_bk;
            const int kk4 = (idx - col * vec_bk) * 4;

            const int g_col = block_col + col;

            Int4 v;
            v.x = 0; v.y = 0; v.z = 0; v.w = 0;

            if (g_col < nr) {
                const Int4* gptr = (const Int4*)(ref_hv + (size_t)g_col * (size_t)d + (size_t)(k0 + kk4));
                v = *gptr;
            }

            int* sptr = Bs + col * STRIDE_B + kk4;
            sptr[0] = v.x;
            sptr[1] = v.y;
            sptr[2] = v.z;
            sptr[3] = v.w;
        }

        const int kk_tail_b = vec_bk * 4;
        if (bk > kk_tail_b) {
            const int tail_cols = bk - kk_tail_b;
            const int total_b_tail = BLOCK_N * tail_cols;
            for (int idx = tid; idx < total_b_tail; idx += nthreads) {
                const int col = idx / tail_cols;
                const int kk  = kk_tail_b + (idx - col * tail_cols);

                const int g_col = block_col + col;
                int v = 0;
                if (g_col < nr) {
                    v = ref_hv[(size_t)g_col * (size_t)d + (size_t)(k0 + kk)];
                }
                Bs[col * STRIDE_B + kk] = v;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            if (kk >= bk) break;

            int a_frag[THREAD_M];
            int b_frag[THREAD_N];

            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                a_frag[i] = As[(local_row0 + i) * STRIDE_A + kk];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_N; ++j) {
                b_frag[j] = Bs[(local_col0 + j) * STRIDE_B + kk];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; ++j) {
                    acc[i][j] += (long long)a_frag[i] * (long long)b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        const int g_row = global_row0 + i;
        if (g_row >= nq) continue;

        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            const int g_col = global_col0 + j;
            if (g_col >= nr) continue;

            out[(size_t)g_row * (size_t)nr + (size_t)g_col] = acc[i][j];
        }
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
    let stride_a = block_k + 1;
    let stride_b = block_k + 1;
    ((block_m * stride_a + block_n * stride_b) * std::mem::size_of::<i32>()) as u32
}

pub struct GpuDotExecutor {
    gpu_id: usize,
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,

    d_query: Option<CudaSlice<i32>>,
    d_ref: Option<CudaSlice<i32>>,
    d_out: Option<CudaSlice<i64>>,

    cap_query: usize,
    cap_ref: usize,
    cap_out: usize,
}

impl GpuDotExecutor {
    pub fn new(gpu_id: usize) -> Result<Self> {
        let t0 = Instant::now();

        let ctx = CudaContext::new(gpu_id)?;
        let ptx = compile_ptx(KERNEL_SRC)?;
        let module = ctx.load_module(ptx)?;

        log::debug!(
            "Initialized GPU dot executor on gpu_id={} in {:.3}s",
            gpu_id,
            t0.elapsed().as_secs_f64()
        );

        Ok(Self {
            gpu_id,
            ctx,
            module,
            d_query: None,
            d_ref: None,
            d_out: None,
            cap_query: 0,
            cap_ref: 0,
            cap_out: 0,
        })
    }

    fn ensure_query_capacity(&mut self, len: usize) -> Result<()> {
        if self.cap_query < len {
            let stream = self.ctx.default_stream();
            self.d_query = Some(stream.alloc_zeros::<i32>(len)?);
            self.cap_query = len;
            log::debug!(
                "gpu_id={} grow d_query to {} elems ({:.2} MiB)",
                self.gpu_id,
                len,
                mib(len * std::mem::size_of::<i32>())
            );
        }
        Ok(())
    }

    fn ensure_ref_capacity(&mut self, len: usize) -> Result<()> {
        if self.cap_ref < len {
            let stream = self.ctx.default_stream();
            self.d_ref = Some(stream.alloc_zeros::<i32>(len)?);
            self.cap_ref = len;
            log::debug!(
                "gpu_id={} grow d_ref to {} elems ({:.2} MiB)",
                self.gpu_id,
                len,
                mib(len * std::mem::size_of::<i32>())
            );
        }
        Ok(())
    }

    fn ensure_out_capacity(&mut self, len: usize) -> Result<()> {
        if self.cap_out < len {
            let stream = self.ctx.default_stream();
            self.d_out = Some(stream.alloc_zeros::<i64>(len)?);
            self.cap_out = len;
            log::debug!(
                "gpu_id={} grow d_out to {} elems ({:.2} MiB)",
                self.gpu_id,
                len,
                mib(len * std::mem::size_of::<i64>())
            );
        }
        Ok(())
    }

    pub fn compute_tile(
        &mut self,
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

        self.ensure_query_capacity(query_hv.len())?;
        self.ensure_ref_capacity(ref_hv.len())?;
        self.ensure_out_capacity(out.len())?;

        let stream = self.ctx.default_stream();
        let func = self
            .module
            .load_function("dot_rect_i32_i64_tiled_rb")
            .context("load function dot_rect_i32_i64_tiled_rb")?;

        let d_query = self
            .d_query
            .as_mut()
            .context("internal error: d_query missing after allocation")?;
        let d_ref = self
            .d_ref
            .as_mut()
            .context("internal error: d_ref missing after allocation")?;
        let d_out = self
            .d_out
            .as_mut()
            .context("internal error: d_out missing after allocation")?;

        // Reuse persistent buffers.
        //
        // If your cudarc version names these differently, the only thing you
        // may need to adapt is these two H2D copies.
        stream.memcpy_htod(query_hv, d_query)?;
        stream.memcpy_htod(ref_hv, d_ref)?;

        const BLOCK_M: usize = 64;
        const BLOCK_N: usize = 32;
        const BLOCK_K: usize = 32;
        const THREAD_M: usize = 4;
        const THREAD_N: usize = 2;

        const BLK_X: usize = BLOCK_N / THREAD_N; // 16
        const BLK_Y: usize = BLOCK_M / THREAD_M; // 16

        let smem_bytes = shared_mem_bytes_i32(BLOCK_M, BLOCK_N, BLOCK_K);

        let cfg = LaunchConfig {
            grid_dim: (nr.div_ceil(BLOCK_N) as u32, nq.div_ceil(BLOCK_M) as u32, 1),
            block_dim: (BLK_X as u32, BLK_Y as u32, 1),
            shared_mem_bytes: smem_bytes,
        };

        let nq_i32 = nq as i32;
        let nr_i32 = nr as i32;
        let d_i32 = d as i32;

        let mut launch = stream.launch_builder(&func);
        launch.arg(d_query);
        launch.arg(d_ref);
        launch.arg(&nq_i32);
        launch.arg(&nr_i32);
        launch.arg(&d_i32);
        launch.arg(&mut *d_out);

        unsafe { launch.launch(cfg) }?;

        // Fast common path: full-size tile matches current capacity.
        if self.cap_out == out.len() {
            stream.memcpy_dtoh(d_out, out)?;
        } else {
            // Edge tiles: copy full persistent buffer to temp, then take prefix.
            let mut tmp = vec![0i64; self.cap_out];
            stream.memcpy_dtoh(d_out, &mut tmp)?;
            out.copy_from_slice(&tmp[..out.len()]);
        }

        log::debug!(
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
