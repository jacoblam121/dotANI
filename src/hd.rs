use log::info;
use std::collections::HashSet;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::types::FileSketch;
use rand::{RngCore, SeedableRng};
use wyhash::WyRng;

extern crate bitpacking;
use bitpacking::{BitPacker, BitPacker8x};

use rayon::prelude::*;

#[inline]
pub fn encode_hash_hd_auto(kmer_hash_set: &HashSet<u64>, sketch: &FileSketch) -> Vec<i32> {
    assert!(
        sketch.hv_d % 64 == 0,
        "hv_d must be a multiple of 64, got {}",
        sketch.hv_d
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                return encode_hash_hd_avx512(kmer_hash_set, sketch);
            }
        }
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return encode_hash_hd_avx2(kmer_hash_set, sketch);
            }
        }
    }

    encode_hash_hd(kmer_hash_set, sketch)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn encode_hash_hd_avx512(
    kmer_hash_set: &HashSet<u64>,
    sketch: &FileSketch,
) -> Vec<i32> {
    let hv_d = sketch.hv_d;
    assert!(hv_d % 64 == 0, "hv_d must be a multiple of 64");

    let num_seed = kmer_hash_set.len();
    let mut hv = vec![-(num_seed as i32); hv_d];

    let num_tail = num_seed % 8;
    let num_seed_round_8 = num_seed + if num_tail == 0 { 0 } else { 8 - num_tail };
    let num_batch_round_8 = num_seed_round_8 / 8;
    let num_chunk = hv_d / 64;

    let mut seed_vec: Vec<u64> = kmer_hash_set.iter().copied().collect();
    seed_vec.resize(num_seed_round_8, 0);

    let mut rng_vec = vec![WyRng::default(); 8];
    let mut rnd_vec = [0u64; 8];
    let mut lane_buf = [0i64; 8];

    for b_i in 0..num_batch_round_8 {
        for j in 0..8 {
            rng_vec[j] = WyRng::seed_from_u64(seed_vec[b_i * 8 + j]);
        }

        for chunk_i in 0..num_chunk {
            for j in 0..8 {
                rnd_vec[j] = rng_vec[j].next_u64();
            }

            if b_i == num_batch_round_8 - 1 && num_tail > 0 {
                for x in rnd_vec.iter_mut().skip(num_tail) {
                    *x = 0;
                }
            }

            let v = _mm512_set_epi64(
                rnd_vec[7] as i64,
                rnd_vec[6] as i64,
                rnd_vec[5] as i64,
                rnd_vec[4] as i64,
                rnd_vec[3] as i64,
                rnd_vec[2] as i64,
                rnd_vec[1] as i64,
                rnd_vec[0] as i64,
            );

            for bit in 0..64usize {
                let shifted = _mm512_srli_epi64(v, bit as u32);
                _mm512_storeu_si512(lane_buf.as_mut_ptr() as *mut __m512i, shifted);

                let mut ones = 0i32;
                for &lane in &lane_buf {
                    ones += ((lane as u64) & 1) as i32;
                }

                hv[chunk_i * 64 + bit] += ones << 1;
            }
        }
    }

    hv
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn encode_hash_hd_avx2(kmer_hash_set: &HashSet<u64>, sketch: &FileSketch) -> Vec<i32> {
    let hv_d = sketch.hv_d;
    assert!(hv_d % 64 == 0, "hv_d must be a multiple of 64");

    let mm256_const_one_epi16 = _mm256_set1_epi16(1);
    let mm256_const_zero = _mm256_setzero_si256();
    let shuffle_mask = _mm256_set_epi8(
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0, 15, 14, 7, 6, 13, 12, 5, 4, 11, 10,
        3, 2, 9, 8, 1, 0,
    );

    let mut rng_vec = vec![WyRng::default(); 4];
    let mut rnd_vec = [0u64; 4];

    let num_seed = kmer_hash_set.len();
    let mut hv = vec![-(num_seed as i32); hv_d];

    let num_tail = num_seed % 4;
    let num_seed_round_4 = num_seed + if num_tail == 0 { 0 } else { 4 - num_tail };
    let num_batch_round_4 = num_seed_round_4 / 4;
    let num_chunk = hv_d / 64;

    let mut seed_vec: Vec<u64> = kmer_hash_set.iter().copied().collect();
    seed_vec.resize(num_seed_round_4, 0);

    for b_i in 0..num_batch_round_4 {
        for j in 0..4 {
            rng_vec[j] = WyRng::seed_from_u64(seed_vec[b_i * 4 + j]);
        }

        for i in 0..num_chunk {
            for j in 0..4 {
                rnd_vec[j] = rng_vec[j].next_u64();
            }

            if b_i == num_batch_round_4 - 1 && num_tail > 0 {
                for x in rnd_vec.iter_mut().skip(num_tail) {
                    *x = 0;
                }
            }

            let simd_rnd_4_shuffle = _mm256_shuffle_epi8(
                _mm256_set_epi64x(
                    rnd_vec[3] as i64,
                    rnd_vec[2] as i64,
                    rnd_vec[1] as i64,
                    rnd_vec[0] as i64,
                ),
                shuffle_mask,
            );

            for k in 0..16usize {
                let shift_and_256 = _mm256_and_si256(
                    _mm256_srl_epi16(simd_rnd_4_shuffle, _mm_set1_epi64x(k as i64)),
                    mm256_const_one_epi16,
                );

                let mut hadd_ = _mm256_hadd_epi16(shift_and_256, mm256_const_zero);
                hadd_ = _mm256_permute4x64_epi64(hadd_, 0xD8);
                hadd_ = _mm256_shuffle_epi8(hadd_, shuffle_mask);
                hadd_ = _mm256_hadd_epi16(hadd_, mm256_const_zero);
                hadd_ = _mm256_slli_epi16(hadd_, 1);

                hv[i * 64 + k * 4] += _mm256_extract_epi16::<0>(hadd_) as i32;
                hv[i * 64 + k * 4 + 1] += _mm256_extract_epi16::<1>(hadd_) as i32;
                hv[i * 64 + k * 4 + 2] += _mm256_extract_epi16::<2>(hadd_) as i32;
                hv[i * 64 + k * 4 + 3] += _mm256_extract_epi16::<3>(hadd_) as i32;
            }
        }
    }

    hv
}

pub fn encode_hash_hd(kmer_hash_set: &HashSet<u64>, sketch: &FileSketch) -> Vec<i32> {
    let hv_d = sketch.hv_d;
    assert!(hv_d % 64 == 0, "hv_d must be a multiple of 64");

    let mut hv = vec![-(kmer_hash_set.len() as i32); hv_d];

    for &hash in kmer_hash_set {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn compress_hd_sketch(sketch: &mut FileSketch, hv: &Vec<i32>) -> u8 {
    let hv_d = sketch.hv_d;
    assert!(hv_d % 32 == 0, "hv_d must be a multiple of 32");

    let min_hv = *hv.iter().min().unwrap();
    let max_hv = *hv.iter().max().unwrap();

    let mut quant_bit: u8 = 6;
    loop {
        let quant_min: i64 = -(1i64 << (quant_bit - 1));
        let quant_max: i64 = (1i64 << (quant_bit - 1)) - 1;

        if quant_min <= min_hv as i64 && quant_max >= max_hv as i64 {
            break;
        }

        if quant_bit == 32 {
            break;
        }
        quant_bit += 1;
    }

    if is_x86_feature_detected!("avx2") {
        let offset: i64 = 1i64 << (quant_bit - 1);
        let hv_u32: Vec<u32> = hv.iter().map(|&x| (x as i64 + offset) as u32).collect();

        let bitpacker = BitPacker8x::new();
        let bits_per_block = quant_bit as usize * 32;
        let mut hv_compress_bits = vec![0u8; quant_bit as usize * (hv_d >> 3)];

        for i in 0..(hv_d / BitPacker8x::BLOCK_LEN) {
            bitpacker.compress(
                &hv_u32[i * BitPacker8x::BLOCK_LEN..(i + 1) * BitPacker8x::BLOCK_LEN],
                &mut hv_compress_bits[(bits_per_block * i)..(bits_per_block * (i + 1))],
                quant_bit,
            );
        }

        sketch.hv = hv_compress_bits
            .chunks_exact(4)
            .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
    } else {
        let total_bits = quant_bit as usize * hv_d;
        let len_bit_vec_u32 = total_bits.div_ceil(32);
        let mut hv_compress_bits: Vec<i32> = vec![0; len_bit_vec_u32];

        let offset: i64 = 1i64 << (quant_bit - 1);
        let hv_u32: Vec<u32> = hv.iter().map(|&x| (x as i64 + offset) as u32).collect();

        for bit_idx in 0..total_bits {
            let src_idx = bit_idx / quant_bit as usize;
            let src_bit = bit_idx % quant_bit as usize;
            let bit = ((hv_u32[src_idx] >> src_bit) & 1) as i32;
            hv_compress_bits[bit_idx / 32] |= bit << (bit_idx % 32);
        }

        sketch.hv = hv_compress_bits;
    }

    quant_bit
}

pub fn decompress_file_sketch(file_sketch: &mut Vec<FileSketch>) {
    let hv_dim = file_sketch[0].hv_d;
    info!("Decompressing sketch with HV dim={}", hv_dim);

    file_sketch.par_iter_mut().for_each(|sketch| {
        let hv_decompressed = unsafe { decompress_hd_sketch(sketch) };
        sketch.hv = hv_decompressed;
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn decompress_hd_sketch(sketch: &mut FileSketch) -> Vec<i32> {
    let hv_d = sketch.hv_d;
    let quant_bit = sketch.hv_quant_bits;

    let mut hv_decompressed: Vec<i32> = vec![0; hv_d];

    if is_x86_feature_detected!("avx2") {
        let bitpacker = BitPacker8x::new();
        let bits_per_block = quant_bit as usize * 32;

        let hv_u8: Vec<u8> = sketch
            .hv
            .iter()
            .flat_map(|&x| x.to_ne_bytes())
            .collect();

        let mut hv_u32: Vec<u32> = vec![0; hv_d];

        for i in 0..(hv_d / BitPacker8x::BLOCK_LEN) {
            bitpacker.decompress(
                &hv_u8[(bits_per_block * i)..(bits_per_block * (i + 1))],
                &mut hv_u32[i * BitPacker8x::BLOCK_LEN..(i + 1) * BitPacker8x::BLOCK_LEN],
                quant_bit,
            );
        }

        let offset: i64 = 1i64 << (quant_bit - 1);
        hv_decompressed = hv_u32
            .into_iter()
            .map(|x| (x as i64 - offset) as i32)
            .collect();
    } else {
        let total_bits = quant_bit as usize * hv_d;
        let mut hv_u32: Vec<u32> = vec![0; hv_d];

        for bit_idx in 0..total_bits {
            let bit = ((sketch.hv[bit_idx / 32] as u32) >> (bit_idx % 32)) & 1;
            hv_u32[bit_idx / quant_bit as usize] |= bit << (bit_idx % quant_bit as usize);
        }

        let offset: i64 = 1i64 << (quant_bit - 1);
        for i in 0..hv_d {
            hv_decompressed[i] = (hv_u32[i] as i64 - offset) as i32;
        }
    }

    hv_decompressed
}