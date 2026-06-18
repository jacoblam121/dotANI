use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::types::{FileSketch, FileUllSketch};

pub const HD_MAGIC: &[u8; 8] = b"DOTCHNK1";
pub const ULL_MAGIC: &[u8; 8] = b"DOTULCH1";
const FORMAT_VERSION: u32 = 1;
pub const DEFAULT_CHUNK_RECORDS: u32 = 32_768;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ChunkedRecordKind {
    Hd,
    Ull,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChunkedHeader {
    pub format_version: u32,
    pub record_kind: ChunkedRecordKind,
    pub record_count: u64,
    pub chunk_count: u32,
    pub chunk_records: u32,
    pub ksize: u8,
    pub canonical: bool,
    pub seed: u64,
    pub scaled: Option<u64>,
    pub hv_d: Option<usize>,
    pub hv_quant_bits: Option<u8>,
    pub ull_p: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkedMetadata {
    pub header: ChunkedHeader,
    pub chunk_offsets: Vec<u64>,
}

pub fn is_chunked_hd_path(path: &Path) -> Result<bool> {
    path_has_magic(path, HD_MAGIC)
}

pub fn is_chunked_ull_path(path: &Path) -> Result<bool> {
    path_has_magic(path, ULL_MAGIC)
}

pub fn convert_legacy_sketch(input_hd: &Path, output_hd: &Path, chunk_records: u32) -> Result<()> {
    if chunk_records == 0 {
        bail!("--chunk-records must be greater than 0");
    }

    if is_chunked_hd_path(input_hd)? {
        bail!(
            "input HD sketch {} is already in chunked format; only legacy-to-chunked conversion is supported",
            input_hd.display()
        );
    }

    let input_ull = ull_sidecar_path(input_hd);
    let output_ull = ull_sidecar_path(output_hd);

    if is_chunked_ull_path(&input_ull)? {
        bail!(
            "input ULL sketch {} is already in chunked format; only legacy-to-chunked conversion is supported",
            input_ull.display()
        );
    }

    let hd = load_legacy_hd(input_hd)?;
    let ull = load_legacy_ull(&input_ull)?;
    validate_hd_ull_pair(&hd, &ull)?;

    write_chunked_hd(output_hd, &hd, chunk_records)?;
    write_chunked_ull(&output_ull, &ull, chunk_records)?;
    Ok(())
}

pub fn load_chunked_hd(path: &Path) -> Result<Vec<FileSketch>> {
    let metadata = read_hd_metadata(path)?;
    let mut records = Vec::with_capacity(metadata.header.record_count as usize);
    for idx in 0..metadata.header.chunk_count {
        records.extend(read_hd_chunk(path, &metadata, idx)?);
    }
    Ok(records)
}

pub fn load_chunked_ull(path: &Path) -> Result<Vec<FileUllSketch>> {
    let metadata = read_ull_metadata(path)?;
    let mut records = Vec::with_capacity(metadata.header.record_count as usize);
    for idx in 0..metadata.header.chunk_count {
        records.extend(read_ull_chunk(path, &metadata, idx)?);
    }
    Ok(records)
}

pub fn read_hd_metadata(path: &Path) -> Result<ChunkedMetadata> {
    read_metadata(path, HD_MAGIC, ChunkedRecordKind::Hd)
}

pub fn read_ull_metadata(path: &Path) -> Result<ChunkedMetadata> {
    read_metadata(path, ULL_MAGIC, ChunkedRecordKind::Ull)
}

pub fn read_hd_chunk(
    path: &Path,
    metadata: &ChunkedMetadata,
    chunk_idx: u32,
) -> Result<Vec<FileSketch>> {
    ensure_kind(&metadata.header, ChunkedRecordKind::Hd)?;
    let payload = read_chunk_payload(path, metadata, chunk_idx)?;
    bincode::deserialize::<Vec<FileSketch>>(&payload).map_err(|e| {
        anyhow!(
            "failed to deserialize HD chunk {} from {}: {}",
            chunk_idx,
            path.display(),
            e
        )
    })
}

pub fn read_ull_chunk(
    path: &Path,
    metadata: &ChunkedMetadata,
    chunk_idx: u32,
) -> Result<Vec<FileUllSketch>> {
    ensure_kind(&metadata.header, ChunkedRecordKind::Ull)?;
    let compressed = read_chunk_payload(path, metadata, chunk_idx)?;
    let payload = zstd::stream::decode_all(compressed.as_slice()).map_err(|e| {
        anyhow!(
            "failed to zstd-decode ULL chunk {} from {}: {}",
            chunk_idx,
            path.display(),
            e
        )
    })?;
    bincode::deserialize::<Vec<FileUllSketch>>(&payload).map_err(|e| {
        anyhow!(
            "failed to deserialize ULL chunk {} from {}: {}",
            chunk_idx,
            path.display(),
            e
        )
    })
}

pub fn write_chunked_hd(path: &Path, records: &[FileSketch], chunk_records: u32) -> Result<()> {
    if chunk_records == 0 {
        bail!("chunk_records must be greater than 0");
    }
    let header = hd_header(records, chunk_records)?;
    write_chunked_file(
        path,
        HD_MAGIC,
        &header,
        records.chunks(chunk_records as usize),
        |chunk| {
            bincode::serialize(&chunk.to_vec())
                .map_err(|e| anyhow!("failed to serialize HD chunk: {}", e))
        },
    )
}

pub fn write_chunked_ull(path: &Path, records: &[FileUllSketch], chunk_records: u32) -> Result<()> {
    if chunk_records == 0 {
        bail!("chunk_records must be greater than 0");
    }
    let header = ull_header(records, chunk_records)?;
    write_chunked_file(
        path,
        ULL_MAGIC,
        &header,
        records.chunks(chunk_records as usize),
        |chunk| {
            let serialized = bincode::serialize(&chunk.to_vec())
                .map_err(|e| anyhow!("failed to serialize ULL chunk: {}", e))?;
            zstd::stream::encode_all(serialized.as_slice(), 3)
                .map_err(|e| anyhow!("failed to zstd-encode ULL chunk: {}", e))
        },
    )
}

fn path_has_magic(path: &Path, magic: &[u8; 8]) -> Result<bool> {
    let mut file =
        File::open(path).map_err(|e| anyhow!("failed to open {}: {}", path.display(), e))?;
    let mut actual = [0u8; 8];
    match file.read_exact(&mut actual) {
        Ok(()) => Ok(&actual == magic),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(anyhow!(
            "failed to read magic from {}: {}",
            path.display(),
            e
        )),
    }
}

fn ull_sidecar_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.ull", path.to_string_lossy()))
}

fn load_legacy_hd(path: &Path) -> Result<Vec<FileSketch>> {
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow!("failed to read legacy HD sketch {}: {}", path.display(), e))?;
    bincode::deserialize::<Vec<FileSketch>>(&bytes).map_err(|e| {
        anyhow!(
            "failed to deserialize legacy HD sketch {}: {}",
            path.display(),
            e
        )
    })
}

fn load_legacy_ull(path: &Path) -> Result<Vec<FileUllSketch>> {
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow!("failed to read legacy ULL sketch {}: {}", path.display(), e))?;

    if let Ok(serialized) = zstd::stream::decode_all(bytes.as_slice()) {
        if let Ok(v) = bincode::deserialize::<Vec<FileUllSketch>>(&serialized) {
            return Ok(v);
        }
    }

    bincode::deserialize::<Vec<FileUllSketch>>(&bytes).map_err(|e| {
        anyhow!(
            "failed to deserialize legacy ULL sketch {} as zstd-compressed or raw bincode: {}",
            path.display(),
            e
        )
    })
}

fn validate_hd_ull_pair(hd: &[FileSketch], ull: &[FileUllSketch]) -> Result<()> {
    if hd.len() != ull.len() {
        bail!(
            "HD/ULL record count mismatch: HD has {} record(s), ULL has {} record(s)",
            hd.len(),
            ull.len()
        );
    }

    validate_hd_params(hd)?;
    validate_ull_params(ull)?;

    for (idx, (h, u)) in hd.iter().zip(ull.iter()).enumerate() {
        if h.file_str != u.file_str {
            bail!(
                "HD/ULL file_str order mismatch at record {}: HD={:?}, ULL={:?}",
                idx,
                h.file_str,
                u.file_str
            );
        }
    }
    Ok(())
}

fn validate_hd_params(records: &[FileSketch]) -> Result<()> {
    let Some(first) = records.first() else {
        return Ok(());
    };

    for (idx, r) in records.iter().enumerate().skip(1) {
        if r.ksize != first.ksize
            || r.canonical != first.canonical
            || r.seed != first.seed
            || r.scaled != first.scaled
            || r.hv_d != first.hv_d
        {
            bail!(
                "mixed HD sketch parameters at record {}: expected ksize={} canonical={} seed={} scaled={} hv_d={}",
                idx,
                first.ksize,
                first.canonical,
                first.seed,
                first.scaled,
                first.hv_d
            );
        }
    }
    Ok(())
}

fn validate_ull_params(records: &[FileUllSketch]) -> Result<()> {
    let Some(first) = records.first() else {
        return Ok(());
    };

    for (idx, r) in records.iter().enumerate().skip(1) {
        if r.ksize != first.ksize
            || r.canonical != first.canonical
            || r.seed != first.seed
            || r.ull_p != first.ull_p
        {
            bail!(
                "mixed ULL sketch parameters at record {}: expected ksize={} canonical={} seed={} ull_p={}",
                idx,
                first.ksize,
                first.canonical,
                first.seed,
                first.ull_p
            );
        }
    }
    Ok(())
}

fn hd_header(records: &[FileSketch], chunk_records: u32) -> Result<ChunkedHeader> {
    validate_hd_params(records)?;
    let chunk_count = chunk_count(records.len(), chunk_records)?;
    let (ksize, canonical, seed, scaled, hv_d, hv_quant_bits) = if let Some(first) = records.first()
    {
        let hv_quant_bits = records
            .iter()
            .all(|r| r.hv_quant_bits == first.hv_quant_bits)
            .then_some(first.hv_quant_bits);
        (
            first.ksize,
            first.canonical,
            first.seed,
            Some(first.scaled),
            Some(first.hv_d),
            hv_quant_bits,
        )
    } else {
        (0, true, 0, None, None, None)
    };

    Ok(ChunkedHeader {
        format_version: FORMAT_VERSION,
        record_kind: ChunkedRecordKind::Hd,
        record_count: records.len() as u64,
        chunk_count,
        chunk_records,
        ksize,
        canonical,
        seed,
        scaled,
        hv_d,
        hv_quant_bits,
        ull_p: None,
    })
}

fn ull_header(records: &[FileUllSketch], chunk_records: u32) -> Result<ChunkedHeader> {
    validate_ull_params(records)?;
    let chunk_count = chunk_count(records.len(), chunk_records)?;
    let (ksize, canonical, seed, ull_p) = if let Some(first) = records.first() {
        (first.ksize, first.canonical, first.seed, Some(first.ull_p))
    } else {
        (0, true, 0, None)
    };

    Ok(ChunkedHeader {
        format_version: FORMAT_VERSION,
        record_kind: ChunkedRecordKind::Ull,
        record_count: records.len() as u64,
        chunk_count,
        chunk_records,
        ksize,
        canonical,
        seed,
        scaled: None,
        hv_d: None,
        hv_quant_bits: None,
        ull_p,
    })
}

fn chunk_count(record_count: usize, chunk_records: u32) -> Result<u32> {
    let chunk_records = chunk_records as usize;
    let count = record_count.div_ceil(chunk_records);
    u32::try_from(count).map_err(|_| anyhow!("chunk count {} exceeds u32::MAX", count))
}

fn write_chunked_file<'a, T, I, F>(
    path: &Path,
    magic: &[u8; 8],
    header: &ChunkedHeader,
    chunks: I,
    mut serialize_payload: F,
) -> Result<()>
where
    I: IntoIterator<Item = &'a [T]>,
    T: 'a,
    F: FnMut(&[T]) -> Result<Vec<u8>>,
{
    let file = File::create(path)
        .map_err(|e| anyhow!("failed to create chunked sketch {}: {}", path.display(), e))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(magic)?;

    let header_bytes = bincode::serialize(header)
        .map_err(|e| anyhow!("failed to serialize chunked header: {}", e))?;
    let header_len = u32::try_from(header_bytes.len())
        .map_err(|_| anyhow!("chunked header is too large: {} bytes", header_bytes.len()))?;
    writer.write_all(&header_len.to_le_bytes())?;
    writer.write_all(&header_bytes)?;

    let mut offsets = Vec::with_capacity(header.chunk_count as usize);
    for chunk in chunks {
        let offset = writer.stream_position()?;
        offsets.push(offset);
        let payload = serialize_payload(chunk)?;
        writer.write_all(&(payload.len() as u64).to_le_bytes())?;
        writer.write_all(&payload)?;
    }

    if offsets.len() != header.chunk_count as usize {
        bail!(
            "internal chunk count mismatch while writing {}: header={} actual={}",
            path.display(),
            header.chunk_count,
            offsets.len()
        );
    }

    for offset in offsets {
        writer.write_all(&offset.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

fn read_metadata(
    path: &Path,
    magic: &[u8; 8],
    expected_kind: ChunkedRecordKind,
) -> Result<ChunkedMetadata> {
    let mut file = BufReader::new(
        File::open(path).map_err(|e| anyhow!("failed to open {}: {}", path.display(), e))?,
    );

    let mut actual_magic = [0u8; 8];
    file.read_exact(&mut actual_magic)
        .map_err(|e| anyhow!("failed to read magic from {}: {}", path.display(), e))?;
    if &actual_magic != magic {
        bail!(
            "{} does not have the expected chunked magic",
            path.display()
        );
    }

    let mut len_bytes = [0u8; 4];
    file.read_exact(&mut len_bytes).map_err(|e| {
        anyhow!(
            "failed to read header length from {}: {}",
            path.display(),
            e
        )
    })?;
    let header_len = u32::from_le_bytes(len_bytes) as usize;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes).map_err(|e| {
        anyhow!(
            "failed to read chunked header from {}: {}",
            path.display(),
            e
        )
    })?;
    let header: ChunkedHeader = bincode::deserialize(&header_bytes).map_err(|e| {
        anyhow!(
            "failed to deserialize chunked header from {}: {}",
            path.display(),
            e
        )
    })?;

    ensure_kind(&header, expected_kind)?;
    if header.format_version != FORMAT_VERSION {
        bail!(
            "unsupported chunked sketch format version {} in {}",
            header.format_version,
            path.display()
        );
    }
    if header.chunk_records == 0 {
        bail!("chunked sketch {} has chunk_records=0", path.display());
    }

    let footer_len = header.chunk_count as u64 * 8;
    let file_len = file.get_ref().metadata()?.len();
    if file_len < footer_len {
        bail!(
            "chunked sketch {} is too short for footer: file_len={} footer_len={}",
            path.display(),
            file_len,
            footer_len
        );
    }
    file.seek(SeekFrom::End(-(footer_len as i64)))?;

    let mut chunk_offsets = Vec::with_capacity(header.chunk_count as usize);
    for _ in 0..header.chunk_count {
        let mut offset_bytes = [0u8; 8];
        file.read_exact(&mut offset_bytes)?;
        chunk_offsets.push(u64::from_le_bytes(offset_bytes));
    }

    Ok(ChunkedMetadata {
        header,
        chunk_offsets,
    })
}

fn ensure_kind(header: &ChunkedHeader, expected: ChunkedRecordKind) -> Result<()> {
    if header.record_kind != expected {
        bail!(
            "chunked sketch record kind mismatch: expected {:?}, found {:?}",
            expected,
            header.record_kind
        );
    }
    Ok(())
}

fn read_chunk_payload(path: &Path, metadata: &ChunkedMetadata, chunk_idx: u32) -> Result<Vec<u8>> {
    let Some(offset) = metadata.chunk_offsets.get(chunk_idx as usize).copied() else {
        bail!(
            "chunk index {} out of range for {} chunk(s)",
            chunk_idx,
            metadata.header.chunk_count
        );
    };

    let mut file = BufReader::new(
        File::open(path).map_err(|e| anyhow!("failed to open {}: {}", path.display(), e))?,
    );
    file.seek(SeekFrom::Start(offset))?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let payload_len = u64::from_le_bytes(len_bytes);
    let mut payload = vec![0u8; payload_len as usize];
    file.read_exact(&mut payload)?;
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_dir(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "dotani_chunked_sketch_test_{}_{}_{}",
            std::process::id(),
            name,
            unique
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn hd_record(file_str: &str) -> FileSketch {
        FileSketch {
            ksize: 16,
            scaled: 1,
            canonical: true,
            seed: 1447,
            hv_d: 4,
            hv_quant_bits: 16,
            hv_norm_2: 30,
            file_str: file_str.to_string(),
            hv: vec![1, -2, 3, -4],
        }
    }

    fn ull_record(file_str: &str) -> FileUllSketch {
        FileUllSketch {
            ksize: 16,
            canonical: true,
            seed: 1447,
            ull_p: 14,
            file_str: file_str.to_string(),
            ull_state: vec![1, 2, 3, 4],
        }
    }

    fn write_legacy_pair(dir: &Path, hd: &[FileSketch], ull: &[FileUllSketch]) -> PathBuf {
        let hd_path = dir.join("legacy.sketch");
        let ull_path = ull_sidecar_path(&hd_path);
        fs::write(&hd_path, bincode::serialize(&hd.to_vec()).unwrap()).unwrap();
        let ull_bytes = bincode::serialize(&ull.to_vec()).unwrap();
        fs::write(
            &ull_path,
            zstd::stream::encode_all(ull_bytes.as_slice(), 3).unwrap(),
        )
        .unwrap();
        hd_path
    }

    fn assert_round_trip(hd: Vec<FileSketch>, ull: Vec<FileUllSketch>, chunk_records: u32) {
        let dir = test_dir("round_trip");
        let hd_path = dir.join("chunked.sketch");
        let ull_path = ull_sidecar_path(&hd_path);

        write_chunked_hd(&hd_path, &hd, chunk_records).unwrap();
        write_chunked_ull(&ull_path, &ull, chunk_records).unwrap();

        assert_eq!(load_chunked_hd(&hd_path).unwrap(), hd);
        assert_eq!(load_chunked_ull(&ull_path).unwrap(), ull);
        assert_eq!(utils::load_sketch(&hd_path), hd);
        assert_eq!(utils::load_ull_sketch(&ull_path), ull);
    }

    #[test]
    fn empty_hd_and_ull_round_trip() {
        assert_round_trip(Vec::new(), Vec::new(), 2);
    }

    #[test]
    fn one_record_hd_and_ull_round_trip() {
        assert_round_trip(vec![hd_record("a")], vec![ull_record("a")], 2);
    }

    #[test]
    fn exact_chunk_size_round_trip() {
        let hd = vec![hd_record("a"), hd_record("b")];
        let ull = vec![ull_record("a"), ull_record("b")];
        assert_round_trip(hd, ull, 2);
    }

    #[test]
    fn chunk_size_plus_one_round_trip() {
        let hd = vec![hd_record("a"), hd_record("b"), hd_record("c")];
        let ull = vec![ull_record("a"), ull_record("b"), ull_record("c")];
        assert_round_trip(hd, ull, 2);
    }

    #[test]
    fn footer_offsets_seek_to_expected_chunks() {
        let dir = test_dir("footer");
        let path = dir.join("chunked.sketch");
        let hd = vec![hd_record("a"), hd_record("b"), hd_record("c")];
        write_chunked_hd(&path, &hd, 2).unwrap();

        let metadata = read_hd_metadata(&path).unwrap();
        assert_eq!(metadata.chunk_offsets.len(), 2);
        assert_eq!(read_hd_chunk(&path, &metadata, 0).unwrap(), hd[..2]);
        assert_eq!(read_hd_chunk(&path, &metadata, 1).unwrap(), hd[2..]);

        let mut file = File::open(&path).unwrap();
        for offset in metadata.chunk_offsets {
            file.seek(SeekFrom::Start(offset)).unwrap();
            let mut len_bytes = [0u8; 8];
            file.read_exact(&mut len_bytes).unwrap();
            assert!(u64::from_le_bytes(len_bytes) > 0);
        }
    }

    #[test]
    fn legacy_loaders_still_load_legacy_files() {
        let dir = test_dir("legacy");
        let hd = vec![hd_record("a")];
        let ull = vec![ull_record("a")];
        let hd_path = write_legacy_pair(&dir, &hd, &ull);
        let ull_path = ull_sidecar_path(&hd_path);

        assert_eq!(utils::load_sketch(&hd_path), hd);
        assert_eq!(utils::load_ull_sketch(&ull_path), ull);
    }

    #[test]
    fn chunked_loaders_return_identical_vectors_after_conversion() {
        let dir = test_dir("convert");
        let hd = vec![hd_record("a"), hd_record("b"), hd_record("c")];
        let ull = vec![ull_record("a"), ull_record("b"), ull_record("c")];
        let legacy_hd = write_legacy_pair(&dir, &hd, &ull);
        let chunked_hd = dir.join("converted.sketch");

        convert_legacy_sketch(&legacy_hd, &chunked_hd, 2).unwrap();

        assert_eq!(
            utils::load_sketch(&legacy_hd),
            utils::load_sketch(&chunked_hd)
        );
        assert_eq!(
            utils::load_ull_sketch(&ull_sidecar_path(&legacy_hd)),
            utils::load_ull_sketch(&ull_sidecar_path(&chunked_hd))
        );
    }

    #[test]
    fn converter_rejects_mixed_hd_parameters() {
        let dir = test_dir("mixed_hd");
        let mut hd = vec![hd_record("a"), hd_record("b")];
        hd[1].hv_d = 8;
        let ull = vec![ull_record("a"), ull_record("b")];
        let legacy_hd = write_legacy_pair(&dir, &hd, &ull);
        let err = convert_legacy_sketch(&legacy_hd, &dir.join("out.sketch"), 2).unwrap_err();
        assert!(err.to_string().contains("mixed HD sketch parameters"));
    }

    #[test]
    fn converter_rejects_mixed_ull_parameters() {
        let dir = test_dir("mixed_ull");
        let hd = vec![hd_record("a"), hd_record("b")];
        let mut ull = vec![ull_record("a"), ull_record("b")];
        ull[1].ull_p = 15;
        let legacy_hd = write_legacy_pair(&dir, &hd, &ull);
        let err = convert_legacy_sketch(&legacy_hd, &dir.join("out.sketch"), 2).unwrap_err();
        assert!(err.to_string().contains("mixed ULL sketch parameters"));
    }

    #[test]
    fn converter_rejects_hd_ull_count_mismatch() {
        let dir = test_dir("count_mismatch");
        let hd = vec![hd_record("a"), hd_record("b")];
        let ull = vec![ull_record("a")];
        let legacy_hd = write_legacy_pair(&dir, &hd, &ull);
        let err = convert_legacy_sketch(&legacy_hd, &dir.join("out.sketch"), 2).unwrap_err();
        assert!(err.to_string().contains("record count mismatch"));
    }

    #[test]
    fn converter_rejects_hd_ull_file_str_order_mismatch() {
        let dir = test_dir("order_mismatch");
        let hd = vec![hd_record("a"), hd_record("b")];
        let ull = vec![ull_record("b"), ull_record("a")];
        let legacy_hd = write_legacy_pair(&dir, &hd, &ull);
        let err = convert_legacy_sketch(&legacy_hd, &dir.join("out.sketch"), 2).unwrap_err();
        assert!(err.to_string().contains("file_str order mismatch"));
    }
}
