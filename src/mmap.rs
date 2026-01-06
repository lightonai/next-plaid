//! Memory-mapped file support for efficient large index loading.
//!
//! This module provides utilities for loading large arrays from disk using
//! memory-mapped files, avoiding the need to load entire arrays into RAM.
//!
//! Two formats are supported:
//! - Custom raw binary format (legacy): 8-byte header with shape, then raw data
//! - NPY format: Standard NumPy format with header, used for index files

use std::fs::File;
use std::path::Path;

#[cfg(feature = "npy")]
use std::collections::HashMap;
#[cfg(feature = "npy")]
use std::fs;
#[cfg(feature = "npy")]
use std::io::{BufReader, BufWriter, Write};

use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{Error, Result};

/// A memory-mapped array of f32 values.
///
/// This struct provides zero-copy access to large arrays stored on disk.
pub struct MmapArray2F32 {
    _mmap: Mmap,
    shape: (usize, usize),
    data_offset: usize,
}

impl MmapArray2F32 {
    /// Load a 2D f32 array from a raw binary file.
    ///
    /// The file format is:
    /// - 8 bytes: nrows (i64 little-endian)
    /// - 8 bytes: ncols (i64 little-endian)
    /// - nrows * ncols * 4 bytes: f32 data (little-endian)
    pub fn from_raw_file(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| Error::IndexLoad(format!("Failed to open file {:?}: {}", path, e)))?;

        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| Error::IndexLoad(format!("Failed to mmap file {:?}: {}", path, e)))?
        };

        if mmap.len() < 16 {
            return Err(Error::IndexLoad("File too small for header".into()));
        }

        // Read shape from header
        let mut cursor = std::io::Cursor::new(&mmap[..16]);
        let nrows = cursor
            .read_i64::<LittleEndian>()
            .map_err(|e| Error::IndexLoad(format!("Failed to read nrows: {}", e)))?
            as usize;
        let ncols = cursor
            .read_i64::<LittleEndian>()
            .map_err(|e| Error::IndexLoad(format!("Failed to read ncols: {}", e)))?
            as usize;

        let expected_size = 16 + nrows * ncols * 4;
        if mmap.len() < expected_size {
            return Err(Error::IndexLoad(format!(
                "File size {} too small for shape ({}, {})",
                mmap.len(),
                nrows,
                ncols
            )));
        }

        Ok(Self {
            _mmap: mmap,
            shape: (nrows, ncols),
            data_offset: 16,
        })
    }

    /// Get the shape of the array.
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get the number of rows.
    pub fn nrows(&self) -> usize {
        self.shape.0
    }

    /// Get the number of columns.
    pub fn ncols(&self) -> usize {
        self.shape.1
    }

    /// Get a view of a row.
    pub fn row(&self, idx: usize) -> ArrayView1<'_, f32> {
        let start = self.data_offset + idx * self.shape.1 * 4;
        let bytes = &self._mmap[start..start + self.shape.1 * 4];

        // Safety: We've verified the bounds and alignment
        let data =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.shape.1) };

        ArrayView1::from_shape(self.shape.1, data).unwrap()
    }

    /// Load a range of rows into an owned Array2.
    pub fn load_rows(&self, start: usize, end: usize) -> Array2<f32> {
        let nrows = end - start;
        let byte_start = self.data_offset + start * self.shape.1 * 4;
        let byte_end = self.data_offset + end * self.shape.1 * 4;
        let bytes = &self._mmap[byte_start..byte_end];

        // Safety: We've verified the bounds
        let data = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, nrows * self.shape.1)
        };

        Array2::from_shape_vec((nrows, self.shape.1), data.to_vec()).unwrap()
    }

    /// Convert to an owned Array2 (loads all data into memory).
    pub fn to_owned(&self) -> Array2<f32> {
        self.load_rows(0, self.shape.0)
    }
}

/// A memory-mapped array of u8 values.
pub struct MmapArray2U8 {
    _mmap: Mmap,
    shape: (usize, usize),
    data_offset: usize,
}

impl MmapArray2U8 {
    /// Load a 2D u8 array from a raw binary file.
    pub fn from_raw_file(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| Error::IndexLoad(format!("Failed to open file {:?}: {}", path, e)))?;

        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| Error::IndexLoad(format!("Failed to mmap file {:?}: {}", path, e)))?
        };

        if mmap.len() < 16 {
            return Err(Error::IndexLoad("File too small for header".into()));
        }

        let mut cursor = std::io::Cursor::new(&mmap[..16]);
        let nrows = cursor
            .read_i64::<LittleEndian>()
            .map_err(|e| Error::IndexLoad(format!("Failed to read nrows: {}", e)))?
            as usize;
        let ncols = cursor
            .read_i64::<LittleEndian>()
            .map_err(|e| Error::IndexLoad(format!("Failed to read ncols: {}", e)))?
            as usize;

        let expected_size = 16 + nrows * ncols;
        if mmap.len() < expected_size {
            return Err(Error::IndexLoad(format!(
                "File size {} too small for shape ({}, {})",
                mmap.len(),
                nrows,
                ncols
            )));
        }

        Ok(Self {
            _mmap: mmap,
            shape: (nrows, ncols),
            data_offset: 16,
        })
    }

    /// Get the shape of the array.
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get a view of the data as ArrayView2.
    pub fn view(&self) -> ArrayView2<'_, u8> {
        let bytes = &self._mmap[self.data_offset..self.data_offset + self.shape.0 * self.shape.1];
        ArrayView2::from_shape(self.shape, bytes).unwrap()
    }

    /// Load a range of rows into an owned Array2.
    pub fn load_rows(&self, start: usize, end: usize) -> Array2<u8> {
        let nrows = end - start;
        let byte_start = self.data_offset + start * self.shape.1;
        let byte_end = self.data_offset + end * self.shape.1;
        let bytes = &self._mmap[byte_start..byte_end];

        Array2::from_shape_vec((nrows, self.shape.1), bytes.to_vec()).unwrap()
    }

    /// Convert to an owned Array2.
    pub fn to_owned(&self) -> Array2<u8> {
        self.load_rows(0, self.shape.0)
    }
}

/// A memory-mapped array of i64 values.
pub struct MmapArray1I64 {
    _mmap: Mmap,
    len: usize,
    data_offset: usize,
}

impl MmapArray1I64 {
    /// Load a 1D i64 array from a raw binary file.
    pub fn from_raw_file(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| Error::IndexLoad(format!("Failed to open file {:?}: {}", path, e)))?;

        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| Error::IndexLoad(format!("Failed to mmap file {:?}: {}", path, e)))?
        };

        if mmap.len() < 8 {
            return Err(Error::IndexLoad("File too small for header".into()));
        }

        let mut cursor = std::io::Cursor::new(&mmap[..8]);
        let len = cursor
            .read_i64::<LittleEndian>()
            .map_err(|e| Error::IndexLoad(format!("Failed to read length: {}", e)))?
            as usize;

        let expected_size = 8 + len * 8;
        if mmap.len() < expected_size {
            return Err(Error::IndexLoad(format!(
                "File size {} too small for length {}",
                mmap.len(),
                len
            )));
        }

        Ok(Self {
            _mmap: mmap,
            len,
            data_offset: 8,
        })
    }

    /// Get the length of the array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a value at an index.
    pub fn get(&self, idx: usize) -> i64 {
        let start = self.data_offset + idx * 8;
        let bytes = &self._mmap[start..start + 8];
        i64::from_le_bytes(bytes.try_into().unwrap())
    }

    /// Convert to an owned Array1.
    pub fn to_owned(&self) -> Array1<i64> {
        let bytes = &self._mmap[self.data_offset..self.data_offset + self.len * 8];

        // Safety: We've verified the bounds
        let data = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i64, self.len) };

        Array1::from_vec(data.to_vec())
    }
}

/// Write an `Array2<f32>` to a raw binary file format.
pub fn write_array2_f32(array: &Array2<f32>, path: &Path) -> Result<()> {
    use std::io::Write;

    let file = File::create(path)
        .map_err(|e| Error::IndexLoad(format!("Failed to create file {:?}: {}", path, e)))?;
    let mut writer = std::io::BufWriter::new(file);

    let nrows = array.nrows() as i64;
    let ncols = array.ncols() as i64;

    writer
        .write_all(&nrows.to_le_bytes())
        .map_err(|e| Error::IndexLoad(format!("Failed to write nrows: {}", e)))?;
    writer
        .write_all(&ncols.to_le_bytes())
        .map_err(|e| Error::IndexLoad(format!("Failed to write ncols: {}", e)))?;

    for val in array.iter() {
        writer
            .write_all(&val.to_le_bytes())
            .map_err(|e| Error::IndexLoad(format!("Failed to write data: {}", e)))?;
    }

    writer
        .flush()
        .map_err(|e| Error::IndexLoad(format!("Failed to flush: {}", e)))?;

    Ok(())
}

/// Write an `Array2<u8>` to a raw binary file format.
pub fn write_array2_u8(array: &Array2<u8>, path: &Path) -> Result<()> {
    use std::io::Write;

    let file = File::create(path)
        .map_err(|e| Error::IndexLoad(format!("Failed to create file {:?}: {}", path, e)))?;
    let mut writer = std::io::BufWriter::new(file);

    let nrows = array.nrows() as i64;
    let ncols = array.ncols() as i64;

    writer
        .write_all(&nrows.to_le_bytes())
        .map_err(|e| Error::IndexLoad(format!("Failed to write nrows: {}", e)))?;
    writer
        .write_all(&ncols.to_le_bytes())
        .map_err(|e| Error::IndexLoad(format!("Failed to write ncols: {}", e)))?;

    for row in array.rows() {
        writer
            .write_all(row.as_slice().unwrap())
            .map_err(|e| Error::IndexLoad(format!("Failed to write data: {}", e)))?;
    }

    writer
        .flush()
        .map_err(|e| Error::IndexLoad(format!("Failed to flush: {}", e)))?;

    Ok(())
}

/// Write an `Array1<i64>` to a raw binary file format.
pub fn write_array1_i64(array: &Array1<i64>, path: &Path) -> Result<()> {
    use std::io::Write;

    let file = File::create(path)
        .map_err(|e| Error::IndexLoad(format!("Failed to create file {:?}: {}", path, e)))?;
    let mut writer = std::io::BufWriter::new(file);

    let len = array.len() as i64;

    writer
        .write_all(&len.to_le_bytes())
        .map_err(|e| Error::IndexLoad(format!("Failed to write length: {}", e)))?;

    for val in array.iter() {
        writer
            .write_all(&val.to_le_bytes())
            .map_err(|e| Error::IndexLoad(format!("Failed to write data: {}", e)))?;
    }

    writer
        .flush()
        .map_err(|e| Error::IndexLoad(format!("Failed to flush: {}", e)))?;

    Ok(())
}

// ============================================================================
// NPY Format Memory-Mapped Arrays
// ============================================================================

/// NPY file magic bytes
const NPY_MAGIC: &[u8] = b"\x93NUMPY";

/// Parse NPY header and return (shape, data_offset, is_fortran_order)
fn parse_npy_header(mmap: &Mmap) -> Result<(Vec<usize>, usize, bool)> {
    if mmap.len() < 10 {
        return Err(Error::IndexLoad("NPY file too small".into()));
    }

    // Check magic
    if &mmap[..6] != NPY_MAGIC {
        return Err(Error::IndexLoad("Invalid NPY magic".into()));
    }

    let major_version = mmap[6];
    let _minor_version = mmap[7];

    // Read header length
    let header_len = if major_version == 1 {
        u16::from_le_bytes([mmap[8], mmap[9]]) as usize
    } else if major_version == 2 {
        if mmap.len() < 12 {
            return Err(Error::IndexLoad("NPY v2 file too small".into()));
        }
        u32::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11]]) as usize
    } else {
        return Err(Error::IndexLoad(format!(
            "Unsupported NPY version: {}",
            major_version
        )));
    };

    let header_start = if major_version == 1 { 10 } else { 12 };
    let header_end = header_start + header_len;

    if mmap.len() < header_end {
        return Err(Error::IndexLoad("NPY header exceeds file size".into()));
    }

    // Parse header dict (simplified Python dict parsing)
    let header_str = std::str::from_utf8(&mmap[header_start..header_end])
        .map_err(|e| Error::IndexLoad(format!("Invalid NPY header encoding: {}", e)))?;

    // Extract shape from header like: {'descr': '<i8', 'fortran_order': False, 'shape': (12345,), }
    let shape = parse_shape_from_header(header_str)?;
    let fortran_order = header_str.contains("'fortran_order': True");

    Ok((shape, header_end, fortran_order))
}

/// Parse shape tuple from NPY header string
fn parse_shape_from_header(header: &str) -> Result<Vec<usize>> {
    // Find 'shape': (...)
    let shape_start = header
        .find("'shape':")
        .ok_or_else(|| Error::IndexLoad("No shape in NPY header".into()))?;

    let after_shape = &header[shape_start + 8..];
    let paren_start = after_shape
        .find('(')
        .ok_or_else(|| Error::IndexLoad("No shape tuple in NPY header".into()))?;
    let paren_end = after_shape
        .find(')')
        .ok_or_else(|| Error::IndexLoad("Unclosed shape tuple in NPY header".into()))?;

    let shape_content = &after_shape[paren_start + 1..paren_end];

    // Parse comma-separated numbers
    let mut shape = Vec::new();
    for part in shape_content.split(',') {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            let dim: usize = trimmed.parse().map_err(|e| {
                Error::IndexLoad(format!("Invalid shape dimension '{}': {}", trimmed, e))
            })?;
            shape.push(dim);
        }
    }

    Ok(shape)
}

/// Memory-mapped NPY array for i64 values (used for codes).
///
/// This struct provides zero-copy access to 1D i64 arrays stored in NPY format.
pub struct MmapNpyArray1I64 {
    _mmap: Mmap,
    len: usize,
    data_offset: usize,
}

impl MmapNpyArray1I64 {
    /// Load a 1D i64 array from an NPY file.
    pub fn from_npy_file(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| Error::IndexLoad(format!("Failed to open NPY file {:?}: {}", path, e)))?;

        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| {
                Error::IndexLoad(format!("Failed to mmap NPY file {:?}: {}", path, e))
            })?
        };

        let (shape, data_offset, _fortran_order) = parse_npy_header(&mmap)?;

        if shape.is_empty() {
            return Err(Error::IndexLoad("Empty shape in NPY file".into()));
        }

        let len = shape[0];

        // Verify file size
        let expected_size = data_offset + len * 8;
        if mmap.len() < expected_size {
            return Err(Error::IndexLoad(format!(
                "NPY file size {} too small for {} elements",
                mmap.len(),
                len
            )));
        }

        Ok(Self {
            _mmap: mmap,
            len,
            data_offset,
        })
    }

    /// Get the length of the array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a slice of the data as &[i64].
    ///
    /// # Safety
    /// The caller must ensure start <= end <= len.
    pub fn slice(&self, start: usize, end: usize) -> &[i64] {
        let byte_start = self.data_offset + start * 8;
        let byte_end = self.data_offset + end * 8;
        let bytes = &self._mmap[byte_start..byte_end];

        // Safety: We've verified bounds and i64 is 8-byte aligned in NPY format
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i64, end - start) }
    }

    /// Get a value at an index.
    pub fn get(&self, idx: usize) -> i64 {
        let start = self.data_offset + idx * 8;
        let bytes = &self._mmap[start..start + 8];
        i64::from_le_bytes(bytes.try_into().unwrap())
    }
}

/// Memory-mapped NPY array for u8 values (used for residuals).
///
/// This struct provides zero-copy access to 2D u8 arrays stored in NPY format.
pub struct MmapNpyArray2U8 {
    _mmap: Mmap,
    shape: (usize, usize),
    data_offset: usize,
}

impl MmapNpyArray2U8 {
    /// Load a 2D u8 array from an NPY file.
    pub fn from_npy_file(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| Error::IndexLoad(format!("Failed to open NPY file {:?}: {}", path, e)))?;

        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| {
                Error::IndexLoad(format!("Failed to mmap NPY file {:?}: {}", path, e))
            })?
        };

        let (shape_vec, data_offset, _fortran_order) = parse_npy_header(&mmap)?;

        if shape_vec.len() != 2 {
            return Err(Error::IndexLoad(format!(
                "Expected 2D array, got {}D",
                shape_vec.len()
            )));
        }

        let shape = (shape_vec[0], shape_vec[1]);

        // Verify file size
        let expected_size = data_offset + shape.0 * shape.1;
        if mmap.len() < expected_size {
            return Err(Error::IndexLoad(format!(
                "NPY file size {} too small for shape {:?}",
                mmap.len(),
                shape
            )));
        }

        Ok(Self {
            _mmap: mmap,
            shape,
            data_offset,
        })
    }

    /// Get the shape of the array.
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get the number of rows.
    pub fn nrows(&self) -> usize {
        self.shape.0
    }

    /// Get the number of columns.
    pub fn ncols(&self) -> usize {
        self.shape.1
    }

    /// Get a view of rows [start..end] as ArrayView2.
    pub fn slice_rows(&self, start: usize, end: usize) -> ArrayView2<'_, u8> {
        let nrows = end - start;
        let byte_start = self.data_offset + start * self.shape.1;
        let byte_end = self.data_offset + end * self.shape.1;
        let bytes = &self._mmap[byte_start..byte_end];

        ArrayView2::from_shape((nrows, self.shape.1), bytes).unwrap()
    }

    /// Get a view of the entire array.
    pub fn view(&self) -> ArrayView2<'_, u8> {
        self.slice_rows(0, self.shape.0)
    }

    /// Get a single row as a slice.
    pub fn row(&self, idx: usize) -> &[u8] {
        let byte_start = self.data_offset + idx * self.shape.1;
        let byte_end = byte_start + self.shape.1;
        &self._mmap[byte_start..byte_end]
    }
}

// ============================================================================
// Merged File Creation
// ============================================================================

/// Manifest entry for tracking chunk files
#[cfg(feature = "npy")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChunkManifestEntry {
    pub rows: usize,
    pub mtime: f64,
}

/// Manifest for merged files
#[cfg(feature = "npy")]
pub type ChunkManifest = HashMap<String, ChunkManifestEntry>;

/// Load manifest from disk if it exists
#[cfg(feature = "npy")]
fn load_manifest(manifest_path: &Path) -> Option<ChunkManifest> {
    if manifest_path.exists() {
        if let Ok(file) = File::open(manifest_path) {
            if let Ok(manifest) = serde_json::from_reader(BufReader::new(file)) {
                return Some(manifest);
            }
        }
    }
    None
}

/// Save manifest to disk
#[cfg(feature = "npy")]
fn save_manifest(manifest_path: &Path, manifest: &ChunkManifest) -> Result<()> {
    let file = File::create(manifest_path)
        .map_err(|e| Error::IndexLoad(format!("Failed to create manifest: {}", e)))?;
    serde_json::to_writer(BufWriter::new(file), manifest)
        .map_err(|e| Error::IndexLoad(format!("Failed to write manifest: {}", e)))?;
    Ok(())
}

/// Get file modification time as f64 seconds since epoch
#[cfg(feature = "npy")]
fn get_mtime(path: &Path) -> Result<f64> {
    let metadata = fs::metadata(path)
        .map_err(|e| Error::IndexLoad(format!("Failed to get metadata for {:?}: {}", path, e)))?;
    let mtime = metadata
        .modified()
        .map_err(|e| Error::IndexLoad(format!("Failed to get mtime: {}", e)))?;
    let duration = mtime
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| Error::IndexLoad(format!("Invalid mtime: {}", e)))?;
    Ok(duration.as_secs_f64())
}

/// Write NPY header for a 1D array
#[cfg(feature = "npy")]
fn write_npy_header_1d(writer: &mut impl Write, len: usize, dtype: &str) -> Result<usize> {
    // Build header dict
    let header_dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({},), }}",
        dtype, len
    );

    // Pad to 64-byte alignment (NPY requirement)
    let header_len = header_dict.len();
    let padding = (64 - ((10 + header_len) % 64)) % 64;
    let padded_header = format!("{}{}\n", header_dict, " ".repeat(padding));

    // Write magic + version
    writer
        .write_all(NPY_MAGIC)
        .map_err(|e| Error::IndexLoad(format!("Failed to write NPY magic: {}", e)))?;
    writer
        .write_all(&[1, 0])
        .map_err(|e| Error::IndexLoad(format!("Failed to write version: {}", e)))?; // v1.0

    // Write header length (2 bytes for v1.0)
    let header_len_bytes = (padded_header.len() as u16).to_le_bytes();
    writer
        .write_all(&header_len_bytes)
        .map_err(|e| Error::IndexLoad(format!("Failed to write header len: {}", e)))?;

    // Write header
    writer
        .write_all(padded_header.as_bytes())
        .map_err(|e| Error::IndexLoad(format!("Failed to write header: {}", e)))?;

    Ok(10 + padded_header.len())
}

/// Write NPY header for a 2D array
#[cfg(feature = "npy")]
fn write_npy_header_2d(
    writer: &mut impl Write,
    nrows: usize,
    ncols: usize,
    dtype: &str,
) -> Result<usize> {
    // Build header dict
    let header_dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({}, {}), }}",
        dtype, nrows, ncols
    );

    // Pad to 64-byte alignment
    let header_len = header_dict.len();
    let padding = (64 - ((10 + header_len) % 64)) % 64;
    let padded_header = format!("{}{}\n", header_dict, " ".repeat(padding));

    // Write magic + version
    writer
        .write_all(NPY_MAGIC)
        .map_err(|e| Error::IndexLoad(format!("Failed to write NPY magic: {}", e)))?;
    writer
        .write_all(&[1, 0])
        .map_err(|e| Error::IndexLoad(format!("Failed to write version: {}", e)))?;

    // Write header length
    let header_len_bytes = (padded_header.len() as u16).to_le_bytes();
    writer
        .write_all(&header_len_bytes)
        .map_err(|e| Error::IndexLoad(format!("Failed to write header len: {}", e)))?;

    // Write header
    writer
        .write_all(padded_header.as_bytes())
        .map_err(|e| Error::IndexLoad(format!("Failed to write header: {}", e)))?;

    Ok(10 + padded_header.len())
}

/// Information about a chunk file for merging
#[cfg(feature = "npy")]
struct ChunkInfo {
    path: std::path::PathBuf,
    filename: String,
    rows: usize,
    mtime: f64,
    #[allow(dead_code)]
    needs_write: bool,
}

/// Merge chunked codes NPY files into a single merged file.
///
/// Uses incremental persistence with manifest tracking to skip unchanged chunks.
/// Returns the path to the merged file.
#[cfg(feature = "npy")]
pub fn merge_codes_chunks(
    index_path: &Path,
    num_chunks: usize,
    padding_rows: usize,
) -> Result<std::path::PathBuf> {
    use ndarray_npy::ReadNpyExt;

    let merged_path = index_path.join("merged_codes.npy");
    let manifest_path = index_path.join("merged_codes.manifest.json");

    // Load previous manifest
    let old_manifest = load_manifest(&manifest_path);

    // Scan chunks and detect changes
    let mut chunks: Vec<ChunkInfo> = Vec::new();
    let mut total_rows = 0usize;
    let mut chain_broken = false;

    for i in 0..num_chunks {
        let filename = format!("{}.codes.npy", i);
        let path = index_path.join(&filename);

        if path.exists() {
            let mtime = get_mtime(&path)?;

            // Get shape by reading header only
            let file = File::open(&path)?;
            let arr: Array1<i64> = Array1::read_npy(file)?;
            let rows = arr.len();

            if rows > 0 {
                total_rows += rows;

                // Check if this chunk changed
                let is_clean = if let Some(ref manifest) = old_manifest {
                    manifest
                        .get(&filename)
                        .is_some_and(|entry| entry.mtime == mtime && entry.rows == rows)
                } else {
                    false
                };

                let needs_write = if chain_broken || !is_clean {
                    chain_broken = true;
                    true
                } else {
                    false
                };

                chunks.push(ChunkInfo {
                    path,
                    filename,
                    rows,
                    mtime,
                    needs_write,
                });
            }
        }
    }

    if total_rows == 0 {
        return Err(Error::IndexLoad("No data to merge".into()));
    }

    let final_rows = total_rows + padding_rows;

    // Check if we need to rewrite
    let needs_full_rewrite = !merged_path.exists() || chain_broken;

    if needs_full_rewrite {
        // Create new merged file
        let file = File::create(&merged_path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        write_npy_header_1d(&mut writer, final_rows, "<i8")?;

        // Write chunk data
        for chunk in &chunks {
            let file = File::open(&chunk.path)?;
            let arr: Array1<i64> = Array1::read_npy(file)?;
            for &val in arr.iter() {
                writer.write_all(&val.to_le_bytes())?;
            }
        }

        // Write padding zeros
        for _ in 0..padding_rows {
            writer.write_all(&0i64.to_le_bytes())?;
        }

        writer.flush()?;
    }

    // Save manifest
    let mut new_manifest = ChunkManifest::new();
    for chunk in &chunks {
        new_manifest.insert(
            chunk.filename.clone(),
            ChunkManifestEntry {
                rows: chunk.rows,
                mtime: chunk.mtime,
            },
        );
    }
    save_manifest(&manifest_path, &new_manifest)?;

    Ok(merged_path)
}

/// Merge chunked residuals NPY files into a single merged file.
#[cfg(feature = "npy")]
pub fn merge_residuals_chunks(
    index_path: &Path,
    num_chunks: usize,
    padding_rows: usize,
) -> Result<std::path::PathBuf> {
    use ndarray_npy::ReadNpyExt;

    let merged_path = index_path.join("merged_residuals.npy");
    let manifest_path = index_path.join("merged_residuals.manifest.json");

    // Load previous manifest
    let old_manifest = load_manifest(&manifest_path);

    // Scan chunks and detect changes
    let mut chunks: Vec<ChunkInfo> = Vec::new();
    let mut total_rows = 0usize;
    let mut ncols = 0usize;
    let mut chain_broken = false;

    for i in 0..num_chunks {
        let filename = format!("{}.residuals.npy", i);
        let path = index_path.join(&filename);

        if path.exists() {
            let mtime = get_mtime(&path)?;

            // Get shape by reading header
            let file = File::open(&path)?;
            let arr: Array2<u8> = Array2::read_npy(file)?;
            let rows = arr.nrows();
            ncols = arr.ncols();

            if rows > 0 {
                total_rows += rows;

                let is_clean = if let Some(ref manifest) = old_manifest {
                    manifest
                        .get(&filename)
                        .is_some_and(|entry| entry.mtime == mtime && entry.rows == rows)
                } else {
                    false
                };

                let needs_write = if chain_broken || !is_clean {
                    chain_broken = true;
                    true
                } else {
                    false
                };

                chunks.push(ChunkInfo {
                    path,
                    filename,
                    rows,
                    mtime,
                    needs_write,
                });
            }
        }
    }

    if total_rows == 0 || ncols == 0 {
        return Err(Error::IndexLoad("No residual data to merge".into()));
    }

    let final_rows = total_rows + padding_rows;

    let needs_full_rewrite = !merged_path.exists() || chain_broken;

    if needs_full_rewrite {
        let file = File::create(&merged_path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        write_npy_header_2d(&mut writer, final_rows, ncols, "|u1")?;

        // Write chunk data
        for chunk in &chunks {
            let file = File::open(&chunk.path)?;
            let arr: Array2<u8> = Array2::read_npy(file)?;
            for row in arr.rows() {
                writer.write_all(row.as_slice().unwrap())?;
            }
        }

        // Write padding zeros
        let zero_row = vec![0u8; ncols];
        for _ in 0..padding_rows {
            writer.write_all(&zero_row)?;
        }

        writer.flush()?;
    }

    // Save manifest
    let mut new_manifest = ChunkManifest::new();
    for chunk in &chunks {
        new_manifest.insert(
            chunk.filename.clone(),
            ChunkManifestEntry {
                rows: chunk.rows,
                mtime: chunk.mtime,
            },
        );
    }
    save_manifest(&manifest_path, &new_manifest)?;

    Ok(merged_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_mmap_array2_f32() {
        // Create a test file
        let mut file = NamedTempFile::new().unwrap();

        // Write header (3 rows, 2 cols)
        file.write_all(&3i64.to_le_bytes()).unwrap();
        file.write_all(&2i64.to_le_bytes()).unwrap();

        // Write data
        for val in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            file.write_all(&val.to_le_bytes()).unwrap();
        }

        file.flush().unwrap();

        // Load and verify
        let mmap = MmapArray2F32::from_raw_file(file.path()).unwrap();
        assert_eq!(mmap.shape(), (3, 2));

        let row0 = mmap.row(0);
        assert_eq!(row0[0], 1.0);
        assert_eq!(row0[1], 2.0);

        let owned = mmap.to_owned();
        assert_eq!(owned[[2, 0]], 5.0);
        assert_eq!(owned[[2, 1]], 6.0);
    }

    #[test]
    fn test_mmap_array1_i64() {
        let mut file = NamedTempFile::new().unwrap();

        // Write header (4 elements)
        file.write_all(&4i64.to_le_bytes()).unwrap();

        // Write data
        for val in [10i64, 20, 30, 40] {
            file.write_all(&val.to_le_bytes()).unwrap();
        }

        file.flush().unwrap();

        let mmap = MmapArray1I64::from_raw_file(file.path()).unwrap();
        assert_eq!(mmap.len(), 4);
        assert_eq!(mmap.get(0), 10);
        assert_eq!(mmap.get(3), 40);

        let owned = mmap.to_owned();
        assert_eq!(owned[1], 20);
        assert_eq!(owned[2], 30);
    }

    #[test]
    fn test_write_read_roundtrip() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();

        // Create test array
        let array = Array2::from_shape_vec((2, 3), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Write
        write_array2_f32(&array, path).unwrap();

        // Read back
        let mmap = MmapArray2F32::from_raw_file(path).unwrap();
        let loaded = mmap.to_owned();

        assert_eq!(array, loaded);
    }
}
