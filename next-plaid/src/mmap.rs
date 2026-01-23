//! Memory-mapped file support for efficient large index loading.
//!
//! This module provides utilities for loading large arrays from disk using
//! memory-mapped files, avoiding the need to load entire arrays into RAM.
//!
//! Two formats are supported:
//! - Custom raw binary format (legacy): 8-byte header with shape, then raw data
//! - NPY format: Standard NumPy format with header, used for index files

use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

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

/// Parse dtype from NPY header string (e.g., "<f2" for float16, "<f4" for float32)
fn parse_dtype_from_header(header: &str) -> Result<String> {
    // Find 'descr': '...'
    let descr_start = header
        .find("'descr':")
        .ok_or_else(|| Error::IndexLoad("No descr in NPY header".into()))?;

    let after_descr = &header[descr_start + 8..];
    let quote_start = after_descr
        .find('\'')
        .ok_or_else(|| Error::IndexLoad("No dtype quote in NPY header".into()))?;
    let rest = &after_descr[quote_start + 1..];
    let quote_end = rest
        .find('\'')
        .ok_or_else(|| Error::IndexLoad("Unclosed dtype quote in NPY header".into()))?;

    Ok(rest[..quote_end].to_string())
}

/// Detect NPY file dtype without loading the entire file
pub fn detect_npy_dtype(path: &Path) -> Result<String> {
    let file = File::open(path)
        .map_err(|e| Error::IndexLoad(format!("Failed to open NPY file {:?}: {}", path, e)))?;

    let mmap = unsafe {
        Mmap::map(&file)
            .map_err(|e| Error::IndexLoad(format!("Failed to mmap NPY file {:?}: {}", path, e)))?
    };

    if mmap.len() < 10 {
        return Err(Error::IndexLoad("NPY file too small".into()));
    }

    // Check magic
    if &mmap[..6] != NPY_MAGIC {
        return Err(Error::IndexLoad("Invalid NPY magic".into()));
    }

    let major_version = mmap[6];

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

    let header_str = std::str::from_utf8(&mmap[header_start..header_end])
        .map_err(|e| Error::IndexLoad(format!("Invalid NPY header encoding: {}", e)))?;

    parse_dtype_from_header(header_str)
}

/// Convert a float16 NPY file to float32 in place
pub fn convert_f16_to_f32_npy(path: &Path) -> Result<()> {
    use half::f16;
    use std::io::Read;

    // Read the entire file
    let mut file = File::open(path)
        .map_err(|e| Error::IndexLoad(format!("Failed to open {:?}: {}", path, e)))?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)
        .map_err(|e| Error::IndexLoad(format!("Failed to read {:?}: {}", path, e)))?;

    if data.len() < 10 || &data[..6] != NPY_MAGIC {
        return Err(Error::IndexLoad("Invalid NPY file".into()));
    }

    let major_version = data[6];
    let header_start = if major_version == 1 { 10 } else { 12 };
    let header_len = if major_version == 1 {
        u16::from_le_bytes([data[8], data[9]]) as usize
    } else {
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
    };
    let header_end = header_start + header_len;

    // Parse header to get shape
    let header_str = std::str::from_utf8(&data[header_start..header_end])
        .map_err(|e| Error::IndexLoad(format!("Invalid header: {}", e)))?;
    let shape = parse_shape_from_header(header_str)?;

    // Calculate total elements
    let total_elements: usize = shape.iter().product();
    let f16_data = &data[header_end..header_end + total_elements * 2];

    // Convert f16 to f32
    let mut f32_data = Vec::with_capacity(total_elements * 4);
    for chunk in f16_data.chunks(2) {
        let f16_val = f16::from_le_bytes([chunk[0], chunk[1]]);
        let f32_val: f32 = f16_val.to_f32();
        f32_data.extend_from_slice(&f32_val.to_le_bytes());
    }

    // Write new file with f32 dtype
    let file = File::create(path)
        .map_err(|e| Error::IndexLoad(format!("Failed to create {:?}: {}", path, e)))?;
    let mut writer = BufWriter::new(file);

    if shape.len() == 1 {
        write_npy_header_1d(&mut writer, shape[0], "<f4")?;
    } else if shape.len() == 2 {
        write_npy_header_2d(&mut writer, shape[0], shape[1], "<f4")?;
    } else {
        return Err(Error::IndexLoad("Unsupported shape dimensions".into()));
    }

    writer
        .write_all(&f32_data)
        .map_err(|e| Error::IndexLoad(format!("Failed to write data: {}", e)))?;
    writer.flush()?;

    Ok(())
}

/// Convert an int64 NPY file to int32 in place
pub fn convert_i64_to_i32_npy(path: &Path) -> Result<()> {
    use std::io::Read;

    // Read the entire file
    let mut file = File::open(path)
        .map_err(|e| Error::IndexLoad(format!("Failed to open {:?}: {}", path, e)))?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)
        .map_err(|e| Error::IndexLoad(format!("Failed to read {:?}: {}", path, e)))?;

    if data.len() < 10 || &data[..6] != NPY_MAGIC {
        return Err(Error::IndexLoad("Invalid NPY file".into()));
    }

    let major_version = data[6];
    let header_start = if major_version == 1 { 10 } else { 12 };
    let header_len = if major_version == 1 {
        u16::from_le_bytes([data[8], data[9]]) as usize
    } else {
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
    };
    let header_end = header_start + header_len;

    // Parse header to get shape
    let header_str = std::str::from_utf8(&data[header_start..header_end])
        .map_err(|e| Error::IndexLoad(format!("Invalid header: {}", e)))?;
    let shape = parse_shape_from_header(header_str)?;

    if shape.len() != 1 {
        return Err(Error::IndexLoad("Expected 1D array for i64->i32".into()));
    }

    let len = shape[0];
    let i64_data = &data[header_end..header_end + len * 8];

    // Convert i64 to i32
    let mut i32_data = Vec::with_capacity(len * 4);
    for chunk in i64_data.chunks(8) {
        let i64_val = i64::from_le_bytes(chunk.try_into().unwrap());
        let i32_val = i64_val as i32;
        i32_data.extend_from_slice(&i32_val.to_le_bytes());
    }

    // Write new file with i32 dtype
    let file = File::create(path)
        .map_err(|e| Error::IndexLoad(format!("Failed to create {:?}: {}", path, e)))?;
    let mut writer = BufWriter::new(file);

    write_npy_header_1d(&mut writer, len, "<i4")?;

    writer
        .write_all(&i32_data)
        .map_err(|e| Error::IndexLoad(format!("Failed to write data: {}", e)))?;
    writer.flush()?;

    Ok(())
}

/// Re-save a u8 NPY file to ensure dtype descriptor is "|u1" (platform-independent)
///
/// Note: We can't use ndarray_npy::ReadNpyExt here because it doesn't accept "<u1"
/// descriptor, so we manually read the raw data and resave with "|u1".
pub fn normalize_u8_npy(path: &Path) -> Result<()> {
    use std::io::Read;

    // Read the entire file
    let mut file = File::open(path)
        .map_err(|e| Error::IndexLoad(format!("Failed to open {:?}: {}", path, e)))?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)
        .map_err(|e| Error::IndexLoad(format!("Failed to read {:?}: {}", path, e)))?;

    if data.len() < 10 || &data[..6] != NPY_MAGIC {
        return Err(Error::IndexLoad("Invalid NPY file".into()));
    }

    let major_version = data[6];
    let header_start = if major_version == 1 { 10 } else { 12 };
    let header_len = if major_version == 1 {
        u16::from_le_bytes([data[8], data[9]]) as usize
    } else {
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
    };
    let header_end = header_start + header_len;

    // Parse header to get shape
    let header_str = std::str::from_utf8(&data[header_start..header_end])
        .map_err(|e| Error::IndexLoad(format!("Invalid header: {}", e)))?;
    let shape = parse_shape_from_header(header_str)?;

    if shape.len() != 2 {
        return Err(Error::IndexLoad(
            "Expected 2D array for u8 normalization".into(),
        ));
    }

    let nrows = shape[0];
    let ncols = shape[1];
    let u8_data = &data[header_end..header_end + nrows * ncols];

    // Re-write with explicit "|u1" dtype
    let new_file = File::create(path)
        .map_err(|e| Error::IndexLoad(format!("Failed to create {:?}: {}", path, e)))?;
    let mut writer = BufWriter::new(new_file);

    write_npy_header_2d(&mut writer, nrows, ncols, "|u1")?;

    writer
        .write_all(u8_data)
        .map_err(|e| Error::IndexLoad(format!("Failed to write data: {}", e)))?;
    writer.flush()?;

    Ok(())
}

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
    /// Returns a `Vec<i64>` instead of &[i64] to handle unaligned data safely.
    ///
    /// # Safety
    /// The caller must ensure start <= end <= len.
    pub fn slice(&self, start: usize, end: usize) -> Vec<i64> {
        let count = end - start;
        let mut result = Vec::with_capacity(count);

        for i in start..end {
            result.push(self.get(i));
        }

        result
    }

    /// Get a value at an index.
    pub fn get(&self, idx: usize) -> i64 {
        let start = self.data_offset + idx * 8;
        let bytes = &self._mmap[start..start + 8];
        i64::from_le_bytes(bytes.try_into().unwrap())
    }
}

/// Memory-mapped NPY array for f32 values (used for centroids).
///
/// This struct provides zero-copy access to 2D f32 arrays stored in NPY format.
/// Unlike loading into an owned `Array2<f32>`, this approach lets the OS manage
/// paging, reducing resident memory usage for large centroid matrices.
pub struct MmapNpyArray2F32 {
    _mmap: Mmap,
    shape: (usize, usize),
    data_offset: usize,
}

impl MmapNpyArray2F32 {
    /// Load a 2D f32 array from an NPY file.
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

        // Verify file size (f32 = 4 bytes)
        let expected_size = data_offset + shape.0 * shape.1 * 4;
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

    /// Get a view of the entire array as ArrayView2.
    ///
    /// This provides zero-copy access to the memory-mapped data.
    pub fn view(&self) -> ArrayView2<'_, f32> {
        let byte_start = self.data_offset;
        let byte_end = self.data_offset + self.shape.0 * self.shape.1 * 4;
        let bytes = &self._mmap[byte_start..byte_end];

        // Safety: We've verified bounds and f32 is 4-byte aligned in NPY format
        let data = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.shape.0 * self.shape.1)
        };

        ArrayView2::from_shape(self.shape, data).unwrap()
    }

    /// Get a view of a single row.
    pub fn row(&self, idx: usize) -> ArrayView1<'_, f32> {
        let byte_start = self.data_offset + idx * self.shape.1 * 4;
        let bytes = &self._mmap[byte_start..byte_start + self.shape.1 * 4];

        // Safety: We've verified bounds and alignment
        let data =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.shape.1) };

        ArrayView1::from_shape(self.shape.1, data).unwrap()
    }

    /// Get a view of rows [start..end] as ArrayView2.
    pub fn slice_rows(&self, start: usize, end: usize) -> ArrayView2<'_, f32> {
        let nrows = end - start;
        let byte_start = self.data_offset + start * self.shape.1 * 4;
        let byte_end = self.data_offset + end * self.shape.1 * 4;
        let bytes = &self._mmap[byte_start..byte_end];

        // Safety: We've verified bounds
        let data = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, nrows * self.shape.1)
        };

        ArrayView2::from_shape((nrows, self.shape.1), data).unwrap()
    }

    /// Convert to an owned Array2 (loads all data into memory).
    ///
    /// Use this only when you need an owned copy; prefer `view()` for read-only access.
    pub fn to_owned(&self) -> Array2<f32> {
        self.view().to_owned()
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChunkManifestEntry {
    pub rows: usize,
    pub mtime: f64,
}

/// Manifest for merged files, including metadata about the merge
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MergeManifest {
    /// Chunk information
    pub chunks: HashMap<String, ChunkManifestEntry>,
    /// Number of padding rows used in the merge
    #[serde(default)]
    pub padding_rows: usize,
    /// Total rows in the merged file (including padding)
    #[serde(default)]
    pub total_rows: usize,
    /// Number of columns (for 2D arrays like residuals)
    #[serde(default)]
    pub ncols: usize,
}

/// Legacy manifest type (for backwards compatibility during migration)
pub type ChunkManifest = HashMap<String, ChunkManifestEntry>;

/// Load manifest from disk if it exists
/// Handles both new MergeManifest format and legacy ChunkManifest format
fn load_merge_manifest(manifest_path: &Path) -> Option<MergeManifest> {
    if manifest_path.exists() {
        if let Ok(file) = File::open(manifest_path) {
            // Try to load as new format first
            let reader = BufReader::new(file);
            if let Ok(manifest) = serde_json::from_reader::<_, MergeManifest>(reader) {
                return Some(manifest);
            }
            // Try legacy format
            if let Ok(file) = File::open(manifest_path) {
                if let Ok(chunks) =
                    serde_json::from_reader::<_, ChunkManifest>(BufReader::new(file))
                {
                    // Convert legacy format - missing padding info means we need to regenerate
                    return Some(MergeManifest {
                        chunks,
                        padding_rows: 0,
                        total_rows: 0,
                        ncols: 0,
                    });
                }
            }
        }
    }
    None
}

/// Save manifest to disk atomically (write to temp file, then rename)
fn save_merge_manifest(manifest_path: &Path, manifest: &MergeManifest) -> Result<()> {
    let temp_path = manifest_path.with_extension("manifest.json.tmp");

    // Write to temp file
    let file = File::create(&temp_path)
        .map_err(|e| Error::IndexLoad(format!("Failed to create temp manifest: {}", e)))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, manifest)
        .map_err(|e| Error::IndexLoad(format!("Failed to write manifest: {}", e)))?;
    writer
        .flush()
        .map_err(|e| Error::IndexLoad(format!("Failed to flush manifest: {}", e)))?;

    // Sync to disk
    writer
        .into_inner()
        .map_err(|e| Error::IndexLoad(format!("Failed to get inner file: {}", e)))?
        .sync_all()
        .map_err(|e| Error::IndexLoad(format!("Failed to sync manifest: {}", e)))?;

    // Atomic rename
    fs::rename(&temp_path, manifest_path)
        .map_err(|e| Error::IndexLoad(format!("Failed to rename manifest: {}", e)))?;

    Ok(())
}

/// Get file modification time as f64 seconds since epoch
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
struct ChunkInfo {
    path: std::path::PathBuf,
    filename: String,
    rows: usize,
    mtime: f64,
}

/// Merge chunked codes NPY files into a single merged file.
///
/// Uses incremental persistence with manifest tracking to skip unchanged chunks.
/// Uses atomic writes to prevent corruption from interrupted writes.
/// Returns the path to the merged file.
pub fn merge_codes_chunks(
    index_path: &Path,
    num_chunks: usize,
    padding_rows: usize,
) -> Result<std::path::PathBuf> {
    use ndarray_npy::ReadNpyExt;

    let merged_path = index_path.join("merged_codes.npy");
    let manifest_path = index_path.join("merged_codes.manifest.json");
    let temp_path = index_path.join("merged_codes.npy.tmp");

    // Load previous manifest
    let old_manifest = load_merge_manifest(&manifest_path);

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
                        .chunks
                        .get(&filename)
                        .is_some_and(|entry| entry.mtime == mtime && entry.rows == rows)
                } else {
                    false
                };

                if !is_clean {
                    chain_broken = true;
                }

                chunks.push(ChunkInfo {
                    path,
                    filename,
                    rows,
                    mtime,
                });
            }
        }
    }

    if total_rows == 0 {
        return Err(Error::IndexLoad("No data to merge".into()));
    }

    let final_rows = total_rows + padding_rows;

    // Check if we need to rewrite:
    // 1. Merged file doesn't exist
    // 2. Chunks have changed
    // 3. Padding has changed (stored in manifest)
    // 4. Total rows don't match (safety check)
    let padding_changed = old_manifest
        .as_ref()
        .map(|m| m.padding_rows != padding_rows)
        .unwrap_or(true);
    let total_rows_mismatch = old_manifest
        .as_ref()
        .map(|m| m.total_rows != final_rows)
        .unwrap_or(true);

    let needs_full_rewrite =
        !merged_path.exists() || chain_broken || padding_changed || total_rows_mismatch;

    if needs_full_rewrite {
        // Write to temp file first (atomic write pattern)
        let file = File::create(&temp_path)
            .map_err(|e| Error::IndexLoad(format!("Failed to create temp merged file: {}", e)))?;
        let mut writer = BufWriter::new(file);

        // Write header
        let header_size = write_npy_header_1d(&mut writer, final_rows, "<i8")?;

        // Write chunk data
        let mut written_rows = 0usize;
        for chunk in &chunks {
            let file = File::open(&chunk.path)?;
            let arr: Array1<i64> = Array1::read_npy(file)?;
            for &val in arr.iter() {
                writer.write_all(&val.to_le_bytes())?;
            }
            written_rows += arr.len();
        }

        // Write padding zeros
        for _ in 0..padding_rows {
            writer.write_all(&0i64.to_le_bytes())?;
        }
        written_rows += padding_rows;

        // Flush and sync to disk
        writer
            .flush()
            .map_err(|e| Error::IndexLoad(format!("Failed to flush merged file: {}", e)))?;
        let file = writer
            .into_inner()
            .map_err(|e| Error::IndexLoad(format!("Failed to get inner file: {}", e)))?;
        file.sync_all()
            .map_err(|e| Error::IndexLoad(format!("Failed to sync merged file to disk: {}", e)))?;

        // Verify file size before renaming
        let expected_size = header_size + written_rows * 8;
        let actual_size = fs::metadata(&temp_path)
            .map_err(|e| Error::IndexLoad(format!("Failed to get temp file metadata: {}", e)))?
            .len() as usize;

        if actual_size != expected_size {
            // Clean up temp file and return error
            let _ = fs::remove_file(&temp_path);
            return Err(Error::IndexLoad(format!(
                "Merged codes file size mismatch: expected {} bytes, got {} bytes",
                expected_size, actual_size
            )));
        }

        // Atomic rename (overwrites existing file)
        fs::rename(&temp_path, &merged_path)
            .map_err(|e| Error::IndexLoad(format!("Failed to rename merged file: {}", e)))?;
    } else {
        // Validate existing merged file before using it
        if merged_path.exists() {
            let file_size = fs::metadata(&merged_path)
                .map_err(|e| {
                    Error::IndexLoad(format!("Failed to get merged file metadata: {}", e))
                })?
                .len() as usize;

            // NPY header is at least 64 bytes, data is final_rows * 8 bytes
            let min_expected_size = 64 + final_rows * 8;
            if file_size < min_expected_size {
                // File is corrupted, force regeneration by recursing with empty manifest
                let _ = fs::remove_file(&merged_path);
                let _ = fs::remove_file(&manifest_path);
                return merge_codes_chunks(index_path, num_chunks, padding_rows);
            }
        }
    }

    // Build and save manifest with full metadata
    let mut chunk_map = HashMap::new();
    for chunk in &chunks {
        chunk_map.insert(
            chunk.filename.clone(),
            ChunkManifestEntry {
                rows: chunk.rows,
                mtime: chunk.mtime,
            },
        );
    }
    let new_manifest = MergeManifest {
        chunks: chunk_map,
        padding_rows,
        total_rows: final_rows,
        ncols: 0, // Not used for 1D codes array
    };
    save_merge_manifest(&manifest_path, &new_manifest)?;

    Ok(merged_path)
}

/// Merge chunked residuals NPY files into a single merged file.
///
/// Uses atomic writes to prevent corruption from interrupted writes.
pub fn merge_residuals_chunks(
    index_path: &Path,
    num_chunks: usize,
    padding_rows: usize,
) -> Result<std::path::PathBuf> {
    use ndarray_npy::ReadNpyExt;

    let merged_path = index_path.join("merged_residuals.npy");
    let manifest_path = index_path.join("merged_residuals.manifest.json");
    let temp_path = index_path.join("merged_residuals.npy.tmp");

    // Load previous manifest
    let old_manifest = load_merge_manifest(&manifest_path);

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
                        .chunks
                        .get(&filename)
                        .is_some_and(|entry| entry.mtime == mtime && entry.rows == rows)
                } else {
                    false
                };

                if !is_clean {
                    chain_broken = true;
                }

                chunks.push(ChunkInfo {
                    path,
                    filename,
                    rows,
                    mtime,
                });
            }
        }
    }

    if total_rows == 0 || ncols == 0 {
        return Err(Error::IndexLoad("No residual data to merge".into()));
    }

    let final_rows = total_rows + padding_rows;

    // Check if we need to rewrite:
    // 1. Merged file doesn't exist
    // 2. Chunks have changed
    // 3. Padding has changed
    // 4. Total rows or ncols don't match
    let padding_changed = old_manifest
        .as_ref()
        .map(|m| m.padding_rows != padding_rows)
        .unwrap_or(true);
    let total_rows_mismatch = old_manifest
        .as_ref()
        .map(|m| m.total_rows != final_rows)
        .unwrap_or(true);
    let ncols_mismatch = old_manifest
        .as_ref()
        .map(|m| m.ncols != ncols && m.ncols != 0)
        .unwrap_or(false);

    let needs_full_rewrite = !merged_path.exists()
        || chain_broken
        || padding_changed
        || total_rows_mismatch
        || ncols_mismatch;

    if needs_full_rewrite {
        // Write to temp file first (atomic write pattern)
        let file = File::create(&temp_path)
            .map_err(|e| Error::IndexLoad(format!("Failed to create temp merged file: {}", e)))?;
        let mut writer = BufWriter::new(file);

        // Write header
        let header_size = write_npy_header_2d(&mut writer, final_rows, ncols, "|u1")?;

        // Write chunk data
        let mut written_rows = 0usize;
        for chunk in &chunks {
            let file = File::open(&chunk.path)?;
            let arr: Array2<u8> = Array2::read_npy(file)?;
            for row in arr.rows() {
                writer.write_all(row.as_slice().unwrap())?;
            }
            written_rows += arr.nrows();
        }

        // Write padding zeros
        let zero_row = vec![0u8; ncols];
        for _ in 0..padding_rows {
            writer.write_all(&zero_row)?;
        }
        written_rows += padding_rows;

        // Flush and sync to disk
        writer
            .flush()
            .map_err(|e| Error::IndexLoad(format!("Failed to flush merged residuals: {}", e)))?;
        let file = writer
            .into_inner()
            .map_err(|e| Error::IndexLoad(format!("Failed to get inner file: {}", e)))?;
        file.sync_all().map_err(|e| {
            Error::IndexLoad(format!("Failed to sync merged residuals to disk: {}", e))
        })?;

        // Verify file size before renaming
        let expected_size = header_size + written_rows * ncols;
        let actual_size = fs::metadata(&temp_path)
            .map_err(|e| Error::IndexLoad(format!("Failed to get temp file metadata: {}", e)))?
            .len() as usize;

        if actual_size != expected_size {
            // Clean up temp file and return error
            let _ = fs::remove_file(&temp_path);
            return Err(Error::IndexLoad(format!(
                "Merged residuals file size mismatch: expected {} bytes, got {} bytes",
                expected_size, actual_size
            )));
        }

        // Atomic rename
        fs::rename(&temp_path, &merged_path)
            .map_err(|e| Error::IndexLoad(format!("Failed to rename merged residuals: {}", e)))?;
    } else {
        // Validate existing merged file before using it
        if merged_path.exists() {
            let file_size = fs::metadata(&merged_path)
                .map_err(|e| {
                    Error::IndexLoad(format!("Failed to get merged file metadata: {}", e))
                })?
                .len() as usize;

            // NPY header is at least 64 bytes, data is final_rows * ncols bytes
            let min_expected_size = 64 + final_rows * ncols;
            if file_size < min_expected_size {
                // File is corrupted, force regeneration
                let _ = fs::remove_file(&merged_path);
                let _ = fs::remove_file(&manifest_path);
                return merge_residuals_chunks(index_path, num_chunks, padding_rows);
            }
        }
    }

    // Build and save manifest with full metadata
    let mut chunk_map = HashMap::new();
    for chunk in &chunks {
        chunk_map.insert(
            chunk.filename.clone(),
            ChunkManifestEntry {
                rows: chunk.rows,
                mtime: chunk.mtime,
            },
        );
    }
    let new_manifest = MergeManifest {
        chunks: chunk_map,
        padding_rows,
        total_rows: final_rows,
        ncols,
    };
    save_merge_manifest(&manifest_path, &new_manifest)?;

    Ok(merged_path)
}

/// Clear merged files and manifests to force regeneration on next load.
///
/// This should be called after index updates to ensure the merged files
/// are regenerated with the latest data. The function silently ignores
/// missing files.
pub fn clear_merged_files(index_path: &Path) -> Result<()> {
    let files_to_remove = [
        "merged_codes.npy",
        "merged_codes.npy.tmp",
        "merged_codes.manifest.json",
        "merged_codes.manifest.json.tmp",
        "merged_residuals.npy",
        "merged_residuals.npy.tmp",
        "merged_residuals.manifest.json",
        "merged_residuals.manifest.json.tmp",
    ];

    for filename in files_to_remove {
        let path = index_path.join(filename);
        if path.exists() {
            fs::remove_file(&path)
                .map_err(|e| Error::IndexLoad(format!("Failed to remove {}: {}", filename, e)))?;
        }
    }

    Ok(())
}

// ============================================================================
// Fast-PLAID Compatibility Conversion
// ============================================================================

/// Convert a fast-plaid index to next-plaid compatible format.
///
/// This function detects and converts:
/// - float16 → float32 for centroids, avg_residual, bucket_cutoffs, bucket_weights
/// - int64 → int32 for ivf_lengths
/// - `<u1` → `|u1` for residuals
///
/// Returns true if any conversion was performed, false if already compatible.
pub fn convert_fastplaid_to_nextplaid(index_path: &Path) -> Result<bool> {
    let mut converted = false;

    // Float files to convert from f16 to f32
    let float_files = [
        "centroids.npy",
        "avg_residual.npy",
        "bucket_cutoffs.npy",
        "bucket_weights.npy",
    ];

    for filename in float_files {
        let path = index_path.join(filename);
        if path.exists() {
            let dtype = detect_npy_dtype(&path)?;
            if dtype == "<f2" {
                eprintln!("  Converting {} from float16 to float32", filename);
                convert_f16_to_f32_npy(&path)?;
                converted = true;
            }
        }
    }

    // Convert ivf_lengths from i64 to i32
    let ivf_lengths_path = index_path.join("ivf_lengths.npy");
    if ivf_lengths_path.exists() {
        let dtype = detect_npy_dtype(&ivf_lengths_path)?;
        if dtype == "<i8" {
            eprintln!("  Converting ivf_lengths.npy from int64 to int32");
            convert_i64_to_i32_npy(&ivf_lengths_path)?;
            converted = true;
        }
    }

    // Normalize residual files to use "|u1" descriptor
    // fast-plaid uses "<u1" which ndarray_npy doesn't accept
    for entry in fs::read_dir(index_path)? {
        let entry = entry?;
        let filename = entry.file_name().to_string_lossy().to_string();
        if filename.ends_with(".residuals.npy") {
            let path = entry.path();
            let dtype = detect_npy_dtype(&path)?;
            if dtype == "<u1" {
                eprintln!(
                    "  Normalizing {} dtype descriptor from <u1 to |u1",
                    filename
                );
                normalize_u8_npy(&path)?;
                converted = true;
            }
        }
    }

    Ok(converted)
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
