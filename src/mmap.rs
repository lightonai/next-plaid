//! Memory-mapped file support for efficient large index loading.
//!
//! This module provides utilities for loading large arrays from disk using
//! memory-mapped files, avoiding the need to load entire arrays into RAM.

use std::fs::File;
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
