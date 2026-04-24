use std::mem::size_of;

use next_plaid_browser_contract::{MemoryUsageBreakdown, SearchIndexPayload};

use crate::keyword_runtime::KeywordIndex;
use crate::WasmError;

pub(crate) fn index_memory_usage_breakdown(
    index: &SearchIndexPayload,
    metadata: Option<&[Option<serde_json::Value>]>,
    keyword_index: Option<&KeywordIndex>,
) -> Result<MemoryUsageBreakdown, WasmError> {
    build_memory_usage_breakdown(
        dense_index_payload_bytes(index)?,
        metadata_json_usage_bytes(metadata)?,
        keyword_runtime_usage_bytes(keyword_index)?,
    )
}

pub(crate) fn compressed_index_memory_usage_breakdown(
    search: &next_plaid_browser_loader::LoadedSearchArtifacts,
    metadata: Option<&[Option<serde_json::Value>]>,
    keyword_index: Option<&KeywordIndex>,
) -> Result<MemoryUsageBreakdown, WasmError> {
    build_memory_usage_breakdown(
        compressed_index_payload_bytes(search)?,
        metadata_json_usage_bytes(metadata)?,
        keyword_runtime_usage_bytes(keyword_index)?,
    )
}

fn build_memory_usage_breakdown(
    index_bytes: u64,
    metadata_json_bytes: u64,
    keyword_runtime_bytes: u64,
) -> Result<MemoryUsageBreakdown, WasmError> {
    let breakdown = MemoryUsageBreakdown {
        index_bytes,
        metadata_json_bytes,
        keyword_runtime_bytes,
    };
    let _ = memory_usage_total_bytes(&breakdown)?;
    Ok(breakdown)
}

struct ByteCounter(u64);

impl ByteCounter {
    fn new() -> Self {
        Self(0)
    }

    fn add_slice<T>(&mut self, slice: &[T]) -> Result<(), WasmError> {
        let bytes = (slice.len() as u64)
            .checked_mul(size_of::<T>() as u64)
            .ok_or(WasmError::ByteCountOverflow)?;
        self.add_bytes(bytes)
    }

    fn add_bytes(&mut self, bytes: u64) -> Result<(), WasmError> {
        self.0 = self
            .0
            .checked_add(bytes)
            .ok_or(WasmError::ByteCountOverflow)?;
        Ok(())
    }

    fn total(&self) -> u64 {
        self.0
    }
}

fn dense_index_payload_bytes(index: &SearchIndexPayload) -> Result<u64, WasmError> {
    let mut counter = ByteCounter::new();
    counter.add_slice(&index.centroids.values)?;
    counter.add_slice(&index.ivf_doc_ids)?;
    counter.add_slice(&index.ivf_lengths)?;
    counter.add_slice(&index.doc_offsets)?;
    counter.add_slice(&index.doc_codes)?;
    counter.add_slice(&index.doc_values)?;
    Ok(counter.total())
}

fn compressed_index_payload_bytes(
    search: &next_plaid_browser_loader::LoadedSearchArtifacts,
) -> Result<u64, WasmError> {
    let mut counter = ByteCounter::new();
    counter.add_slice(&search.centroids)?;
    counter.add_slice(&search.ivf)?;
    counter.add_slice(&search.ivf_lengths)?;
    counter.add_slice(&search.doc_lengths)?;
    counter.add_slice(&search.doc_offsets)?;
    counter.add_slice(&search.merged_codes)?;
    counter.add_bytes(search.merged_residuals.len() as u64)?;
    counter.add_slice(&search.bucket_weights)?;
    Ok(counter.total())
}

fn metadata_json_usage_bytes(
    metadata: Option<&[Option<serde_json::Value>]>,
) -> Result<u64, WasmError> {
    metadata.map_or(Ok(0), |metadata| {
        metadata.iter().try_fold(0u64, |acc, value| {
            let bytes = serde_json::to_vec(value)?;
            acc.checked_add(bytes.len() as u64)
                .ok_or(WasmError::ByteCountOverflow)
        })
    })
}

fn keyword_runtime_usage_bytes(keyword_index: Option<&KeywordIndex>) -> Result<u64, WasmError> {
    keyword_index
        .map(|keyword_index| keyword_index.memory_usage_bytes().map_err(WasmError::from))
        .transpose()
        .map(|bytes| bytes.unwrap_or(0))
}

fn memory_usage_total_bytes(breakdown: &MemoryUsageBreakdown) -> Result<u64, WasmError> {
    breakdown
        .index_bytes
        .checked_add(breakdown.metadata_json_bytes)
        .and_then(|total| total.checked_add(breakdown.keyword_runtime_bytes))
        .ok_or(WasmError::ByteCountOverflow)
}

pub(crate) fn saturating_memory_usage_total_bytes(breakdown: &MemoryUsageBreakdown) -> u64 {
    breakdown
        .index_bytes
        .saturating_add(breakdown.metadata_json_bytes)
        .saturating_add(breakdown.keyword_runtime_bytes)
}
