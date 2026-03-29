use anyhow::{Context, Result};
use next_plaid_onnx::{Colbert, ExecutionProvider, GpuMemoryInfo};
use serde::Deserialize;
use std::env;
use std::process::Command;

fn main() -> Result<()> {
    let mut model_dir: Option<String> = None;
    let mut batch_size: Option<usize> = None;
    let mut document_length: Option<usize> = None;
    let mut num_docs: usize = 32;
    let mut exact = false;
    let mut parallel = 1usize;
    let mut provider = ExecutionProvider::Auto;
    let mut quantized = false;
    let mut shared_spill_allowance_mib: u64 = 256;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_dir = args.next(),
            "--batch-size" => batch_size = args.next().and_then(|v| v.parse().ok()),
            "--document-length" => document_length = args.next().and_then(|v| v.parse().ok()),
            "--num-docs" => {
                num_docs = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(num_docs)
            }
            "--exact" => exact = true,
            "--parallel" => {
                parallel = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(parallel)
            }
            "--cuda" => provider = ExecutionProvider::Cuda,
            "--cpu" => provider = ExecutionProvider::Cpu,
            "--int8" => quantized = true,
            "--shared-spill-allowance-mib" => {
                shared_spill_allowance_mib = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(shared_spill_allowance_mib)
            }
            other => anyhow::bail!("Unknown argument: {other}"),
        }
    }

    let model_dir = model_dir.context("--model is required")?;
    let mut builder = Colbert::builder(model_dir)
        .with_parallel(parallel)
        .with_execution_provider(provider)
        .with_quantized(quantized);
    if let Some(batch_size) = batch_size {
        builder = builder.with_batch_size(batch_size);
    }
    if let Some(document_length) = document_length {
        builder = builder.with_document_length(document_length);
    }

    let model = builder.build()?;
    println!("planner context: {:#?}", model.batch_planner_context());
    println!("static model estimate: {:#?}", model.static_model_estimate());

    let docs = make_probe_documents(num_docs, model.config().document_length);
    let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
    let before = probe_nvidia_gpu_memory();
    let before_windows = probe_windows_gpu_memory();
    let prepared_batches = if exact {
        build_exact_batches(&model, &doc_refs)?
    } else {
        model.tokenize_documents_in_batches(&doc_refs)?
    };
    println!("prepared batches: {}", prepared_batches.len());
    for (idx, batch) in prepared_batches.iter().enumerate() {
        println!(
            "prepared batch {idx}: docs={}, max_len={}",
            batch.batch_size(),
            batch.batch_max_len()
        );
    }

    let mut min_free_mib = before.as_ref().map(|m| m.free_mib);
    let mut peak_windows_memory = before_windows.clone();
    for (idx, batch) in prepared_batches.into_iter().enumerate() {
        let _embeddings = model.encode_prepared_documents(batch)?;
        if let Some(mem) = probe_nvidia_gpu_memory() {
            min_free_mib = Some(min_free_mib.map(|v| v.min(mem.free_mib)).unwrap_or(mem.free_mib));
            println!(
                "after batch {idx}: gpu_free_mib={}, gpu_total_mib={}",
                mem.free_mib, mem.total_mib
            );
        }
        if let Some(mem) = probe_windows_gpu_memory() {
            println!(
                "after batch {idx}: windows_gpu_instance={}, dedicated_mib={}, shared_mib={}, total_committed_mib={}",
                mem.instance_name,
                mem.dedicated_usage_bytes / (1024 * 1024),
                mem.shared_usage_bytes / (1024 * 1024),
                mem.total_committed_bytes / (1024 * 1024),
            );
            peak_windows_memory = Some(match peak_windows_memory {
                Some(current) => current.peak_with(mem),
                None => mem,
            });
        }
    }
    if let (Some(before), Some(min_free_mib)) = (before, min_free_mib) {
        let observed_incremental_peak_mib = before.free_mib.saturating_sub(min_free_mib);
        println!(
            "observed incremental peak mib ~= {}",
            observed_incremental_peak_mib
        );
        let static_reference_batch_incremental_mib = model
            .static_model_estimate()
            .map(|estimate| estimate.estimated_reference_batch_incremental_bytes / (1024 * 1024))
            .unwrap_or(0);
        let windows_peak_dedicated_mib = peak_windows_memory
            .as_ref()
            .map(|mem| mem.dedicated_usage_bytes / (1024 * 1024))
            .unwrap_or(0);
        let windows_peak_shared_mib = peak_windows_memory
            .as_ref()
            .map(|mem| mem.shared_usage_bytes / (1024 * 1024))
            .unwrap_or(0);
        let windows_peak_total_committed_mib = peak_windows_memory
            .as_ref()
            .map(|mem| mem.total_committed_bytes / (1024 * 1024))
            .unwrap_or(0);
        let windows_before_shared_mib = before_windows
            .as_ref()
            .map(|mem| mem.shared_usage_bytes / (1024 * 1024))
            .unwrap_or(0);
        let windows_incremental_shared_peak_mib =
            windows_peak_shared_mib.saturating_sub(windows_before_shared_mib);
        let fits_dedicated_vram =
            windows_incremental_shared_peak_mib <= shared_spill_allowance_mib;
        if let Some(mem) = &peak_windows_memory {
            println!(
                "windows peak: instance={}, dedicated_mib={}, shared_mib={}, total_committed_mib={}, incremental_shared_peak_mib={}, fits_dedicated_vram={}",
                mem.instance_name,
                windows_peak_dedicated_mib,
                windows_peak_shared_mib,
                windows_peak_total_committed_mib,
                windows_incremental_shared_peak_mib,
                fits_dedicated_vram,
            );
        }
        println!(
            "PROBE_SUMMARY batch_size={} document_length={} num_docs={} exact={} observed_incremental_peak_mib={} static_reference_batch_incremental_mib={} windows_peak_dedicated_mib={} windows_peak_shared_mib={} windows_peak_total_committed_mib={} windows_incremental_shared_peak_mib={} fits_dedicated_vram={}",
            model.batch_size(),
            model.config().document_length,
            num_docs,
            exact,
            observed_incremental_peak_mib,
            static_reference_batch_incremental_mib,
            windows_peak_dedicated_mib,
            windows_peak_shared_mib,
            windows_peak_total_committed_mib,
            windows_incremental_shared_peak_mib,
            fits_dedicated_vram,
        );
    }

    Ok(())
}

fn make_probe_documents(num_docs: usize, document_length: usize) -> Vec<String> {
    let repeats = (document_length * 2).max(32);
    let body = "token ".repeat(repeats);
    (0..num_docs)
        .map(|idx| format!("Document {idx}\n{body}"))
        .collect()
}

fn build_exact_batches(model: &Colbert, documents: &[&str]) -> Result<Vec<next_plaid_onnx::PreparedDocumentBatch>> {
    let mut batches = Vec::new();
    for chunk in documents.chunks(model.batch_size()) {
        batches.push(model.tokenize_documents(chunk)?);
    }
    Ok(batches)
}

fn probe_nvidia_gpu_memory() -> Option<GpuMemoryInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let first_line = stdout.lines().find(|line| !line.trim().is_empty())?;
    let mut parts = first_line.split(',').map(|part| part.trim());
    let free_mib = parts.next()?.parse::<usize>().ok()?;
    let total_mib = parts.next()?.parse::<usize>().ok()?;
    Some(GpuMemoryInfo {
        free_mib,
        total_mib,
    })
}

#[derive(Debug, Clone)]
struct WindowsGpuMemoryInfo {
    instance_name: String,
    dedicated_usage_bytes: u64,
    shared_usage_bytes: u64,
    total_committed_bytes: u64,
}

impl WindowsGpuMemoryInfo {
    fn peak_with(self, other: WindowsGpuMemoryInfo) -> WindowsGpuMemoryInfo {
        WindowsGpuMemoryInfo {
            instance_name: if other.total_committed_bytes >= self.total_committed_bytes {
                other.instance_name
            } else {
                self.instance_name
            },
            dedicated_usage_bytes: self
                .dedicated_usage_bytes
                .max(other.dedicated_usage_bytes),
            shared_usage_bytes: self.shared_usage_bytes.max(other.shared_usage_bytes),
            total_committed_bytes: self.total_committed_bytes.max(other.total_committed_bytes),
        }
    }
}

#[derive(Debug, Deserialize)]
struct WindowsGpuCounterSample {
    #[serde(rename = "Path")]
    path: String,
    #[serde(rename = "InstanceName")]
    instance_name: String,
    #[serde(rename = "Value")]
    value: u64,
}

fn probe_windows_gpu_memory() -> Option<WindowsGpuMemoryInfo> {
    let script = r#"
$samples = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage','\GPU Adapter Memory(*)\Total Committed').CounterSamples |
  ForEach-Object {
    [pscustomobject]@{
      Path = $_.Path
      InstanceName = $_.InstanceName
      Value = [uint64]$_.RawValue
    }
  }
$samples | ConvertTo-Json -Compress
"#;
    let output = Command::new("powershell")
        .args(["-NoProfile", "-Command", script])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let stdout = stdout.trim();
    if stdout.is_empty() {
        return None;
    }
    let samples: Vec<WindowsGpuCounterSample> = match serde_json::from_str(stdout) {
        Ok(samples) => samples,
        Err(_) => vec![serde_json::from_str(stdout).ok()?],
    };

    let mut best: Option<WindowsGpuMemoryInfo> = None;
    let mut by_instance = std::collections::BTreeMap::<String, WindowsGpuMemoryInfo>::new();
    for sample in samples {
        let entry = by_instance
            .entry(sample.instance_name.clone())
            .or_insert_with(|| WindowsGpuMemoryInfo {
                instance_name: sample.instance_name.clone(),
                dedicated_usage_bytes: 0,
                shared_usage_bytes: 0,
                total_committed_bytes: 0,
            });
        let path = sample.path.to_ascii_lowercase();
        if path.contains("dedicated usage") {
            entry.dedicated_usage_bytes = sample.value;
        } else if path.contains("shared usage") {
            entry.shared_usage_bytes = sample.value;
        } else if path.contains("total committed") {
            entry.total_committed_bytes = sample.value;
        }
    }

    for info in by_instance.into_values() {
        match &best {
            Some(current) if current.total_committed_bytes >= info.total_committed_bytes => {}
            _ => best = Some(info),
        }
    }
    best
}
