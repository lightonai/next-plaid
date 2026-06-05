//! Cheap host hardware inspection for production acceleration heuristics.
//!
//! This module intentionally uses `/proc` and `/sys` on Linux instead of ROCm
//! APIs. The auto path can call it before deciding whether touching the GPU
//! stack is worthwhile.

use serde::Serialize;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

pub const DEFAULT_MIGRAPHX_AUTO_MIN_UNITS: usize = 10_000;

#[derive(Debug, Clone, Serialize)]
pub struct CpuInfo {
    pub logical_cores: usize,
    pub model_name: Option<String>,
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_fma: bool,
    pub has_neon: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct AmdGpuInfo {
    pub drm_node: String,
    pub vendor_id: Option<String>,
    pub device_id: Option<String>,
    pub driver: Option<String>,
    pub vram_total_bytes: Option<u64>,
    pub gtt_total_bytes: Option<u64>,
    pub max_sclk_mhz: Option<u64>,
    pub integrated_guess: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct MigraphxAutoPolicy {
    pub min_units: usize,
    pub source: String,
    pub reason: String,
    pub cpu: CpuInfo,
    pub gpu: Option<AmdGpuInfo>,
    pub model: Option<ModelInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub model_name: Option<String>,
    pub model_class: Option<String>,
    pub embedding_dim: Option<usize>,
    pub query_length: Option<usize>,
    pub document_length: Option<usize>,
    pub hidden_size: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub local_attention: Option<usize>,
    pub global_attn_every_n_layers: Option<usize>,
    pub estimated_query_macs: Option<u64>,
    pub estimated_document_macs: Option<u64>,
    pub model_onnx_bytes: Option<u64>,
    pub model_fp16_onnx_bytes: Option<u64>,
    pub model_int8_onnx_bytes: Option<u64>,
}

pub fn migraphx_auto_policy() -> MigraphxAutoPolicy {
    migraphx_auto_policy_for_model(None)
}

pub fn migraphx_auto_policy_for_model(model_dir: Option<&Path>) -> MigraphxAutoPolicy {
    let cpu = detect_cpu_info();
    let gpu = detect_amd_gpus().into_iter().next();
    let model = model_dir.map(detect_model_info);

    if let Some(value) = std::env::var("NEXT_PLAID_MIGRAPHX_AUTO_MIN_UNITS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
    {
        return MigraphxAutoPolicy {
            min_units: value,
            source: "env".to_string(),
            reason: "NEXT_PLAID_MIGRAPHX_AUTO_MIN_UNITS override".to_string(),
            cpu,
            gpu,
            model,
        };
    }

    let (min_units, reason) = estimate_migraphx_auto_min_units(&cpu, gpu.as_ref(), model.as_ref());
    MigraphxAutoPolicy {
        min_units,
        source: "hardware".to_string(),
        reason,
        cpu,
        gpu,
        model,
    }
}

fn detect_model_info(model_dir: &Path) -> ModelInfo {
    let onnx_config = fs::read_to_string(model_dir.join("onnx_config.json"))
        .ok()
        .and_then(|contents| serde_json::from_str::<serde_json::Value>(&contents).ok());
    let model_config = fs::read_to_string(model_dir.join("config.json"))
        .ok()
        .and_then(|contents| serde_json::from_str::<serde_json::Value>(&contents).ok());
    let get_onnx_string = |key: &str| {
        onnx_config
            .as_ref()
            .and_then(|value| value.get(key))
            .and_then(|value| value.as_str())
            .map(ToString::to_string)
    };
    let get_onnx_usize = |key: &str| {
        onnx_config
            .as_ref()
            .and_then(|value| value.get(key))
            .and_then(|value| value.as_u64())
            .and_then(|value| usize::try_from(value).ok())
    };
    let get_model_usize = |key: &str| {
        model_config
            .as_ref()
            .and_then(|value| value.get(key))
            .and_then(|value| value.as_u64())
            .and_then(|value| usize::try_from(value).ok())
    };

    let query_length = get_onnx_usize("query_length");
    let document_length = get_onnx_usize("document_length");
    let hidden_size = get_model_usize("hidden_size");
    let intermediate_size = get_model_usize("intermediate_size");
    let num_hidden_layers = get_model_usize("num_hidden_layers");
    let num_attention_heads = get_model_usize("num_attention_heads");
    let local_attention = get_model_usize("local_attention");
    let global_attn_every_n_layers = get_model_usize("global_attn_every_n_layers");
    let estimated_query_macs = estimate_transformer_macs(
        query_length,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        local_attention,
        global_attn_every_n_layers,
    );
    let estimated_document_macs = estimate_transformer_macs(
        document_length,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        local_attention,
        global_attn_every_n_layers,
    );

    ModelInfo {
        model_name: get_onnx_string("model_name"),
        model_class: get_onnx_string("model_class"),
        embedding_dim: get_onnx_usize("embedding_dim"),
        query_length,
        document_length,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        local_attention,
        global_attn_every_n_layers,
        estimated_query_macs,
        estimated_document_macs,
        model_onnx_bytes: file_len(model_dir.join("model.onnx")),
        model_fp16_onnx_bytes: file_len(model_dir.join("model_fp16.onnx")),
        model_int8_onnx_bytes: file_len(model_dir.join("model_int8.onnx")),
    }
}

fn estimate_transformer_macs(
    seq_len: Option<usize>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    local_attention: Option<usize>,
    global_attn_every_n_layers: Option<usize>,
) -> Option<u64> {
    let seq_len = seq_len?;
    let hidden_size = hidden_size?;
    let intermediate_size = intermediate_size?;
    let num_hidden_layers = num_hidden_layers?;
    if seq_len == 0 || hidden_size == 0 || intermediate_size == 0 || num_hidden_layers == 0 {
        return None;
    }

    let seq = seq_len as u128;
    let hidden = hidden_size as u128;
    let intermediate = intermediate_size as u128;
    let mut total = 0u128;

    for layer in 0..num_hidden_layers {
        // Transformer block approximation in MACs, not FLOPs:
        // - Q/K/V/O projections: 4 × S × H × H
        // - MLP up/down projections: 2 × S × H × I
        // - attention score and value matmuls: 2 × S × attention_window × H
        let linear_macs = seq * ((4 * hidden * hidden) + (2 * hidden * intermediate));
        let is_global_attention = global_attn_every_n_layers
            .filter(|every| *every > 0)
            .is_none_or(|every| layer % every == 0);
        let attention_window = if is_global_attention {
            seq_len
        } else {
            local_attention.unwrap_or(seq_len).min(seq_len)
        } as u128;
        let attention_macs = 2 * seq * attention_window * hidden;
        total = total.saturating_add(linear_macs.saturating_add(attention_macs));
    }

    Some(total.min(u64::MAX as u128) as u64)
}

fn file_len(path: PathBuf) -> Option<u64> {
    fs::metadata(path).ok().map(|metadata| metadata.len())
}

fn detect_cpu_info() -> CpuInfo {
    let logical_cores = std::thread::available_parallelism()
        .map(|cores| cores.get())
        .unwrap_or(1);

    let mut model_name = None;
    let mut flags = HashSet::new();

    #[cfg(target_os = "linux")]
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        for line in cpuinfo.lines() {
            if model_name.is_none() {
                if let Some((key, value)) = line.split_once(':') {
                    if matches!(key.trim(), "model name" | "Hardware") {
                        let value = value.trim();
                        if !value.is_empty() {
                            model_name = Some(value.to_string());
                        }
                    }
                }
            }
            if flags.is_empty() {
                if let Some((key, value)) = line.split_once(':') {
                    if matches!(key.trim(), "flags" | "Features") {
                        flags.extend(value.split_whitespace().map(|flag| flag.to_string()));
                    }
                }
            }
            if model_name.is_some() && !flags.is_empty() {
                break;
            }
        }
    }

    CpuInfo {
        logical_cores,
        model_name,
        has_avx2: flags.contains("avx2"),
        has_avx512: flags.iter().any(|flag| flag.starts_with("avx512")),
        has_fma: flags.contains("fma"),
        has_neon: flags.contains("neon") || flags.contains("asimd"),
    }
}

fn detect_amd_gpus() -> Vec<AmdGpuInfo> {
    #[cfg(target_os = "linux")]
    {
        detect_amd_gpus_linux()
    }
    #[cfg(not(target_os = "linux"))]
    {
        Vec::new()
    }
}

#[cfg(target_os = "linux")]
fn detect_amd_gpus_linux() -> Vec<AmdGpuInfo> {
    let mut gpus = Vec::new();
    let mut seen_devices = HashSet::new();
    let Ok(entries) = fs::read_dir("/sys/class/drm") else {
        return gpus;
    };

    let mut drm_nodes = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            (name.starts_with("renderD") || (name.starts_with("card") && !name.contains('-')))
                .then_some((name, entry.path()))
        })
        .collect::<Vec<_>>();
    // Prefer render nodes, which are less likely to be display connector dirs.
    drm_nodes.sort_by_key(|(name, _)| if name.starts_with("renderD") { 0 } else { 1 });

    for (name, path) in drm_nodes {
        let device_path = path.join("device");
        let canonical = fs::canonicalize(&device_path).unwrap_or_else(|_| device_path.clone());
        if !seen_devices.insert(canonical) {
            continue;
        }
        let vendor_id = read_trimmed(device_path.join("vendor"));
        if !vendor_id
            .as_deref()
            .is_some_and(|vendor| vendor.eq_ignore_ascii_case("0x1002"))
        {
            continue;
        }

        let device_id = read_trimmed(device_path.join("device"));
        let driver = read_uevent_value(&device_path, "DRIVER");
        let vram_total_bytes = read_u64(device_path.join("mem_info_vram_total"));
        let gtt_total_bytes = read_u64(device_path.join("mem_info_gtt_total"));
        let max_sclk_mhz = read_max_clock_mhz(device_path.join("pp_dpm_sclk"));
        let integrated_guess = looks_integrated_gpu(vram_total_bytes, gtt_total_bytes);

        gpus.push(AmdGpuInfo {
            drm_node: name,
            vendor_id,
            device_id,
            driver,
            vram_total_bytes,
            gtt_total_bytes,
            max_sclk_mhz,
            integrated_guess,
        });
    }

    gpus.sort_by_key(|gpu| std::cmp::Reverse(gpu.vram_total_bytes.unwrap_or(0)));
    gpus
}

#[cfg(target_os = "linux")]
fn read_trimmed<P: AsRef<Path>>(path: P) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(target_os = "linux")]
fn read_u64(path: PathBuf) -> Option<u64> {
    read_trimmed(path).and_then(|value| value.parse::<u64>().ok())
}

#[cfg(target_os = "linux")]
fn read_uevent_value(device_path: &Path, key: &str) -> Option<String> {
    let uevent = fs::read_to_string(device_path.join("uevent")).ok()?;
    uevent.lines().find_map(|line| {
        line.split_once('=').and_then(|(line_key, value)| {
            (line_key == key && !value.trim().is_empty()).then(|| value.trim().to_string())
        })
    })
}

#[cfg(target_os = "linux")]
fn read_max_clock_mhz(path: PathBuf) -> Option<u64> {
    let contents = fs::read_to_string(path).ok()?;
    contents.lines().filter_map(parse_clock_mhz).max()
}

#[cfg(target_os = "linux")]
fn parse_clock_mhz(line: &str) -> Option<u64> {
    let lower = line.to_ascii_lowercase();
    let idx = lower.find("mhz")?;
    let before = &lower[..idx];
    let digits = before
        .chars()
        .rev()
        .skip_while(|ch| ch.is_whitespace() || *ch == '*')
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        return None;
    }
    digits.chars().rev().collect::<String>().parse().ok()
}

fn looks_integrated_gpu(vram_total_bytes: Option<u64>, gtt_total_bytes: Option<u64>) -> bool {
    const TWO_GIB: u64 = 2 * 1024 * 1024 * 1024;
    match (vram_total_bytes, gtt_total_bytes) {
        (Some(vram), Some(gtt)) => vram < TWO_GIB || (vram < gtt / 4),
        (Some(vram), None) => vram < TWO_GIB,
        _ => false,
    }
}

fn estimate_migraphx_auto_min_units(
    cpu: &CpuInfo,
    gpu: Option<&AmdGpuInfo>,
    model: Option<&ModelInfo>,
) -> (usize, String) {
    let Some(gpu) = gpu else {
        return (
            DEFAULT_MIGRAPHX_AUTO_MIN_UNITS,
            "no AMD GPU detected via sysfs; provider availability still decides final fallback"
                .to_string(),
        );
    };

    let gib = 1024 * 1024 * 1024u64;
    let vram = gpu.vram_total_bytes.unwrap_or(0);
    let mut threshold = if gpu.integrated_guess {
        2_000usize
    } else if vram >= 12 * gib {
        1_000
    } else if vram >= 8 * gib {
        1_500
    } else if vram >= 4 * gib {
        2_500
    } else if vram >= 2 * gib {
        4_000
    } else {
        6_000
    };

    let mut reasons = Vec::new();
    if gpu.integrated_guess {
        reasons.push("integrated/UMA AMD GPU".to_string());
    } else {
        reasons.push(format!("AMD GPU with {} GiB VRAM", vram / gib));
    }

    if cpu.logical_cores >= 24 && (cpu.has_avx2 || cpu.has_avx512) {
        threshold = threshold.saturating_mul(3) / 2;
        reasons.push(format!(
            "strong CPU ({} logical cores with SIMD)",
            cpu.logical_cores
        ));
    } else if cpu.logical_cores >= 16 && (cpu.has_avx2 || cpu.has_neon) {
        threshold = threshold.saturating_mul(5) / 4;
        reasons.push(format!(
            "multi-core CPU ({} logical cores)",
            cpu.logical_cores
        ));
    } else if cpu.logical_cores <= 8 {
        threshold = threshold.saturating_mul(3) / 4;
        reasons.push(format!("smaller CPU ({} logical cores)", cpu.logical_cores));
    }

    if let Some(max_sclk_mhz) = gpu.max_sclk_mhz {
        if !gpu.integrated_guess && max_sclk_mhz >= 2400 {
            threshold = threshold.saturating_mul(9) / 10;
            reasons.push(format!("high GPU clock ({max_sclk_mhz} MHz)"));
        } else if max_sclk_mhz < 1500 {
            threshold = threshold.saturating_mul(5) / 4;
            reasons.push(format!("low GPU clock ({max_sclk_mhz} MHz)"));
        }
    }

    if let Some(model) = model {
        let before = threshold;
        threshold = apply_model_complexity_adjustment(threshold, model);
        if threshold < before {
            reasons.push(model_complexity_reason(model, "larger model favors GPU"));
        } else if threshold > before {
            reasons.push(model_complexity_reason(model, "small model favors CPU"));
        }
    }

    threshold = threshold.clamp(500, 25_000);
    (threshold, reasons.join("; "))
}

fn apply_model_complexity_adjustment(threshold: usize, model: &ModelInfo) -> usize {
    // Estimated 1×2048 document MACs for LateOn-Code-edge from config.json
    // (hidden=256, intermediate=384, layers=7, local attention=128,
    // global attention every 3 layers). This anchors the model scaling to a
    // known lightweight ColGREP model instead of raw file size.
    const LATEON_EDGE_DOC_MACS: f64 = 13_555_990_528.0;
    const EDGE_MODEL_BYTES: u64 = 65 * 1024 * 1024;
    const LARGE_MODEL_BYTES: u64 = 400 * 1024 * 1024;
    const MID_MODEL_BYTES: u64 = 150 * 1024 * 1024;

    if let Some(document_macs) = model.estimated_document_macs.filter(|macs| *macs > 0) {
        let factor = (1.25 * (LATEON_EDGE_DOC_MACS / document_macs as f64).sqrt()).clamp(0.4, 1.25);
        return ((threshold as f64) * factor).round() as usize;
    }

    let complexity_bytes = model
        .model_onnx_bytes
        .or(model
            .model_fp16_onnx_bytes
            .map(|bytes| bytes.saturating_mul(2)))
        .or(model
            .model_int8_onnx_bytes
            .map(|bytes| bytes.saturating_mul(4)));

    if model.embedding_dim.is_some_and(|dim| dim >= 128)
        || complexity_bytes.is_some_and(|bytes| bytes >= LARGE_MODEL_BYTES)
    {
        threshold.saturating_mul(2) / 5
    } else if model.embedding_dim.is_some_and(|dim| dim >= 96)
        || complexity_bytes.is_some_and(|bytes| bytes >= MID_MODEL_BYTES)
    {
        threshold.saturating_mul(3) / 5
    } else if model.embedding_dim.is_some_and(|dim| dim <= 64)
        || complexity_bytes.is_some_and(|bytes| bytes <= EDGE_MODEL_BYTES)
    {
        threshold.saturating_mul(5) / 4
    } else {
        threshold
    }
}

fn model_complexity_reason(model: &ModelInfo, prefix: &str) -> String {
    let name = model.model_name.as_deref().unwrap_or("unknown model");
    let dim = model
        .embedding_dim
        .map(|dim| dim.to_string())
        .unwrap_or_else(|| "?".to_string());
    let mib = model
        .model_onnx_bytes
        .map(|bytes| (bytes as f64 / (1024.0 * 1024.0)).round() as u64)
        .map(|mib| mib.to_string())
        .unwrap_or_else(|| "?".to_string());
    let doc_gmacs = model
        .estimated_document_macs
        .map(|macs| format!(", doc≈{:.1} GMAC", macs as f64 / 1_000_000_000.0))
        .unwrap_or_default();
    format!("{prefix}: {name} (dim={dim}, model.onnx≈{mib} MiB{doc_gmacs})")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu(logical_cores: usize, simd: bool) -> CpuInfo {
        CpuInfo {
            logical_cores,
            model_name: None,
            has_avx2: simd,
            has_avx512: false,
            has_fma: simd,
            has_neon: false,
        }
    }

    fn gpu(vram_gib: u64, integrated_guess: bool) -> AmdGpuInfo {
        AmdGpuInfo {
            drm_node: "renderD128".to_string(),
            vendor_id: Some("0x1002".to_string()),
            device_id: Some("0x0000".to_string()),
            driver: Some("amdgpu".to_string()),
            vram_total_bytes: Some(vram_gib * 1024 * 1024 * 1024),
            gtt_total_bytes: None,
            max_sclk_mhz: None,
            integrated_guess,
        }
    }

    fn model(embedding_dim: usize, model_mib: u64) -> ModelInfo {
        ModelInfo {
            model_name: Some(format!("test-dim-{embedding_dim}")),
            model_class: Some("ModernBertModel".to_string()),
            embedding_dim: Some(embedding_dim),
            query_length: Some(256),
            document_length: Some(2048),
            hidden_size: None,
            intermediate_size: None,
            num_hidden_layers: None,
            num_attention_heads: None,
            local_attention: None,
            global_attn_every_n_layers: None,
            estimated_query_macs: None,
            estimated_document_macs: None,
            model_onnx_bytes: Some(model_mib * 1024 * 1024),
            model_fp16_onnx_bytes: None,
            model_int8_onnx_bytes: None,
        }
    }

    #[test]
    fn integrated_gpu_with_strong_cpu_uses_high_threshold() {
        let (threshold, reason) =
            estimate_migraphx_auto_min_units(&cpu(32, true), Some(&gpu(1, true)), None);

        assert_eq!(threshold, 3_000);
        assert!(reason.contains("integrated"));
    }

    #[test]
    fn discrete_gpu_with_small_cpu_uses_lower_threshold() {
        let (threshold, reason) =
            estimate_migraphx_auto_min_units(&cpu(8, true), Some(&gpu(16, false)), None);

        assert_eq!(threshold, 750);
        assert!(reason.contains("16 GiB"));
    }

    #[test]
    fn larger_model_lowers_gpu_auto_threshold() {
        let base_cpu = cpu(32, true);
        let integrated = gpu(1, true);
        let (edge_threshold, edge_reason) =
            estimate_migraphx_auto_min_units(&base_cpu, Some(&integrated), Some(&model(48, 65)));
        let (full_threshold, full_reason) =
            estimate_migraphx_auto_min_units(&base_cpu, Some(&integrated), Some(&model(128, 570)));

        assert_eq!(edge_threshold, 3_750);
        assert_eq!(full_threshold, 1_200);
        assert!(edge_reason.contains("small model"));
        assert!(full_reason.contains("larger model"));
    }

    #[test]
    fn transformer_mac_estimate_tracks_lateon_model_scale() {
        let edge_doc_macs = estimate_transformer_macs(
            Some(2048),
            Some(256),
            Some(384),
            Some(7),
            Some(128),
            Some(3),
        )
        .unwrap();
        let full_doc_macs = estimate_transformer_macs(
            Some(2048),
            Some(768),
            Some(1152),
            Some(22),
            Some(128),
            Some(3),
        )
        .unwrap();
        let ratio = full_doc_macs as f64 / edge_doc_macs as f64;

        assert!((13_000_000_000..14_000_000_000).contains(&edge_doc_macs));
        assert!((17.0..19.0).contains(&ratio));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_amdgpu_clock_lines() {
        assert_eq!(parse_clock_mhz("2: 2900Mhz *"), Some(2900));
        assert_eq!(parse_clock_mhz("0: 600Mhz"), Some(600));
        assert_eq!(parse_clock_mhz("not a clock"), None);
    }
}
