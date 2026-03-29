use anyhow::{Context, Result};
use colgrep::{build_call_graph, build_embedding_text, detect_language, extract_units};
use ignore::WalkBuilder;
use next_plaid_onnx::{Colbert, ExecutionProvider};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::env;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct Shape {
    docs: usize,
    len: usize,
}

#[derive(Clone, Copy, Debug)]
struct Step {
    prev: usize,
    docs: usize,
    len: usize,
}

struct ShapeCatalog {
    name: &'static str,
    shapes: Vec<Shape>,
}

fn main() -> Result<()> {
    let mut model_dir: Option<String> = None;
    let mut path = PathBuf::from(".");
    let mut batch_size = 16usize;
    let mut document_length: Option<usize> = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_dir = args.next(),
            "--path" => {
                path = PathBuf::from(args.next().context("--path requires a value")?);
            }
            "--batch-size" => {
                batch_size = args
                    .next()
                    .and_then(|v| v.parse::<usize>().ok())
                    .context("--batch-size requires a positive integer")?;
            }
            "--document-length" => {
                document_length = Some(
                    args.next()
                        .and_then(|v| v.parse::<usize>().ok())
                        .context("--document-length requires a positive integer")?,
                );
            }
            other => anyhow::bail!("Unknown argument: {other}"),
        }
    }

    let model_dir = model_dir.context("--model is required")?;
    let path = std::fs::canonicalize(&path)
        .with_context(|| format!("Path does not exist: {}", path.display()))?;

    let files = collect_files(&path)?;
    let mut units = parse_units(&path, &files)?;
    build_call_graph(&mut units);

    let mut seen = HashSet::new();
    let mut texts = Vec::new();
    for unit in &units {
        let text = build_embedding_text(unit);
        if seen.insert(text.clone()) {
            texts.push(text);
        }
    }

    let mut builder = Colbert::builder(model_dir)
        .with_execution_provider(ExecutionProvider::Cpu)
        .with_parallel(1)
        .with_batch_size(batch_size);
    if let Some(document_length) = document_length {
        builder = builder.with_document_length(document_length);
    }
    let model = builder.build()?;

    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let mut lengths = model.tokenize_document_lengths(&refs)?;
    lengths.sort_unstable();

    let catalogs = build_shape_catalogs(model.batch_size(), model.config().document_length);

    println!("path: {}", path.display());
    println!("files: {}", files.len());
    println!("units: {}", units.len());
    println!("unique_texts: {}", texts.len());
    println!("batch_size: {}", model.batch_size());
    println!("document_length: {}", model.config().document_length);
    println!("documents_analyzed: {}", lengths.len());
    for catalog in catalogs {
        let analysis = analyze_shapes(&lengths, &catalog.shapes)?;
        println!();
        println!("catalog: {}", catalog.name);
        println!("candidate_shapes: {}", catalog.shapes.len());
        println!("total_actual_tokens: {}", analysis.total_actual_tokens);
        println!("total_padded_tokens: {}", analysis.total_padded_tokens);
        println!("padding_overhead_tokens: {}", analysis.total_overhead_tokens);
        println!(
            "padding_overhead_pct: {:.2}",
            if analysis.total_actual_tokens == 0 {
                0.0
            } else {
                (analysis.total_overhead_tokens as f64 * 100.0) / analysis.total_actual_tokens as f64
            }
        );
        println!("batches: {}", analysis.steps.len());
        println!("distinct_shapes_used: {}", analysis.shape_counts.len());
        println!("used_shapes:");
        for (shape, count) in &analysis.shape_counts {
            println!("  {}x{}: {}", shape.docs, shape.len, count);
        }
    }

    Ok(())
}

fn collect_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let walker = WalkBuilder::new(root)
        .standard_filters(true)
        .hidden(false)
        .build();

    for entry in walker {
        let entry = entry?;
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let full_path = entry.into_path();
        let rel_path = full_path
            .strip_prefix(root)
            .unwrap_or(&full_path)
            .to_path_buf();
        if detect_language(&rel_path).is_some() {
            files.push(rel_path);
        }
    }

    Ok(files)
}

fn parse_units(root: &Path, files: &[PathBuf]) -> Result<Vec<colgrep::CodeUnit>> {
    let parsed = files
        .par_iter()
        .map(|path| -> Result<Vec<colgrep::CodeUnit>> {
            let full_path = root.join(path);
            let lang = match detect_language(path) {
                Some(lang) => lang,
                None => return Ok(Vec::new()),
            };
            let source = std::fs::read_to_string(&full_path)
                .with_context(|| format!("Failed to read {}", full_path.display()))?;
            Ok(extract_units(path, &source, lang))
        })
        .collect::<Vec<_>>();

    let mut units = Vec::new();
    for result in parsed {
        units.extend(result?);
    }
    Ok(units)
}

fn round_up_len(len: usize) -> usize {
    if len <= 8 {
        return len.max(1);
    }
    len.div_ceil(32) * 32
}

fn build_shapes(batch_size: usize, document_length: usize) -> Vec<Shape> {
    let total_budget = batch_size.max(1).saturating_mul(document_length.max(1));
    let mut shapes = Vec::new();
    let mut len = round_up_len(document_length.max(1));
    let min_len = 128.min(len.max(1));

    loop {
        shapes.push(Shape {
            docs: total_budget.checked_div(len).unwrap_or(0).max(1),
            len,
        });
        if len <= min_len {
            break;
        }
        let next_len = round_up_len((len / 2).max(min_len));
        if next_len == len {
            break;
        }
        len = next_len;
    }

    shapes.sort_by_key(|shape| shape.len);
    shapes
}

fn build_shape_catalogs(batch_size: usize, document_length: usize) -> Vec<ShapeCatalog> {
    let total_budget = batch_size.max(1).saturating_mul(document_length.max(1));
    let current = build_shapes(batch_size, document_length);

    let len32 = build_len_step_catalog(total_budget, document_length, 32, false);
    let len64 = build_len_step_catalog(total_budget, document_length, 64, false);
    let len32_pow2_docs = build_len_step_catalog(total_budget, document_length, 32, true);

    vec![
        ShapeCatalog {
            name: "current_ladder",
            shapes: current,
        },
        ShapeCatalog {
            name: "len_step_64",
            shapes: len64,
        },
        ShapeCatalog {
            name: "len_step_32",
            shapes: len32,
        },
        ShapeCatalog {
            name: "len_step_32_pow2_docs",
            shapes: len32_pow2_docs,
        },
    ]
}

fn build_len_step_catalog(
    total_budget: usize,
    document_length: usize,
    step: usize,
    include_pow2_docs: bool,
) -> Vec<Shape> {
    let mut shapes = BTreeSet::new();
    let mut len = 128usize;
    let max_len = round_up_len(document_length.max(128));
    while len <= max_len {
        let max_docs = total_budget.checked_div(len).unwrap_or(0).max(1);
        shapes.insert(Shape { docs: max_docs, len });
        if include_pow2_docs {
            let mut docs = 1usize;
            while docs <= max_docs {
                shapes.insert(Shape { docs, len });
                docs *= 2;
            }
            if !max_docs.is_power_of_two() {
                shapes.insert(Shape { docs: max_docs, len });
            }
        }
        len += step;
    }
    shapes.into_iter().collect()
}

struct AnalysisResult {
    total_actual_tokens: u64,
    total_padded_tokens: u64,
    total_overhead_tokens: u64,
    steps: Vec<Step>,
    shape_counts: BTreeMap<Shape, usize>,
}

fn analyze_shapes(lengths: &[usize], shapes: &[Shape]) -> Result<AnalysisResult> {
    let n = lengths.len();
    let mut prefix = vec![0u64; n + 1];
    for (i, &len) in lengths.iter().enumerate() {
        prefix[i + 1] = prefix[i] + len as u64;
    }

    let mut best_cost = vec![u64::MAX; n + 1];
    let mut best_batches = vec![usize::MAX; n + 1];
    let mut prev: Vec<Option<Step>> = vec![None; n + 1];
    best_cost[0] = 0;
    best_batches[0] = 0;

    for i in 0..n {
        if best_cost[i] == u64::MAX {
            continue;
        }
        for shape in shapes {
            let fit_count = lengths[i..].partition_point(|&len| len <= shape.len);
            let take = fit_count.min(shape.docs);
            if take == 0 {
                continue;
            }
            let next = i + take;
            let batch_sum = prefix[next] - prefix[i];
            let padded = shape.len as u64 * take as u64;
            let overhead = padded.saturating_sub(batch_sum);
            let candidate_cost = best_cost[i].saturating_add(overhead);
            let candidate_batches = best_batches[i] + 1;
            if candidate_cost < best_cost[next]
                || (candidate_cost == best_cost[next] && candidate_batches < best_batches[next])
            {
                best_cost[next] = candidate_cost;
                best_batches[next] = candidate_batches;
                prev[next] = Some(Step {
                    prev: i,
                    docs: take,
                    len: shape.len,
                });
            }
        }
    }

    if best_cost[n] == u64::MAX {
        anyhow::bail!("No feasible batching solution found");
    }

    let mut steps = Vec::new();
    let mut idx = n;
    while idx > 0 {
        let step = prev[idx].context("Broken DP backtrace")?;
        idx = step.prev;
        steps.push(step);
    }
    steps.reverse();

    let mut shape_counts = BTreeMap::new();
    let mut total_padded_tokens = 0u64;
    for step in &steps {
        *shape_counts
            .entry(Shape {
                docs: step.docs,
                len: step.len,
            })
            .or_insert(0) += 1;
        total_padded_tokens += step.docs as u64 * step.len as u64;
    }

    Ok(AnalysisResult {
        total_actual_tokens: prefix[n],
        total_padded_tokens,
        total_overhead_tokens: best_cost[n],
        steps,
        shape_counts,
    })
}
