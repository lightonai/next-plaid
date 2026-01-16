//! Compare Rust ONNX embeddings with PyLate outputs.
//!
//! This example encodes test documents and outputs the results in JSON format
//! for comparison with PyLate embeddings.
//!
//! Run with:
//!     cargo run --example compare_with_pylate --release
//!
//! Note: Requires the model to be exported first. See python/export_onnx.py

use next_plaid_onnx::Colbert;

fn main() -> anyhow::Result<()> {
    // Test documents (same as in Python compare_embeddings.py)
    let test_docs = vec![
        "Paris is the capital of France.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Hello, world!",
    ];

    let test_queries = vec![
        "What is the capital of France?",
        "Tell me about machine learning.",
    ];

    // Model path
    let model_path = "models/GTE-ModernColBERT-v1";

    println!("Loading model from: {}", model_path);
    let model = Colbert::new(model_path)?;

    // Encode documents
    println!("\nEncoding {} documents...", test_docs.len());
    let doc_embeddings = model.encode_documents(&test_docs, None)?;

    // Encode queries
    println!("Encoding {} queries...", test_queries.len());
    let query_embeddings = model.encode_queries(&test_queries)?;

    // Output results as JSON
    println!("\n{}", "=".repeat(60));
    println!("RUST ONNX DOCUMENT EMBEDDINGS");
    println!("{}", "=".repeat(60));

    for (i, (doc, emb)) in test_docs.iter().zip(doc_embeddings.iter()).enumerate() {
        let shape = emb.dim();
        println!(
            "\nDocument {}: \"{}\"",
            i,
            doc.chars().take(50).collect::<String>()
        );
        println!("  Shape: ({}, {})", shape.0, shape.1);

        // Print first 5 values of first token
        if shape.0 > 0 && shape.1 >= 5 {
            let first_5: Vec<f32> = (0..5).map(|j| emb[[0, j]]).collect();
            println!(
                "  First token [0,:5]: {:?}",
                first_5
                    .iter()
                    .map(|v| format!("{:.4}", v))
                    .collect::<Vec<_>>()
            );
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("RUST ONNX QUERY EMBEDDINGS");
    println!("{}", "=".repeat(60));

    for (i, (query, emb)) in test_queries.iter().zip(query_embeddings.iter()).enumerate() {
        let shape = emb.dim();
        println!("\nQuery {}: \"{}\"", i, query);
        println!("  Shape: ({}, {})", shape.0, shape.1);

        if shape.0 > 0 && shape.1 >= 5 {
            let first_5: Vec<f32> = (0..5).map(|j| emb[[0, j]]).collect();
            println!(
                "  First token [0,:5]: {:?}",
                first_5
                    .iter()
                    .map(|v| format!("{:.4}", v))
                    .collect::<Vec<_>>()
            );
        }
    }

    // Output full embeddings to a JSON file for detailed comparison
    let output_path = "rust_embeddings.json";
    println!("\n{}", "=".repeat(60));
    println!("Saving full embeddings to: {}", output_path);

    let mut output = serde_json::Map::new();

    // Convert document embeddings to JSON
    let doc_json: Vec<serde_json::Value> = doc_embeddings
        .iter()
        .map(|emb| {
            let shape = emb.dim();
            let data: Vec<Vec<f32>> = (0..shape.0)
                .map(|i| (0..shape.1).map(|j| emb[[i, j]]).collect())
                .collect();
            serde_json::json!({
                "shape": [shape.0, shape.1],
                "data": data
            })
        })
        .collect();
    output.insert("documents".to_string(), serde_json::Value::Array(doc_json));

    // Convert query embeddings to JSON
    let query_json: Vec<serde_json::Value> = query_embeddings
        .iter()
        .map(|emb| {
            let shape = emb.dim();
            let data: Vec<Vec<f32>> = (0..shape.0)
                .map(|i| (0..shape.1).map(|j| emb[[i, j]]).collect())
                .collect();
            serde_json::json!({
                "shape": [shape.0, shape.1],
                "data": data
            })
        })
        .collect();
    output.insert("queries".to_string(), serde_json::Value::Array(query_json));

    let json_str = serde_json::to_string_pretty(&output)?;
    std::fs::write(output_path, json_str)?;
    println!("Done!");

    Ok(())
}
