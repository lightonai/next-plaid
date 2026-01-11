use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;

const ONNXRUNTIME_VERSION: &str = "1.23.0";

fn main() {
    // Only download for x86_64 Linux when building binaries
    let target = env::var("TARGET").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if target_os != "linux" || !target.contains("x86_64") {
        println!("cargo:warning=ONNX Runtime auto-download only supported on x86_64-linux");
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let onnx_dir = out_dir.join("onnxruntime");
    let lib_path = onnx_dir.join("lib").join("libonnxruntime.so");

    // Skip if already downloaded
    if lib_path.exists() {
        println!(
            "cargo:rustc-env=ORT_DYLIB_PATH={}",
            lib_path.to_string_lossy()
        );
        return;
    }

    // Create directory
    fs::create_dir_all(&onnx_dir).expect("Failed to create onnxruntime directory");

    // Download URL
    let url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{version}/onnxruntime-linux-x64-{version}.tgz",
        version = ONNXRUNTIME_VERSION
    );

    println!("cargo:warning=Downloading ONNX Runtime from {}", url);

    // Download the file
    let response = ureq::get(&url).call().expect("Failed to download ONNX Runtime");

    let mut bytes = Vec::new();
    response
        .into_body()
        .into_reader()
        .read_to_end(&mut bytes)
        .expect("Failed to read response");

    // Extract the tarball
    let decoder = flate2::read::GzDecoder::new(&bytes[..]);
    let mut archive = tar::Archive::new(decoder);

    // Extract to temp location
    let temp_dir = out_dir.join("temp_onnx");
    fs::create_dir_all(&temp_dir).ok();
    archive
        .unpack(&temp_dir)
        .expect("Failed to extract ONNX Runtime");

    // Move the lib directory to our target location
    let extracted_dir = temp_dir.join(format!("onnxruntime-linux-x64-{}", ONNXRUNTIME_VERSION));
    let extracted_lib = extracted_dir.join("lib");

    // Create lib directory and copy files
    let target_lib_dir = onnx_dir.join("lib");
    fs::create_dir_all(&target_lib_dir).ok();

    // Copy library files
    for entry in fs::read_dir(&extracted_lib).expect("Failed to read lib dir") {
        let entry = entry.expect("Failed to read entry");
        let src_path = entry.path();
        let file_name = src_path.file_name().unwrap();
        let dest_path = target_lib_dir.join(file_name);

        if src_path.is_file() {
            fs::copy(&src_path, &dest_path).expect("Failed to copy library file");
        }
    }

    // Clean up temp directory
    fs::remove_dir_all(&temp_dir).ok();

    // Set environment variable for runtime
    println!(
        "cargo:rustc-env=ORT_DYLIB_PATH={}",
        lib_path.to_string_lossy()
    );

    // Write path to a file that lib.rs can read
    let path_file = out_dir.join("ort_lib_path.txt");
    let mut file = fs::File::create(&path_file).expect("Failed to create path file");
    file.write_all(lib_path.to_string_lossy().as_bytes())
        .expect("Failed to write path");

    println!("cargo:warning=ONNX Runtime downloaded to {}", lib_path.display());
}
