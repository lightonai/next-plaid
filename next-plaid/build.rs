fn main() {
    // Only run this when the cuda feature is enabled
    #[cfg(feature = "cuda")]
    {
        // Common CUDA library search paths
        let cuda_paths = [
            "/usr/local/cuda/lib64/stubs",
            "/usr/local/cuda/targets/x86_64-linux/lib/stubs",
            "/usr/local/cuda-12/lib64/stubs",
            "/usr/local/cuda-12.9/lib64/stubs",
            "/usr/local/cuda-12.9/targets/x86_64-linux/lib/stubs",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
        ];

        for path in cuda_paths {
            if std::path::Path::new(path).exists() {
                println!("cargo:rustc-link-search=native={}", path);
            }
        }

        // Also check CUDA_PATH environment variable
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            let stubs_path = format!("{}/lib64/stubs", cuda_path);
            if std::path::Path::new(&stubs_path).exists() {
                println!("cargo:rustc-link-search=native={}", stubs_path);
            }
            let targets_stubs = format!("{}/targets/x86_64-linux/lib/stubs", cuda_path);
            if std::path::Path::new(&targets_stubs).exists() {
                println!("cargo:rustc-link-search=native={}", targets_stubs);
            }
        }
    }
}
