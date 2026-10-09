//! Tests for CUDA code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

const KERNELS: &str = r#"#include <cuda_runtime.h>

/// Adds two vectors element-wise.
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__device__ __forceinline__ float square(float x) {
    return x * x;
}

template <typename T>
__global__ void scale(T* data, T factor, int n) {
    __shared__ T tile[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] * factor;
}

void launch_vector_add(const float* a, const float* b, float* c, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(a, b, c, n);
    cudaDeviceSynchronize();
}
"#;

#[test]
fn test_kernel() {
    let units = parse(KERNELS, Language::Cuda, "kernels.cu");

    let unit = get_unit_by_name(&units, "vector_add").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: vector_add
Signature: __global__ void vector_add(const float* a, const float* b, float* c, int n) {
Description: Adds two vectors element-wise.
Parameters: a, b, c, n
Returns: void
Variables: i
File: kernels kernels.cu
Code:
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}"#;
    assert_eq!(text, expected);
}

#[test]
fn test_device_function_and_template_kernel() {
    let units = parse(KERNELS, Language::Cuda, "kernels.cu");

    let square = get_unit_by_name(&units, "square").unwrap();
    assert_eq!((square.line, square.end_line), (11, 13));
    assert_eq!(square.parameters, vec!["x"]);
    assert_eq!(square.return_type.as_deref(), Some("float"));

    let scale = get_unit_by_name(&units, "scale").unwrap();
    assert_eq!((scale.line, scale.end_line), (16, 20));
    assert_eq!(scale.parameters, vec!["data", "factor", "n"]);
    assert!(scale.variables.contains(&"tile".to_string()));
}

/// The `<<<grid, block>>>` launch parses as a call to the kernel, so the call
/// graph links host code to the kernels it launches.
#[test]
fn test_kernel_launch_is_a_call() {
    let units = parse(KERNELS, Language::Cuda, "kernels.cu");

    let launch = get_unit_by_name(&units, "launch_vector_add").unwrap();
    assert!(
        launch.calls.contains(&"vector_add".to_string()),
        "{:?}",
        launch.calls
    );
    assert!(launch.calls.contains(&"cudaDeviceSynchronize".to_string()));
}

/// The CUDA grammar extends C++'s, so plain C++ in a `.cu` file is split
/// exactly as it would be in a `.cpp` file.
#[test]
fn test_plain_cpp_matches_cpp() {
    let source = r#"namespace math {
class Calculator {
public:
    int add(int a, int b) { return a + b; }
};
}

template <typename T>
T identity(T x) {
    return x;
}"#;
    let shape = |lang, file| {
        parse(source, lang, file)
            .into_iter()
            .map(|u| (format!("{:?}", u.unit_type), u.name, u.line, u.end_line))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        shape(Language::Cuda, "math.cu"),
        shape(Language::Cpp, "math.cpp")
    );
}

/// AMD HIP sources (`.hip`) use CUDA's syntax and parse with the CUDA grammar.
#[test]
fn test_hip_kernel() {
    let source = r#"#include <hip/hip_runtime.h>

__global__ void saxpy(float a, const float* x, float* y, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    saxpy<<<dim3(64), dim3(256), 0, hipStreamDefault>>>(2.f, d_x, d_y, n);
    HIP_CHECK(hipDeviceSynchronize());
}
"#;
    let lang = crate::parser::detect_language(std::path::Path::new("saxpy/main.hip")).unwrap();
    assert_eq!(lang, Language::Cuda);
    let units = parse(source, lang, "main.hip");
    let saxpy = get_unit_by_name(&units, "saxpy").unwrap();
    assert_eq!((saxpy.line, saxpy.end_line), (3, 6));
    assert_eq!(saxpy.parameters, vec!["a", "x", "y", "n"]);
    let main = get_unit_by_name(&units, "main").unwrap();
    assert!(
        main.calls.contains(&"saxpy".to_string()),
        "{:?}",
        main.calls
    );
}
