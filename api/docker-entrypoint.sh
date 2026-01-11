#!/bin/bash
# =============================================================================
# Lategrep API Docker Entrypoint
# =============================================================================
# Downloads models from HuggingFace Hub if they don't exist locally.
#
# Environment variables:
#   HF_MODEL_ID: HuggingFace model repository (e.g., "lightonai/GTE-ModernColBERT-v1-onnx")
#   HF_TOKEN: Optional HuggingFace token for private models
#   MODEL_QUANTIZED: Set to "true" to download INT8 quantized model (default: false)
#
# The script looks for --model flag in the arguments and downloads the model
# from HuggingFace if the path doesn't exist.
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Download a file from HuggingFace Hub
download_hf_file() {
    local repo_id="$1"
    local filename="$2"
    local dest_path="$3"
    local token="$4"

    local url="https://huggingface.co/${repo_id}/resolve/main/${filename}"

    log_info "Downloading ${filename}..."

    local curl_opts="-fSL --progress-bar"
    if [ -n "$token" ]; then
        curl_opts="$curl_opts -H \"Authorization: Bearer ${token}\""
    fi

    if eval curl $curl_opts -o "${dest_path}/${filename}" "$url"; then
        log_info "Downloaded ${filename}"
        return 0
    else
        log_error "Failed to download ${filename}"
        return 1
    fi
}

# Download model from HuggingFace Hub
download_model() {
    local model_path="$1"
    local repo_id="$2"
    local token="${3:-}"
    local use_quantized="${4:-false}"

    log_info "Downloading model from HuggingFace: ${repo_id}"
    log_info "Target directory: ${model_path}"

    # Create directory
    mkdir -p "$model_path"

    # Required files
    local files=("tokenizer.json" "config_sentence_transformers.json")

    # Add model file based on quantization preference
    if [ "$use_quantized" = "true" ]; then
        files+=("model_int8.onnx")
        log_info "Using INT8 quantized model"
    else
        files+=("model.onnx")
        log_info "Using FP32 model"
    fi

    # Download each file
    local failed=0
    for file in "${files[@]}"; do
        if ! download_hf_file "$repo_id" "$file" "$model_path" "$token"; then
            failed=1
        fi
    done

    if [ $failed -eq 1 ]; then
        log_error "Some files failed to download"
        return 1
    fi

    log_info "Model download complete"
    return 0
}

# Check if model directory has required files
model_exists() {
    local model_path="$1"
    local use_quantized="${2:-false}"

    if [ ! -d "$model_path" ]; then
        return 1
    fi

    # Check for tokenizer
    if [ ! -f "${model_path}/tokenizer.json" ]; then
        return 1
    fi

    # Check for model file
    if [ "$use_quantized" = "true" ]; then
        if [ ! -f "${model_path}/model_int8.onnx" ]; then
            return 1
        fi
    else
        if [ ! -f "${model_path}/model.onnx" ]; then
            return 1
        fi
    fi

    return 0
}

# Extract --model path from arguments
get_model_path() {
    local args=("$@")
    local i=0
    while [ $i -lt ${#args[@]} ]; do
        if [ "${args[$i]}" = "--model" ] || [ "${args[$i]}" = "-m" ]; then
            echo "${args[$((i+1))]}"
            return 0
        fi
        ((i++))
    done
    return 1
}

# Main entrypoint logic
main() {
    local model_path
    model_path=$(get_model_path "$@") || true

    # If --model is specified and HF_MODEL_ID is set, check if we need to download
    if [ -n "$model_path" ] && [ -n "$HF_MODEL_ID" ]; then
        local use_quantized="${MODEL_QUANTIZED:-false}"

        if ! model_exists "$model_path" "$use_quantized"; then
            log_info "Model not found at ${model_path}"

            if download_model "$model_path" "$HF_MODEL_ID" "${HF_TOKEN:-}" "$use_quantized"; then
                log_info "Model ready at ${model_path}"
            else
                log_error "Failed to download model from ${HF_MODEL_ID}"
                exit 1
            fi
        else
            log_info "Model already exists at ${model_path}"
        fi
    elif [ -n "$model_path" ] && [ -z "$HF_MODEL_ID" ]; then
        # Model path specified but no HF_MODEL_ID - check if model exists
        if [ ! -d "$model_path" ]; then
            log_warn "Model path ${model_path} does not exist and HF_MODEL_ID is not set"
            log_warn "Set HF_MODEL_ID to auto-download from HuggingFace Hub"
        fi
    fi

    # Execute the API
    exec lategrep-api "$@"
}

main "$@"
