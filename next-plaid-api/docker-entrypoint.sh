#!/bin/bash
# =============================================================================
# Lategrep API Docker Entrypoint
# =============================================================================
# Automatically downloads models from HuggingFace Hub if needed.
#
# Usage:
#   --model <path>              Use local model directory
#   --model <org/model>         Auto-download from HuggingFace Hub
#   --model <org/model> --int8  Download INT8 quantized version
#
# Examples:
#   --model /models/my-model                        # Local path
#   --model lightonai/GTE-ModernColBERT-v1-onnx     # Download from HF
#   --model lightonai/GTE-ModernColBERT-v1-onnx --int8  # Download INT8 version
#
# Environment variables (optional):
#   HF_TOKEN: HuggingFace token for private models
#   MODELS_DIR: Directory to store downloaded models (default: /models)
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

# Check if a string looks like a HuggingFace model ID (org/model format)
is_hf_model_id() {
    local path="$1"
    # Must contain exactly one slash, not start with / or ., and have content on both sides
    if [[ "$path" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$ ]]; then
        return 0
    fi
    return 1
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

    # Check for model file (either INT8 or FP32)
    if [ "$use_quantized" = "true" ]; then
        if [ ! -f "${model_path}/model_int8.onnx" ]; then
            return 1
        fi
    else
        # Accept either model.onnx or model_int8.onnx for non-quantized check
        if [ ! -f "${model_path}/model.onnx" ] && [ ! -f "${model_path}/model_int8.onnx" ]; then
            return 1
        fi
    fi

    return 0
}

# Extract --model value and its index from arguments
get_model_info() {
    local args=("$@")
    local i=0
    while [ $i -lt ${#args[@]} ]; do
        if [ "${args[$i]}" = "--model" ] || [ "${args[$i]}" = "-m" ]; then
            echo "$((i+1)):${args[$((i+1))]}"
            return 0
        fi
        ((i++))
    done
    return 1
}

# Check if --int8 flag is present
has_int8_flag() {
    for arg in "$@"; do
        if [ "$arg" = "--int8" ]; then
            return 0
        fi
    done
    return 1
}

# Remove --int8 flag from arguments
remove_int8_flag() {
    local result=()
    for arg in "$@"; do
        if [ "$arg" != "--int8" ]; then
            result+=("$arg")
        fi
    done
    echo "${result[@]}"
}

# Main entrypoint logic
main() {
    local model_info
    model_info=$(get_model_info "$@") || true

    local use_quantized="false"
    if has_int8_flag "$@"; then
        use_quantized="true"
    fi

    if [ -n "$model_info" ]; then
        local model_idx="${model_info%%:*}"
        local model_path="${model_info#*:}"
        local models_dir="${MODELS_DIR:-/models}"

        # Check if it's a HuggingFace model ID
        if is_hf_model_id "$model_path"; then
            local repo_id="$model_path"
            local model_name="${repo_id#*/}"  # Extract name after /
            local local_path="${models_dir}/${model_name}"

            log_info "Detected HuggingFace model: ${repo_id}"

            if ! model_exists "$local_path" "$use_quantized"; then
                log_info "Model not found locally at ${local_path}"

                if download_model "$local_path" "$repo_id" "${HF_TOKEN:-}" "$use_quantized"; then
                    log_info "Model ready at ${local_path}"
                else
                    log_error "Failed to download model from ${repo_id}"
                    exit 1
                fi
            else
                log_info "Model already exists at ${local_path}"
            fi

            # Rebuild arguments with local path instead of HF model ID
            local new_args=()
            local i=0
            for arg in "$@"; do
                if [ $i -eq $model_idx ]; then
                    new_args+=("$local_path")
                elif [ "$arg" != "--int8" ]; then
                    new_args+=("$arg")
                fi
                i=$((i + 1))
            done

            # Execute with modified arguments
            exec next-plaid-api "${new_args[@]}"
        else
            # Local path - check if it exists
            if [ ! -d "$model_path" ]; then
                log_warn "Model path ${model_path} does not exist"
                log_warn "Tip: Use a HuggingFace model ID like 'lightonai/GTE-ModernColBERT-v1-onnx'"
            fi
        fi
    fi

    # Remove --int8 flag if present (not recognized by next-plaid-api)
    local clean_args
    clean_args=$(remove_int8_flag "$@")

    # Execute the API
    exec next-plaid-api $clean_args
}

main "$@"
