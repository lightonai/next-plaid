#!/bin/bash
#
# Run API integration tests
# This script builds and starts the API server, runs Python tests, then cleans up
#
# Usage: ./scripts/run-api-tests.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
API_PORT=8080
API_PID=""
TEST_INDEX_DIR="$REPO_ROOT/next-plaid-api/test_indices"

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

cleanup() {
    log_info "Cleaning up..."

    # Kill the API server if running
    if [ -n "$API_PID" ] && kill -0 "$API_PID" 2>/dev/null; then
        log_info "Stopping API server (PID: $API_PID)..."
        kill "$API_PID" 2>/dev/null || true
        wait "$API_PID" 2>/dev/null || true
    fi

    # Kill any process on the API port
    if lsof -t -i:"$API_PORT" >/dev/null 2>&1; then
        log_warn "Killing process on port $API_PORT..."
        kill -9 $(lsof -t -i:"$API_PORT") 2>/dev/null || true
    fi

    # Remove test indices
    if [ -d "$TEST_INDEX_DIR" ]; then
        log_info "Removing test indices..."
        rm -rf "$TEST_INDEX_DIR"
    fi
}

# Set up trap to clean up on exit
trap cleanup EXIT

# Start from repo root
cd "$REPO_ROOT"

# Kill any existing server on the port
if lsof -t -i:"$API_PORT" >/dev/null 2>&1; then
    log_warn "Killing existing process on port $API_PORT..."
    kill -9 $(lsof -t -i:"$API_PORT") 2>/dev/null || true
    sleep 2
fi

# Build the API server
log_info "Building API server..."
cargo build --release -p next-plaid-api

# Create test index directory
mkdir -p "$TEST_INDEX_DIR"

# Start the API server in the background
log_info "Starting API server on port $API_PORT..."
"$REPO_ROOT/target/release/next-plaid-api" -d "$TEST_INDEX_DIR" -p "$API_PORT" &
API_PID=$!

# Wait for the server to be ready
log_info "Waiting for API server to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0
while ! curl -s "http://localhost:$API_PORT/health" >/dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        log_error "API server failed to start after $MAX_RETRIES seconds"
        exit 1
    fi
    sleep 1
done
log_info "API server is ready!"

# Run the Python tests
log_info "Running API integration tests..."
cd "$REPO_ROOT/next-plaid-api/tests"

# Sync dependencies with uv
uv sync

# Run pytest
if uv run pytest test_api.py -v; then
    log_info "All API integration tests passed!"
    exit 0
else
    log_error "API integration tests failed!"
    exit 1
fi
