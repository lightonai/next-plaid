#!/usr/bin/env bash
# Usage: scripts/build-migraphx-ort.sh <work-dir> <out-dir> <rocm-prefix> [jobs]
# Requires Git, Python 3.8+, CMake 3.28+, a C++ compiler, and ROCm 7.1.1
# with MIGraphX development files. Installs nothing on the host.
set -euo pipefail

ORT_VERSION=1.23.1
ORT_COMMIT=d9b2048791efb5804fe3d53a04b4971256addebf
ROCM_VERSION=7.1.1
NAME="onnxruntime-linux-x64-migraphx-${ORT_VERSION}-rocm${ROCM_VERSION}-r4"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCHES=("$SCRIPT_DIR/ort-migraphx-lifetime.patch" "$SCRIPT_DIR/ort-migraphx-cache.patch" "$SCRIPT_DIR/ort-migraphx-integer.patch")

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <work-dir> <out-dir> <rocm-prefix> [jobs]" >&2
  exit 2
fi
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  echo "This runtime is built for Linux x86_64." >&2
  exit 2
fi
JOBS=${4:-$(nproc)}
if [[ ! $JOBS =~ ^[1-9][0-9]*$ ]]; then
  echo "jobs must be a positive integer." >&2
  exit 2
fi
PREFIX=$(cd "$3" && pwd)
mkdir -p "$1" "$2"
WORK=$(cd "$1" && pwd)
OUT=$(cd "$2" && pwd)
if [[ -e "$OUT/$NAME.tgz" || -e "$OUT/$NAME.tgz.sha256" ]]; then
  echo "Output already exists for $NAME; choose an empty output directory." >&2
  exit 2
fi
# Each invocation owns a new directory; keep it on failure for diagnosis.
BUILD=$(mktemp -d "$WORK/migraphx-ort.XXXXXXXX")
echo "Build directory: $BUILD"
SOURCE="$BUILD/source"
git init -q "$SOURCE"
git -C "$SOURCE" fetch --depth 1 https://github.com/microsoft/onnxruntime.git "$ORT_COMMIT"
git -C "$SOURCE" checkout -q --detach FETCH_HEAD
[[ $(git -C "$SOURCE" rev-parse HEAD) == "$ORT_COMMIT" ]]
git -C "$SOURCE" submodule update --init --recursive --depth 1
for PATCH in "${PATCHES[@]}"; do
  git -C "$SOURCE" apply --check "$PATCH"
  git -C "$SOURCE" apply "$PATCH"
done
EPOCH=$(git -C "$SOURCE" show -s --format=%ct HEAD)
export SOURCE_DATE_EPOCH="$EPOCH"

ROOT_ARGS=()
if [[ $(id -u) == 0 ]]; then
  ROOT_ARGS+=(--allow_running_as_root)
fi
python3 "$SOURCE/tools/ci_build/build.py" \
  --build_dir "$BUILD/build" --config Release --update --build \
  --parallel "$JOBS" --skip_tests --skip_submodule_sync \
  --compile_no_warning_as_error --build_shared_lib --use_migraphx \
  --migraphx_home "$PREFIX" \
  --cmake_extra_defines "CMAKE_PREFIX_PATH=$PREFIX" onnxruntime_BUILD_UNIT_TESTS=OFF \
  "${ROOT_ARGS[@]}"

PACKAGE="$BUILD/package/$NAME"
mkdir -p "$PACKAGE/lib"
for LIB in "libonnxruntime.so.$ORT_VERSION" libonnxruntime_providers_shared.so libonnxruntime_providers_migraphx.so; do
  cp -L "$BUILD/build/Release/$LIB" "$PACKAGE/lib/$LIB"
done
cp "$SOURCE/LICENSE" "$SOURCE/ThirdPartyNotices.txt" "$PACKAGE/"
for PATCH in "${PATCHES[@]}"; do
  cp "$PATCH" "$PACKAGE/$(basename "$PATCH")"
done
cat > "$PACKAGE/SOURCE.txt" <<EOF
ONNX Runtime $ORT_VERSION from https://github.com/microsoft/onnxruntime
Source revision: $ORT_COMMIT
Applied modifications: ort-migraphx-lifetime.patch, ort-migraphx-cache.patch, ort-migraphx-integer.patch (included in this archive)
ROCm build version: $ROCM_VERSION
EOF
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$EPOCH" \
  -C "$BUILD/package" -cf - "$NAME" | gzip -n > "$BUILD/$NAME.tgz"

# Verify core loading, compiled providers, and ELF dependencies on a CPU runner.
# ORT initializes the provider bridge during session registration; dlopen of the
# provider itself before that initialization is not supported.
mkdir "$BUILD/unpacked"
tar -xzf "$BUILD/$NAME.tgz" -C "$BUILD/unpacked"
cat > "$BUILD/check-provider.cc" <<'CPP'
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include "onnxruntime_c_api.h"

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  void* runtime = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
  if (!runtime) {
    std::fprintf(stderr, "%s\n", dlerror());
    return 1;
  }
  using GetApiBase = const OrtApiBase* (*)();
  auto get_api_base = reinterpret_cast<GetApiBase>(dlsym(runtime, "OrtGetApiBase"));
  if (!get_api_base) return 1;
  const OrtApi* api = get_api_base()->GetApi(ORT_API_VERSION);
  if (!api) return 1;
  char** providers = nullptr;
  int count = 0;
  OrtStatus* status = api->GetAvailableProviders(&providers, &count);
  if (status) {
    std::fprintf(stderr, "%s\n", api->GetErrorMessage(status));
    api->ReleaseStatus(status);
    return 1;
  }
  bool found = false;
  for (int i = 0; i < count; ++i) {
    std::puts(providers[i]);
    found |= std::strcmp(providers[i], "MIGraphXExecutionProvider") == 0;
  }
  status = api->ReleaseAvailableProviders(providers, count);
  if (status) {
    api->ReleaseStatus(status);
    return 1;
  }
  dlclose(runtime);
  return found ? 0 : 1;
}
CPP
LIB_DIR="$BUILD/unpacked/$NAME/lib"
"${CXX:-c++}" "$BUILD/check-provider.cc" \
  -I "$SOURCE/include/onnxruntime/core/session" \
  -ldl -o "$BUILD/check-provider"
LD_LIBRARY_PATH="$LIB_DIR:$PREFIX/lib:$PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$BUILD/check-provider" "$LIB_DIR/libonnxruntime.so.$ORT_VERSION"
LD_LIBRARY_PATH="$LIB_DIR:$PREFIX/lib:$PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ldd "$LIB_DIR/libonnxruntime_providers_migraphx.so" > "$BUILD/provider-dependencies.txt"
cat "$BUILD/provider-dependencies.txt"
if grep -q 'not found' "$BUILD/provider-dependencies.txt"; then
  echo "The packaged MIGraphX provider has unresolved host dependencies." >&2
  exit 1
fi

mv "$BUILD/$NAME.tgz" "$OUT/"
(cd "$OUT" && sha256sum "$NAME.tgz" > "$NAME.tgz.sha256")
echo "Created $OUT/$NAME.tgz and $OUT/$NAME.tgz.sha256"
