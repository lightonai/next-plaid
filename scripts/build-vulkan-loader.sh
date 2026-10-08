#!/usr/bin/env bash
# Build the Khronos Vulkan loader (libvulkan.so.1) that `colgrep --agent` downloads on
# Linux machines whose GPU driver ships a Vulkan ICD but whose distribution left out
# the loader (agent/src/runtime.rs, `vulkan_loader`).
#
# Built in a manylinux2014 container (glibc 2.17) so it runs on any distribution, with
# window-system support off: llama.cpp only needs compute.
#
# usage: scripts/build-vulkan-loader.sh <version, e.g. v1.4.361> <x64|arm64> <out-dir>
# Writes <out-dir>/vulkan-loader-<version>-linux-<arch>.tar.gz and prints its SHA-256.
# Run it on a host of the same architecture (CI: ubuntu-22.04 / ubuntu-22.04-arm).
set -euo pipefail

VERSION=${1:?version, e.g. v1.4.361}
ARCH=${2:?x64 or arm64}
OUT=$(mkdir -p "${3:?out dir}" && cd "$3" && pwd)
case "$ARCH" in
  x64) IMAGE=quay.io/pypa/manylinux2014_x86_64 ;;
  arm64) IMAGE=quay.io/pypa/manylinux2014_aarch64 ;;
  *) echo "unknown arch: $ARCH" >&2; exit 1 ;;
esac

NAME="vulkan-loader-${VERSION}-linux-${ARCH}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
git clone -q --depth 1 -b "$VERSION" https://github.com/KhronosGroup/Vulkan-Headers "$WORK/Vulkan-Headers"
git clone -q --depth 1 -b "$VERSION" https://github.com/KhronosGroup/Vulkan-Loader "$WORK/Vulkan-Loader"

docker run --rm -v "$WORK:/work" "$IMAGE" bash -euo pipefail -c '
  cmake -S /work/Vulkan-Headers -B /tmp/hb -DCMAKE_INSTALL_PREFIX=/tmp/headers >/dev/null
  cmake --install /tmp/hb >/dev/null
  cmake -S /work/Vulkan-Loader -B /tmp/lb -DCMAKE_BUILD_TYPE=Release \
    -DVULKAN_HEADERS_INSTALL_DIR=/tmp/headers \
    -DBUILD_WSI_XCB_SUPPORT=OFF -DBUILD_WSI_XLIB_SUPPORT=OFF \
    -DBUILD_WSI_WAYLAND_SUPPORT=OFF -DBUILD_WSI_DIRECTFB_SUPPORT=OFF \
    -DBUILD_TESTS=OFF -DENABLE_WERROR=OFF >/dev/null
  cmake --build /tmp/lb -j"$(nproc)" >/dev/null
  mkdir -p /work/out
  cp -L /tmp/lb/loader/libvulkan.so.1 /work/out/libvulkan.so.1
  strip /work/out/libvulkan.so.1
  echo "needs $(objdump -T /work/out/libvulkan.so.1 | grep -oE "GLIBC_[0-9.]+" | sort -V | tail -1)"
  chown -R '"$(id -u):$(id -g)"' /work/out
'

mkdir -p "$WORK/pkg/$NAME"
cp "$WORK/out/libvulkan.so.1" "$WORK/Vulkan-Loader/LICENSE.txt" "$WORK/pkg/$NAME/"
tar --owner=0 --group=0 --mtime=2025-01-01 --sort=name -C "$WORK/pkg" -czf "$OUT/$NAME.tar.gz" "$NAME"
(cd "$OUT" && sha256sum "$NAME.tar.gz")
