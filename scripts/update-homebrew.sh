#!/usr/bin/env bash
set -euo pipefail

# Update the Homebrew formula for colgrep with a new version.
#
# Usage:
#   ./scripts/update-homebrew.sh                  # uses version from Cargo.toml
#   ./scripts/update-homebrew.sh 1.0.7            # explicit version
#   HOMEBREW_TAP_DIR=/path/to/tap ./scripts/update-homebrew.sh

REPO="lightonai/next-plaid"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HOMEBREW_TAP_DIR="${HOMEBREW_TAP_DIR:-$ROOT_DIR/../homebrew-colgrep}"
FORMULA="$HOMEBREW_TAP_DIR/Formula/colgrep.rb"

# Get version from argument or Cargo.toml
if [ $# -ge 1 ]; then
    VERSION="$1"
else
    VERSION=$(grep -A5 '^\[workspace\.package\]' "$ROOT_DIR/Cargo.toml" | grep '^version' | head -1 | sed 's/.*"\(.*\)".*/\1/')
fi

echo "Updating Homebrew formula to v${VERSION}..."

# Check that the formula exists
if [ ! -f "$FORMULA" ]; then
    echo "Error: Formula not found at $FORMULA"
    echo "Set HOMEBREW_TAP_DIR to the path of the homebrew-colgrep repository."
    exit 1
fi

# Compute sha256 of source archive
echo "Fetching sha256 of source archive..."
SHA256=$(curl -sL "https://github.com/${REPO}/archive/refs/tags/${VERSION}.tar.gz" | shasum -a 256 | awk '{print $1}')
echo "  sha256: $SHA256"

# Generate the formula
cat > "$FORMULA" << RUBY
class Colgrep < Formula
  desc "Semantic code search powered by late-interaction models"
  homepage "https://github.com/lightonai/next-plaid"
  url "https://github.com/${REPO}/archive/refs/tags/${VERSION}.tar.gz"
  sha256 "${SHA256}"
  license "Apache-2.0"
  head "https://github.com/${REPO}.git", branch: "main"

  livecheck do
    url :stable
    strategy :github_latest
  end

  depends_on "rust" => :build

  def install
    features = []
    if OS.mac?
      features << "accelerate"
      features << "coreml" if Hardware::CPU.arm?
    end

    args = std_cargo_args(path: "colgrep")
    if features.any?
      system "cargo", "install", "--features", features.join(","), *args
    else
      system "cargo", "install", *args
    end
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/colgrep --version")
  end
end
RUBY

echo "✓ Updated $FORMULA to v${VERSION}"
