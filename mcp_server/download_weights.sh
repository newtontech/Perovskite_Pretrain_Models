#!/bin/bash
# Download the fine-tuned UniMol perovskite weights (fixed_hyper_param.tgz, ~843 MB).
# Primary source: GitHub Release asset of this repository (tag: unimol-perovskite-v1).
# Fallback: the original Bohrium OSS upload (project 657926).
# Usage: ./download_weights.sh [dest_dir]   (default dest_dir: ./weights)
set -euo pipefail

SHA256="65771d5f4e373415b1420636400fd252c48d7c5be4e5a5c7c9235ff8da4ab647"
TAG="unimol-perovskite-v1"
REPO="newtontech/Perovskite_Pretrain_Models"
FILE="fixed_hyper_param.tgz"
OSS_URL="https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/654582/657926/store/upload/71121830-4116-4d0c-847f-4b7cd64c4c85/fixed_hyper_param.tgz"
DEST_DIR="${1:-./weights}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if command -v gh >/dev/null 2>&1; then
  echo ">> Trying GitHub Release (${TAG})..."
  gh release download "$TAG" -R "$REPO" -p "$FILE" -O "$TMP/$FILE" || echo ">> gh download failed, falling back to OSS"
fi

if [ ! -s "$TMP/$FILE" ]; then
  echo ">> Downloading from Bohrium OSS..."
  curl -fL --retry 3 -C - -o "$TMP/$FILE" "$OSS_URL"
fi

echo ">> Verifying sha256..."
echo "$SHA256  $TMP/$FILE" | sha256sum -c -

mkdir -p "$DEST_DIR"
tar -xzf "$TMP/$FILE" -C "$DEST_DIR"
echo ">> Done. Weights extracted to: $DEST_DIR"
echo ">> Then set: export PEROVSKITE_WEIGHT_DIR=\"$(cd "$DEST_DIR" && pwd)/fixed_hyper_param\""
