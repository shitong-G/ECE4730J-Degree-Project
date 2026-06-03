#!/usr/bin/env bash
# Clone upstream RT-DETR into third_party/ (not vendored in repo root).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${ROOT}/third_party/RT-DETR"
REPO_URL="https://github.com/lyuwenyu/RT-DETR.git"

mkdir -p "${ROOT}/third_party"

if [ -d "$TARGET/.git" ]; then
  echo "RT-DETR already cloned at $TARGET"
  exit 0
fi

echo "Cloning RT-DETR to $TARGET ..."
git clone "$REPO_URL" "$TARGET"
echo "Done. See third_party/README.md for export instructions."
