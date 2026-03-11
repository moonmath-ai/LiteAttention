#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <host> <target-dir>" >&2
  exit 2
fi

host="$1"
target_dir="$2"

source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ssh -o BatchMode=yes "$host" "mkdir -p '$target_dir'"

rsync -az --delete \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='.act-cache/' \
  --exclude='build/' \
  --exclude='dist/' \
  --exclude='dist-*' \
  --exclude='*.so' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='.pytest_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='lastcheck.txt' \
  --exclude='wiggum.mg' \
  --exclude='"' \
  --exclude='*.swp' \
  "$source_dir/" "$host:$target_dir/"
