#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <run-prefix> <workdir>" >&2
  exit 2
fi

prefix="$1"
workdir="$2"
run_id="${prefix}-$(date -u +%Y%m%dT%H%M%SZ)"
outdir="/tmp/${run_id}"
mkdir -p "$outdir"

set +e
/usr/bin/time -f 'WALL_SEC=%e' -o "$outdir/time.txt" \
  timeout -k 2m 45m sudo -n docker run --rm --network host \
    -v /tmp/ltafanout:/tmp/ltafanout \
    -w "$workdir" \
    -e CCACHE_DIR=/tmp/ltafanout/LiteAttention/.act-cache/ccache \
    -e CCACHE_BASEDIR=/tmp/ltafanout \
    -e CCACHE_NOHASHDIR=true \
    -e CCACHE_COMPILERCHECK=content \
    -e CUDA_HOME=/usr/local/cuda \
    nvidia/cuda:12.8.1-devel-ubuntu22.04 bash -lc '
      set -euo pipefail
      export DEBIAN_FRONTEND=noninteractive
      export CUDA_HOME=/usr/local/cuda
      export GIT_DISCOVERY_ACROSS_FILESYSTEM=1
      export PATH=/usr/local/cuda/bin:$PATH

      apt-get update >/dev/null
      apt-get install -y git python3-pip ccache time >/dev/null
      python3 -m pip install --upgrade pip >/dev/null
      python3 -m pip install setuptools==75.8.0 typing-extensions==4.12.2 packaging ninja wheel >/dev/null
      python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 >/dev/null

      rm -rf build dist dist-FALSE *.egg-info

      git config --global --add safe.directory "$PWD" || true

      mkdir -p /tmp/liteattention-ccache-wrappers
      for tool in gcc g++ cc c++; do
        real="$(command -v "$tool")"
        cat >/tmp/liteattention-ccache-wrappers/$tool <<WRAP
#!/usr/bin/env bash
set -euo pipefail
exec ccache "$real" "\$@"
WRAP
        chmod +x /tmp/liteattention-ccache-wrappers/$tool
      done

      if [ ! -x /usr/local/cuda/bin/nvcc.real ]; then
        mv /usr/local/cuda/bin/nvcc /usr/local/cuda/bin/nvcc.real
        cat >/usr/local/cuda/bin/nvcc <<WRAP
#!/usr/bin/env bash
set -euo pipefail
exec ccache /usr/local/cuda/bin/nvcc.real "\$@"
WRAP
        chmod +x /usr/local/cuda/bin/nvcc
      fi

      export PATH=/tmp/liteattention-ccache-wrappers:$PATH
      export PYTORCH_NVCC=/usr/local/cuda/bin/nvcc

      ccache -z >/dev/null
      python3 setup.py bdist_wheel --dist-dir=dist-FALSE
      ccache -sv > ccache-stats-after.txt
      cp ccache-stats-after.txt /tmp/ltafanout/LiteAttention/ccache-stats-after-latest.txt
    ' >"$outdir/run.log" 2>&1
status=$?
set -e

printf 'exit_code=%s\nend_utc=%s\n' "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$outdir/meta.txt"
cp /tmp/ltafanout/LiteAttention/ccache-stats-after-latest.txt "$outdir/ccache-stats-after.txt" 2>/dev/null || true
echo "$outdir"
