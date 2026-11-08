#!/usr/bin/env bash
# Build gsplat 1.5.3 from source for the optional fast splat back end, on Linux, into the active Python environment.
#
# Why a source build: gsplat publishes no wheel for Python 3.14 or for torch 2.14, its PyPI source lacks the glm
# submodule, and its setup.py asks nvcc for C++17 while torch 2.14's headers need C++20. This script clones the
# v1.5.3 tag with its submodules, changes that one flag to C++20, and builds with a CUDA 13.2 toolchain taken from
# NVIDIA's pip wheels into a throw-away folder (nothing is installed system-wide). All parts of the toolchain must be
# the same release: an nvvm newer than ptxas fails with "Unsupported .version".
#
# DexVantage works without gsplat: the plain-torch rasteriser (core/splat_raster.py) is the default CUDA path and the
# fitting path; gsplat, when it imports and has CUDA, is used for pinhole renders ('auto' back end).
#
# Usage:   scripts/build_gsplat.sh [python]        (default: .venv/bin/python; needs the 'splat' extra installed)
# Env:     ARCH (default 12.0, the compute capability to build for), JOBS (default 8), WORK (default out/gsplat-build)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${1:-.venv/bin/python}
ARCH=${ARCH:-12.0}
JOBS=${JOBS:-8}
WORK=${WORK:-out/gsplat-build}
CUDA_VER=13.2.86
mkdir -p "$WORK/tmp"
WORK=$(cd "$WORK" && pwd)
export TMPDIR="$WORK/tmp" PIP_CACHE_DIR="$WORK/pipcache"

"$PY" -c 'import torch, sys; sys.exit(0 if torch.version.cuda else "torch has no CUDA: install the splat extra first")'

if [ ! -d "$WORK/src" ]; then
  git clone -q --depth 1 --recursive --branch v1.5.3 https://github.com/nerfstudio-project/gsplat "$WORK/src"
fi
sed -i 's/"-std=c++17"/"-std=c++20"/' "$WORK/src/setup.py"

if [ ! -x "$WORK/cuda/nvidia/cu13/bin/nvcc" ]; then
  "$PY" -m pip install -q --target "$WORK/cuda" "nvidia-cuda-nvcc==$CUDA_VER" "nvidia-cuda-cccl==$CUDA_VER" \
    "nvidia-cuda-runtime==$CUDA_VER" "nvidia-cuda-crt==$CUDA_VER" "nvidia-nvvm==$CUDA_VER"
fi
C="$WORK/cuda/nvidia/cu13"
ln -sfn lib "$C/lib64"
ln -sf libcudart.so.13 "$C/lib/libcudart.so"

rm -rf "$WORK/src/build" "$WORK/wheel"
CUDA_HOME="$C" PATH="$C/bin:$PATH" TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS="$JOBS" \
  "$PY" -m pip wheel -q --no-build-isolation --no-deps -w "$WORK/wheel" "$WORK/src"
"$PY" -m pip install -q "$WORK"/wheel/gsplat-1.5.3-*.whl
"$PY" -c 'from dexvantage.core import splat_raster as SR; import sys; sys.exit(0 if SR.gsplat_ok() else "gsplat built but does not run here")'
echo "gsplat 1.5.3 installed; the 'auto' splat back end now uses it for pinhole renders."
