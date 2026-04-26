#!/usr/bin/env bash
# Build cuvs_search_client (one-shot). Run inside the cuvs-faiss-26.04 env, e.g.:
#   CONDA_PREFIX=/home/ubuntu/micromamba/envs/cuvs-faiss-26.04 bash build_cuvs_search_client.sh
set -euo pipefail
PFX="${CONDA_PREFIX:-/home/ubuntu/micromamba/envs/cuvs-faiss-26.04}"
SRC="$(dirname "$(readlink -f "$0")")/cuvs_search_client.cpp"
OUT="$(dirname "$(readlink -f "$0")")/cuvs_search_client"
g++ -O3 -std=c++17 -pthread \
  -I"$PFX/include" \
  -I"$PFX/targets/x86_64-linux/include" \
  -L"$PFX/lib" \
  -Wl,-rpath,"$PFX/lib" \
  "$SRC" -o "$OUT" \
  -lcuvs_c -lcudart
echo "Built: $OUT"
