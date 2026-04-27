# cuVS search throughput benchmark

End-to-end benchmark for cuVS GPU vector search with realistic concurrent online-serving throughput. Supports **CAGRA** (graph-based, with persistent kernel) and **IVF-PQ** (inverted-list + product quantization).

The notebook builds the index in Python, then runs **all** cuVS searches (every `c`, including `1`) through a small C++ master client (`cuvs_search_client`) so many concurrent in-flight searches share one CUDA context — the same pattern `cuvs-bench` uses for high QPS at `bs=1` online-serving workloads.

## Files

| Path | Purpose |
|---|---|
| `index_size_analysis.ipynb` | Main benchmark notebook |
| `index_size_params.yaml` | Algorithm parameters (CAGRA + IVF-PQ + FAISS) |
| `cuvs_search_client.cpp` | Master C++ client (CAGRA / IVF-PQ via `--algo`) |
| `build_cuvs_search_client.sh` | Compiles `cuvs_search_client` from `.cpp` |
| `docs/cagra_vs_hnsw_concurrency_qps_branded.html` | Standalone chart: CAGRA vs FAISS QPS vs concurrency (open locally in a browser) |

## Diagram (HTML)

Open [`docs/cagra_vs_hnsw_concurrency_qps_branded.html`](docs/cagra_vs_hnsw_concurrency_qps_branded.html) in a browser (clone the repo or use **Raw** → save as file → open). GitHub’s file viewer does not execute inline HTML; for in-repo rendering, enable [**GitHub Pages**](https://docs.github.com/en/pages) with the `/docs` folder and open the published URL to the same path.

To use your own branded export from another machine, copy it over this path:

`scp /Users/manass/Downloads/cagra_vs_hnsw_concurrency_qps_branded.html user@host:/path/to/cuvs-search-bench/docs/cagra_vs_hnsw_concurrency_qps_branded.html`

## Setup

1. **Install cuVS 26.04 + dependencies** (Python 3.11, conda/micromamba):

   ```bash
   micromamba create -y -n cuvs-search -c rapidsai -c conda-forge \
     python=3.11 cuvs=26.04 pylibraft=26.04 rmm=26.04 cupy faiss-cpu \
     dlpack cuda-version=12 \
     numpy pandas matplotlib tqdm pyyaml ipykernel
   ```

2. **Build the C++ client**:

   ```bash
   CONDA_PREFIX=$HOME/micromamba/envs/cuvs-search bash build_cuvs_search_client.sh
   ```

3. **Point the notebook at your dataset** (`.fbin` format from `cuvs-bench`):

   ```bash
   export CUVS_DATASET_PATH=/path/to/dataset_dir   # contains base.fbin + query.fbin
   ```

4. Open `index_size_analysis.ipynb` and run all cells.

## Configuration

Edit cell 2 of the notebook:

```python
INDEX_TYPE = "cagra"               # or "ivf_pq"
SIZES = [1_000_000]                # dataset sizes to sweep
BENCHMARK_DIMS = [1024]            # truncate to these dims
SEARCH_CONCURRENCY = [1, 32, 64]   # in-flight searches (c)
SEARCH_BATCH_SIZES = [1]           # queries per client search call (bs)
```

All cuVS search runs (including `c=1`) use the **`cuvs_search_client`** binary; the notebook only builds the index in Python.

Edit `index_size_params.yaml` for algorithm tuning:

```yaml
cagra:
  graph_degree: 32
  intermediate_graph_degree: 64
  itopk_size: 64
  persistent: true   # required by C++ client for high QPS

ivf_pq:
  n_lists: 1024
  n_probes: 32
```

## Why this is fast

cuVS Python at `bs=1` is limited to **~600 QPS per process** because each call is host-blocking — only one query in flight at a time. The C++ client launches `c` concurrent threads (default 32-64), each dropping 1-query searches into a single persistent CAGRA kernel queue. The kernel never restarts between queries, so the GPU is always doing useful work. Result: **~30K QPS at 90% recall on an L4** — matching `cuvs-bench`.

This works around a known cuVS Python thread-safety bug (`device_ndarray.empty` segfaults with per-thread `Resources(stream=...)`); the C++ path uses cuVS's C API directly which doesn't have the issue.

## Outputs

- `index_size_analysis_results.csv` — one row per (size, dim, c, bs) configuration
- 3-panel plot: build time, search QPS, search latency
