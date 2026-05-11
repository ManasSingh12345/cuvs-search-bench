// Master cuVS search client (CAGRA or IVF-PQ) using the cuVS C API.
// Concurrent in-flight searches via std::async for high-throughput benchmarking.
//
// CLI:
//   --algo {cagra,ivf_pq}     index/search algorithm (default: cagra)
//   --index-file <path>       saved index (Python: cagra.save / ivf_pq.save)
//   --queries-file <path>     raw float32 binary, shape (Q, D)
//   --out-neighbors <path>    output: raw binary; uint32 for cagra, int64 for ivf_pq
//   --out-meta <path>         output: JSON search_wall_s (avg wall per iter), batch_latency_p95_s
//   --shape Q,D               int64 query rows and dims
//   --k <int>                 neighbors per query
//   --c <int>                 max concurrent in-flight searches
//   --batch-size <int>        queries per cuvs*Search call
//   --iters <int>             timed iterations (averaged)
//   --itopk-size <int>        CAGRA itopk
//   --search-width <int>      CAGRA search_width (default 1; try 2 for recall)
//   --persistent {0,1}        CAGRA persistent kernel (CAGRA only; ignored for ivf_pq)
//   --n-probes <int>          IVF-PQ n_probes

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/common.h>
#include <cuvs/neighbors/ivf_pq.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#define CUDA_CHECK(x)                                                                              \
  do {                                                                                             \
    cudaError_t e = (x);                                                                           \
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); } \
  } while (0)

#define CUVS_CHECK(x)                                                          \
  do {                                                                         \
    cuvsError_t e = (x);                                                       \
    if (e != CUVS_SUCCESS) { std::fprintf(stderr, "cuVS: %s\n", cuvsGetLastErrorText()); std::exit(1); } \
  } while (0)

static DLManagedTensor make_dl(void* ptr, int64_t* shape, int ndim, DLDataType dtype) {
  DLManagedTensor t{};
  t.dl_tensor.data = ptr;
  t.dl_tensor.device.device_type = kDLCUDA;
  t.dl_tensor.device.device_id = 0;
  t.dl_tensor.ndim = ndim;
  t.dl_tensor.dtype = dtype;
  t.dl_tensor.shape = shape;
  return t;
}

/** Linear interpolation percentile (numpy.percentile(..., p, method="linear")). */
static double percentile_linear(std::vector<double> v, double p) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  if (n == 1) return v[0];
  const double pos = (p / 100.0) * static_cast<double>(n - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  if (lo >= hi) return v[lo];
  return v[lo] + (pos - static_cast<double>(lo)) * (v[hi] - v[lo]);
}

int main(int argc, char** argv) {
  std::string algo = "cagra", index_file, queries_file, out_neighbors, out_meta;
  int64_t Q = 0, D = 0;
  int k = 10, c = 1, batch_size = 1, iters = 5, persistent = 1, itopk = 64, n_probes = 32;
  int search_width = 1;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--algo") algo = next();
    else if (a == "--index-file") index_file = next();
    else if (a == "--queries-file") queries_file = next();
    else if (a == "--out-neighbors") out_neighbors = next();
    else if (a == "--out-meta") out_meta = next();
    else if (a == "--shape") std::sscanf(argv[++i], "%ld,%ld", &Q, &D);
    else if (a == "--itopk-size") itopk = std::atoi(argv[++i]);
    else if (a == "--search-width") search_width = std::atoi(argv[++i]);
    else if (a == "--k") k = std::atoi(argv[++i]);
    else if (a == "--c") c = std::atoi(argv[++i]);
    else if (a == "--batch-size") batch_size = std::atoi(argv[++i]);
    else if (a == "--iters") iters = std::atoi(argv[++i]);
    else if (a == "--persistent") persistent = std::atoi(argv[++i]);
    else if (a == "--n-probes") n_probes = std::atoi(argv[++i]);
  }

  cuvsResources_t res;
  CUVS_CHECK(cuvsResourcesCreate(&res));

  // Load queries to GPU.
  std::vector<float> q_host(static_cast<size_t>(Q) * D);
  std::ifstream(queries_file, std::ios::binary)
      .read(reinterpret_cast<char*>(q_host.data()),
            static_cast<std::streamsize>(q_host.size() * sizeof(float)));
  float* q_dev = nullptr;
  CUDA_CHECK(cudaMalloc(&q_dev, q_host.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(q_dev, q_host.data(), q_host.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Output buffers (different neighbor dtype per algo).
  const bool is_cagra = (algo == "cagra");
  const size_t neigh_elt = is_cagra ? sizeof(uint32_t) : sizeof(int64_t);
  void* n_dev = nullptr;
  float* d_dev = nullptr;
  CUDA_CHECK(cudaMalloc(&n_dev, static_cast<size_t>(Q) * k * neigh_elt));
  CUDA_CHECK(cudaMalloc(&d_dev, static_cast<size_t>(Q) * k * sizeof(float)));

  const DLDataType dt_f32 = {kDLFloat, 32, 1};
  const DLDataType dt_neigh = is_cagra ? DLDataType{kDLUInt, 32, 1} : DLDataType{kDLInt, 64, 1};

  std::function<void(int64_t, int64_t)> search_slice;

  // Algo-specific setup.
  cuvsCagraIndex_t cagra_index = nullptr;
  cuvsCagraSearchParams_t cagra_sp = nullptr;
  cuvsIvfPqIndex_t ivf_index = nullptr;
  cuvsIvfPqSearchParams_t ivf_sp = nullptr;

  if (is_cagra) {
    CUVS_CHECK(cuvsCagraIndexCreate(&cagra_index));
    CUVS_CHECK(cuvsCagraDeserialize(res, index_file.c_str(), cagra_index));
    CUVS_CHECK(cuvsCagraSearchParamsCreate(&cagra_sp));
    cagra_sp->itopk_size = itopk;
    cagra_sp->search_width = static_cast<size_t>(search_width < 1 ? 1 : search_width);
    cagra_sp->algo = SINGLE_CTA;
    cagra_sp->persistent = persistent != 0;
    cagra_sp->persistent_device_usage = 0.95f;

    search_slice = [&](int64_t offset, int64_t bs) {
      int64_t qs[2] = {bs, D};
      int64_t os[2] = {bs, (int64_t)k};
      DLManagedTensor qt = make_dl(q_dev + offset * D, qs, 2, dt_f32);
      DLManagedTensor nt = make_dl(static_cast<uint32_t*>(n_dev) + offset * k, os, 2, dt_neigh);
      DLManagedTensor dt = make_dl(d_dev + offset * k, os, 2, dt_f32);
      cuvsFilter no_filter{0, NO_FILTER};
      CUVS_CHECK(cuvsCagraSearch(res, cagra_sp, cagra_index, &qt, &nt, &dt, no_filter));
    };
  } else {
    CUVS_CHECK(cuvsIvfPqIndexCreate(&ivf_index));
    CUVS_CHECK(cuvsIvfPqDeserialize(res, index_file.c_str(), ivf_index));
    CUVS_CHECK(cuvsIvfPqSearchParamsCreate(&ivf_sp));
    ivf_sp->n_probes = n_probes;

    search_slice = [&](int64_t offset, int64_t bs) {
      int64_t qs[2] = {bs, D};
      int64_t os[2] = {bs, (int64_t)k};
      DLManagedTensor qt = make_dl(q_dev + offset * D, qs, 2, dt_f32);
      DLManagedTensor nt = make_dl(static_cast<int64_t*>(n_dev) + offset * k, os, 2, dt_neigh);
      DLManagedTensor dt = make_dl(d_dev + offset * k, os, 2, dt_f32);
      CUVS_CHECK(cuvsIvfPqSearch(res, ivf_sp, ivf_index, &qt, &nt, &dt));
    };
  }

  // Warmup (excluded from timing).
  search_slice(0, Q);
  CUDA_CHECK(cudaDeviceSynchronize());

  const int64_t num_jobs = (Q + batch_size - 1) / batch_size;
  std::vector<std::future<void>> futures(c);
  std::vector<double> batch_latencies;
  batch_latencies.reserve(static_cast<size_t>(num_jobs * std::max(1, iters)));
  std::mutex lat_mu;
  auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < iters; it++) {
    for (int64_t j = 0; j < num_jobs + c; j++) {
      const int slot = j % c;
      if (j >= c) futures[slot].wait();
      if (j < num_jobs) {
        const int64_t offset = j * batch_size;
        const int64_t bs = std::min<int64_t>(batch_size, Q - offset);
        futures[slot] = std::async(std::launch::async, [&, offset, bs]() {
          const auto t_batch0 = std::chrono::steady_clock::now();
          search_slice(offset, bs);
          const auto t_batch1 = std::chrono::steady_clock::now();
          const double dt = std::chrono::duration<double>(t_batch1 - t_batch0).count();
          std::lock_guard<std::mutex> lock(lat_mu);
          batch_latencies.push_back(dt);
        });
      }
    }
    if (!is_cagra || !persistent) CUDA_CHECK(cudaDeviceSynchronize());
  }
  auto t1 = std::chrono::steady_clock::now();
  double wall = std::chrono::duration<double>(t1 - t0).count() / std::max(1, iters);
  const double p95_batch_s = percentile_linear(batch_latencies, 95.0);

  // Copy neighbors back.
  std::vector<char> n_host(static_cast<size_t>(Q) * k * neigh_elt);
  CUDA_CHECK(cudaMemcpy(n_host.data(), n_dev, n_host.size(), cudaMemcpyDeviceToHost));
  {
    std::ofstream nf(out_neighbors, std::ios::binary);
    nf.write(n_host.data(), static_cast<std::streamsize>(n_host.size()));
  }
  {
    std::ofstream mf(out_meta);
    mf << "{\"search_wall_s\": " << wall << ", \"batch_latency_p95_s\": " << p95_batch_s << "}";
  }
  std::fflush(nullptr);
  std::quick_exit(0);
}
