#include <random>
#include <benchmark/benchmark.h>

#include "mgard_api.h"

static void BM_MGARD(benchmark::State& state) {
  std::mt19937_64 gen(124124);
  std::normal_distribution<double> dis{0,1};

  std::vector<double> v(state.range(0));
  for (auto & x : v) {
    x = dis(gen);
  }

  int out_size;
  double tol = 1e-3;
  for (auto _ : state) {
    // This code gets timed
    benchmark::DoNotOptimize(mgard_compress(/* double = 0 */ 0, v.data(), out_size,  4,  v.size()/4,  1, tol));
  }

  state.SetComplexityN(state.range(0));
  state.SetBytesProcessed(int64_t(state.range(0))*int64_t(state.iterations())*int64_t(sizeof(double)));
}

BENCHMARK(BM_MGARD)->RangeMultiplier(2)->Range(1<<4, 1<<22)->Complexity();

BENCHMARK_MAIN();
