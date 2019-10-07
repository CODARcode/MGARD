#include <random>
#include <benchmark/benchmark.h>

static void BM_MGARD(benchmark::State& state) {
  std::mt19937_64 gen(124124);
  std::normal_distribution<double> dis{0,1};
  for (auto _ : state) {
    // This code gets timed
    benchmark::DoNotOptimize(dis(gen));
  }
}

BENCHMARK(BM_MGARD);

BENCHMARK_MAIN();
