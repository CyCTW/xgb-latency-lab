// Direct single-row calls through the same ABI; no Python in timed regions.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using Predict = void (*)(const float*, float*);
using PredictTL = void (*)(const float*, int, float*);
static volatile double sink;
struct Engine {
  std::string name;
  void* handle;
  Predict predict;
  PredictTL predict_tl;
  bool packed;
  std::vector<double> calls, blocks;
  double checksum = 0;
};
double ns(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::nano>(b-a).count();
}
double quantile(std::vector<double> a, double q) {
  std::sort(a.begin(), a.end());
  return a[static_cast<size_t>(q * (a.size()-1))];
}

template <class Call>
void measure(Engine& e, const float* data, size_t nf,
             const std::vector<size_t>& indices, Call call) {
  const size_t samples = indices.size();
  float out;
  double checksum = 0;
  for (size_t i = 0; i < samples; ++i) {
    call(data + indices[i]*nf, &out); checksum += out;
  }
  sink = checksum;
  for (size_t i = 0; i < samples; ++i) {
    const float* row = data + indices[i]*nf;
    auto start = Clock::now();
    call(row, &out);
    auto stop = Clock::now();
    e.calls.push_back(ns(start, stop)); checksum += out;
  }
  auto start = Clock::now();
  for (size_t i = 0; i < samples; ++i) {
    call(data + indices[i]*nf, &out); checksum += out;
  }
  auto stop = Clock::now();
  e.blocks.push_back(ns(start,stop)/samples);
  e.checksum += checksum;
  sink = checksum;
}

int main(int argc, char** argv) try {
  if (argc < 11 || (argc-7) % 4) {
    throw std::runtime_error("usage: native data.f32 rows features samples rounds seed [name lib symbol packed]...");
  }
  const size_t nr = std::stoull(argv[2]), nf = std::stoull(argv[3]);
  const size_t samples = std::stoull(argv[4]), rounds = std::stoull(argv[5]);
  if (!nr || !nf || !samples || !rounds) throw std::runtime_error("counts must be positive");
  std::mt19937 rng(std::stoul(argv[6]));
  std::vector<float> rows(nr*nf), packed(nr*nf);
  std::ifstream file(argv[1], std::ios::binary);
  if (!file.read(reinterpret_cast<char*>(rows.data()), rows.size()*sizeof(float)))
    throw std::runtime_error("cannot read input matrix");
  packed = rows;
  for (size_t i = 0; i < rows.size(); ++i) if (std::isnan(rows[i])) {
    int32_t missing = -1;
    std::memcpy(&packed[i], &missing, sizeof(float));
  }
  std::vector<Engine> engines;
  for (int i = 7; i < argc; i += 4) {
    void* h = dlopen(argv[i+1], RTLD_NOW | RTLD_LOCAL);
    if (!h) throw std::runtime_error(dlerror());
    auto symbol = dlsym(h, argv[i+2]);
    if (!symbol) throw std::runtime_error(dlerror());
    engines.push_back({argv[i], h, reinterpret_cast<Predict>(symbol),
                       reinterpret_cast<PredictTL>(symbol), std::stoi(argv[i+3]) != 0, {}, {}});
  }
  std::vector<size_t> indices(samples), order(engines.size());
  std::iota(order.begin(), order.end(), 0);
  for (auto& e: engines) e.calls.reserve(samples*rounds);
  std::vector<double> timer;
  for (int i = 0; i < 10000; ++i) {
    auto a = Clock::now(); auto b = Clock::now(); timer.push_back(ns(a,b));
  }
  // Rotate engine order and use a fresh common row sequence each round.
  for (size_t round = 0; round < rounds; ++round) {
    for (auto& i: indices) i = rng() % nr;
    std::shuffle(order.begin(), order.end(), rng);
    for (auto ei: order) {
      auto& e = engines[ei];
      const float* data = e.packed ? packed.data() : rows.data();
      if (e.packed) {
        // Exactly one indirect call to TL2cgen's generated predict, no adapter.
        auto predict = e.predict_tl;
        measure(e,data,nf,indices,[predict](const float* row,float* out) {
          *out = 0.0f; predict(row,1,out);
        });
      } else {
        auto predict = e.predict;
        measure(e,data,nf,indices,[predict](const float* row,float* out) { predict(row,out); });
      }
    }
  }
  std::cout << std::setprecision(10) << "{\"clock_pair_p50_ns\":" << quantile(timer,.5)
            << ",\"samples_per_engine\":" << samples*rounds << ",\"engines\":[";
  for (size_t i = 0; i < engines.size(); ++i) {
    auto& e = engines[i];
    if (i) std::cout << ',';
    std::cout << "{\"name\":\"" << e.name << "\",\"p50_ns\":" << quantile(e.calls,.50)
              << ",\"p95_ns\":" << quantile(e.calls,.95)
              << ",\"p99_ns\":" << quantile(e.calls,.99)
              << ",\"block_median_ns_per_row\":" << quantile(e.blocks,.5)
              << ",\"block_ns_per_row\":[";
    for (size_t j = 0; j < e.blocks.size(); ++j) {
      if (j) std::cout << ',';
      std::cout << e.blocks[j];
    }
    std::cout << "],\"checksum\":" << e.checksum << '}';
    dlclose(e.handle);
  }
  std::cout << "]}\n";
} catch (const std::exception& e) {
  std::cerr << e.what() << '\n';
  return 1;
}
