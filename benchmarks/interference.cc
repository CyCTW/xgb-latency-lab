// Same-thread feature generation and controlled code/data interference.
#include <algorithm>
#include <atomic>
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
extern "C" uint64_t feature_code_work(uint64_t, size_t);
static volatile uint64_t integer_sink;
static volatile double prediction_sink;
static void fence() { std::atomic_signal_fence(std::memory_order_seq_cst); }
static double elapsed(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::nano>(b-a).count();
}
static double quantile(std::vector<double> a, double q) {
  std::sort(a.begin(), a.end()); return a[size_t(q*(a.size()-1))];
}
static void array_json(const std::vector<double>& values) {
  std::cout << '[';
  for (size_t i=0; i<values.size(); ++i) { if (i) std::cout << ','; std::cout << values[i]; }
  std::cout << ']';
}
static std::vector<float> read_rows(const char* path, size_t count) {
  std::vector<float> values(count); std::ifstream f(path, std::ios::binary);
  if (!f.read(reinterpret_cast<char*>(values.data()), count*sizeof(float)))
    throw std::runtime_error("cannot read float32 input");
  return values;
}
static bool power_two(size_t n) { return n && !(n&(n-1)); }
struct Work {
  std::vector<float> history;
  std::vector<uint64_t> eviction;
  size_t code_blocks;
  Work(size_t history_bytes, size_t eviction_bytes, size_t blocks) : code_blocks(blocks) {
    if ((history_bytes && (!power_two(history_bytes) || history_bytes<64)) ||
        (eviction_bytes && (!power_two(eviction_bytes) || eviction_bytes<64)) || blocks>256)
      throw std::runtime_error("work sizes must be powers of two >=64 bytes; code blocks <=256");
    history.resize(history_bytes/4); eviction.resize(eviction_bytes/8);
    uint32_t state=0x12345678u;
    for (auto& value: history) { state=state*1664525u+1013904223u; value=float(state&65535u)*(1.0f/65536.0f); }
    for (auto& value: eviction) { state=state*1664525u+1013904223u; value=state; }
  }
  __attribute__((noinline)) void disturb(size_t row) const {
    // One volatile read per 64 bytes, visiting every line once in permuted order.
    // This is software pressure, not a guarantee of hardware cache invalidation.
    if (!eviction.empty()) {
      const volatile uint64_t* data=eviction.data();
      size_t lines=eviction.size()/8; uint64_t sum=0;
      for (size_t i=0; i<lines; ++i) sum += data[((i*8191+row)&(lines-1))*8];
      integer_sink=sum;
    }
    if (code_blocks) integer_sink=feature_code_work(uint64_t(row)+17, code_blocks);
  }
  __attribute__((noinline)) void features(const float* raw, size_t row, size_t nf, float* out, bool packed) const {
    for (size_t j=0; j<nf; ++j) {
      float value=raw[j];
      if (!history.empty()) {
        uint32_t key=uint32_t(row)*0x9e3779b9u+uint32_t(j)*0x85ebca6bu;
        float sum=0;
        for (unsigned k=0; k<16; ++k) {
          key=key*1664525u+1013904223u;
          sum += history[(key>>4)&(history.size()-1)];
        }
        value += (sum*(1.0f/16.0f)-0.5f)*(1.0f/64.0f);
      }
      if (packed && std::isnan(value)) {
        int32_t missing=-1; std::memcpy(out+j,&missing,4);
      } else out[j]=value;
    }
  }
};
struct Engine {
  std::string name; void* handle; Predict dense; PredictTL prepared; bool packed;
  std::vector<double> model_calls, prep_calls, total_calls, model_means, end_to_end_blocks;
  double checksum=0, max_error=0;
  void call(const float* row, float* output) const {
    if (packed) { *output=0; prepared(row,1,output); } else dense(row,output);
  }
};
int main(int argc, char** argv) try {
  if (argc==8 && std::string(argv[1])=="generate") {
    size_t nr=std::stoull(argv[4]), nf=std::stoull(argv[5]);
    if (!nr || !nf) throw std::runtime_error("counts must be positive");
    auto raw=read_rows(argv[2],nr*nf); std::vector<float> output(nr*nf);
    Work work(std::stoull(argv[6]),0,0);
    bool packed=std::stoi(argv[7])!=0;
    for (size_t i=0; i<nr; ++i) work.features(raw.data()+i*nf,i,nf,output.data()+i*nf,packed);
    std::ofstream f(argv[3],std::ios::binary); f.write(reinterpret_cast<const char*>(output.data()),output.size()*4);
    if (!f) throw std::runtime_error("cannot write generated features");
    return 0;
  }
  // bench raw expected rows features samples rounds seed history_bytes eviction_bytes code_blocks [name lib symbol packed]...
  bool paired = argc>1 && std::string(argv[1])=="bench_paired";
  if (argc<16 || (!paired && std::string(argv[1])!="bench") || (argc-12)%4)
    throw std::runtime_error("usage: interference bench raw expected rows features samples rounds seed history_bytes eviction_bytes code_blocks [name lib symbol packed]...");
  size_t nr=std::stoull(argv[4]), nf=std::stoull(argv[5]), samples=std::stoull(argv[6]), rounds=std::stoull(argv[7]);
  if (!nr || !nf || !samples || !rounds) throw std::runtime_error("counts must be positive");
  if (paired && rounds%2) throw std::runtime_error("paired rounds must be even");
  size_t history_bytes=std::stoull(argv[9]), eviction_bytes=std::stoull(argv[10]), blocks=std::stoull(argv[11]);
  auto raw=read_rows(argv[2],nr*nf), expected=read_rows(argv[3],nr);
  Work work(history_bytes,eviction_bytes,blocks); std::vector<float> features(nf);
  std::vector<Engine> engines;
  for (int i=12; i<argc; i+=4) {
    void* h=dlopen(argv[i+1],RTLD_NOW|RTLD_LOCAL); if (!h) throw std::runtime_error(dlerror());
    void* symbol=dlsym(h,argv[i+2]); if (!symbol) throw std::runtime_error(dlerror());
    engines.push_back({argv[i],h,reinterpret_cast<Predict>(symbol),reinterpret_cast<PredictTL>(symbol),std::stoi(argv[i+3])!=0});
  }
  // Validate actual generated inputs outside timed regions. Every engine sees
  // identical features; prepared TL packs missing values in the producer.
  for (auto& engine:engines) {
    for (size_t i=0;i<nr;++i) {
      work.features(raw.data()+i*nf,i,nf,features.data(),engine.packed); float result;
      engine.call(features.data(),&result);
      double error=std::abs(double(result)-expected[i]);
      if (!std::isfinite(result) || error>2e-6+2e-5*std::abs(expected[i]))
        throw std::runtime_error("prediction validation failed: "+engine.name);
      engine.max_error=std::max(engine.max_error,error);
    }
  }
  std::mt19937 rng(std::stoul(argv[8]));
  std::vector<size_t> indices(samples), order(engines.size()); std::iota(order.begin(),order.end(),0);
  std::vector<std::vector<size_t>> round_orders;
  std::vector<uint64_t> row_sequence_hashes;
  std::vector<double> clock_pairs;
  for (int i=0;i<10000;++i) { auto a=Clock::now(); fence(); auto b=Clock::now(); clock_pairs.push_back(elapsed(a,b)); }
  for (size_t round=0;round<rounds;++round) {
    if (!paired || round%2==0) {
      for (auto& row:indices) row=rng()%nr;
      std::shuffle(order.begin(),order.end(),rng);
    } else {
      // Same rows, opposite engine order: balance position within each pair.
      std::reverse(order.begin(),order.end());
    }
    round_orders.push_back(order);
    uint64_t row_hash=14695981039346656037ULL;
    for (auto row:indices) { row_hash^=row; row_hash*=1099511628211ULL; }
    row_sequence_hashes.push_back(row_hash);
    for (auto index:order) {
      auto& e=engines[index]; double model_sum=0, checksum=0;
      // Warm the workload, but do not make an unconditioned prediction call.
      for (size_t k=0;k<8;++k) { size_t row=indices[k%samples]; work.disturb(row); work.features(raw.data()+row*nf,row,nf,features.data(),e.packed); }
      for (size_t row:indices) {
        fence(); auto begin=Clock::now();
        work.disturb(row); work.features(raw.data()+row*nf,row,nf,features.data(),e.packed);
        fence(); auto ready=Clock::now(); fence();
        float result; e.call(features.data(),&result);
        fence(); auto end=Clock::now();
        double model=elapsed(ready,end); model_sum+=model;
        e.model_calls.push_back(model); e.prep_calls.push_back(elapsed(begin,ready));
        e.total_calls.push_back(elapsed(begin,end)); checksum+=result;
      }
      e.model_means.push_back(model_sum/samples);
      // Separate block timing avoids charging per-call clock reads to pipeline throughput.
      fence(); auto begin=Clock::now();
      for (size_t row:indices) {
        work.disturb(row); work.features(raw.data()+row*nf,row,nf,features.data(),e.packed);
        float result; e.call(features.data(),&result); checksum+=result;
      }
      fence(); auto end=Clock::now();
      e.end_to_end_blocks.push_back(elapsed(begin,end)/samples); e.checksum+=checksum;
      prediction_sink=checksum;
    }
  }
  std::cout<<std::setprecision(10)<<"{\"history_bytes\":"<<history_bytes<<",\"eviction_bytes\":"<<eviction_bytes
    <<",\"code_blocks\":"<<blocks<<",\"samples_per_engine\":"<<samples*rounds
    <<",\"clock_pair_p50_ns\":"<<quantile(clock_pairs,.5)
    <<",\"paired_rounds\":"<<(paired ? "true" : "false")<<",\"round_orders\":[";
  for (size_t r=0;r<round_orders.size();++r) {
    if (r)std::cout<<',';std::cout<<'[';
    for (size_t i=0;i<round_orders[r].size();++i) { if (i)std::cout<<',';std::cout<<round_orders[r][i]; }
    std::cout<<']';
  }
  std::cout<<"],\"row_sequence_hashes\":[";
  for (size_t r=0;r<row_sequence_hashes.size();++r) { if(r)std::cout<<',';std::cout<<'\"'<<row_sequence_hashes[r]<<'\"'; }
  std::cout<<"],\"engines\":[";
  for (size_t i=0;i<engines.size();++i) {
    auto& e=engines[i]; if (i)std::cout<<',';
    std::cout<<"{\"name\":\""<<e.name<<"\",\"model_mean_median_ns\":"<<quantile(e.model_means,.5)
      <<",\"model_p50_ns\":"<<quantile(e.model_calls,.5)<<",\"model_p95_ns\":"<<quantile(e.model_calls,.95)
      <<",\"model_p99_ns\":"<<quantile(e.model_calls,.99)<<",\"preparation_p50_ns\":"<<quantile(e.prep_calls,.5)
      <<",\"pipeline_p50_ns\":"<<quantile(e.total_calls,.5)<<",\"pipeline_p99_ns\":"<<quantile(e.total_calls,.99)
      <<",\"pipeline_block_median_ns\":"<<quantile(e.end_to_end_blocks,.5)<<",\"model_round_means\":";
    array_json(e.model_means); std::cout<<",\"pipeline_blocks\":"; array_json(e.end_to_end_blocks);
    std::cout<<",\"checksum\":"<<e.checksum<<",\"max_abs_error\":"<<e.max_error<<'}'; dlclose(e.handle);
  }
  std::cout<<"]}\n";
} catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
