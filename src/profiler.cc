#include "ctranslate2/profiler.h"

#ifndef CT2_ENABLE_PROFILING

#include <stdexcept>

namespace ctranslate2 {

  void init_profiling(Device, size_t) {
    throw std::runtime_error("CTranslate2 was not compiled with profiling support, "
                             "enable it with -DENABLE_PROFILING=ON during cmake configuration.");
  }

  void dump_profiling(std::ostream&) {
  }

}

#else

#include <algorithm>
#include <iomanip>
#include <memory>
#include <mutex>
#include <vector>
#include <unordered_map>

#ifdef CT2_WITH_METAL
#include "metal/utils.h"
#endif

namespace ctranslate2 {

  static void print_as_percentage(std::ostream& os, double ratio) {
    os << std::right << std::setw(6) << std::fixed << std::setprecision(2)
       << ratio * 100 << '%';
  }


  class ScopeProfile {
  public:
    std::chrono::microseconds time_in_scope;
    std::chrono::microseconds time_in_scope_and_callees;
    std::chrono::microseconds gpu_time;  // M11.4: GPU execution time
  };

  class Profiler {
  private:
    Device _device;
    size_t _num_threads;
    std::chrono::high_resolution_clock::time_point _global_start;
    std::unordered_map<std::string, ScopeProfile> _cumulated;
    std::mutex _mutex;

    ScopeProfile& get_scope_profile(const std::string& name) {
      auto it = _cumulated.find(name);
      if (it == _cumulated.end())
        it = _cumulated.emplace(name, ScopeProfile()).first;
      return it->second;
    }

  public:
    Profiler(Device device, size_t num_threads)
      : _device(device)
      , _num_threads(num_threads)
      , _global_start(std::chrono::high_resolution_clock::now()) {
    }

    Device device() const {
      return _device;
    }

    void add_scope_time(const std::string& name,
                        const std::chrono::microseconds& elapsed,
                        const std::chrono::microseconds& gpu_elapsed,
                        const std::string* parent_name) {
      std::lock_guard<std::mutex> lock(_mutex);
      auto& scope_profile = get_scope_profile(name);
      scope_profile.time_in_scope += elapsed;
      scope_profile.time_in_scope_and_callees += elapsed;
      scope_profile.gpu_time += gpu_elapsed;
      if (parent_name) {
        auto& parent_scope_profile = get_scope_profile(*parent_name);
        parent_scope_profile.time_in_scope -= elapsed;
      }
    }

    void dump(std::ostream& os) const {
      if (_cumulated.empty())
        return;

      auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - _global_start);
      total_time *= _num_threads;

      // Sort from largest to smallest accumulated time.
      std::vector<std::pair<std::string, ScopeProfile>> sorted_cumulated(_cumulated.begin(),
                                                                         _cumulated.end());
      std::sort(sorted_cumulated.begin(), sorted_cumulated.end(),
                [] (const std::pair<std::string, ScopeProfile>& a,
                    const std::pair<std::string, ScopeProfile>& b) {
                  return a.second.time_in_scope > b.second.time_in_scope;
                });

      // Get the longest profiler name to pretty print the output.
      size_t longest_name = 0;
      for (const auto& pair : sorted_cumulated)
        longest_name = std::max(longest_name, pair.first.length());

      // M11.4: Check if any scope has GPU time recorded.
      bool has_gpu_time = false;
      for (const auto& pair : sorted_cumulated) {
        if (pair.second.gpu_time.count() > 0) {
          has_gpu_time = true;
          break;
        }
      }

      double total_time_us = total_time.count();
      double ratio_printed_so_far = 0;
      for (const auto& pair : sorted_cumulated) {
        const auto& name = pair.first;
        const auto& result = pair.second;

        double time_in_scope_us = result.time_in_scope.count();
        double time_in_scope_and_callees_us = result.time_in_scope_and_callees.count();
        double time_in_scope_ratio = time_in_scope_us / total_time_us;
        double time_in_scope_and_callees_ratio = time_in_scope_and_callees_us / total_time_us;
        ratio_printed_so_far += time_in_scope_ratio;

        print_as_percentage(os, time_in_scope_ratio);
        os << ' ';
        print_as_percentage(os, time_in_scope_and_callees_ratio);
        os << ' ';
        print_as_percentage(os, ratio_printed_so_far);
        os << ' ' << std::left << std::setw(longest_name) << name
           << ' ' << (time_in_scope_us / 1000) << "ms";

        // M11.4: Append GPU time column when available.
        if (has_gpu_time) {
          double gpu_time_us = result.gpu_time.count();
          os << "  " << (gpu_time_us / 1000) << "ms(gpu)";
        }

        os << std::endl;
      }
    }
  };


  static std::unique_ptr<Profiler> profiler;


  void init_profiling(Device device, size_t num_threads) {
    profiler = std::make_unique<Profiler>(device, num_threads);
  }

  void dump_profiling(std::ostream& os) {
    if (profiler) {
      profiler->dump(os);
      profiler.reset();
    }
  }


  // Track active scope in the current thread.
  static thread_local ScopeProfiler* current_scope = nullptr;

  ScopeProfiler::ScopeProfiler(const std::string& name) {
    if (!profiler)
      return;
    _parent = current_scope;
    _name = name;
    synchronize_stream(profiler->device());
#ifdef CT2_WITH_METAL
    if (profiler->device() == Device::METAL)
      _gpu_start = metal::gpu_time_elapsed();
#endif
    _start = std::chrono::high_resolution_clock::now();
    current_scope = this;
  }

  ScopeProfiler::~ScopeProfiler() {
    if (!profiler)
      return;
    synchronize_stream(profiler->device());
    auto diff = std::chrono::high_resolution_clock::now() - _start;
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(diff);

    std::chrono::microseconds gpu_elapsed{0};
#ifdef CT2_WITH_METAL
    if (profiler->device() == Device::METAL) {
      double gpu_secs = metal::gpu_time_elapsed() - _gpu_start;
      gpu_elapsed = std::chrono::microseconds(static_cast<int64_t>(gpu_secs * 1e6));
    }
#endif

    profiler->add_scope_time(_name, elapsed, gpu_elapsed, _parent ? &_parent->_name : nullptr);
    current_scope = _parent;
  }

}

#endif
