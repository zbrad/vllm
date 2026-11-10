#pragma once

#include <climits>
#include <cstdint>
#include <cstdlib>

// Count leading/trailing zero bits of a nonzero 32-bit value. MSVC has no
// __builtin_clz/__builtin_ctz and its _BitScan* intrinsics are not constexpr,
// so it gets a plain loop (zbrad windows-dev); GCC/Clang keep the builtins.
#if defined(_MSC_VER) && !defined(__clang__)
inline constexpr int vllm_clz32(uint32_t x) {
  int n = 0;
  while (!(x & 0x80000000u)) {
    x <<= 1;
    ++n;
  }
  return n;
}
inline constexpr int vllm_ctz32(uint32_t x) {
  int n = 0;
  while (!(x & 1u)) {
    x >>= 1;
    ++n;
  }
  return n;
}
#else
inline constexpr int vllm_clz32(uint32_t x) { return __builtin_clz(x); }
inline constexpr int vllm_ctz32(uint32_t x) { return __builtin_ctz(x); }
#endif

inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - vllm_clz32(num - 1));
}

template <typename A, typename B>
static inline constexpr auto div_ceil(A a, B b) {
  return (a + b - 1) / b;
}

// Round a down to the next multiple of b. The caller is responsible for making
// sure that b is non-zero
template <typename T>
inline constexpr T round_to_previous_multiple_of(T a, T b) {
  return a % b == 0 ? a : (a / b) * b;
}

// Round a up to the next multiple of b. The caller is responsible for making
// sure that b is non-zero
template <typename T>
inline constexpr T round_to_next_multiple_of(T a, T b) {
  return a % b == 0 ? a : ((a / b) + 1) * b;
}

namespace vllm {

// Returns true if env VLLM_BATCH_INVARIANT=1
inline bool vllm_is_batch_invariant() {
  static bool cached = []() {
    const char* val = std::getenv("VLLM_BATCH_INVARIANT");
    return val && std::atoi(val) != 0;
  }();
  return cached;
}

}  // namespace vllm
