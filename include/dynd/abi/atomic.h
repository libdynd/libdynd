#if !defined(DYND_ABI_ATOMIC_H)
#define DYND_ABI_ATOMIC_H

#include "dynd/abi/integers.h"

// Defines for an atomic size_t.
// This currently only includes the type itself and
// some additional macros since it's unlikely we'll
// actually need everything that's normally available
// via intrinsics or the standard header.
// For the time being there is no standard way
// to refer to C11 atomics from C++, but
// we want this type to be usable in both C and C++
// code, so we have to rely on the fact that
// C and C++ atomics currently happen to be
// binary compatible. This is unlikely to be
// fixed until at least c++23. See also:
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p0943r5.html
#if !defined(__cplusplus) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
// C11 atomics
#include <stdatomic.h>
typedef atomic_size_t dynd_atomic_size_t;
#define dynd_atomic_memory_order_relaxed (memory_order_relazed)
#define dynd_atomic_meomry_order_acquire (memory_order_acquire)
#define dynd_atomic_memory_order_release (memory_order_release)
#define dynd_atomic_memory_order_acq_rel (memory_order_acq_rel)
#define dynd_atomic_fetch_add(val, increment, consistency) (atomic_fetch_add_explicit(val, increment, consistency))
#define dynd_atomic_fetch_sub(val, increment, consistency) (atomic_fetch_sub_explicit(val, increment, consistency))
#define dynd_atomic_thread_fence(consistency) (atomic_thread_fence(consistency))
#elif defined(__cplusplus) && (__cplusplus >= 201103L)
// C++11 atomics
#include <atomic>
#include <cstddef>
using dynd_atomic_size_t = std::atomic<dynd_size_t>;
#define dynd_atomic_memory_order_relaxed (std::memory_order_relaxed)
#define dynd_atomic_memory_order_acquire (std::memory_order_acquire)
#define dynd_atomic_memory_order_release (std::memory_order_release)
#define dynd_atomic_memory_order_acq_rel (std::memory_order_acq_rel)
#define dynd_atomic_fetch_add(val, increment, consistency) (std::atomic_fetch_add_explicit(val, increment, consistency))
#define dynd_atomic_fetch_sub(val, increment, consistency) (std::atomic_fetch_sub_explicit(val, increment, consistency))
#define dynd_atomic_thread_fence(consistency) (std::atomic_thread_fence(consistency))
#elif defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7))
// Neither c11 nor C++11, but equivalent gcc intrinsics are still available with gcc >= 4.7.
#include <stddef.h>
typedef dynd_size_t dynd_atomic_size_t;
#define dynd_atomic_memory_order_relaxed __ATOMIC_RELAXED
#define dynd_atomic_memory_order_acquire __ATOMIC_ACQUIRE
#define dynd_atomic_memory_order_release __ATOMIC_RELEASE
#define dynd_atomic_memory_order_acq_rel __ATOMIC_ACQ_REL
#define dynd_atomic_fetch_add(val, increment, consistency) (__atomic_add_fetch(val, increment, consistency))
#define dynd_atomic_fetch_sub(val, increment, consistency) (__atomic_sub_fetch(val, increment, consistency))
#define dynd_atomic_thread_fence(consistency) (__atomic_thread_fence(consistency))
#elif defined(_MSC_VER)
// MSVC in C mode doesn't provide all the atomic operations we need
// with documented intrinsics, but we can get what we need by using
// the same intrinsics that are used in their c++ standard library
// implementation. Hopefully someday soon MSVC will have its
// own implementation of stdatomic.h, but this at least gets things
// working without horrible performance consequences.

#include <Windows.h>

typedef dynd_size_t dynd_atomic_size_t;

// Match the number of enum items and the order
// of their declaration in the MSVC C++ standard
// library to hopefully make them binary compatible.
// TODO: check this and specify the actual values if needed.
typedef enum {
  dynd_atomic_memory_order_relaxed,
  dynd_internal_atomic_memory_order_consume, // not used/supported
  dynd_atomic_memory_order_acquire,
  dynd_atomic_memory_order_release,
  dynd_atomic_memory_order_acq_rel,
  dynd_internal_atomic_memory_order_seq_cst // not used/supported
} dynd_atomic_memory_order;

inline dynd_size_t dynd_internal_atomic_fetch_add(dynd_atomic_size_t *val, dynd_size_t increment, dynd_atomic_memory_order consistency) {
  // In terms of the raw bytes, signed and unsigned addition are the same,
  // and the overflow behavior is correct for unsigned.
  // Here we rely on this fact to use the signed intrinsics
  // to implement unsigned fetch-add operations.
  // This does rely on implementation details, but it
  // is really only a stopgap until MSVC adds support for
  // C11 atomics. Once there's a standard-supported way
  // to do this we can use that.
  if (sizeof(dynd_atomic_size_t) == 4) {
    if (consistency == dynd_atomic_memory_order_relaxed)
      return (dynd_atomic_size_t) InterlockedAddNoFence((LONG*)val, (LONG)increment);
    if (consistency == dynd_atomic_memory_order_acquire)
      return (dynd_atomic_size_t) InterlockedAddAcquire((LONG*)val, (LONG)increment);
    if (consistency == dynd_atomic_memory_order_release)
      return (dynd_atomic_size_t) InterlockedAddRelease((LONG*)val, (LONG)increment);
    if (consistency == dynd_atomic_memory_order_acq_rel)
      // Fallback to sequential consistency here since
      // that's what the C++ standard library does too.
      return (dynd_atomic_size_t) InterlockedAdd((LONG*)val, (LONG)increment);
  } else {
    if (consistency == dynd_atomic_memory_order_relaxed)
      return (dynd_atomic_size_t) InterlockedAddNoFence64((LONG64*)val, (LONG64)increment);
    if (consistency == dynd_atomic_memory_order_acquire)
      return (dynd_atomic_size_t) InterlockedAddAcquire64((LONG64*)val, (LONG64)increment);
    if (consistency == dynd_atomic_memory_order_release)
      return (dynd_atomic_size_t) InterlockedAddRelease64((LONG64*)val, (LONG64)increment);
    if (consistency == dynd_atomic_memory_order_acq_rel)
      // Fallback to sequential consistency here since
      // that's what the C++ standard library does too.
      return (dynd_atomic_size_t) InterlockedAdd64((LONG64*)val, (LONG64)increment);
  }
}
#define dynd_atomic_fetch_add(val, increment, consistency) dynd_internal_atomic_fetch_add(val, increment, consistency)

inline dynd_size_t dynd_internal_atomic_fetch_sub(dynd_atomic_size_t *val, dynd_size_t decrement, dynd_atomic_memory_order consistency) {
  // Rely on wraparound arithmetic with unsigned integers.
  dynd_size_t increment = -decrement;
  return dynd_atomic_fetch_add(val, increment, consistency);
}
#define dynd_atomic_fetch_sub(val, increment, consistency) dynd_internal_atomic_fetch_sub(val, increment, consistency)

// Simplified version of the logic in atomic_thread_fence
// from the MSVC C++ standard library that ignores the
// consistencies not supported here.
inline void dynd_internal_atomic_thread_fence(dynd_atomic_memory_order consistency) {
  if (consistency == dynd_atomic_memory_order_relaxed)
    return;
#if defined(_M_IX86) || defined(_M_X64)
  _Compiler_barrier();
#elif defined(_M_ARM) || defined(_M_ARM64)
  _Memory_barrier();
#else
// As of this writing, Windows only works on x86 and arm
// based architectures, so there's nothing else to compile for.
#error Unrecognized architecture
#endif
}

#define dynd_atomic_thread_fence(consistency) (dynd_internal_atomic_thread_fence(consistency))

#else
// Unknown compiler with no standard atomics available.
#error Don't know how to perform atomic operations with the current compiler.
#endif

#endif // !defined(DYND_ABI_ATOMIC_H)
