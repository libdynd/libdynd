#if !defined(DYND_ABI_REFCOUNT_H)
#define DYND_ABI_REFCOUNT_H

#include "dynd/abi/atomic.h"
#include "dynd/abi/noexcept.h"
#include "dynd/abi/resource.h"
#include "dynd/abi/version.h"

#if defined(__cplusplus)
extern "C" {
#endif // defined(__cplusplus)

typedef dynd_atomic_size_t dynd_refcount;

// A header for refcounting.
// Incref and decref operations are provided below.
// If the reference count hits 0 during a decref operation,
// the resource is released.
// Note: resource-specific metadata may be stored in memory
// after the end of this struct.
#define dynd_refcounted DYND_ABI(refcounted)
typedef struct {
  dynd_refcount refcount;
  dynd_resource resource;
} dynd_refcounted;

#define dynd_incref DYND_ABI(incref)
inline void dynd_incref(dynd_refcounted *ref) dynd_noexcept {
  dynd_atomic_fetch_add((dynd_refcount*)ref, dynd_size_t(1u), dynd_memory_order_relaxed);
}

#define dynd_decref DYND_ABI(decref)
inline void dynd_decref(dynd_refcounted *ref) dynd_noexcept {
  dynd_size_t val = dynd_atomic_fetch_sub((dynd_refcount*)ref, dynd_size_t(1u), dynd_memory_order_release);
  if (val == 1) {
    dynd_atomic_thread_fence(dynd_memory_order_acquire);
    ref->resource.release();
  }
}

#if defined(__cplusplus)
}
#endif // defined(__cplusplus)

#endif // !defined(DYND_ABI_REFCOUNT_H)
