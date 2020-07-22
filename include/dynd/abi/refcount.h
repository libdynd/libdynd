#if !defined(DYND_ABI_REFCOUNT_H)
#define DYND_ABI_REFCOUNT_H

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
struct dynd_refcounted {
  dynd_refcount refcount;
  dynd_resource resource;
};

// TODO: incref and decref operations here!

#if defined(__cplusplus)
}
#endif // defined(__cplusplus)

#endif // !defined(DYND_ABI_REFCOUNT_H)
