#if !defined(DYND_ABI_ARRAY_H)
#define DYND_ABI_ARRAY_H

#include "dynd/abi/resource.h"
#include "dynd/abi/version.h"

// TODO: dynd type forward declaration header.
#ifndef dynd_type
#define dynd_type DYND_ABI(type)
#endif
struct dynd_type;

#define dynd_array DYND_ABI(array)
struct dynd_array {
  dynd_resource resource;
  // Note: This may be an owned reference to the
  // type used here. That's expected to be the
  // most common case, however whether or not
  // the reference is owned, the actual in-memory
  // layout is the same and that's what is specified
  // here.
  // TODO: the corresponding C++ interface
  // should automatically tie the lifetime of
  // this reference to the lifetime of this
  // metadata buffer.
  dynd_type *type;
};
// Note: the arrmeta specified by this type
// should be laid out in memory immediately
// after the type pointer. This is all done
// inside the buffer managed by the resource
// in the array struct.

// If the array itself
// is refcounted, the refcount appears
// in-memory before the resource.
#define dynd_refcounted_array DYND_ABI(refcounted_array)
struct dynd_refcounted_array {
  dynd_refcounted resource;
  dynd_type *type;
};

#endif // !defined(DYND_ABI_ARRAY_H)
