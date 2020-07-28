#if !defined(DYND_ABI_ARRAY_H)
#define DYND_ABI_ARRAY_H

#include "dynd/abi/metadata.h"
#include "dynd/abi/resource.h"
#include "dynd/abi/version.h"

// TODO: dynd type forward declaration header.
#ifndef dynd_type
#define dynd_type DYND_ABI(type)
#endif
struct dynd_type;

// Similar to dynd_allocated,
// but brings in the additional metadata needed
// to manage type-specific destructors.
#define dynd_array_header DYND_ABI(array_header)
struct dynd_array_header {
  dynd_allocated allocated;
  void *base;
  dynd_type *type;
};

#define dynd_array DYND_ABI(array)
struct dynd_array {
  // This resource encapsulates the allocation
  // of this particular buffer as well as
  // any additional custom deallocation required
  // by the type. Note: the type-specific deallocation
  // routine will only be called if the base pointer
  // is null.
  // TODO: the corresponding C++ interface
  // should automatically tie the lifetime of
  // this reference to the lifetime of this
  // metadata buffer.
  dynd_refcounted refcount;
  dynd_array_header header;
};
// Note: the arrmeta specified by this type
// should be laid out in memory immediately
// after the type pointer. This is all done
// inside the buffer managed by the resource
// in the array struct.

// Same as previous, but without reference counting.
// The layout is kept intentionally compatible.
#define dynd_array_ref DYND_ABI(array)
struct dynd_array_ref {
  dynd_resource resource;
  dynd_array_header header;
};

#endif // !defined(DYND_ABI_ARRAY_H)
