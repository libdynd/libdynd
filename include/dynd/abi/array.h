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
typedef struct {
  dynd_allocated allocated;
  void *base;
  dynd_type *type;
} dynd_array_header;

#define dynd_array DYND_ABI(array)
typedef struct {
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
} dynd_array;

// Note: the arrmeta specified by this type
// should be laid out in memory immediately
// after the array header. This is all done
// inside the buffer managed by the resource
// in the array struct.
// Note: the arrmeta will have the alignment
// of a function pointer or of size_t, whichever
// is greater. Use a macro so it works with
// both dynd_array and dynd_array_ref.
#define dynd_arrmeta(a) ((void*)(a + 1))

// Same as previous, but without reference counting.
// The layout is kept intentionally compatible.
#define dynd_array_ref DYND_ABI(array)
typedef struct {
  dynd_resource resource;
  dynd_array_header header;
} dynd_array_ref;

#endif // !defined(DYND_ABI_ARRAY_H)
