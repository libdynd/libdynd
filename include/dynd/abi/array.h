#if !defined(DYND_ABI_ARRAY_H)
#define DYND_ABI_ARRAY_H

#include "dynd/abi/metadata.h"
#include "dynd/abi/refcount.h"
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
  // TODO: would it be better to just full-on store
  // an additional resource pointer here to avoid
  // having to have a separate resource release
  // function specific to the array type for each
  // allocator used? For now, assume there are many
  // more arrays than allocators.
  dynd_allocated allocated;
  // The array metadata object this one was generated
  // from (if any) via slicing or some similar operation.
  // If no such array exists, this entry is null.
  // If an array header has no base_array and it contains
  // types with nontrivial destruction routines,
  // it calls the corresponding destructors when
  // it is destroyed. If it has a base_array, it instead
  // decrefs its base array.
  void *base_array;
  // An owned reference to the type of this array.
  dynd_type *type;
  // Access rights associated with this metadata header.
  // For now this is primarily reserved space until
  // access flags have a settled design or until something
  // more sophisticated can be done.
  // Flags representing read/write access.
  // Only a couple of bits of this space are used,
  // but alignment constraints mean that a full size_t
  // of space will be used anyway, so there's no reason
  // to use a smaller int.
  // TODO: Given that the base_array pointer
  // will always be an address aligned
  // to at least 4 bytes, can we use the two bits
  dynd_size_t access;
  // Base pointer used for all strided accesses
  // into the data described by this metadata header.
  // All accesses are computed as offsets from this
  // pointer.
  // In theory this could be moved into the
  // type-specific metadata and then composing container
  // types could be done relative some interface that
  // they all agree on, but the benefits of doing so
  // seem very small relative to a significant increase
  // in the corresponding complexity for the code managing
  // what gets stored in the array-specific metadata.
  void *base;
} dynd_array_header;

#define dynd_array DYND_ABI(array)
typedef struct {
  // This resource encapsulates the allocation
  // of this particular buffer as well as
  // any additional custom deallocation required
  // by the type. Note: the type-specific deallocation
  // routine will only be called if the base pointer
  // is null.
  // NOTE: the resource here is also in-charge
  // of decrefing the corresponding type.
  // TODO: the corresponding C++ interface
  // should automatically tie the lifetime of
  // this reference to the lifetime of this
  // metadata buffer.
  dynd_refcounted refcount;
  dynd_array_header header;
} dynd_array;

// Note: the array-specific metadata specified by
// the type should be laid out in memory immediately
// after the array header. This is all done
// inside the buffer managed by the resource
// in the array struct.
// Note: the metadata will have the alignment
// of a function pointer or of size_t, whichever
// is greater. Use a macro so it works with
// both dynd_array and dynd_array_ref.
// This alignment is guaranteed by making
// the alignment of the metadata the same as
// the alignment of the dynd_array struct.
#define dynd_array_metadata(a) ((void*)(a + 1))

// Same as previous, but without reference counting.
// The layout is kept intentionally compatible.
#define dynd_array_ref DYND_ABI(array_ref)
typedef struct {
  dynd_resource resource;
  dynd_array_header header;
} dynd_array_ref;

#endif // !defined(DYND_ABI_ARRAY_H)
