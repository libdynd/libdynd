#if !defined(DYND_ABI_METADATA_H)
#define DYND_ABI_METADATA_H

#include "dynd/abi/resource.h"

// A descriptor for an allocated block of memory
// meant to be stored in-memory after the resource struct.
// This is the most common type of resource.
// The information about the allocator it came
// from is expected to be handled in whatever
// release routine is used to release it.
// TODO: it would be possible to have a
// single deallocation routine for all allocated
// buffers that looks up an additional pointer to an
// allocator vtable and uses that to free the buffer.
// This is a good design, but it's not yet possible
// since, as of this writing, the allocator interface
// hasn't been concretely defined yet.
#define dynd_allocated DYND_ABI(allocated)
struct dynd_allocated {
  void *base_ptr;
  dynd_size_t size;
  // dynd_allocator allocator;
};

// Metadata header for an externally allocated buffer.
#define dynd_buffer DYND_ABI(buffer)
struct dynd_buffer {
  dynd_resource resource;
  dynd_allocated allocated;
};

// TODO: Facilities for external vs non-external buffers.

#endif // !defined(DYND_ABI_METADATA_H)
