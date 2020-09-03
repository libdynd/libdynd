#include "stdlib.h"

#include "dynd/abi/metadata.h"

DYND_ABI_EXPORT dynd_buffer *dynd_malloc_buffer(dynd_size_t size) dynd_noexcept {
  dynd_buffer *buffer = malloc(size);
  buffer->resource.release = &dynd_release_malloc_allocated;
  buffer->allocated.base_ptr = buffer;
  buffer->allocated.size = size;
  return buffer;
}

DYND_ABI_EXPORT void dynd_release_malloc_allocated(dynd_resource* resource) dynd_noexcept {
  free(resource);
}
