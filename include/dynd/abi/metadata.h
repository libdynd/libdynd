#if !defined(DYND_ABI_METADATA_H)
#define DYND_ABI_METADATA_H

#include "dynd/abi/integers.h"
#include "dynd/abi/resource.h"

// A descriptor for an allocated block of memory
// meant to be stored in-memory after the resource struct.
// This is the most common type of resource.
// The information about the allocator it came
// from is expected to be handled in whatever
// release routine is used to release it.
#define dynd_allocated DYND_ABI(allocated)
typedef struct {
  void *base_ptr;
  dynd_size_t size;
  // dynd_allocator allocator;
} dynd_allocated;

// Metadata header for an externally allocated buffer.
#define dynd_buffer DYND_ABI(buffer)
typedef struct {
  dynd_resource resource;
  dynd_allocated allocated;
} dynd_buffer;

// In a context where the type is statically known, the type-specific metadata is really all that needs to be stored
// and that has a size that's known at compile time, so there's no need for inline resource management where the metadata is allocated.
//
// In the case where the type is not statically known, two cases may apply: the metadata block may be allocated separately, so it is a resource in its own right.
// It may be managed using whatever lifetime system a given laguage already has on-hand, or it may be managed via reference counting.
// if the metadata is allocated using calloc, the resource management is already handled by the stack itself so
// no resource struct is necessary and the program can just allocate the type-specified metadata.
//
// This means there are 3 types of buffers:
// externally managed
// inline resource management but no refcounting
// inline resource management with refcounting
//
// There are two ways to handle these three cases.
// We can have each "allocated" object store the base pointer for the buffer even when it's inlined into the allocation.
// In this case there's only need for one function to free resources.
// We can instead have each "allocated" object store the base pointer only when that's actually necessary.
// In that case ther are three different kinds of functions needed:
//  - one to free external memory blocks
//  - one to free memory blocks with inlined resource metadata
//  - one to free memory blocks with reference counting and inlined resource metadata.
// The first approach is simpler and could conceivably have benefits WRT branch prediction.
// The second approach saves storing an extra pointer.
// Allocation is generally a moderately expensive operation,
// so we're going with simplicity here. Support for the two additional
// specialized buffer types for the two additional inline cases
// can be added later as needed since the external memory block approach
// can be used for those cases even when the special cased
// versions are present.
//
// Here we define the two additional specialized buffer layouts
// but for the time being we won't add the necessary support
// for them elsewhere.
//
// Because the external memory block case is the most universal
// of the three types, that also needs to be the one with the
// most generic name.
//
// Layout: refcount inline_resource type_metadata(includes base_pointer and external_resource_ref)
// The only time inline_resource isn't needed is when something has a lifetime managed by something else (e.g. if it's stack allocated), in which case the refcount doesn't matter either.
// The layout also allows inline resource metadata but no refcount for cases where ownership is managed by something else, but reseource release still needs to be virtualized.

#define dynd_inline_allocated DYND_ABI(inline_allocated)
typedef struct {
  dynd_size_t size;
} dynd_inline_allocated;

// Metadata descriptor used for an allocated block
// of memory that has its corresponding resource
// struct at the beginning of the allocated block.
// In this case it's not necessary to store the
// base pointer since the pointer to the resource
// is the base pointer.
#define dynd_inline_buffer DYND_ABI(inline_buffer)
typedef struct {
  dynd_resource resource;
  dynd_inline_allocated allocated;
} dynd_inline_buffer;

#define dynd_inline_refcounted_allocated DYND_ABI(inline_refcounted_allocated)
typedef struct {
  dynd_size_t size;
} dynd_inline_refcounted_allocated;

// Metadata descriptor used for an allocated block
// of memory that is managed via inline reference
// counting where the reference count is the first
// address in the block, immediately followed by
// the corresponding resource struct.
#define dynd_inline_refcounted_buffer DYND_ABI(inline_refcounted_buffer)
typedef struct {
  dynd_resource resource;
  dynd_inline_refcounted_allocated allocated;
} dynd_inline_refcounted_buffer;

#endif // !defined(DYND_ABI_METADATA_H)
