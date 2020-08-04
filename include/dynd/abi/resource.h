#if !defined(DYND_ABI_RESOURCE_H)
#define DYND_ABI_RESOURCE_H

#include "dynd/abi/function_pointer.h"
#include "dynd/abi/version.h"

// Some abstract resource (usually an allocated buffer)
// that needs to be destroyed when it is no longer referred to,
// but which does not track its reference count internally.
#define dynd_resource_impl DYND_ABI(resource_impl);
struct dynd_resource_impl;

DYND_ABI_NOEXCEPT_FUNC(dynd_resource_release, void, dynd_resource_impl*)

// Note: The intent is that a given resource
// be able to store whatever additional metadata it needs
// immediately after this struct.
// When the resource is released, the release function is
// passed the address of the release member here and
// any corresponding metadata can be read using offsets
// from that base address.
struct dynd_resource_impl {
  dynd_resource_release release;
};

#define dynd_resource DYND_ABI(resource)
typedef dynd_resource_impl dynd_resource;

#endif // !defined(DYND_ABI_RESOURCE_H)
