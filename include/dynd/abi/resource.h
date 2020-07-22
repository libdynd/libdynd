#if !defined(DYND_ABI_RESOURCE_H)
#define DYND_ABI_RESOURCE_H

#if defined(__cplusplus) && __cplusplus >= 201103L && __cplusplus < 201703L
// Needed for function pointer noexcept typedef trick.
#include <type_traits>
#endif

#include "dynd/abi/version.h"

#if defined(__cplusplus)
extern "C" {
#endif // defined(__cplusplus)

// Some abstract resource (usually an allocated buffer)
// that needs to be destroyed when it is no longer referred to,
// but which does not track its reference count internally.
#define dynd_resource DYND_ABI(resource)
struct dynd_resource;

// Call the function pointer provided by the resource and
// pass it to the resource itself to release the resource.
#define dynd_resource_release DYND_ABI(resource_release)
#if defined(__cplusplus) && __cplusplus >= 201103L && __cplusplus < 201703L
}

#if __cplusplus < 201103L
// No way to include throw() in the function pointer type
// with old versions of C++.
typedef void (*dynd_resource_release)(dynd_resource*);
#elif __cplusplus < 201703L
// C++ equivalent of C typedef with the noexcept specifier added.
// Note: this isn't possible with a plain typedef.
// Use the technique from: https://stackoverflow.com/a/53674998
// Noexcept is a full part of the type system in c++17 and later,
// but this trickery is needed to get the right behavior
// with earlier versions that still support noexcept.
using dynd_resource_release = decltype(std::declval<void (*)(dynd_resource*) dynd_noexcept>());
#elif __cplusplus >= 201703L
typedef void (*dynd_resource_release)(dynd_resource*) dynd_noexcept;
#endif

extern "C" {
#else

typedef void (*dynd_resource_release)(dynd_resource*);

#endif // defined(__cplusplus)

// Note: The intent is that a given resource
// be able to store whatever additional metadata it needs
// immediately after this struct.
// When the resource is released, the release function is
// passed the address of the release member here and
// any corresponding metadata can be read using offsets
// from that base address.
struct dynd_resource {
  dynd_resource_release release;
};

#if defined(__cplusplus)
}
#endif // defined(__cplusplus)

#endif // !defined(DYND_ABI_RESOURCE_H)
