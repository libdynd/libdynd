#if !defined(DYND_ABI_TYPE_H)
#define DYND_ABI_TYPE_H

#include "dynd/abi/integers.h"
#include "dynd/abi/metadata.h"
#include "dynd/abi/resource.h"
#include "dynd/abi/version.h"

#if defined(__cplusplus)
extern "C" {
#endif // defined(__cplusplus)

#define dynd_type_header_impl DYND_ABI(type_header_impl)
struct dynd_type_header_impl;

// TODO: The abstract resource freeing ABI seems
// like a poor way to do the decrefs for each array/type/handle/etc.
// These should probably have destruction routines that happen to also
// call the corresponding destruction routines.
// That'll mean custom incref and decref for each type though?
// Having each allocator provide a destruction routine for
// arrays, types, vtables, and type constructors seems dumb though.

#define dynd_type_impl DYND_ABI(type_impl)
struct dynd_type_impl;

#define dynd_type_alignment DYND_ABI(type_alignment)
DYND_ABI_NOEXCEPT_FUNC(dynd_type_alignment, size_t, dynd_type_header_impl)

// A pair of begin and end pointers used for returning
// ranges of supertypes or type parameters.
#define dynd_type_range DYND_ABI(type_range)
typedef struct {
  dynd_type_impl *begin;
  dynd_type_impl *end;
} dynd_type_range;

#define dynd_type_parameters DYND_ABI(type_parameters)
DYND_ABI_NOEXCEPT_FUNC(dynd_type_parameters, dynd_type_range, dynd_type_header_impl)

#define dynd_type_superclasses DYND_ABI(type_superclasses)
DYND_ABI_NOEXCEPT_FUNC(dynd_type_superclasses, dynd_type_range, dynd_type_header_impl)

#define dynd_type_vtable_entries DYND_ABI(type_vtable_entries)
typedef struct {
  dynd_type_alignment alignment;
  dynd_type_parameters parameters
  dynd_type_superclasses superclasses;
} dynd_type_vtable_entries;

#define dynd_type_vtable DYND_ABI(type_vtable)
typedef struct {
  dynd_vtable vtable_header;
  dynd_type_vtable_entries entries;
} dynd_type_vtable;

#define dynd_type_constructor_impl DYND_ABI(type_constructor_impl)
struct dynd_type_constructor_impl;

struct dynd_type_header_impl{
  dynd_allocated allocated;
  dynd_type_vtable *vtable;
  dynd_type_constructor_impl *constructor;
};
#define dynd_type_header DYND_ABI(type_header)
typedef struct dynd_type_header_impl dynd_type_header;

struct dynd_type_impl {
  dynd_refcounted refcount;
  dynd_array_header header;
};
#define dynd_type DYND_ABI(type)
typedef struct dynd_type_impl dynd_type;

#define dynd_type_metadata(a) ((void*)(a + 1))

// Same as previous, but without reference counting.
// The layout is kept intentionally compatible.
#define dynd_type_ref DYND_ABI(type_ref)
typedef struct {
  dynd_resource resource;
  dynd_type_header header;
} dynd_type_ref;

#if defined(__cplusplus)
}
#endif // defined(__cplusplus)

#endif // !defined(DYND_ABI_TYPE_H)
