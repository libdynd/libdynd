#if !defined(DYND_ABI_TYPE_CONSTRUCTOR_H)
#define DYND_ABI_TYPE_CONSTRUCTOR_H

#include "dynd/abi/type.h"
#include "dynd/abi/version.h"
#include "dynd/abi/vtable.h"

#if defined(__cplusplus)
extern "C" {
#endif // defined(__cplusplus)

#define dynd_type_constructor_header_impl DYND_ABI(type_constructor_header)
struct dynd_type_constructor_header_impl;

DYND_ABI_NOEXCEPT_FUNC(dynd_type_constructor_make, dynd_type*, dynd_type_constructor_header_impl*, dynd_type_range)

#define dynd_type_constructor_vtable_entries DYND_ABI(type_constructor_vtable_entries)
typedef struct {
  dynd_type_constructor_make make;
} dynd_type_constructor_vtable_entries;

#define dynd_type_constructor_vtable DYND_ABI(type_constructor_vtable)
typedef struct {
  dynd_vtable header;
  dynd_type_constructor_vtable_entries entries;
} dynd_type_constructor_vtable;

struct dynd_type_constructor_header_impl {
  dynd_allocated allocated;
  dynd_type_constructor_vtable *vtable;
};
typedef struct dynd_type_constructor_header_impl dynd_type_constructor_header;

// dynd_type_constructor_impl is a macro defined in abi/type.h
// since a forward declaration for this struct type is needed there.
struct dynd_type_constructor_impl {
  dynd_refcounted refcount;
  dynd_type_constructor_header header;
};
typedef struct dynd_type_constructor_impl dynd_type_constructor;

#if defined(__cplusplus)
}
#endif // defined(__cplusplus)

#endif // !defined(DYND_ABI_TYPE_CONSTRUCTOR_H)
