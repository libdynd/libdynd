#if !defined(DYND_ABI_VTABLE_H)
#define DYND_ABI_VTABLE_H

#include "dynd/abi/function_pointer.h"
#include "dynd/abi/refcount.h"
#include "dynd/abi/resource.h"
#include "dynd/abi/version.h"

// Interfaces aren't supported yet, so use a
// generic filler function pointer to save
// space for that later once the function
// input and output types have been sorted out.
typedef dynd_generic_func_ptr dynd_interface_cast;

#define dynd_vtable_header DYND_ABI(vtable_header)
typedef struct {
  dynd_interface_cast build_interface;
} dynd_vtable_header;

#define dynd_vtable DYND_ABI(vtable)
typedef struct {
  dynd_refcounted refcount;
  dynd_vtable_header header;
} dynd_vtable;

#endif // !defined(DYND_ABI_VTABLE_H)
