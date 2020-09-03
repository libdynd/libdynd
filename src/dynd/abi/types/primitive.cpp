#include <cassert>

#define DYND_ABI_TYPES_PRIMITIVE_CPP

#include "dynd/abi/noexcept.h"
#include "dynd/abi/types/primitive.h"

extern "C" {

// TODO: We could build a factory type constructor to generate all the
// builtin fixed-width types from their size and alignment requirements.
// That'd require that we have types that represent integers or some notion
// of dependent typing, which isn't currently present.

// TODO: This is currently a part of a type's ABI. This probably should be moved
// to be a part of an interface specific to primitive types.
static dynd_size_t primitive_type_alignment(dynd_type_header *header) dynd_noexcept {
  return ((dynd_type_primitive_typemeta*)(dynd_type_metadata(header))) -> alignment;
}

static dynd_type_range empty_range(dynd_type_header*) dynd_noexcept {
  return {nullptr, nullptr};
}

struct dynd_primitive_vtable : dynd_type_vtable {
  dynd_primitive_vtable() dynd_noexcept {
    header.refcount.resource.release = dynd_abi_resource_never_release;
    dynd_atomic_store(&header.refcount.refcount, dynd_size_t(1u), dynd_memory_order_relaxed);
    header.header.allocated.base_ptr = this;
    header.header.allocated.size = sizeof(dynd_primitive_vtable);
    header.header.build_interface = nullptr;
    entries.alignment = primitive_type_alignment;
    entries.parameters = empty_range;
    entries.superclasses = empty_range;
  }
  ~dynd_primitive_vtable() dynd_noexcept {
    assert(dynd_atomic_load(&header.refcount.refcount, dynd_memory_order_relaxed) == 1);
  }
};

static dynd_primitive_vtable primitive_vtable{};

// Once we have dependent typing, set this to be something real.
// For now, it's address is really all that matters.
DYND_ABI_EXPORT dynd_type_constructor dynd_type_make_primitive;

struct dynd_builtin_primitive_type : dynd_type_primitive {
  dynd_builtin_primitive_type(std::size_t size, std::size_t alignment) dynd_noexcept {
    typemeta.size = size;
    typemeta.alignment = alignment;
    dynd_atomic_store(&prefix.refcount.refcount, dynd_size_t(1u), dynd_memory_order_relaxed);
    prefix.refcount.resource.release = &dynd_abi_resource_never_release;
    // These will likely be never used, but initialize them correctly anyway.
    prefix.header.allocated.base_ptr = this;
    prefix.header.allocated.size = sizeof(dynd_builtin_primitive_type);
    prefix.header.vtable = reinterpret_cast<dynd_type_vtable*>(&primitive_vtable);
    prefix.header.constructor = &dynd_type_make_primitive;
  }
  ~dynd_builtin_primitive_type() dynd_noexcept {
    assert(dynd_atomic_load(&prefix.refcount.refcount, dynd_memory_order_relaxed) == 1u);
  }
};

DYND_ABI_EXPORT dynd_builtin_primitive_type dynd_types_float16{2u, 2u};
DYND_ABI_EXPORT dynd_builtin_primitive_type dynd_types_float32{4u, 4u};
DYND_ABI_EXPORT dynd_builtin_primitive_type dynd_types_float64{8u, 8u};

DYND_ABI_EXPORT dynd_builtin_primitive_type dynd_types_uint8{1u, 1u};
DYND_ABI_EXPORT dynd_builtin_primitive_type dynd_types_uint16{2u, 2u};
DYND_ABI_EXPORT dynd_builtin_primitive_type dynd_types_uint32{4u, 4u};
DYND_ABI_EXPORT dynd_builtin_primitive_type dynd_types_uint64{8u, 8u};
DYND_ABI_EXPORT dynd_builtin_primitive_type dynd_types_int8{1u, 1u};
DYND_ABI_EXPORT dynd_builtin_primitive_type dynd_types_int16{2u, 2u};
DYND_ABI_EXPORT dynd_builtin_primitive_type dynd_types_int32{4u, 4u};
DYND_ABI_EXPORT dynd_builtin_primitive_type dynd_types_int64{8u, 8u};

DYND_ABI_EXPORT dynd_builtin_primitive_type dynd_types_size_t{sizeof(dynd_size_t), alignof(dynd_size_t)};

}
