#include <cassert>

#define DYND_ABI_TYPES_DENSE_CPP

#include "dynd/abi/metadata.h"
#include "dynd/abi/types/dense.h"

static dynd_type **dense_get_parameter(dynd_type_header_impl *type_header) noexcept {
  return &reinterpret_cast<dynd_type_dense_typemeta*>(dynd_type_metadata(type_header))->parameter;
}

static dynd_type_range dense_type_parameter(dynd_type_header_impl *type_header) noexcept {
  dynd_type **child_ptr = dense_get_parameter(type_header);
  return {child_ptr, child_ptr + 1u};
}

static dynd_size_t dense_alignment(dynd_type_header_impl *type_header) noexcept {
  dynd_type **child_ptr = dense_get_parameter(type_header);
  return (*child_ptr)->header.vtable->entries.alignment(&(*child_ptr)->header);
}

struct dynd_dense_concrete_vtable : dynd_type_vtable {
  dynd_dense_concrete_vtable() dynd_noexcept {
    header.refcount.resource.release = dynd_abi_resource_never_release;
    dynd_atomic_store(&header.refcount.refcount, dynd_size_t(1u), dynd_memory_order_relaxed);
    header.header.allocated.base_ptr = this;
    header.header.allocated.size = sizeof(dynd_dense_concrete_vtable);
    header.header.build_interface = nullptr;
    entries.alignment = dense_alignment;
    entries.parameters = dense_type_parameter;
    entries.superclasses = dynd_type_range_empty;
  }
  ~dynd_dense_concrete_vtable() dynd_noexcept {
    assert(dynd_atomic_load(&header.refcount.refcount, dynd_memory_order_relaxed) == 1);
  }
};

dynd_dense_concrete_vtable dense_vtable{};

struct dynd_type_dense_impl;

extern "C" {
extern dynd_type_dense_impl dynd_type_dense;
}

static dynd_type *make_dense(dynd_type_constructor_header *, dynd_type_range parameters) noexcept {
  // The resource management system needs an overhaul.
  // It's too complicated right now.
  // There should just be a function pointer for deallocating the buffer containing the metadata
  // and a *separate* function pointer for releasing the underlying resource.
  // Right now things are all rolled together into the single "resource release"
  // function that does too many things.
  // As a shortcut to get this initial pass working, just track deallocation info for
  // the metadata buffers. This means that it's okay to just use dynd_malloc_buffer
  // out of the box.
  dynd_type_dense_concrete *buffer = reinterpret_cast<dynd_type_dense_concrete*>(dynd_malloc_buffer(sizeof(dynd_type_dense_concrete)));
  dynd_atomic_store(&(buffer->prefix.refcount.refcount), dynd_size_t(1u), dynd_memory_order_relaxed);
  // These will likely be never used, but initialize them correctly anyway.
  buffer->prefix.header.vtable = reinterpret_cast<dynd_type_vtable*>(&dense_vtable);
  buffer->prefix.header.constructor = reinterpret_cast<dynd_type_constructor*>(&dynd_type_dense);
  assert(parameters.end == parameters.begin + 1);
  buffer->typemeta.parameter = *parameters.begin;
  // Return owned reference to the current buffer.
  return reinterpret_cast<dynd_type*>(buffer);
}

// The vtable for the type constructor 
struct dynd_dense_constructor_vtable : dynd_type_constructor_vtable {
  dynd_dense_constructor_vtable() dynd_noexcept {
    header.refcount.resource.release = dynd_abi_resource_never_release;
    dynd_atomic_store(&header.refcount.refcount, dynd_size_t(1u), dynd_memory_order_relaxed);
    header.header.allocated.base_ptr = this;
    header.header.allocated.size = sizeof(decltype(*this));
    header.header.build_interface = nullptr;
    entries.make = &make_dense;
  }
  ~dynd_dense_constructor_vtable() dynd_noexcept {
    assert(dynd_atomic_load(&header.refcount.refcount, dynd_memory_order_relaxed) == 1);
  }
};

dynd_dense_constructor_vtable dense_constructor_vtable{};

struct dynd_type_dense_impl : dynd_type_constructor {
  dynd_type_dense_impl() noexcept {
    refcount.resource.release = dynd_abi_resource_never_release;
    dynd_atomic_store(&refcount.refcount, dynd_size_t(1u), dynd_memory_order_relaxed);
    header.allocated.base_ptr = this;
    header.allocated.size = sizeof(decltype(*this));
    header.vtable = reinterpret_cast<dynd_type_constructor_vtable*>(&dense_constructor_vtable);
  }
  ~dynd_type_dense_impl() noexcept {
    assert(dynd_atomic_load(&refcount.refcount, dynd_memory_order_relaxed) == 1);
  }
};

extern "C" {

dynd_type_dense_impl dynd_type_dense{};

}
