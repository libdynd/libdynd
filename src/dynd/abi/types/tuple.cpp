#include <cassert>
#include <iterator>

#define DYND_ABI_TYPES_TUPLE_CPP

#include "dynd/abi/metadata.h"
#include "dynd/abi/types/tuple.h"

static dynd_type_range tuple_parameters(dynd_type_header_impl *type_header) noexcept {
  dynd_type_tuple_typemeta_header *typemeta = reinterpret_cast<dynd_type_tuple_typemeta_header*>(dynd_type_metadata(type_header));
  dynd_size_t num_entries = typemeta->num_entries;
  dynd_type **first_child = reinterpret_cast<dynd_type**>(typemeta + 1);
  return dynd_type_range{first_child, first_child + num_entries};
}

static dynd_size_t tuple_alignment(dynd_type_header_impl *type_header) noexcept {
  dynd_type_range children = tuple_parameters(type_header);
  // Follow C++ convention for alignment of empty structs.
  dynd_size_t alignment = 1;
  // Alignment of the tuple as a whole is the max of the alignments
  // of its children types.
  while (children.begin < children.end) {
    dynd_size_t child_alignment = (*children.begin)->header.vtable->entries.alignment(&(*children.begin)->header);
    if (child_alignment > alignment) alignment = child_alignment;
    children.begin++;
  }
  return alignment;
}

namespace {
struct dynd_tuple_concrete_vtable : dynd_type_vtable {
  dynd_tuple_concrete_vtable() dynd_noexcept {
    header.refcount.resource.release = dynd_abi_resource_never_release;
    dynd_atomic_store(&header.refcount.refcount, dynd_size_t(1u), dynd_memory_order_relaxed);
    header.header.allocated.base_ptr = this;
    header.header.allocated.size = sizeof(dynd_tuple_concrete_vtable);
    header.header.build_interface = nullptr;
    entries.alignment = tuple_alignment;
    entries.parameters = tuple_parameters;
    entries.superclasses = dynd_type_range_empty;
  }
  ~dynd_tuple_concrete_vtable() dynd_noexcept {
    assert(dynd_atomic_load(&header.refcount.refcount, dynd_memory_order_relaxed) == 1);
  }
};
}

static dynd_tuple_concrete_vtable tuple_vtable{};

namespace {
struct dynd_type_tuple_impl;
}

extern "C" {
extern dynd_type_tuple_impl dynd_type_tuple;
}

static dynd_type *make_tuple(dynd_type_constructor_header *, dynd_type_range parameters) noexcept {
  dynd_size_t num_entries = std::distance(parameters.begin, parameters.end);
  dynd_type_tuple_concrete_header *buffer = reinterpret_cast<dynd_type_tuple_concrete_header*>(dynd_malloc_buffer(sizeof(dynd_type_tuple_concrete_header) + 8 * num_entries));
  dynd_atomic_store(&(buffer->prefix.refcount.refcount), dynd_size_t(1u), dynd_memory_order_relaxed);
  // These will likely be never used, but initialize them correctly anyway.
  buffer->prefix.header.vtable = reinterpret_cast<dynd_type_vtable*>(&tuple_vtable);
  buffer->prefix.header.constructor = reinterpret_cast<dynd_type_constructor*>(&dynd_type_tuple);
  buffer->typemeta_header.num_entries = num_entries;
  dynd_type **parameters_copy = reinterpret_cast<dynd_type**>(buffer + 1);
  while (parameters.begin < parameters.end) {
    *parameters_copy = *parameters.begin;
    parameters_copy++;
    parameters.begin++;
  }
  // Return owned reference to the current buffer.
  return reinterpret_cast<dynd_type*>(buffer);
}

// The vtable for the type constructor
namespace {
struct dynd_tuple_constructor_vtable : dynd_type_constructor_vtable {
  dynd_tuple_constructor_vtable() dynd_noexcept {
    header.refcount.resource.release = dynd_abi_resource_never_release;
    dynd_atomic_store(&header.refcount.refcount, dynd_size_t(1u), dynd_memory_order_relaxed);
    header.header.allocated.base_ptr = this;
    header.header.allocated.size = sizeof(decltype(*this));
    header.header.build_interface = nullptr;
    entries.make = &make_tuple;
  }
  ~dynd_tuple_constructor_vtable() dynd_noexcept {
    assert(dynd_atomic_load(&header.refcount.refcount, dynd_memory_order_relaxed) == 1);
  }
};
}

static dynd_tuple_constructor_vtable tuple_constructor_vtable{};

namespace {
struct dynd_type_tuple_impl : dynd_type_constructor {
  dynd_type_tuple_impl() noexcept {
    refcount.resource.release = dynd_abi_resource_never_release;
    dynd_atomic_store(&refcount.refcount, dynd_size_t(1u), dynd_memory_order_relaxed);
    header.allocated.base_ptr = this;
    header.allocated.size = sizeof(decltype(*this));
    header.vtable = reinterpret_cast<dynd_type_constructor_vtable*>(&tuple_constructor_vtable);
  }
  ~dynd_type_tuple_impl() noexcept {
    assert(dynd_atomic_load(&refcount.refcount, dynd_memory_order_relaxed) == 1);
  }
};
}

extern "C" {

dynd_type_tuple_impl dynd_type_tuple{};

}
