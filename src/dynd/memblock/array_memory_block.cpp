//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/memblock/array_memory_block.hpp>
#include <dynd/types/base_memory_type.hpp>
#include <dynd/array.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/shape_tools.hpp>

using namespace std;
using namespace dynd;

namespace dynd {
namespace detail {

  void free_array_memory_block(memory_block_data *memblock)
  {
    array_preamble *preamble = reinterpret_cast<array_preamble *>(memblock);
    char *arrmeta = reinterpret_cast<char *>(preamble + 1);

    // Call the data destructor if necessary (i.e. the nd::array owns
    // the data memory, and the type has a data destructor)
    if (preamble->data.ref == NULL && !preamble->is_builtin_type() &&
        (preamble->m_type->get_flags() & type_flag_destructor) != 0) {
      preamble->m_type->data_destruct(arrmeta, preamble->data.ptr);
    }

    // Free the ndobject data if it wasn't allocated together with the memory block
    if (preamble->data.ref == NULL && !preamble->is_builtin_type() && !preamble->m_type->is_expression()) {
      const ndt::type &dtp = preamble->m_type->get_type_at_dimension(NULL, preamble->m_type->get_ndim());
      if (dtp.get_kind() == memory_kind) {
        dtp.extended<ndt::base_memory_type>()->data_free(preamble->data.ptr);
      }
    }

    // Free the references contained in the arrmeta
    if (!preamble->is_builtin_type()) {
      preamble->m_type->arrmeta_destruct(arrmeta);
      base_type_decref(preamble->m_type);
    }

    // Free the reference to the nd::array data
    if (preamble->data.ref != NULL) {
      memory_block_decref(preamble->data.ref);
    }

    // Finally free the memory block itself
    free(reinterpret_cast<void *>(memblock));
  }
}
} // namespace dynd::detail

memory_block_ptr dynd::make_array_memory_block(size_t arrmeta_size)
{
  char *result = (char *)malloc(sizeof(memory_block_data) + sizeof(array_preamble) + arrmeta_size);
  if (result == 0) {
    throw bad_alloc();
  }
  // Zero out all the arrmeta to start
  memset(result + sizeof(memory_block_data), 0, sizeof(array_preamble) + arrmeta_size);
  return memory_block_ptr(new (result) memory_block_data(1, array_memory_block_type), false);
}

memory_block_ptr dynd::make_array_memory_block(size_t arrmeta_size, size_t extra_size, size_t extra_alignment,
                                               char **out_extra_ptr)
{
  size_t extra_offset =
      inc_to_alignment(sizeof(memory_block_data) + sizeof(array_preamble) + arrmeta_size, extra_alignment);
  char *result = (char *)malloc(extra_offset + extra_size);
  if (result == 0) {
    throw bad_alloc();
  }
  // Zero out all the arrmeta to start
  memset(result + sizeof(memory_block_data), 0, sizeof(array_preamble) + arrmeta_size);
  // Return a pointer to the extra allocated memory
  *out_extra_ptr = result + extra_offset;
  return memory_block_ptr(new (result) memory_block_data(1, array_memory_block_type), false);
}

memory_block_ptr dynd::shallow_copy_array_memory_block(const memory_block_ptr &ndo)
{
  // Allocate the new memory block.
  const array_preamble *preamble = reinterpret_cast<const array_preamble *>(ndo.get());
  size_t arrmeta_size = 0;
  if (!preamble->is_builtin_type()) {
    arrmeta_size = preamble->m_type->get_arrmeta_size();
  }
  memory_block_ptr result = make_array_memory_block(arrmeta_size);
  array_preamble *result_preamble = reinterpret_cast<array_preamble *>(result.get());

  // Clone the data pointer
  result_preamble->data.ptr = preamble->data.ptr;
  result_preamble->data.ref = preamble->data.ref;
  if (result_preamble->data.ref == NULL) {
    result_preamble->data.ref = ndo.get();
  }
  memory_block_incref(result_preamble->data.ref);

  // Copy the flags
  result_preamble->m_flags = preamble->m_flags;

  // Clone the type
  result_preamble->m_type = preamble->m_type;
  if (!preamble->is_builtin_type()) {
    base_type_incref(preamble->m_type);
    preamble->m_type->arrmeta_copy_construct(reinterpret_cast<char *>(result.get()) + sizeof(array_preamble),
                                             reinterpret_cast<const char *>(ndo.get()) + sizeof(array_preamble),
                                             ndo.get());
  }

  return result;
}

void dynd::array_memory_block_debug_print(const memory_block_data *memblock, std::ostream &o, const std::string &indent)
{
  const array_preamble *preamble = reinterpret_cast<const array_preamble *>(memblock);
  if (preamble->m_type != NULL) {
    ndt::type tp = preamble->is_builtin_type() ? ndt::type(preamble->get_type_id()) : ndt::type(preamble->m_type, true);
    o << indent << " type: " << tp << "\n";
  } else {
    o << indent << " uninitialized nd::array\n";
  }
}
