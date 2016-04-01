//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <string>

#include <dynd/memblock/external_memory_block.hpp>

using namespace dynd;

intrusive_ptr<memory_block_data> dynd::make_external_memory_block(void *object, external_memory_block_free_t free_fn)
{
  external_memory_block *emb = new external_memory_block(object, free_fn);
  return intrusive_ptr<memory_block_data>(reinterpret_cast<memory_block_data *>(emb), false);
}

namespace dynd {
namespace detail {

  void free_external_memory_block(memory_block_data *memblock)
  {
    external_memory_block *emb = reinterpret_cast<external_memory_block *>(memblock);
    emb->m_free_fn(emb->m_object);
    delete emb;
  }
}
} // namespace dynd::detail

void dynd::external_memory_block_debug_print(const memory_block_data *memblock, std::ostream &o,
                                             const std::string &indent)
{
  const external_memory_block *mb = reinterpret_cast<const external_memory_block *>(memblock);
  o << indent << " object void pointer: " << mb->m_object << "\n";
  o << indent << " free function: " << (const void *)mb->m_free_fn << "\n";
}
