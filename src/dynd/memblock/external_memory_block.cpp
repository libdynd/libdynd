//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <string>

#include <dynd/memblock/external_memory_block.hpp>

using namespace std;
using namespace dynd;

intrusive_ptr<memory_block_data> dynd::make_external_memory_block(void *object, external_memory_block_free_t free_fn) {
  external_memory_block *emb = new external_memory_block(object, free_fn);
  return intrusive_ptr<memory_block_data>(reinterpret_cast<memory_block_data *>(emb), false);
}

void external_memory_block::debug_print(std::ostream &o, const std::string &indent) {
  o << indent << "------ memory_block at " << static_cast<const void *>(this) << "\n";
  o << indent << " reference count: " << m_use_count << "\n";
  o << indent << " object void pointer: " << m_object << "\n";
  o << indent << " free function: " << (const void *)m_free_fn << "\n";
  o << indent << "------" << endl;
}
