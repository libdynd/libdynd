//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

#include <dynd/memblock/memmap_memory_block.hpp>

using namespace std;
using namespace dynd;

intrusive_ptr<memory_block_data> dynd::make_memmap_memory_block(const std::string &filename, uint32_t access,
                                                                char **out_pointer, intptr_t *out_size, intptr_t begin,
                                                                intptr_t end) {
  memmap_memory_block *pmb = new memmap_memory_block(filename, access, out_pointer, out_size, begin, end);
  return intrusive_ptr<memory_block_data>(reinterpret_cast<memory_block_data *>(pmb), false);
}

void memmap_memory_block::debug_print(std::ostream &o, const std::string &indent) {
  o << indent << "------ memory_block at " << static_cast<const void *>(this) << "\n";
  o << indent << " reference count: " << m_use_count << "\n";
  o << indent << " filename: " << m_filename << "\n";
  o << indent << " begin: " << m_begin << "\n";
  o << indent << " end: " << m_end << "\n";
  o << indent << "------" << endl;
}
