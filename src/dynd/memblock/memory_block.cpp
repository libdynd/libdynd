//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/memblock/array_memory_block.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/memblock/fixed_size_pod_memory_block.hpp>
#include <dynd/memblock/memmap_memory_block.hpp>
#include <dynd/memblock/memory_block.hpp>
#include <dynd/memblock/objectarray_memory_block.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/memblock/zeroinit_memory_block.hpp>

using namespace std;
using namespace dynd;

memory_block_data::~memory_block_data() {}

std::ostream &dynd::operator<<(std::ostream &o, memory_block_type_t mbt) {
  switch (mbt) {
  case external_memory_block_type:
    o << "external";
    break;
  case fixed_size_pod_memory_block_type:
    o << "fixed_size_pod";
    break;
  case pod_memory_block_type:
    o << "pod";
    break;
  case zeroinit_memory_block_type:
    o << "zeroinit";
    break;
  case objectarray_memory_block_type:
    o << "objectarray";
    break;
  case array_memory_block_type:
    o << "array";
    break;
  case memmap_memory_block_type:
    o << "memmap";
    break;
  default:
    o << "unknown memory_block_type(" << (int)mbt << ")";
  }
  return o;
}

void dynd::memory_block_debug_print(const memory_block_data *memblock, std::ostream &o, const std::string &indent) {
  if (memblock != NULL) {
    o << indent << "------ memory_block at " << (const void *)memblock << "\n";
    o << indent << " reference count: " << memblock->get_use_count() << "\n";
    o << indent << " type: " << (memory_block_type_t)memblock->m_type << "\n";
    switch ((memory_block_type_t)memblock->m_type) {
    case external_memory_block_type:
      external_memory_block_debug_print(memblock, o, indent);
      break;
    case fixed_size_pod_memory_block_type:
      fixed_size_pod_memory_block_debug_print(memblock, o, indent);
      break;
    case pod_memory_block_type:
      pod_memory_block_debug_print(memblock, o, indent);
      break;
    case zeroinit_memory_block_type:
      zeroinit_memory_block_debug_print(memblock, o, indent);
      break;
    case objectarray_memory_block_type:
      // objectarray_memory_block_debug_print(memblock, o, indent);
      break;
    case array_memory_block_type:
      array_memory_block_debug_print(memblock, o, indent);
      break;
    case memmap_memory_block_type:
      memmap_memory_block_debug_print(memblock, o, indent);
      break;
    }
    o << indent << "------" << endl;
  } else {
    o << indent << "------ NULL memory block" << endl;
  }
}
