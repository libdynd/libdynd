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
