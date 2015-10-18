//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/type.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/bytes_assignment_kernels.hpp>
#include <dynd/types/bytes_type.hpp>

using namespace std;
using namespace dynd;

/////////////////////////////////////////
// blockref bytes to blockref bytes assignment

namespace {
struct blockref_bytes_kernel : nd::base_kernel<blockref_bytes_kernel, 1> {
  size_t dst_alignment, src_alignment;
  const bytes_type_arrmeta *dst_arrmeta, *src_arrmeta;

  /** Does a single blockref-string copy */
  void single(char *dst, char *const *src)
  {
    const bytes_type_arrmeta *dst_md = dst_arrmeta;
    const bytes_type_arrmeta *src_md = src_arrmeta;
    bytes_type_data *dst_d = reinterpret_cast<bytes_type_data *>(dst);
    bytes_type_data *src_d = reinterpret_cast<bytes_type_data *>(src[0]);

    if (dst_d->begin() != NULL) {
      throw runtime_error("Cannot assign to an already initialized dynd string");
    } else if (src_d->begin() == NULL) {
      // Allow uninitialized -> uninitialized assignment as a special case, for
      // (future) missing data support
      return;
    }

    // If the blockrefs are different, require a copy operation
    if (dst_md->blockref != src_md->blockref) {
      char *dst_begin = NULL, *dst_end = NULL;
      char *src_begin = src_d->begin();
      char *src_end = src_d->end();
      memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(dst_md->blockref);

      allocator->allocate(dst_md->blockref, src_end - src_begin, dst_alignment, &dst_begin, &dst_end);
      memcpy(dst_begin, src_begin, src_end - src_begin);

      // Set the output
      dst_d->assign(dst_begin, dst_end - dst_begin);
    } else if (dst_alignment <= src_alignment) {
      // Copy the pointers from the source bytes
      *dst_d = *src_d;
    } else {
      throw runtime_error("Attempted to reference source data when increasing bytes alignment");
    }
  }
};
} // anonymous namespace

size_t dynd::make_blockref_bytes_assignment_kernel(void *ckb, intptr_t ckb_offset, size_t dst_alignment,
                                                   const char *dst_arrmeta, size_t src_alignment,
                                                   const char *src_arrmeta, kernel_request_t kernreq,
                                                   const eval::eval_context *DYND_UNUSED(ectx))
{
  // Adapt the incoming request to a 'single' kernel
  blockref_bytes_kernel *e = blockref_bytes_kernel::make(ckb, kernreq, ckb_offset);
  e->dst_alignment = dst_alignment;
  e->src_alignment = src_alignment;
  e->dst_arrmeta = reinterpret_cast<const bytes_type_arrmeta *>(dst_arrmeta);
  e->src_arrmeta = reinterpret_cast<const bytes_type_arrmeta *>(src_arrmeta);
  return ckb_offset;
}

/////////////////////////////////////////
// fixed_bytes to blockref bytes assignment

namespace {
struct fixed_bytes_to_blockref_bytes_kernel : nd::base_kernel<fixed_bytes_to_blockref_bytes_kernel, 1> {
  size_t dst_alignment;
  intptr_t src_data_size;
  size_t src_alignment;
  const bytes_type_arrmeta *dst_arrmeta;

  /** Does a single fixed-bytes copy */
  void single(char *dst, char *const *src)
  {
    const bytes_type_arrmeta *dst_md = dst_arrmeta;
    // TODO: With some additional mechanism to track the source memory block,
    // could
    //       avoid copying the bytes data.
    char *dst_begin = NULL, *dst_end = NULL;
    char *src_begin = src[0];
    char *src_end = src_begin + src_data_size;
    bytes_type_data *dst_d = reinterpret_cast<bytes_type_data *>(dst);

    if (dst_d->begin() != NULL) {
      throw runtime_error("Cannot assign to an already initialized dynd string");
    }

    memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(dst_md->blockref);

    allocator->allocate(dst_md->blockref, src_end - src_begin, dst_alignment, &dst_begin, &dst_end);
    memcpy(dst_begin, src_begin, src_end - src_begin);

    // Set the output
    dst_d->assign(dst_begin, dst_end - dst_begin);
  }
};
} // anonymous namespace

size_t dynd::make_fixed_bytes_to_blockref_bytes_assignment_kernel(void *ckb, intptr_t ckb_offset, size_t dst_alignment,
                                                                  const char *dst_arrmeta, intptr_t src_data_size,
                                                                  size_t src_alignment, kernel_request_t kernreq,
                                                                  const eval::eval_context *DYND_UNUSED(ectx))
{
  // Adapt the incoming request to a 'single' kernel
  fixed_bytes_to_blockref_bytes_kernel *e = fixed_bytes_to_blockref_bytes_kernel::make(ckb, kernreq, ckb_offset);
  e->dst_alignment = dst_alignment;
  e->src_data_size = src_data_size;
  e->src_alignment = src_alignment;
  e->dst_arrmeta = reinterpret_cast<const bytes_type_arrmeta *>(dst_arrmeta);
  return ckb_offset;
}
