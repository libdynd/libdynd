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

  /** Does a single blockref-string copy */
  void single(char *dst, char *const *src)
  {
    *reinterpret_cast<dynd::bytes *>(dst) = *reinterpret_cast<dynd::bytes *>(src[0]);
  }
};
} // anonymous namespace

size_t dynd::make_blockref_bytes_assignment_kernel(void *ckb, intptr_t ckb_offset, size_t dst_alignment,
                                                   const char *DYND_UNUSED(dst_arrmeta), size_t src_alignment,
                                                   const char *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                                   const eval::eval_context *DYND_UNUSED(ectx))
{
  // Adapt the incoming request to a 'single' kernel
  blockref_bytes_kernel *e = blockref_bytes_kernel::make(ckb, kernreq, ckb_offset);
  e->dst_alignment = dst_alignment;
  e->src_alignment = src_alignment;
  return ckb_offset;
}

/////////////////////////////////////////
// fixed_bytes to blockref bytes assignment

namespace {
struct fixed_bytes_to_blockref_bytes_kernel : nd::base_kernel<fixed_bytes_to_blockref_bytes_kernel, 1> {
  size_t dst_alignment;
  intptr_t src_data_size;
  size_t src_alignment;

  /** Does a single fixed-bytes copy */
  void single(char *dst, char *const *src)
  {
    // TODO: With some additional mechanism to track the source memory block,
    // could avoid copying the bytes data.
    char *src_begin = src[0];
    char *src_end = src_begin + src_data_size;
    bytes_type_data *dst_d = reinterpret_cast<bytes_type_data *>(dst);

    if (dst_d->begin() != NULL) {
      throw runtime_error("Cannot assign to an already initialized dynd string");
    }

    dst_d->assign(src_begin, src_end - src_begin);
  }
};
} // anonymous namespace

size_t dynd::make_fixed_bytes_to_blockref_bytes_assignment_kernel(void *ckb, intptr_t ckb_offset, size_t dst_alignment,
                                                                  const char *DYND_UNUSED(dst_arrmeta),
                                                                  intptr_t src_data_size, size_t src_alignment,
                                                                  kernel_request_t kernreq,
                                                                  const eval::eval_context *DYND_UNUSED(ectx))
{
  // Adapt the incoming request to a 'single' kernel
  fixed_bytes_to_blockref_bytes_kernel *e = fixed_bytes_to_blockref_bytes_kernel::make(ckb, kernreq, ckb_offset);
  e->dst_alignment = dst_alignment;
  e->src_data_size = src_data_size;
  e->src_alignment = src_alignment;
  return ckb_offset;
}
