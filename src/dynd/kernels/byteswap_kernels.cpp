//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>

#include <dynd/kernels/byteswap_kernels.hpp>

using namespace std;
using namespace dynd;

size_t dynd::make_byteswap_assignment_function(void *ckb, intptr_t ckb_offset, intptr_t data_size, intptr_t,
                                               kernel_request_t kernreq)
{
  // Otherwise use the general case ckernel
  nd::byteswap_ck::make(ckb, kernreq, ckb_offset, data_size);
  return ckb_offset;
}

size_t dynd::make_pairwise_byteswap_assignment_function(void *ckb, intptr_t ckb_offset, intptr_t data_size, intptr_t,
                                                        kernel_request_t kernreq)
{
  // Otherwise use the general case ckernel
  nd::pairwise_byteswap_ck::make(ckb, kernreq, ckb_offset, data_size);
  return ckb_offset;
}

nd::callable nd::byteswap::make() { return callable::make<byteswap_ck>(ndt::type("(Any) -> Any")); }

struct nd::byteswap nd::byteswap;
