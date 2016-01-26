//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>
#include <dynd/types/struct_type.hpp>

namespace dynd {

/**
 * Gets a kernel which converts from one struct to another.
 *
 * \param dst_struct_tp  The struct-kind dtype of the destination.
 * \param src_struct_tp  The struct-kind dtype of the source.
 */
DYND_API void make_struct_assignment_kernel(nd::kernel_builder *ckb, const ndt::type &dst_struct_tp,
                                            const char *dst_arrmeta, const ndt::type &src_struct_tp,
                                            const char *src_arrmeta, kernel_request_t kernreq);

} // namespace dynd
