//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DATE_EXPR_KERNELS_HPP_
#define _DYND__DATE_EXPR_KERNELS_HPP_

#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/dtypes/date_dtype.hpp>

namespace dynd {

expr_kernel_generator *make_strftime_kernelgen(const std::string& format);

} // namespace dynd

#endif // _DYND__DATE_EXPR_KERNELS_HPP_

