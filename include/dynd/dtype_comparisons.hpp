//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DTYPE_COMPARISONS_HPP_
#define _DYND__DTYPE_COMPARISONS_HPP_

#include <dynd/auxiliary_data.hpp>
#include <dynd/kernels/single_compare_kernel_instance.hpp>


namespace dynd {

extern single_compare_operation_table_t builtin_dtype_comparisons_table[13];

}


#endif // _DYND__DTYPE_COMPARISONS_HPP_
