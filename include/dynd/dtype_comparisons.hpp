//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DTYPE_COMPARISONS_HPP_
#define _DYND__DTYPE_COMPARISONS_HPP_

#include <dynd/auxiliary_data.hpp>
#include <dynd/kernels/kernel_instance.hpp>


namespace dynd {

extern single_compare_operation_t builtin_dtype_comparisons_table[builtin_type_id_count-2][6];

}


#endif // _DYND__DTYPE_COMPARISONS_HPP_
