//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__DTYPE_COMPARISONS_HPP_
#define _DND__DTYPE_COMPARISONS_HPP_

#include <dnd/auxiliary_data.hpp>
#include <dnd/kernels/kernel_instance.hpp>



namespace dnd {

enum comparison_id_t {
    less_id,
    less_equal_id,
    equal_id,
    not_equal_id,
    greater_equal_id,
    greater_id
};

typedef compare_operation_t comparisons_table_t[6];

extern comparisons_table_t builtin_dtype_comparisons_table[8];

}


#endif // _DND__DTYPE_COMPARISONS_HPP_
