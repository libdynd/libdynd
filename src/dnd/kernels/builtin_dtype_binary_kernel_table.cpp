//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/kernels/builtin_dtype_binary_kernel_table.hpp>

using namespace std;
using namespace dynd;

binary_operation_t dynd::get_binary_operation_from_builtin_dtype_table(
                                specialized_binary_operation_table_t *builtin_optable,
                                const dtype& dt, intptr_t dst_fixedstride,
                                intptr_t src0_fixedstride, intptr_t src1_fixedstride)
{
    static int compress_type_id[builtin_type_id_count] = {-1, -1, -1, 0, 1, -1, -1, 2, 3, 4, 5, 6, 7};
    intptr_t element_size = dt.element_size();
    int cid = compress_type_id[dt.type_id()];

    // Pick out a specialized inner loop based on the strides
    if (dst_fixedstride == element_size) {
        if (src0_fixedstride == element_size) {
            if (src1_fixedstride == element_size) {
                return builtin_optable[cid][3];
            } else if (src1_fixedstride == 0) {
                return builtin_optable[cid][5];
            }
        } else if (src0_fixedstride == 0 && src1_fixedstride == element_size) {
            return builtin_optable[cid][4];
        }
    }

    if (src0_fixedstride == 0) {
        return builtin_optable[cid][1];
    } else if (src1_fixedstride == 0) {
        return builtin_optable[cid][2];
    } else {
        return builtin_optable[cid][0];
    }
}

