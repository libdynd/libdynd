//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/dtypes/dtype_alignment.hpp>
#include <dnd/dtypes/view_dtype.hpp>
#include <dnd/dtype_assign.hpp>
#include <dnd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dnd;

dtype dnd::make_unaligned_dtype(const dtype& value_dtype)
{
    if (value_dtype.alignment() > 1) {
        // Only do something if it requires alignment
        if (value_dtype.kind() != expression_kind) {
            return make_view_dtype(value_dtype, make_bytes_dtype(value_dtype.itemsize(), 1));
        } else {
            const dtype& sdt = value_dtype.storage_dtype();
            return dtype(value_dtype.extended()->with_replaced_storage_dtype(make_view_dtype(sdt, make_bytes_dtype(sdt.itemsize(), 1))));
        }
    } else {
        return value_dtype;
    }
}
