//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/dtypes/fixedbytes_dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

dtype dynd::make_unaligned_dtype(const dtype& value_dtype)
{
    if (value_dtype.alignment() > 1) {
        // Only do something if it requires alignment
        if (value_dtype.kind() != expression_kind) {
            return make_view_dtype(value_dtype, make_fixedbytes_dtype(value_dtype.element_size(), 1));
        } else {
            const dtype& sdt = value_dtype.storage_dtype();
            return dtype(static_cast<const extended_expression_dtype *>(value_dtype.extended())->with_replaced_storage_dtype(make_view_dtype(sdt, make_fixedbytes_dtype(sdt.element_size(), 1))));
        }
    } else {
        return value_dtype;
    }
}
