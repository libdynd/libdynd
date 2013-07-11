//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/dtypes/fixedbytes_dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

ndt::type ndt::make_unaligned_dtype(const ndt::type& value_type)
{
    if (value_type.get_data_alignment() > 1) {
        // Only do something if it requires alignment
        if (value_type.get_kind() != expression_kind) {
            return make_view_dtype(value_type, make_fixedbytes_dtype(value_type.get_data_size(), 1));
        } else {
            const ndt::type& sdt = value_type.storage_type();
            return ndt::type(static_cast<const base_expression_type *>(value_type.extended())->with_replaced_storage_type(make_view_dtype(sdt, make_fixedbytes_dtype(sdt.get_data_size(), 1))));
        }
    } else {
        return value_type;
    }
}
