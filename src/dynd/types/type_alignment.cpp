//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/type_alignment.hpp>
#include <dynd/types/view_type.hpp>
#include <dynd/types/fixedbytes_type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

ndt::type ndt::make_unaligned(const ndt::type& value_type)
{
    if (value_type.get_data_alignment() > 1) {
        // Only do something if it requires alignment
        if (value_type.get_kind() != expr_kind) {
            return ndt::make_view(value_type, ndt::make_fixedbytes(value_type.get_data_size(), 1));
        } else {
            const ndt::type& sdt = value_type.storage_type();
            return ndt::type(value_type.tcast<base_expr_type>()->with_replaced_storage_type(ndt::make_view(sdt, ndt::make_fixedbytes(sdt.get_data_size(), 1))));
        }
    } else {
        return value_type;
    }
}
