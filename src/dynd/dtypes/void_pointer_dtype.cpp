//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/void_pointer_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/exceptions.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

void dynd::void_pointer_dtype::print_data(std::ostream& o, const char *DYND_UNUSED(metadata), const char *data) const
{
    uintptr_t target_ptr = *reinterpret_cast<const uintptr_t *>(data);
    o << "0x";
    hexadecimal_print(o, target_ptr);
}

void dynd::void_pointer_dtype::print_dtype(std::ostream& o) const {

    o << "pointer<void>";
}

dtype dynd::void_pointer_dtype::apply_linear_index(int nindices, const irange *DYND_UNUSED(indices), int current_i, const dtype& DYND_UNUSED(root_dt)) const
{
    if (nindices == 0) {
        return dtype(this, true);
    } else {
        throw too_many_indices(dtype(this, true), current_i + nindices, current_i);
    }
}

void dynd::void_pointer_dtype::get_shape(size_t DYND_UNUSED(i), std::vector<intptr_t>& DYND_UNUSED(out_shape)) const
{
}

bool dynd::void_pointer_dtype::is_lossless_assignment(const dtype& DYND_UNUSED(dst_dt), const dtype& DYND_UNUSED(src_dt)) const
{
    return false;
}

void dynd::void_pointer_dtype::get_single_compare_kernel(kernel_instance<compare_operations_t>& DYND_UNUSED(out_kernel)) const {
    throw std::runtime_error("void_pointer_dtype::get_single_compare_kernel not supported yet");
}

bool dynd::void_pointer_dtype::operator==(const base_dtype& rhs) const
{
    return rhs.get_type_id() == void_pointer_type_id;
}

void dynd::void_pointer_dtype::get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                assign_error_mode DYND_UNUSED(errmode),
                kernel_instance<unary_operation_pair_t>& out_kernel) const
{
    if (this == dst_dt.extended()) {
        if (src_dt.get_type_id() == void_type_id) {
            // Get a POD assignment kernel. The code handling the blockref should see
            // that this kernel doesn't define a kernel_api, and raise an error if
            // a copy is attempted instead of maintaining existing blockrefs.
            // TODO: Validate this, needs more work fleshing out blockrefs in general.
            get_pod_dtype_assignment_kernel(sizeof(void *), sizeof(void *), out_kernel);
        }
    }

    stringstream ss;
    ss << "Cannot assign from " << src_dt << " to " << dst_dt;
    throw runtime_error(ss.str());
}
