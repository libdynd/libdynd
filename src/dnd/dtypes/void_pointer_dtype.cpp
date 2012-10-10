//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/dtypes/void_pointer_dtype.hpp>
#include <dnd/kernels/single_compare_kernel_instance.hpp>
#include <dnd/kernels/assignment_kernels.hpp>

#include <algorithm>

using namespace std;
using namespace dnd;

void dnd::void_pointer_dtype::print_element(std::ostream& o, const char *data) const
{
    uintptr_t target_ptr = *reinterpret_cast<const uintptr_t *>(data);
    o << "0x";
    hexadecimal_print(o, target_ptr);
}

void dnd::void_pointer_dtype::print_dtype(std::ostream& o) const {

    o << "pointer<void>";

}

dtype dnd::void_pointer_dtype::apply_linear_index(int ndim, const irange *indices, int dtype_ndim) const
{
    if (ndim == 0) {
        return dtype(this);
    } else {
        throw runtime_error("not implemented yet");
    }
}

bool dnd::void_pointer_dtype::is_lossless_assignment(const dtype& DND_UNUSED(dst_dt), const dtype& DND_UNUSED(src_dt)) const
{
    return false;
}

void dnd::void_pointer_dtype::get_single_compare_kernel(single_compare_kernel_instance& DND_UNUSED(out_kernel)) const {
    throw std::runtime_error("void_pointer_dtype::get_single_compare_kernel not supported yet");
}

bool dnd::void_pointer_dtype::operator==(const extended_dtype& rhs) const
{
    return rhs.type_id() == void_pointer_type_id;
}

void dnd::void_pointer_dtype::get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                assign_error_mode DND_UNUSED(errmode),
                unary_specialization_kernel_instance& out_kernel) const
{
    if (this == dst_dt.extended()) {
        if (src_dt.type_id() == void_type_id) {
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
