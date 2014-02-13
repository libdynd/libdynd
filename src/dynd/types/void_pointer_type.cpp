//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/void_pointer_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/exceptions.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

void void_pointer_type::print_data(std::ostream& o, const char *DYND_UNUSED(metadata), const char *data) const
{
    uintptr_t target_ptr = *reinterpret_cast<const uintptr_t *>(data);
    o << "0x";
    hexadecimal_print(o, target_ptr);
}

void void_pointer_type::print_type(std::ostream& o) const {

    o << "pointer(void)";
}

bool void_pointer_type::is_lossless_assignment(const ndt::type& DYND_UNUSED(dst_tp), const ndt::type& DYND_UNUSED(src_tp)) const
{
    return false;
}

bool void_pointer_type::operator==(const base_type& rhs) const
{
    return rhs.get_type_id() == void_pointer_type_id;
}

size_t void_pointer_type::make_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& dst_tp, const char *dst_metadata,
                const ndt::type& src_tp, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_tp.extended()) {
        if (src_tp.get_type_id() == void_pointer_type_id) {
            return ::make_pod_typed_data_assignment_kernel(out, offset_out,
                    get_data_size(), get_data_alignment(),
                    kernreq);
        } else if (!src_tp.is_builtin()) {
            src_tp.extended()->make_assignment_kernel(out, offset_out,
                            dst_tp, dst_metadata,
                            src_tp, src_metadata,
                            kernreq, errmode, ectx);
        }
    }

    stringstream ss;
    ss << "Cannot assign from " << src_tp << " to " << dst_tp;
    throw dynd::type_error(ss.str());
}

