//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/void_pointer_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/exceptions.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

void void_pointer_type::print_data(std::ostream& o, const char *DYND_UNUSED(arrmeta), const char *data) const
{
    uintptr_t target_ptr = *reinterpret_cast<const uintptr_t *>(data);
    o << "0x";
    hexadecimal_print(o, target_ptr);
}

void void_pointer_type::print_type(std::ostream& o) const {

    o << "pointer[void]";
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
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    if (this == dst_tp.extended()) {
        if (src_tp.get_type_id() == void_pointer_type_id) {
            return ::make_pod_typed_data_assignment_kernel(ckb, ckb_offset,
                    get_data_size(), get_data_alignment(),
                    kernreq);
        } else if (!src_tp.is_builtin()) {
            src_tp.extended()->make_assignment_kernel(
                ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta,
                kernreq, ectx);
        }
    }

    stringstream ss;
    ss << "Cannot assign from " << src_tp << " to " << dst_tp;
    throw dynd::type_error(ss.str());
}
