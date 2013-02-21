//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/void_pointer_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/exceptions.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

void void_pointer_dtype::print_data(std::ostream& o, const char *DYND_UNUSED(metadata), const char *data) const
{
    uintptr_t target_ptr = *reinterpret_cast<const uintptr_t *>(data);
    o << "0x";
    hexadecimal_print(o, target_ptr);
}

void void_pointer_dtype::print_dtype(std::ostream& o) const {

    o << "pointer<void>";
}

void void_pointer_dtype::get_shape(size_t DYND_UNUSED(i),
                intptr_t *DYND_UNUSED(out_shape)) const
{
}

void void_pointer_dtype::get_shape(size_t DYND_UNUSED(i),
                intptr_t *DYND_UNUSED(out_shape), const char *DYND_UNUSED(metadata)) const
{
}

bool void_pointer_dtype::is_lossless_assignment(const dtype& DYND_UNUSED(dst_dt), const dtype& DYND_UNUSED(src_dt)) const
{
    return false;
}

bool void_pointer_dtype::operator==(const base_dtype& rhs) const
{
    return rhs.get_type_id() == void_pointer_type_id;
}

size_t void_pointer_dtype::make_assignment_kernel(
                assignment_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_dt.extended()) {
        if (src_dt.get_type_id() == void_pointer_type_id) {
            return ::make_pod_dtype_assignment_kernel(out, offset_out,
                    get_data_size(), get_alignment(),
                    kernreq);
        } else if (!src_dt.is_builtin()) {
            src_dt.extended()->make_assignment_kernel(out, offset_out,
                            dst_dt, dst_metadata,
                            src_dt, src_metadata,
                            kernreq, errmode, ectx);
        }
    }

    stringstream ss;
    ss << "Cannot assign from " << src_dt << " to " << dst_dt;
    throw runtime_error(ss.str());
}

