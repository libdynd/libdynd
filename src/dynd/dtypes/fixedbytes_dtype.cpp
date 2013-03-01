//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dynd/dtypes/fixedbytes_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>

using namespace std;
using namespace dynd;

fixedbytes_dtype::fixedbytes_dtype(intptr_t data_size, intptr_t alignment)
    : base_bytes_dtype(fixedbytes_type_id, bytes_kind, data_size,
                    alignment, dtype_flag_scalar, 0)
{
    if (alignment > data_size) {
        std::stringstream ss;
        ss << "Cannot make a fixedbytes<" << data_size << "," << alignment << "> dtype, its alignment is greater than its size";
        throw std::runtime_error(ss.str());
    }
    if (alignment != 1 && alignment != 2 && alignment != 4 && alignment != 8 && alignment != 16) {
        std::stringstream ss;
        ss << "Cannot make a fixedbytes<" << data_size << "," << alignment << "> dtype, its alignment is not a small power of two";
        throw std::runtime_error(ss.str());
    }
    if ((data_size&(alignment-1)) != 0) {
        std::stringstream ss;
        ss << "Cannot make a fixedbytes<" << data_size << "," << alignment << "> dtype, its alignment does not divide into its element size";
        throw std::runtime_error(ss.str());
    }
}

fixedbytes_dtype::~fixedbytes_dtype()
{
}

void fixedbytes_dtype::get_bytes_range(const char **out_begin, const char**out_end,
                const char *DYND_UNUSED(metadata), const char *data) const
{
    *out_begin = data;
    *out_end = data + get_data_size();
}

void fixedbytes_dtype::print_data(std::ostream& o, const char *DYND_UNUSED(metadata), const char *data) const
{
    o << "0x";
    hexadecimal_print(o, data, get_data_size());
}

void fixedbytes_dtype::print_dtype(std::ostream& o) const
{
    o << "fixedbytes<" << get_data_size() << "," << get_alignment() << ">";
}

bool fixedbytes_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.extended() == this) {
            return true;
        } else if (src_dt.get_type_id() == fixedbytes_type_id) {
            const fixedbytes_dtype *src_fs = static_cast<const fixedbytes_dtype*>(src_dt.extended());
            return get_data_size() == src_fs->get_data_size();
        } else {
            return false;
        }
    } else {
        return false;
    }
}

bool fixedbytes_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != fixedbytes_type_id) {
        return false;
    } else {
        const fixedbytes_dtype *dt = static_cast<const fixedbytes_dtype*>(&rhs);
        return get_data_size() == dt->get_data_size() && get_alignment() == dt->get_alignment();
    }
}

size_t fixedbytes_dtype::make_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_dt.extended()) {
        switch (src_dt.get_type_id()) {
            case fixedbytes_type_id: {
                const fixedbytes_dtype *src_fs = static_cast<const fixedbytes_dtype *>(src_dt.extended());
                if (get_data_size() != src_fs->get_data_size()) {
                    throw runtime_error("cannot assign to a fixedbytes dtype of a different size");
                }
                return ::make_pod_dtype_assignment_kernel(out, offset_out,
                                get_data_size(), std::min(get_alignment(), src_fs->get_alignment()),
                                kernreq);
            }
            default: {
                return src_dt.extended()->make_assignment_kernel(out, offset_out,
                                dst_dt, dst_metadata,
                                src_dt, src_metadata,
                                kernreq, errmode, ectx);
            }
        }
    } else {
        stringstream ss;
        ss << "Cannot assign from " << src_dt << " to " << dst_dt;
        throw runtime_error(ss.str());
    }
}

