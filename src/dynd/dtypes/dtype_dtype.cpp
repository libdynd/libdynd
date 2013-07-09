//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/dtypes/dtype_dtype.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

dtype_dtype::dtype_dtype()
    : base_dtype(dtype_type_id, custom_kind, sizeof(const base_dtype *),
                    sizeof(const base_dtype *),
                    type_flag_scalar|type_flag_zeroinit|type_flag_destructor,
                    0, 0)
{
}

dtype_dtype::~dtype_dtype()
{
}

void dtype_dtype::print_data(std::ostream& o,
                const char *DYND_UNUSED(metadata), const char *data) const
{
    const dtype_dtype_data *ddd = reinterpret_cast<const dtype_dtype_data *>(data);
    // This tests avoids the atomic increment/decrement of
    // always constructing a dtype object
    if (is_builtin_type(ddd->dt)) {
        o << ndt::type(ddd->dt, true);
    } else {
        ddd->dt->print_dtype(o);
    }
}

void dtype_dtype::print_dtype(std::ostream& o) const
{
    o << "type";
}

bool dtype_dtype::operator==(const base_dtype& rhs) const
{
    return this == &rhs || rhs.get_type_id() == dtype_type_id;
}

void dtype_dtype::metadata_default_construct(char *DYND_UNUSED(metadata),
                size_t DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const
{
}

void dtype_dtype::metadata_copy_construct(char *DYND_UNUSED(dst_metadata),
                const char *DYND_UNUSED(src_metadata), memory_block_data *DYND_UNUSED(embedded_reference)) const
{
}

void dtype_dtype::metadata_reset_buffers(char *DYND_UNUSED(metadata)) const
{
}

void dtype_dtype::metadata_finalize_buffers(char *DYND_UNUSED(metadata)) const
{
}

void dtype_dtype::metadata_destruct(char *DYND_UNUSED(metadata)) const
{
}

void dtype_dtype::data_destruct(const char *DYND_UNUSED(metadata), char *data) const
{
    const base_dtype *bd = reinterpret_cast<dtype_dtype_data *>(data)->dt;
    if (!is_builtin_type(bd)) {
        base_dtype_decref(bd);
    }
}

void dtype_dtype::data_destruct_strided(const char *DYND_UNUSED(metadata), char *data,
                intptr_t stride, size_t count) const
{
    for (size_t i = 0; i != count; ++i, data += stride) {
        const base_dtype *bd = reinterpret_cast<dtype_dtype_data *>(data)->dt;
        if (!is_builtin_type(bd)) {
            base_dtype_decref(bd);
        }
    }
}

static void dtype_assignment_kernel_single(char *dst, const char *src,
                kernel_data_prefix *DYND_UNUSED(extra))
{
    // Free the destination reference
    base_dtype_xdecref(reinterpret_cast<const dtype_dtype_data *>(dst)->dt);
    // Copy the pointer and count the reference
    const base_dtype *bd = reinterpret_cast<const dtype_dtype_data *>(src)->dt;
    reinterpret_cast<dtype_dtype_data *>(dst)->dt = bd;
    base_dtype_xincref(bd);
}

namespace {
    struct string_to_dtype_kernel_extra {
        typedef string_to_dtype_kernel_extra extra_type;

        kernel_data_prefix base;
        const base_string_dtype *src_string_dt;
        const char *src_metadata;
        assign_error_mode errmode;

        static void single(char *dst, const char *src, kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const string& s = e->src_string_dt->get_utf8_string(e->src_metadata, src, e->errmode);
            ndt::type(s).swap(reinterpret_cast<dtype_dtype_data *>(dst)->dt);
        }

        static void destruct(kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            base_dtype_xdecref(e->src_string_dt);
        }
    };

    struct dtype_to_string_kernel_extra {
        typedef dtype_to_string_kernel_extra extra_type;

        kernel_data_prefix base;
        const base_string_dtype *dst_string_dt;
        const char *dst_metadata;
        assign_error_mode errmode;

        static void single(char *dst, const char *src, kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const base_dtype *bd = reinterpret_cast<const dtype_dtype_data *>(src)->dt;
            stringstream ss;
            if (is_builtin_type(bd)) {
                ss << ndt::type(bd, true);
            } else {
                bd->print_dtype(ss);
            }
            e->dst_string_dt->set_utf8_string(e->dst_metadata, dst, e->errmode,
                            ss.str());
        }

        static void destruct(kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            base_dtype_xdecref(e->dst_string_dt);
        }
    };
} // anonymous namespace



size_t dtype_dtype::make_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const ndt::type& dst_dt, const char *dst_metadata,
                const ndt::type& src_dt, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);

    if (this == dst_dt.extended()) {
        if (src_dt.get_type_id() == dtype_type_id) {
            kernel_data_prefix *e = out->get_at<kernel_data_prefix>(offset_out);
            e->set_function<unary_single_operation_t>(dtype_assignment_kernel_single);
            return offset_out + sizeof(kernel_data_prefix);
        } else if (src_dt.get_kind() == string_kind) {
            // String to dtype
            out->ensure_capacity(offset_out + sizeof(string_to_dtype_kernel_extra));
            string_to_dtype_kernel_extra *e = out->get_at<string_to_dtype_kernel_extra>(offset_out);
            e->base.set_function<unary_single_operation_t>(&string_to_dtype_kernel_extra::single);
            e->base.destructor = &string_to_dtype_kernel_extra::destruct;
            // The kernel data owns a reference to this dtype
            e->src_string_dt = static_cast<const base_string_dtype *>(ndt::type(src_dt).release());
            e->src_metadata = src_metadata;
            e->errmode = errmode;
            return offset_out + sizeof(string_to_dtype_kernel_extra);
        } else if (!src_dt.is_builtin()) {
            return src_dt.extended()->make_assignment_kernel(out, offset_out,
                            dst_dt, dst_metadata,
                            src_dt, src_metadata,
                            kernreq, errmode, ectx);
        }
    } else {
        if (dst_dt.get_kind() == string_kind) {
            // Dtype to string
            out->ensure_capacity(offset_out + sizeof(dtype_to_string_kernel_extra));
            dtype_to_string_kernel_extra *e = out->get_at<dtype_to_string_kernel_extra>(offset_out);
            e->base.set_function<unary_single_operation_t>(&dtype_to_string_kernel_extra::single);
            e->base.destructor = &dtype_to_string_kernel_extra::destruct;
            // The kernel data owns a reference to this dtype
            e->dst_string_dt = static_cast<const base_string_dtype *>(ndt::type(dst_dt).release());
            e->dst_metadata = dst_metadata;
            e->errmode = errmode;
            return offset_out + sizeof(dtype_to_string_kernel_extra);
        }
    }

    stringstream ss;
    ss << "Cannot assign from " << src_dt << " to " << dst_dt;
    throw runtime_error(ss.str());
}
