//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

type_type::type_type()
    : base_type(type_type_id, custom_kind, sizeof(const base_type *),
                    sizeof(const base_type *),
                    type_flag_scalar|type_flag_zeroinit|type_flag_destructor,
                    0, 0)
{
}

type_type::~type_type()
{
}

void type_type::print_data(std::ostream& o,
                const char *DYND_UNUSED(metadata), const char *data) const
{
    const type_type_data *ddd = reinterpret_cast<const type_type_data *>(data);
    // This tests avoids the atomic increment/decrement of
    // always constructing a type object
    if (is_builtin_type(ddd->dt)) {
        o << ndt::type(ddd->dt, true);
    } else {
        ddd->dt->print_type(o);
    }
}

void type_type::print_type(std::ostream& o) const
{
    o << "type";
}

bool type_type::operator==(const base_type& rhs) const
{
    return this == &rhs || rhs.get_type_id() == type_type_id;
}

void type_type::metadata_default_construct(char *DYND_UNUSED(metadata),
                size_t DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const
{
}

void type_type::metadata_copy_construct(char *DYND_UNUSED(dst_metadata),
                const char *DYND_UNUSED(src_metadata), memory_block_data *DYND_UNUSED(embedded_reference)) const
{
}

void type_type::metadata_reset_buffers(char *DYND_UNUSED(metadata)) const
{
}

void type_type::metadata_finalize_buffers(char *DYND_UNUSED(metadata)) const
{
}

void type_type::metadata_destruct(char *DYND_UNUSED(metadata)) const
{
}

void type_type::data_destruct(const char *DYND_UNUSED(metadata), char *data) const
{
    const base_type *bd = reinterpret_cast<type_type_data *>(data)->dt;
    if (!is_builtin_type(bd)) {
        base_type_decref(bd);
    }
}

void type_type::data_destruct_strided(const char *DYND_UNUSED(metadata), char *data,
                intptr_t stride, size_t count) const
{
    for (size_t i = 0; i != count; ++i, data += stride) {
        const base_type *bd = reinterpret_cast<type_type_data *>(data)->dt;
        if (!is_builtin_type(bd)) {
            base_type_decref(bd);
        }
    }
}

static void typed_data_assignment_kernel_single(char *dst, const char *src,
                kernel_data_prefix *DYND_UNUSED(extra))
{
    // Free the destination reference
    base_type_xdecref(reinterpret_cast<const type_type_data *>(dst)->dt);
    // Copy the pointer and count the reference
    const base_type *bd = reinterpret_cast<const type_type_data *>(src)->dt;
    reinterpret_cast<type_type_data *>(dst)->dt = bd;
    base_type_xincref(bd);
}

namespace {
    struct string_to_type_kernel_extra {
        typedef string_to_type_kernel_extra extra_type;

        kernel_data_prefix base;
        const base_string_type *src_string_dt;
        const char *src_metadata;
        assign_error_mode errmode;

        static void single(char *dst, const char *src, kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const string& s = e->src_string_dt->get_utf8_string(e->src_metadata, src, e->errmode);
            ndt::type(s).swap(reinterpret_cast<type_type_data *>(dst)->dt);
        }

        static void destruct(kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            base_type_xdecref(e->src_string_dt);
        }
    };

    struct type_to_string_kernel_extra {
        typedef type_to_string_kernel_extra extra_type;

        kernel_data_prefix base;
        const base_string_type *dst_string_dt;
        const char *dst_metadata;
        assign_error_mode errmode;

        static void single(char *dst, const char *src, kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const base_type *bd = reinterpret_cast<const type_type_data *>(src)->dt;
            stringstream ss;
            if (is_builtin_type(bd)) {
                ss << ndt::type(bd, true);
            } else {
                bd->print_type(ss);
            }
            e->dst_string_dt->set_utf8_string(e->dst_metadata, dst, e->errmode,
                            ss.str());
        }

        static void destruct(kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            base_type_xdecref(e->dst_string_dt);
        }
    };
} // anonymous namespace



size_t type_type::make_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const ndt::type& dst_tp, const char *dst_metadata,
                const ndt::type& src_tp, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);

    if (this == dst_tp.extended()) {
        if (src_tp.get_type_id() == type_type_id) {
            kernel_data_prefix *e = out->get_at<kernel_data_prefix>(offset_out);
            e->set_function<unary_single_operation_t>(typed_data_assignment_kernel_single);
            return offset_out + sizeof(kernel_data_prefix);
        } else if (src_tp.get_kind() == string_kind) {
            // String to type
            out->ensure_capacity(offset_out + sizeof(string_to_type_kernel_extra));
            string_to_type_kernel_extra *e = out->get_at<string_to_type_kernel_extra>(offset_out);
            e->base.set_function<unary_single_operation_t>(&string_to_type_kernel_extra::single);
            e->base.destructor = &string_to_type_kernel_extra::destruct;
            // The kernel data owns a reference to this type
            e->src_string_dt = static_cast<const base_string_type *>(ndt::type(src_tp).release());
            e->src_metadata = src_metadata;
            e->errmode = errmode;
            return offset_out + sizeof(string_to_type_kernel_extra);
        } else if (!src_tp.is_builtin()) {
            return src_tp.extended()->make_assignment_kernel(out, offset_out,
                            dst_tp, dst_metadata,
                            src_tp, src_metadata,
                            kernreq, errmode, ectx);
        }
    } else {
        if (dst_tp.get_kind() == string_kind) {
            // Type to string
            out->ensure_capacity(offset_out + sizeof(type_to_string_kernel_extra));
            type_to_string_kernel_extra *e = out->get_at<type_to_string_kernel_extra>(offset_out);
            e->base.set_function<unary_single_operation_t>(&type_to_string_kernel_extra::single);
            e->base.destructor = &type_to_string_kernel_extra::destruct;
            // The kernel data owns a reference to this type
            e->dst_string_dt = static_cast<const base_string_type *>(ndt::type(dst_tp).release());
            e->dst_metadata = dst_metadata;
            e->errmode = errmode;
            return offset_out + sizeof(type_to_string_kernel_extra);
        }
    }

    stringstream ss;
    ss << "Cannot assign from " << src_tp << " to " << dst_tp;
    throw runtime_error(ss.str());
}
