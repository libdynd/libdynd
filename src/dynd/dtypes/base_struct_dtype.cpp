//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtype.hpp>
#include <dynd/dtypes/base_struct_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;


base_struct_dtype::~base_struct_dtype() {
}

size_t base_struct_dtype::get_elwise_property_index(const std::string& property_name) const
{
    size_t field_count = get_field_count();
    const string *field_names = get_field_names();
    for (size_t i = 0; i != field_count; ++i) {
        if (field_names[i] == property_name) {
            return i;
        }
    }

    stringstream ss;
    ss << "dtype " << dtype(this, true) << " does not have a kernel for property " << property_name;
    throw runtime_error(ss.str());
}

dtype base_struct_dtype::get_elwise_property_dtype(size_t elwise_property_index,
                bool& out_readable, bool& out_writable) const
{
    size_t field_count = get_field_count();
    if (elwise_property_index < field_count) {
        out_readable = true;
        out_writable = false;
        return get_field_types()[elwise_property_index].value_dtype();
    } else {
        return make_dtype<void>();
    }
}

namespace {
    struct struct_property_getter_extra {
        typedef struct_property_getter_extra extra_type;

        kernel_data_prefix base;
        size_t field_offset;

        static void single(char *dst, const char *src, kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            kernel_data_prefix *echild = &(e + 1)->base;
            unary_single_operation_t opchild = echild->get_function<unary_single_operation_t>();
            opchild(dst, src + e->field_offset, echild);
        }
        static void strided(char *dst, intptr_t dst_stride,
                        const char *src, intptr_t src_stride,
                        size_t count, kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            kernel_data_prefix *echild = &(e + 1)->base;
            unary_strided_operation_t opchild = echild->get_function<unary_strided_operation_t>();
            opchild(dst, dst_stride, src + e->field_offset, src_stride, count, echild);
        }

        static void destruct(kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            kernel_data_prefix *echild = &(e + 1)->base;
            if (echild->destructor) {
                echild->destructor(echild);
            }
        }
    };
} // anonymous namespace

size_t base_struct_dtype::make_elwise_property_getter_kernel(
                assignment_kernel *out, size_t offset_out,
                const char *dst_metadata,
                const char *src_metadata, size_t src_elwise_property_index,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    size_t field_count = get_field_count();
    if (src_elwise_property_index < field_count) {
        const size_t *metadata_offsets = get_metadata_offsets();
        const dtype& field_type = get_field_types()[src_elwise_property_index];
        out->ensure_capacity(offset_out + sizeof(struct_property_getter_extra));
        struct_property_getter_extra *e = out->get_at<struct_property_getter_extra>(offset_out);
        switch (kernreq) {
            case kernel_request_single:
                e->base.set_function<unary_single_operation_t>(&struct_property_getter_extra::single);
                break;
            case kernel_request_strided:
                e->base.set_function<unary_strided_operation_t>(&struct_property_getter_extra::strided);
                break;
            default: {
                stringstream ss;
                ss << "base_struct_dtype::make_elwise_property_getter_kernel: ";
                ss << "unrecognized request " << (int)kernreq;
                throw runtime_error(ss.str());
            }   
        }
        e->base.destructor = &struct_property_getter_extra::destruct;
        e->field_offset = get_data_offsets(src_metadata)[src_elwise_property_index];
        return ::make_assignment_kernel(out, offset_out + sizeof(struct_property_getter_extra),
                        field_type.value_dtype(), dst_metadata,
                        field_type, src_metadata,
                        kernreq, assign_error_none, ectx);
    } else {
        stringstream ss;
        ss << "dynd dtype " << dtype(this, true);
        ss << " given an invalid property index" << src_elwise_property_index;
        throw runtime_error(ss.str());
    }
}

size_t base_struct_dtype::make_elwise_property_setter_kernel(
                assignment_kernel *DYND_UNUSED(out), size_t DYND_UNUSED(offset_out),
                const char *DYND_UNUSED(dst_metadata), size_t dst_elwise_property_index,
                const char *DYND_UNUSED(src_metadata),
                kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context *DYND_UNUSED(ectx)) const
{
    // No writable properties
    stringstream ss;
    ss << "dynd dtype " << dtype(this, true);
    ss << " given an invalid property index" << dst_elwise_property_index;
    throw runtime_error(ss.str());
}

