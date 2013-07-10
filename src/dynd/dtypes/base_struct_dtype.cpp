//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/dtypes/base_struct_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/shortvector.hpp>

using namespace std;
using namespace dynd;


base_struct_dtype::~base_struct_dtype() {
}

void base_struct_dtype::get_shape(size_t ndim, size_t i, intptr_t *out_shape, const char *metadata) const
{
    out_shape[i] = m_field_count;
    if (i < ndim-1) {
        const ndt::type *field_types = get_field_types();
        const size_t *metadata_offsets = get_metadata_offsets();
        dimvector tmpshape(ndim);
        // Accumulate the shape from all the field shapes
        for (size_t fi = 0; fi != m_field_count; ++fi) {
            const ndt::type& ft = field_types[i];
            if (!ft.is_builtin()) {
                ft.extended()->get_shape(ndim, i+1, tmpshape.get(),
                                metadata ? (metadata + metadata_offsets[fi]) : NULL);
            } else {
                stringstream ss;
                ss << "requested too many dimensions from type " << ft;
                throw runtime_error(ss.str());
            }
            if (fi == 0) {
                // Copy the shape from the first field
                memcpy(out_shape + i + 1, tmpshape.get() + i + 1, (ndim - i - 1) * sizeof(intptr_t));
            } else {
                // Merge the shape from the rest
                for (size_t k = i + 1; k <ndim; ++k) {
                    // If we see different sizes, make the output -1
                    if (out_shape[k] != -1 && out_shape[k] != tmpshape[k]) {
                        out_shape[k] = -1;
                    }
                }
            }
        }
    }
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
    ss << "dynd type " << ndt::type(this, true) << " does not have a kernel for property " << property_name;
    throw runtime_error(ss.str());
}

ndt::type base_struct_dtype::get_elwise_property_dtype(size_t elwise_property_index,
                bool& out_readable, bool& out_writable) const
{
    size_t field_count = get_field_count();
    if (elwise_property_index < field_count) {
        out_readable = true;
        out_writable = false;
        return get_field_types()[elwise_property_index].value_type();
    } else {
        return ndt::make_dtype<void>();
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
                hierarchical_kernel *out, size_t offset_out,
                const char *dst_metadata,
                const char *src_metadata, size_t src_elwise_property_index,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    size_t field_count = get_field_count();
    if (src_elwise_property_index < field_count) {
        const size_t *metadata_offsets = get_metadata_offsets();
        const ndt::type& field_type = get_field_types()[src_elwise_property_index];
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
                        field_type.value_type(), dst_metadata,
                        field_type, src_metadata + metadata_offsets[src_elwise_property_index],
                        kernreq, assign_error_none, ectx);
    } else {
        stringstream ss;
        ss << "dynd type " << ndt::type(this, true);
        ss << " given an invalid property index" << src_elwise_property_index;
        throw runtime_error(ss.str());
    }
}

size_t base_struct_dtype::make_elwise_property_setter_kernel(
                hierarchical_kernel *DYND_UNUSED(out), size_t DYND_UNUSED(offset_out),
                const char *DYND_UNUSED(dst_metadata), size_t dst_elwise_property_index,
                const char *DYND_UNUSED(src_metadata),
                kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context *DYND_UNUSED(ectx)) const
{
    // No writable properties
    stringstream ss;
    ss << "dynd type " << ndt::type(this, true);
    ss << " given an invalid property index" << dst_elwise_property_index;
    throw runtime_error(ss.str());
}

void base_struct_dtype::data_destruct(const char *metadata, char *data) const
{
    const ndt::type *field_types = get_field_types();
    const size_t *metadata_offsets = get_metadata_offsets();
    const size_t *data_offsets = get_data_offsets(metadata);
    size_t field_count = get_field_count();
    for (size_t i = 0; i != field_count; ++i) {
        const ndt::type& dt = field_types[i];
        if (dt.get_flags()&type_flag_destructor) {
            dt.extended()->data_destruct(
                            metadata + metadata_offsets[i],
                            data + data_offsets[i]);
        }
    }
}

void base_struct_dtype::data_destruct_strided(const char *metadata, char *data,
                intptr_t stride, size_t count) const
{
    const ndt::type *field_types = get_field_types();
    const size_t *metadata_offsets = get_metadata_offsets();
    const size_t *data_offsets = get_data_offsets(metadata);
    size_t field_count = get_field_count();
    // Destruct all the fields a chunk at a time, in an
    // attempt to have some kind of locality
    while (count > 0) {
        size_t chunk_size = min(count, DYND_BUFFER_CHUNK_SIZE);
        for (size_t i = 0; i != field_count; ++i) {
            const ndt::type& dt = field_types[i];
            if (dt.get_flags()&type_flag_destructor) {
                dt.extended()->data_destruct_strided(
                                metadata + metadata_offsets[i],
                                data + data_offsets[i],
                                stride, chunk_size);
            }
        }
        data += stride * chunk_size;
        count -= chunk_size;
    }
}


