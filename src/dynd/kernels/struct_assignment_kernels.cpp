//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>
#include <algorithm>

#include <dynd/type.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/struct_assignment_kernels.hpp>

using namespace std;
using namespace dynd;

namespace {
    struct struct_kernel_extra {
        typedef struct_kernel_extra extra_type;

        ckernel_data_prefix base;
        size_t field_count;
        // After this, there are 'field_count' of
        // the following in a row
        struct field_items {
            size_t child_kernel_offset;
            size_t dst_data_offset;
            size_t src_data_offset;
        };

        static void single(char *dst, const char *src, ckernel_data_prefix *extra)
        {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const field_items *fi = reinterpret_cast<const field_items *>(e + 1);
            size_t field_count = e->field_count;
            ckernel_data_prefix *echild;
            unary_single_operation_t opchild;

            for (size_t i = 0; i < field_count; ++i) {
                const field_items& item = fi[i];
                echild  = reinterpret_cast<ckernel_data_prefix *>(eraw + item.child_kernel_offset);
                opchild = echild->get_function<unary_single_operation_t>();
                opchild(dst + item.dst_data_offset, src + item.src_data_offset, echild);
            }
        }

        static void destruct(ckernel_data_prefix *extra)
        {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            ckernel_data_prefix *echild;
            const field_items *fi = reinterpret_cast<const field_items *>(e + 1);
            size_t field_count = e->field_count;
            for (size_t i = 0; i < field_count; ++i) {
                const field_items& item = fi[i];
                if (item.child_kernel_offset != 0) {
                    echild  = reinterpret_cast<ckernel_data_prefix *>(eraw + item.child_kernel_offset);
                    if (echild->destructor != NULL) {
                        echild->destructor(echild);
                    }
                }
            }
        }
    };
} // anonymous namespace

/////////////////////////////////////////
// struct to identical struct assignment

size_t dynd::make_struct_identical_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& val_struct_tp,
                const char *dst_metadata, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx)
{
    if (val_struct_tp.get_kind() != struct_kind) {
        stringstream ss;
        ss << "make_struct_identical_assignment_kernel: provided type " << val_struct_tp << " is not of struct kind";
        throw runtime_error(ss.str());
    }
    if (val_struct_tp.is_pod()) {
        // For POD structs, get a trivial memory copy kernel
        return make_pod_typed_data_assignment_kernel(out, offset_out,
                        val_struct_tp.get_data_size(), val_struct_tp.get_data_alignment(),
                        kernreq);
    }

    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);

    const base_struct_type *sd = static_cast<const base_struct_type *>(val_struct_tp.extended());
    size_t field_count = sd->get_field_count();

    size_t extra_size = sizeof(struct_kernel_extra) +
                    field_count * sizeof(struct_kernel_extra::field_items);
    out->ensure_capacity(offset_out + extra_size);
    struct_kernel_extra *e = out->get_at<struct_kernel_extra>(offset_out);
    e->base.set_function<unary_single_operation_t>(&struct_kernel_extra::single);
    e->base.destructor = &struct_kernel_extra::destruct;
    e->field_count = field_count;

    const size_t *dst_data_offsets = sd->get_data_offsets(dst_metadata);
    const size_t *src_data_offsets = sd->get_data_offsets(src_metadata);

    // Create the kernels and dst offsets for copying individual fields
    size_t current_offset = offset_out + extra_size;
    struct_kernel_extra::field_items *fi;
    for (size_t i = 0; i != field_count; ++i) {
        out->ensure_capacity(current_offset);
        // Adding another kernel may have invalidated 'e', so get it again
        e = out->get_at<struct_kernel_extra>(offset_out);
        fi = reinterpret_cast<struct_kernel_extra::field_items *>(e + 1) + i;
        fi->child_kernel_offset = current_offset - offset_out;
        fi->dst_data_offset = dst_data_offsets[i];
        fi->src_data_offset = src_data_offsets[i];
        current_offset = ::make_assignment_kernel(out, current_offset,
                        sd->get_field_types()[i], dst_metadata + sd->get_metadata_offsets()[i],
                        sd->get_field_types()[i], src_metadata + sd->get_metadata_offsets()[i],
                        kernel_request_single, errmode, ectx);
    }
    return current_offset;
}

/////////////////////////////////////////
// struct to different struct assignment

size_t dynd::make_struct_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& dst_struct_tp, const char *dst_metadata,
                const ndt::type& src_struct_tp, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx)
{
    if (src_struct_tp.get_kind() != struct_kind) {
        stringstream ss;
        ss << "make_struct_assignment_kernel: provided source type " << src_struct_tp << " is not of struct kind";
        throw runtime_error(ss.str());
    }
    if (dst_struct_tp.get_kind() != struct_kind) {
        stringstream ss;
        ss << "make_struct_assignment_kernel: provided destination type " << dst_struct_tp << " is not of struct kind";
        throw runtime_error(ss.str());
    }
    const base_struct_type *dst_sd = static_cast<const base_struct_type *>(dst_struct_tp.extended());
    const base_struct_type *src_sd = static_cast<const base_struct_type *>(src_struct_tp.extended());
    size_t field_count = dst_sd->get_field_count();

    if (field_count != src_sd->get_field_count()) {
        stringstream ss;
        ss << "cannot assign dynd struct " << src_struct_tp << " to " << dst_struct_tp;
        ss << " because they have different numbers of fields";
        throw runtime_error(ss.str());
    }

    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);

    size_t extra_size = sizeof(struct_kernel_extra) +
                    field_count * sizeof(struct_kernel_extra::field_items);
    out->ensure_capacity(offset_out + extra_size);
    struct_kernel_extra *e = out->get_at<struct_kernel_extra>(offset_out);
    e->base.set_function<unary_single_operation_t>(&struct_kernel_extra::single);
    e->base.destructor = &struct_kernel_extra::destruct;
    e->field_count = field_count;

    // Match up the fields
    const string *dst_field_names = dst_sd->get_field_names();
    const string *src_field_names = src_sd->get_field_names();
    vector<size_t> field_reorder(field_count);
    for (size_t i = 0; i != field_count; ++i) {
        const std::string& dst_name = dst_field_names[i];
        // TODO: accelerate this linear search if there are lots of fields?
        const string *it = std::find(src_field_names, src_field_names + field_count, dst_name);
        if (it == src_field_names + field_count) {
            stringstream ss;
            ss << "cannot assign dynd struct " << src_struct_tp << " to " << dst_struct_tp;
            ss << " because they have different field names";
            throw runtime_error(ss.str());
        }
        field_reorder[i] = it - src_field_names;
    }

    const ndt::type *src_field_types = src_sd->get_field_types();
    const ndt::type *dst_field_types = dst_sd->get_field_types();
    const size_t *src_data_offsets = src_sd->get_data_offsets(src_metadata);
    const size_t *dst_data_offsets = dst_sd->get_data_offsets(dst_metadata);
    const size_t *src_metadata_offsets = src_sd->get_metadata_offsets();
    const size_t *dst_metadata_offsets = dst_sd->get_metadata_offsets();

    // Create the kernels and dst offsets for copying individual fields
    size_t current_offset = offset_out + extra_size;
    struct_kernel_extra::field_items *fi;
    for (size_t i = 0; i != field_count; ++i) {
        size_t i_src = field_reorder[i];
        out->ensure_capacity(current_offset);
        // Ensuring capacity may have invalidated 'e', so get it again
        e = out->get_at<struct_kernel_extra>(offset_out);
        fi = reinterpret_cast<struct_kernel_extra::field_items *>(e + 1) + i;
        fi->child_kernel_offset = current_offset - offset_out;
        fi->dst_data_offset = dst_data_offsets[i];
        fi->src_data_offset = src_data_offsets[i_src];
        current_offset = ::make_assignment_kernel(out, current_offset,
                        dst_field_types[i], dst_metadata + dst_metadata_offsets[i],
                        src_field_types[i_src], src_metadata + src_metadata_offsets[i_src],
                        kernel_request_single, errmode, ectx);
    }
    return current_offset;
}
