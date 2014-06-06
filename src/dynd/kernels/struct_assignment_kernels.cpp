//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
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

        ckernel_prefix base;
        size_t field_count;
        // After this, there are 'field_count' of
        // the following in a row
        struct field_items {
            size_t child_kernel_offset;
            size_t dst_data_offset;
            size_t src_data_offset;
        };

        static void single(char *dst, const char *src, ckernel_prefix *extra)
        {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const field_items *fi = reinterpret_cast<const field_items *>(e + 1);
            size_t field_count = e->field_count;
            ckernel_prefix *echild;
            unary_single_operation_t child_fn;

            for (size_t i = 0; i < field_count; ++i) {
                const field_items& item = fi[i];
                echild  = reinterpret_cast<ckernel_prefix *>(eraw + item.child_kernel_offset);
                child_fn = echild->get_function<unary_single_operation_t>();
                child_fn(dst + item.dst_data_offset, src + item.src_data_offset, echild);
            }
        }

        static void destruct(ckernel_prefix *self)
        {
            extra_type *e = reinterpret_cast<extra_type *>(self);
            const field_items *fi = reinterpret_cast<const field_items *>(e + 1);
            size_t field_count = e->field_count;
            for (size_t i = 0; i < field_count; ++i) {
                const field_items& item = fi[i];
                self->destroy_child_ckernel(item.child_kernel_offset);
            }
        }
    };
} // anonymous namespace

/////////////////////////////////////////
// struct to identical struct assignment

size_t dynd::make_struct_identical_assignment_kernel(
    ckernel_builder *ckb, size_t ckb_offset, const ndt::type &val_struct_tp,
    const char *dst_arrmeta, const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
    if (val_struct_tp.get_kind() != struct_kind) {
        stringstream ss;
        ss << "make_struct_identical_assignment_kernel: provided type " << val_struct_tp << " is not of struct kind";
        throw runtime_error(ss.str());
    }
    if (val_struct_tp.is_pod()) {
        // For POD structs, get a trivial memory copy kernel
        return make_pod_typed_data_assignment_kernel(ckb, ckb_offset,
                        val_struct_tp.get_data_size(), val_struct_tp.get_data_alignment(),
                        kernreq);
    }

    ckb_offset = make_kernreq_to_single_kernel_adapter(ckb, ckb_offset, kernreq);

    const base_struct_type *sd = val_struct_tp.tcast<base_struct_type>();
    size_t field_count = sd->get_field_count();

    size_t extra_size = sizeof(struct_kernel_extra) +
                    field_count * sizeof(struct_kernel_extra::field_items);
    ckb->ensure_capacity(ckb_offset + extra_size);
    struct_kernel_extra *e = ckb->get_at<struct_kernel_extra>(ckb_offset);
    e->base.set_function<unary_single_operation_t>(&struct_kernel_extra::single);
    e->base.destructor = &struct_kernel_extra::destruct;
    e->field_count = field_count;

    const uintptr_t *dst_data_offsets = sd->get_data_offsets(dst_arrmeta);
    const uintptr_t *src_data_offsets = sd->get_data_offsets(src_arrmeta);

    // Create the kernels and dst offsets for copying individual fields
    size_t current_offset = ckb_offset + extra_size;
    struct_kernel_extra::field_items *fi;
    for (size_t i = 0; i != field_count; ++i) {
        ckb->ensure_capacity(current_offset);
        // Adding another kernel may have invalidated 'e', so get it again
        e = ckb->get_at<struct_kernel_extra>(ckb_offset);
        fi = reinterpret_cast<struct_kernel_extra::field_items *>(e + 1) + i;
        fi->child_kernel_offset = current_offset - ckb_offset;
        fi->dst_data_offset = dst_data_offsets[i];
        fi->src_data_offset = src_data_offsets[i];
        const ndt::type &ft = sd->get_field_type(i);
        uintptr_t arrmeta_offset = sd->get_arrmeta_offset(i);
        current_offset = ::make_assignment_kernel(ckb, current_offset,
                        ft, dst_arrmeta + arrmeta_offset,
                        ft, src_arrmeta + arrmeta_offset,
                        kernel_request_single, ectx);
    }
    return current_offset;
}

/////////////////////////////////////////
// struct to different struct assignment

size_t dynd::make_struct_assignment_kernel(
    ckernel_builder *ckb, size_t ckb_offset, const ndt::type &dst_struct_tp,
    const char *dst_arrmeta, const ndt::type &src_struct_tp,
    const char *src_arrmeta, kernel_request_t kernreq,
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
    const base_struct_type *dst_sd = dst_struct_tp.tcast<base_struct_type>();
    const base_struct_type *src_sd = src_struct_tp.tcast<base_struct_type>();
    size_t field_count = dst_sd->get_field_count();

    if (field_count != src_sd->get_field_count()) {
        stringstream ss;
        ss << "cannot assign dynd struct " << src_struct_tp << " to " << dst_struct_tp;
        ss << " because they have different numbers of fields";
        throw runtime_error(ss.str());
    }

    ckb_offset = make_kernreq_to_single_kernel_adapter(ckb, ckb_offset, kernreq);

    size_t extra_size = sizeof(struct_kernel_extra) +
                    field_count * sizeof(struct_kernel_extra::field_items);
    ckb->ensure_capacity(ckb_offset + extra_size);
    struct_kernel_extra *e = ckb->get_at<struct_kernel_extra>(ckb_offset);
    e->base.set_function<unary_single_operation_t>(&struct_kernel_extra::single);
    e->base.destructor = &struct_kernel_extra::destruct;
    e->field_count = field_count;

    // Match up the fields
    vector<size_t> field_reorder(field_count);
    for (size_t i = 0; i != field_count; ++i) {
        const string_type_data& dst_name = dst_sd->get_field_name_raw(i);
        intptr_t src_i = src_sd->get_field_index(dst_name.begin, dst_name.end);
        if (src_i < 0) {
            stringstream ss;
            ss << "cannot assign dynd struct " << src_struct_tp << " to " << dst_struct_tp;
            ss << " because they have different field names";
            throw runtime_error(ss.str());
        }
        field_reorder[i] = src_i;
    }

    const uintptr_t *src_data_offsets = src_sd->get_data_offsets(src_arrmeta);
    const uintptr_t *dst_data_offsets = dst_sd->get_data_offsets(dst_arrmeta);
    const uintptr_t *src_arrmeta_offsets = src_sd->get_arrmeta_offsets_raw();
    const uintptr_t *dst_arrmeta_offsets = dst_sd->get_arrmeta_offsets_raw();

    // Create the kernels and dst offsets for copying individual fields
    size_t current_offset = ckb_offset + extra_size;
    struct_kernel_extra::field_items *fi;
    for (size_t i = 0; i != field_count; ++i) {
        size_t i_src = field_reorder[i];
        ckb->ensure_capacity(current_offset);
        // Ensuring capacity may have invalidated 'e', so get it again
        e = ckb->get_at<struct_kernel_extra>(ckb_offset);
        fi = reinterpret_cast<struct_kernel_extra::field_items *>(e + 1) + i;
        fi->child_kernel_offset = current_offset - ckb_offset;
        fi->dst_data_offset = dst_data_offsets[i];
        fi->src_data_offset = src_data_offsets[i_src];
        current_offset = ::make_assignment_kernel(
            ckb, current_offset, dst_sd->get_field_type(i),
            dst_arrmeta + dst_arrmeta_offsets[i], src_sd->get_field_type(i_src),
            src_arrmeta + src_arrmeta_offsets[i_src], kernel_request_single,
            ectx);
    }
    return current_offset;
}

/////////////////////////////////////////
// value to each struct field assignment

size_t dynd::make_broadcast_to_struct_assignment_kernel(
    ckernel_builder *ckb, size_t ckb_offset, const ndt::type &dst_struct_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
    // This implementation uses the same struct to struct kernel, just with
    // an offset of 0 for each source value. A kernel tailored to this
    // case can be made if better performance is needed.

    if (dst_struct_tp.get_kind() != struct_kind) {
        stringstream ss;
        ss << "make_struct_assignment_kernel: provided destination type " << dst_struct_tp << " is not of struct kind";
        throw runtime_error(ss.str());
    }
    const base_struct_type *dst_sd = dst_struct_tp.tcast<base_struct_type>();
    size_t field_count = dst_sd->get_field_count();

    ckb_offset = make_kernreq_to_single_kernel_adapter(ckb, ckb_offset, kernreq);

    size_t extra_size = sizeof(struct_kernel_extra) +
                    field_count * sizeof(struct_kernel_extra::field_items);
    ckb->ensure_capacity(ckb_offset + extra_size);
    struct_kernel_extra *e = ckb->get_at<struct_kernel_extra>(ckb_offset);
    e->base.set_function<unary_single_operation_t>(&struct_kernel_extra::single);
    e->base.destructor = &struct_kernel_extra::destruct;
    e->field_count = field_count;

    const uintptr_t *dst_data_offsets = dst_sd->get_data_offsets(dst_arrmeta);
    const uintptr_t *dst_arrmeta_offsets = dst_sd->get_arrmeta_offsets_raw();

    // Create the kernels and dst offsets for copying individual fields
    size_t current_offset = ckb_offset + extra_size;
    struct_kernel_extra::field_items *fi;
    for (size_t i = 0; i != field_count; ++i) {
        ckb->ensure_capacity(current_offset);
        // Ensuring capacity may have invalidated 'e', so get it again
        e = ckb->get_at<struct_kernel_extra>(ckb_offset);
        fi = reinterpret_cast<struct_kernel_extra::field_items *>(e + 1) + i;
        fi->child_kernel_offset = current_offset - ckb_offset;
        fi->dst_data_offset = dst_data_offsets[i];
        fi->src_data_offset = 0;
        current_offset = ::make_assignment_kernel(
            ckb, current_offset, dst_sd->get_field_type(i),
            dst_arrmeta + dst_arrmeta_offsets[i], src_tp, src_arrmeta,
            kernel_request_single, ectx);
    }
    return current_offset;
}
