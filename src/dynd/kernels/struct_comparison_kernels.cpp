//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/dtype.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/struct_comparison_kernels.hpp>
#include <dynd/dtypes/base_struct_dtype.hpp>

using namespace std;
using namespace dynd;

namespace {
    // Sorting less operation when the metadata is different
    struct struct_compare_sorting_less_matching_metadata_kernel {
        typedef struct_compare_sorting_less_matching_metadata_kernel extra_type;

        kernel_data_prefix base;
        size_t field_count;
        const size_t *src_data_offsets;
        // After this are field_count sorting_less kernel offsets, for
        // src#.field_i < src#.field_i with each 0 <= i < field_count

        static bool sorting_less(const char *src0, const char *src1, kernel_data_prefix *extra) {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            size_t field_count = e->field_count;
            const size_t *src_data_offsets = e->src_data_offsets;
            const size_t *kernel_offsets = reinterpret_cast<const size_t *>(e + 1);
            for (size_t i = 0; i != field_count; ++i) {
                kernel_data_prefix *sorting_less_kdp =
                                reinterpret_cast<kernel_data_prefix *>(eraw + kernel_offsets[i]);
                binary_single_predicate_t opchild =
                                sorting_less_kdp->get_function<binary_single_predicate_t>();
                size_t data_offset = src_data_offsets[i];
                // if (src0.field_i < src1.field_i) return true
                if (opchild(src0 + data_offset, src1 + data_offset,
                                sorting_less_kdp)) {
                    return true;
                }
                // if (src1.field_i < src0.field_i) return false
                if (opchild(src1 + data_offset, src0 + data_offset,
                                sorting_less_kdp)) {
                    return false;
                }
            }
            return false;
        }

        static void destruct(kernel_data_prefix *extra)
        {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const size_t *kernel_offsets = reinterpret_cast<const size_t *>(e + 1);
            size_t field_count = e->field_count;
            kernel_data_prefix *echild;
            for (size_t i = 0; i != field_count; ++i) {
                echild = reinterpret_cast<kernel_data_prefix *>(eraw + kernel_offsets[i]);
                if (echild->destructor) {
                    echild->destructor(echild);
                }
            }
        }
    };

    // Sorting less operation when the metadata is different
    struct struct_compare_sorting_less_diff_metadata_kernel {
        typedef struct_compare_sorting_less_diff_metadata_kernel extra_type;

        kernel_data_prefix base;
        size_t field_count;
        const size_t *src0_data_offsets, *src1_data_offsets;
        // After this are 2*field_count sorting_less kernel offsets, for
        // src0.field_i < src1.field_i and src1.field_i < src0.field_i
        // with each 0 <= i < field_count

        static bool sorting_less(const char *src0, const char *src1, kernel_data_prefix *extra) {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            size_t field_count = e->field_count;
            const size_t *src0_data_offsets = e->src0_data_offsets;
            const size_t *src1_data_offsets = e->src1_data_offsets;
            const size_t *kernel_offsets = reinterpret_cast<const size_t *>(e + 1);
            for (size_t i = 0; i != field_count; ++i) {
                kernel_data_prefix *src0_sorting_less_src1 =
                                reinterpret_cast<kernel_data_prefix *>(eraw + kernel_offsets[2*i]);
                binary_single_predicate_t opchild =
                                src0_sorting_less_src1->get_function<binary_single_predicate_t>();
                // if (src0.field_i < src1.field_i) return true
                if (opchild(src0 + src0_data_offsets[i],
                                src1 + src1_data_offsets[i],
                                src0_sorting_less_src1)) {
                    return true;
                }
                kernel_data_prefix *src1_sorting_less_src0 =
                                reinterpret_cast<kernel_data_prefix *>(eraw + kernel_offsets[2*i+1]);
                opchild = src1_sorting_less_src0->get_function<binary_single_predicate_t>();
                // if (src1.field_i < src0.field_i) return false
                if (opchild(src1 + src1_data_offsets[i],
                                src0 + src0_data_offsets[i],
                                src1_sorting_less_src0)) {
                    return false;
                }
            }
            return false;
        }

        static void destruct(kernel_data_prefix *extra)
        {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const size_t *kernel_offsets = reinterpret_cast<const size_t *>(e + 1);
            size_t field_count = e->field_count;
            kernel_data_prefix *echild;
            for (size_t i = 0; i != 2 * field_count; ++i) {
                echild = reinterpret_cast<kernel_data_prefix *>(eraw + kernel_offsets[i]);
                if (echild->destructor) {
                    echild->destructor(echild);
                }
            }
        }
    };

    // Equality comparison kernels
    struct struct_compare_equality_kernel {
        typedef struct_compare_equality_kernel extra_type;

        kernel_data_prefix base;
        size_t field_count;
        const size_t *src0_data_offsets, *src1_data_offsets;
        // After this are field_count sorting_less kernel offsets, for
        // src0.field_i <op> src1.field_i
        // with each 0 <= i < field_count

        static bool equal(const char *src0, const char *src1, kernel_data_prefix *extra) {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            size_t field_count = e->field_count;
            const size_t *src0_data_offsets = e->src0_data_offsets;
            const size_t *src1_data_offsets = e->src1_data_offsets;
            const size_t *kernel_offsets = reinterpret_cast<const size_t *>(e + 1);
            for (size_t i = 0; i != field_count; ++i) {
                kernel_data_prefix *echild =
                                reinterpret_cast<kernel_data_prefix *>(eraw + kernel_offsets[i]);
                binary_single_predicate_t opchild =
                                echild->get_function<binary_single_predicate_t>();
                // if (src0.field_i < src1.field_i) return true
                if (!opchild(src0 + src0_data_offsets[i],
                                src1 + src1_data_offsets[i],
                                echild)) {
                    return false;
                }
            }
            return true;
        }

        static bool not_equal(const char *src0, const char *src1, kernel_data_prefix *extra) {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            size_t field_count = e->field_count;
            const size_t *src0_data_offsets = e->src0_data_offsets;
            const size_t *src1_data_offsets = e->src1_data_offsets;
            const size_t *kernel_offsets = reinterpret_cast<const size_t *>(e + 1);
            for (size_t i = 0; i != field_count; ++i) {
                kernel_data_prefix *echild =
                                reinterpret_cast<kernel_data_prefix *>(eraw + kernel_offsets[i]);
                binary_single_predicate_t opchild =
                                echild->get_function<binary_single_predicate_t>();
                // if (src0.field_i < src1.field_i) return true
                if (opchild(src0 + src0_data_offsets[i],
                                src1 + src1_data_offsets[i],
                                echild)) {
                    return true;
                }
            }
            return false;
        }

        static void destruct(kernel_data_prefix *extra)
        {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const size_t *kernel_offsets = reinterpret_cast<const size_t *>(e + 1);
            size_t field_count = e->field_count;
            kernel_data_prefix *echild;
            for (size_t i = 0; i != field_count; ++i) {
                echild = reinterpret_cast<kernel_data_prefix *>(eraw + kernel_offsets[i]);
                if (echild->destructor) {
                    echild->destructor(echild);
                }
            }
        }
    };
} // anonymous namespace

size_t dynd::make_struct_comparison_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const dtype& src_dt,
                const char *src0_metadata, const char *src1_metadata,
                comparison_type_t comptype,
                const eval::eval_context *ectx)
{
    const base_struct_dtype *bsd = static_cast<const base_struct_dtype *>(src_dt.extended());
    size_t field_count = bsd->get_field_count();
    if (comptype == comparison_type_sorting_less) {
        if (src0_metadata == src1_metadata ||
                        src_dt.get_metadata_size() == 0 ||
                        memcmp(src0_metadata, src1_metadata, src_dt.get_metadata_size()) == 0) {
            // The metadata is identical, so can use a more specialized comparison function
            size_t field_kernel_offset = offset_out +
                            sizeof(struct_compare_sorting_less_matching_metadata_kernel) +
                            field_count * sizeof(size_t);
            out->ensure_capacity(field_kernel_offset);
            struct_compare_sorting_less_matching_metadata_kernel *e =
                            out->get_at<struct_compare_sorting_less_matching_metadata_kernel>(offset_out);
            e->base.set_function<binary_single_predicate_t>(
                            &struct_compare_sorting_less_matching_metadata_kernel::sorting_less);
            e->base.destructor = &struct_compare_sorting_less_matching_metadata_kernel::destruct;
            e->field_count = field_count;
            e->src_data_offsets = bsd->get_data_offsets(src0_metadata);
            size_t *field_kernel_offsets;
            const size_t *metadata_offsets = bsd->get_metadata_offsets();
            const dtype *field_types = bsd->get_field_types();
            for (size_t i = 0; i != field_count; ++i) {
                // Reserve space for the child, and save the offset to this
                // field comparison kernel. Have to re-get
                // the pointer because creating the field comparison kernel may
                // move the memory.
                out->ensure_capacity(field_kernel_offset);
                e = out->get_at<struct_compare_sorting_less_matching_metadata_kernel>(offset_out);
                field_kernel_offsets = reinterpret_cast<size_t *>(e + 1);
                field_kernel_offsets[i] = field_kernel_offset - offset_out;
                const char *field_metadata = src0_metadata + metadata_offsets[i];
                field_kernel_offset = make_comparison_kernel(out, field_kernel_offset,
                                field_types[i], field_metadata,
                                field_types[i], field_metadata,
                                comparison_type_sorting_less, ectx);
            }
            return field_kernel_offset;
        } else {
            // The metadata is different, so have to get the kernels both ways for the fields
            size_t field_kernel_offset = offset_out +
                            sizeof(struct_compare_sorting_less_diff_metadata_kernel) +
                            2 * field_count * sizeof(size_t);
            out->ensure_capacity(field_kernel_offset);
            struct_compare_sorting_less_diff_metadata_kernel *e =
                            out->get_at<struct_compare_sorting_less_diff_metadata_kernel>(offset_out);
            e->base.set_function<binary_single_predicate_t>(
                            &struct_compare_sorting_less_diff_metadata_kernel::sorting_less);
            e->base.destructor = &struct_compare_sorting_less_diff_metadata_kernel::destruct;
            e->field_count = field_count;
            e->src0_data_offsets = bsd->get_data_offsets(src0_metadata);
            e->src1_data_offsets = bsd->get_data_offsets(src1_metadata);
            size_t *field_kernel_offsets;
            const size_t *metadata_offsets = bsd->get_metadata_offsets();
            const dtype *field_types = bsd->get_field_types();
            for (size_t i = 0; i != field_count; ++i) {
                // Reserve space for the child, and save the offset to this
                // field comparison kernel. Have to re-get
                // the pointer because creating the field comparison kernel may
                // move the memory.
                out->ensure_capacity(field_kernel_offset);
                e = out->get_at<struct_compare_sorting_less_diff_metadata_kernel>(offset_out);
                field_kernel_offsets = reinterpret_cast<size_t *>(e + 1);
                field_kernel_offsets[2*i] = field_kernel_offset - offset_out;
                field_kernel_offset = make_comparison_kernel(out, field_kernel_offset,
                                field_types[i], src0_metadata + metadata_offsets[i],
                                field_types[i], src1_metadata + metadata_offsets[i],
                                comparison_type_sorting_less, ectx);
                // Repeat for comparing the other way
                out->ensure_capacity(field_kernel_offset);
                e = out->get_at<struct_compare_sorting_less_diff_metadata_kernel>(offset_out);
                field_kernel_offsets = reinterpret_cast<size_t *>(e + 1);
                field_kernel_offsets[2*i+1] = field_kernel_offset - offset_out;
                field_kernel_offset = make_comparison_kernel(out, field_kernel_offset,
                                field_types[i], src1_metadata + metadata_offsets[i],
                                field_types[i], src0_metadata + metadata_offsets[i],
                                comparison_type_sorting_less, ectx);
            }
            return field_kernel_offset;
        }
    } else if (comptype == comparison_type_equal || comptype == comparison_type_not_equal) {
        size_t field_kernel_offset = offset_out +
                        sizeof(struct_compare_equality_kernel) +
                        field_count * sizeof(size_t);
        out->ensure_capacity(field_kernel_offset);
        struct_compare_equality_kernel *e =
                        out->get_at<struct_compare_equality_kernel>(offset_out);
        if (comptype == comparison_type_equal) {
            e->base.set_function<binary_single_predicate_t>(
                            &struct_compare_equality_kernel::equal);
        } else {
            e->base.set_function<binary_single_predicate_t>(
                            &struct_compare_equality_kernel::not_equal);
        }
        e->base.destructor = &struct_compare_equality_kernel::destruct;
        e->field_count = field_count;
        e->src0_data_offsets = bsd->get_data_offsets(src0_metadata);
        e->src1_data_offsets = bsd->get_data_offsets(src1_metadata);
        size_t *field_kernel_offsets;
        const size_t *metadata_offsets = bsd->get_metadata_offsets();
        const dtype *field_types = bsd->get_field_types();
        for (size_t i = 0; i != field_count; ++i) {
            // Reserve space for the child, and save the offset to this
            // field comparison kernel. Have to re-get
            // the pointer because creating the field comparison kernel may
            // move the memory.
            out->ensure_capacity(field_kernel_offset);
            e = out->get_at<struct_compare_equality_kernel>(offset_out);
            field_kernel_offsets = reinterpret_cast<size_t *>(e + 1);
            field_kernel_offsets[i] = field_kernel_offset - offset_out;
            const char *field_metadata = src0_metadata + metadata_offsets[i];
            field_kernel_offset = make_comparison_kernel(out, field_kernel_offset,
                            field_types[i], field_metadata,
                            field_types[i], field_metadata,
                            comptype, ectx);
        }
        return field_kernel_offset;
    } else {
        throw not_comparable_error(src_dt, src_dt, comptype);
    }
}

size_t dynd::make_general_struct_comparison_kernel(
                hierarchical_kernel *DYND_UNUSED(out), size_t DYND_UNUSED(offset_out),
                const dtype& DYND_UNUSED(src0_dt), const char *DYND_UNUSED(src0_metadata),
                const dtype& DYND_UNUSED(src1_dt), const char *DYND_UNUSED(src1_metadata),
                comparison_type_t DYND_UNUSED(comptype),
                const eval::eval_context *DYND_UNUSED(ectx))
{
    throw runtime_error("TODO: make_general_struct_comparison_kernel is not implemented");
}
