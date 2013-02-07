//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/dtype.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/dtypes/string_dtype.hpp>

using namespace std;
using namespace dynd;

/////////////////////////////////////////
// fixedstring to fixedstring assignment

namespace {
    struct fixedstring_assign_kernel_extra {
        typedef fixedstring_assign_kernel_extra extra_type;

        hierarchical_kernel_common_base base;
        next_unicode_codepoint_t next_fn;
        append_unicode_codepoint_t append_fn;
        intptr_t dst_data_size, src_data_size;
        bool overflow_check;

        static void single(char *dst, const char *src,
                        hierarchical_kernel_common_base *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            char *dst_end = dst + e->dst_data_size;
            const char *src_end = src + e->src_data_size;
            next_unicode_codepoint_t next_fn = e->next_fn;
            append_unicode_codepoint_t append_fn = e->append_fn;
            uint32_t cp;

            while (src < src_end && dst < dst_end) {
                cp = next_fn(src, src_end);
                // The fixedstring dtype uses null-terminated strings
                if (cp == 0) {
                    // Null-terminate the destination string, and we're done
                    memset(dst, 0, dst_end - dst);
                    return;
                } else {
                    append_fn(cp, dst, dst_end);
                }
            }
            if (src < src_end) {
                if (e->overflow_check) {
                    throw std::runtime_error("Input string is too large to convert to destination fixed-size string");
                }
            } else if (dst < dst_end) {
                memset(dst, 0, dst_end - dst);
            }
        }
    };
} // anonymous namespace

size_t dynd::make_fixedstring_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                intptr_t dst_data_size, string_encoding_t dst_encoding,
                intptr_t src_data_size, string_encoding_t src_encoding,
                assign_error_mode errmode,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    out->ensure_capacity_leaf(offset_out + sizeof(fixedstring_assign_kernel_extra));
    fixedstring_assign_kernel_extra *e = out->get_at<fixedstring_assign_kernel_extra>(offset_out);
    e->base.set_function<unary_single_operation_t>(&fixedstring_assign_kernel_extra::single);
    e->next_fn = get_next_unicode_codepoint_function(src_encoding, errmode);
    e->append_fn = get_append_unicode_codepoint_function(dst_encoding, errmode);
    e->dst_data_size = dst_data_size;
    e->src_data_size = src_data_size;
    e->overflow_check = (errmode != assign_error_none);
    return offset_out + sizeof(fixedstring_assign_kernel_extra);
}

/////////////////////////////////////////
// blockref string to blockref string assignment

namespace {
    struct blockref_string_assign_kernel_extra {
        typedef blockref_string_assign_kernel_extra extra_type;

        hierarchical_kernel_common_base base;
        string_encoding_t dst_encoding, src_encoding;
        next_unicode_codepoint_t next_fn;
        append_unicode_codepoint_t append_fn;
        const string_dtype_metadata *dst_metadata, *src_metadata;

        static void single(char *dst, const char *src,
                        hierarchical_kernel_common_base *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const string_dtype_metadata *dst_md = e->dst_metadata;
            const string_dtype_metadata *src_md = e->src_metadata;
            string_dtype_data *dst_d = reinterpret_cast<string_dtype_data *>(dst);
            const string_dtype_data *src_d = reinterpret_cast<const string_dtype_data *>(src);
            intptr_t src_charsize = string_encoding_char_size_table[e->src_encoding];
            intptr_t dst_charsize = string_encoding_char_size_table[e->dst_encoding];

            if (dst_d->begin != NULL) {
                throw runtime_error("Cannot assign to an already initialized dynd string");
            } else if (src_d->begin == NULL) {
                // Allow uninitialized -> uninitialized assignment as a special case, for
                // (future) missing data support
                return;
            }

            // If the blockrefs are different, require a copy operation
            if (dst_md->blockref != src_md->blockref) {
                char *dst_begin = NULL, *dst_current, *dst_end = NULL;
                const char *src_begin = src_d->begin;
                const char *src_end = src_d->end;
                next_unicode_codepoint_t next_fn = e->next_fn;
                append_unicode_codepoint_t append_fn = e->append_fn;
                uint32_t cp;

                memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(dst_md->blockref);

                // Allocate the initial output as the src number of characters + some padding
                // TODO: Don't add padding if the output is not a multi-character encoding
                allocator->allocate(dst_md->blockref, ((src_end - src_begin) / src_charsize + 16) * dst_charsize * 1124 / 1024,
                                dst_charsize, &dst_begin, &dst_end);

                dst_current = dst_begin;
                while (src_begin < src_end) {
                    cp = next_fn(src_begin, src_end);
                    // Append the codepoint, or increase the allocated memory as necessary
                    if (dst_end - dst_current >= 8) {
                        append_fn(cp, dst_current, dst_end);
                    } else {
                        char *dst_begin_saved = dst_begin;
                        allocator->resize(dst_md->blockref, 2 * (dst_end - dst_begin), &dst_begin, &dst_end);
                        dst_current = dst_begin + (dst_current - dst_begin_saved);

                        append_fn(cp, dst_current, dst_end);
                    }
                }

                // Shrink-wrap the memory to just fit the string
                allocator->resize(dst_md->blockref, dst_current - dst_begin, &dst_begin, &dst_end);

                // Set the output
                dst_d->begin = dst_begin;
                dst_d->end = dst_end;
            } else if (e->dst_encoding == e->src_encoding) {
                // Copy the pointers from the source string
                *dst_d = *src_d;
            } else {
                throw runtime_error("Attempted to reference source data when changing string encoding");
            }
        }
    };
} // anonymous namespace

size_t dynd::make_blockref_string_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                const char *dst_metadata, string_encoding_t dst_encoding,
                const char *src_metadata, string_encoding_t src_encoding,
                assign_error_mode errmode,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    out->ensure_capacity_leaf(offset_out + sizeof(blockref_string_assign_kernel_extra));
    blockref_string_assign_kernel_extra *e = out->get_at<blockref_string_assign_kernel_extra>(offset_out);
    e->base.set_function<unary_single_operation_t>(&blockref_string_assign_kernel_extra::single);
    e->dst_encoding = dst_encoding;
    e->src_encoding = src_encoding;
    e->next_fn = get_next_unicode_codepoint_function(src_encoding, errmode);
    e->append_fn = get_append_unicode_codepoint_function(dst_encoding, errmode);
    e->dst_metadata = reinterpret_cast<const string_dtype_metadata *>(dst_metadata);
    e->src_metadata = reinterpret_cast<const string_dtype_metadata *>(src_metadata);
    return offset_out + sizeof(blockref_string_assign_kernel_extra);
}

/////////////////////////////////////////
// fixedstring to blockref string assignment

namespace {
    struct fixedstring_to_blockref_string_assign_kernel_extra {
        typedef fixedstring_to_blockref_string_assign_kernel_extra extra_type;

        hierarchical_kernel_common_base base;
        string_encoding_t dst_encoding, src_encoding;
        intptr_t src_element_size;
        next_unicode_codepoint_t next_fn;
        append_unicode_codepoint_t append_fn;
        const string_dtype_metadata *dst_metadata;

        static void single(char *dst, const char *src,
                        hierarchical_kernel_common_base *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const string_dtype_metadata *dst_md = e->dst_metadata;
            string_dtype_data *dst_d = reinterpret_cast<string_dtype_data *>(dst);
            intptr_t src_charsize = string_encoding_char_size_table[e->src_encoding];
            intptr_t dst_charsize = string_encoding_char_size_table[e->dst_encoding];

            if (dst_d->begin != NULL) {
                throw runtime_error("Cannot assign to an already initialized dynd string");
            }

            char *dst_begin = NULL, *dst_current, *dst_end = NULL;
            const char *src_begin = src;
            const char *src_end = src + e->src_element_size;
            next_unicode_codepoint_t next_fn = e->next_fn;
            append_unicode_codepoint_t append_fn = e->append_fn;
            uint32_t cp;

            memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(dst_md->blockref);

            // Allocate the initial output as the src number of characters + some padding
            // TODO: Don't add padding if the output is not a multi-character encoding
            allocator->allocate(dst_md->blockref, ((src_end - src_begin) / src_charsize + 16) * dst_charsize * 1124 / 1024,
                            dst_charsize, &dst_begin, &dst_end);

            dst_current = dst_begin;
            while (src_begin < src_end) {
                cp = next_fn(src_begin, src_end);
                // Append the codepoint, or increase the allocated memory as necessary
                if (cp != 0) {
                    if (dst_end - dst_current >= 8) {
                        append_fn(cp, dst_current, dst_end);
                    } else {
                        char *dst_begin_saved = dst_begin;
                        allocator->resize(dst_md->blockref, 2 * (dst_end - dst_begin), &dst_begin, &dst_end);
                        dst_current = dst_begin + (dst_current - dst_begin_saved);

                        append_fn(cp, dst_current, dst_end);
                    }
                } else {
                    break;
                }
            }

            // Shrink-wrap the memory to just fit the string
            allocator->resize(dst_md->blockref, dst_current - dst_begin, &dst_begin, &dst_end);

            // Set the output
            dst_d->begin = dst_begin;
            dst_d->end = dst_end;
        }
    };
} // anonymous namespace

size_t dynd::make_fixedstring_to_blockref_string_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                const char *dst_metadata, string_encoding_t dst_encoding,
                intptr_t src_element_size, string_encoding_t src_encoding,
                assign_error_mode errmode,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    out->ensure_capacity_leaf(offset_out + sizeof(blockref_string_assign_kernel_extra));
    fixedstring_to_blockref_string_assign_kernel_extra *e =
                    out->get_at<fixedstring_to_blockref_string_assign_kernel_extra>(offset_out);
    e->base.set_function<unary_single_operation_t>(&fixedstring_to_blockref_string_assign_kernel_extra::single);
    e->dst_encoding = dst_encoding;
    e->src_encoding = src_encoding;
    e->src_element_size = src_element_size;
    e->next_fn = get_next_unicode_codepoint_function(src_encoding, errmode);
    e->append_fn = get_append_unicode_codepoint_function(dst_encoding, errmode);
    e->dst_metadata = reinterpret_cast<const string_dtype_metadata *>(dst_metadata);
    return offset_out + sizeof(blockref_string_assign_kernel_extra);
}

/////////////////////////////////////////
// blockref string to fixedstring assignment

namespace {
    struct blockref_string_to_fixedstring_assign_kernel_extra {
        typedef blockref_string_to_fixedstring_assign_kernel_extra extra_type;

        hierarchical_kernel_common_base base;
        next_unicode_codepoint_t next_fn;
        append_unicode_codepoint_t append_fn;
        intptr_t dst_data_size, src_element_size;
        bool overflow_check;

        static void single(char *dst, const char *src,
                        hierarchical_kernel_common_base *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            char *dst_end = dst + e->dst_data_size;
            const string_dtype_data *src_d = reinterpret_cast<const string_dtype_data *>(src);
            const char *src_begin = src_d->begin;
            const char *src_end = src_d->end;
            next_unicode_codepoint_t next_fn = e->next_fn;
            append_unicode_codepoint_t append_fn = e->append_fn;
            uint32_t cp;

            while (src_begin < src_end && dst < dst_end) {
                cp = next_fn(src_begin, src_end);
                append_fn(cp, dst, dst_end);
            }
            if (src_begin < src_end) {
                if (e->overflow_check) {
                    throw std::runtime_error("Input string is too large to convert to destination fixed-size string");
                }
            } else if (dst < dst_end) {
                memset(dst, 0, dst_end - dst);
            }
        }
    };
} // anonymous namespace

size_t dynd::make_blockref_string_to_fixedstring_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                intptr_t dst_data_size, string_encoding_t dst_encoding,
                string_encoding_t src_encoding,
                assign_error_mode errmode,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    out->ensure_capacity_leaf(offset_out + sizeof(blockref_string_to_fixedstring_assign_kernel_extra));
    blockref_string_to_fixedstring_assign_kernel_extra *e = out->get_at<blockref_string_to_fixedstring_assign_kernel_extra>(offset_out);
    e->base.set_function<unary_single_operation_t>(&blockref_string_to_fixedstring_assign_kernel_extra::single);
    e->next_fn = get_next_unicode_codepoint_function(src_encoding, errmode);
    e->append_fn = get_append_unicode_codepoint_function(dst_encoding, errmode);
    e->dst_data_size = dst_data_size;
    e->overflow_check = (errmode != assign_error_none);
    return offset_out + sizeof(blockref_string_to_fixedstring_assign_kernel_extra);
}
