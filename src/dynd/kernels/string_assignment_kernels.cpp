//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/type.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/types/string_type.hpp>

using namespace std;
using namespace dynd;

/////////////////////////////////////////
// fixedstring to fixedstring assignment

namespace {
    struct fixedstring_assign_ck : public kernels::assignment_ck<fixedstring_assign_ck> {
        next_unicode_codepoint_t m_next_fn;
        append_unicode_codepoint_t m_append_fn;
        intptr_t m_dst_data_size, m_src_data_size;
        bool m_overflow_check;

        inline void single(char *dst, const char *src)
        {
            char *dst_end = dst + m_dst_data_size;
            const char *src_end = src + m_src_data_size;
            next_unicode_codepoint_t next_fn = m_next_fn;
            append_unicode_codepoint_t append_fn = m_append_fn;
            uint32_t cp = 0;

            while (src < src_end && dst < dst_end) {
                cp = next_fn(src, src_end);
                // The fixedstring type uses null-terminated strings
                if (cp == 0) {
                    // Null-terminate the destination string, and we're done
                    memset(dst, 0, dst_end - dst);
                    return;
                } else {
                    append_fn(cp, dst, dst_end);
                }
            }
            if (src < src_end) {
                if (m_overflow_check) {
                    throw std::runtime_error("Input string is too large to convert to destination fixed-size string");
                }
            } else if (dst < dst_end) {
                memset(dst, 0, dst_end - dst);
            }
        }
    };
} // anonymous namespace

size_t dynd::make_fixedstring_assignment_kernel(
                ckernel_builder *out_ckb, size_t ckb_offset,
                intptr_t dst_data_size, string_encoding_t dst_encoding,
                intptr_t src_data_size, string_encoding_t src_encoding,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    typedef fixedstring_assign_ck self_type;
    self_type *self = self_type::create_leaf(out_ckb, ckb_offset, kernreq);
    self->m_next_fn = get_next_unicode_codepoint_function(src_encoding, errmode);
    self->m_append_fn = get_append_unicode_codepoint_function(dst_encoding, errmode);
    self->m_dst_data_size = dst_data_size;
    self->m_src_data_size = src_data_size;
    self->m_overflow_check = (errmode != assign_error_none);
    return ckb_offset + sizeof(self_type);
}

/////////////////////////////////////////
// blockref string to blockref string assignment

namespace {
    struct blockref_string_assign_ck : public kernels::assignment_ck<blockref_string_assign_ck> {
        string_encoding_t m_dst_encoding, m_src_encoding;
        next_unicode_codepoint_t m_next_fn;
        append_unicode_codepoint_t m_append_fn;
        const string_type_metadata *m_dst_metadata, *m_src_metadata;

        inline void single(char *dst, const char *src)
        {
            const string_type_metadata *dst_md = m_dst_metadata;
            const string_type_metadata *src_md = m_src_metadata;
            string_type_data *dst_d = reinterpret_cast<string_type_data *>(dst);
            const string_type_data *src_d = reinterpret_cast<const string_type_data *>(src);
            intptr_t src_charsize = string_encoding_char_size_table[m_src_encoding];
            intptr_t dst_charsize = string_encoding_char_size_table[m_dst_encoding];

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
                next_unicode_codepoint_t next_fn = m_next_fn;
                append_unicode_codepoint_t append_fn = m_append_fn;
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
            } else if (m_dst_encoding == m_src_encoding) {
                // Copy the pointers from the source string
                *dst_d = *src_d;
            } else {
                throw runtime_error("Attempted to reference source data when changing string encoding");
            }
        }
    };
} // anonymous namespace

size_t dynd::make_blockref_string_assignment_kernel(
                ckernel_builder *out_ckb, size_t ckb_offset,
                const char *dst_metadata, string_encoding_t dst_encoding,
                const char *src_metadata, string_encoding_t src_encoding,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    typedef blockref_string_assign_ck self_type;
    self_type *self = self_type::create_leaf(out_ckb, ckb_offset, kernreq);
    self->m_dst_encoding = dst_encoding;
    self->m_src_encoding = src_encoding;
    self->m_next_fn = get_next_unicode_codepoint_function(src_encoding, errmode);
    self->m_append_fn = get_append_unicode_codepoint_function(dst_encoding, errmode);
    self->m_dst_metadata = reinterpret_cast<const string_type_metadata *>(dst_metadata);
    self->m_src_metadata = reinterpret_cast<const string_type_metadata *>(src_metadata);
    return ckb_offset + sizeof(self_type);
}

/////////////////////////////////////////
// fixedstring to blockref string assignment

namespace {
    struct fixedstring_to_blockref_string_assign_ck : public kernels::assignment_ck<fixedstring_to_blockref_string_assign_ck> {
        string_encoding_t m_dst_encoding, m_src_encoding;
        intptr_t m_src_element_size;
        next_unicode_codepoint_t m_next_fn;
        append_unicode_codepoint_t m_append_fn;
        const string_type_metadata *m_dst_metadata;

        inline void single(char *dst, const char *src)
        {
            const string_type_metadata *dst_md = m_dst_metadata;
            string_type_data *dst_d = reinterpret_cast<string_type_data *>(dst);
            intptr_t src_charsize = string_encoding_char_size_table[m_src_encoding];
            intptr_t dst_charsize = string_encoding_char_size_table[m_dst_encoding];

            if (dst_d->begin != NULL) {
                throw runtime_error("Cannot assign to an already initialized dynd string");
            }

            char *dst_begin = NULL, *dst_current, *dst_end = NULL;
            const char *src_begin = src;
            const char *src_end = src + m_src_element_size;
            next_unicode_codepoint_t next_fn = m_next_fn;
            append_unicode_codepoint_t append_fn = m_append_fn;
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
                ckernel_builder *out_ckb, size_t ckb_offset,
                const char *dst_metadata, string_encoding_t dst_encoding,
                intptr_t src_element_size, string_encoding_t src_encoding,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    typedef fixedstring_to_blockref_string_assign_ck self_type;
    self_type *self = self_type::create_leaf(out_ckb, ckb_offset, kernreq);
    self->m_dst_encoding = dst_encoding;
    self->m_src_encoding = src_encoding;
    self->m_src_element_size = src_element_size;
    self->m_next_fn = get_next_unicode_codepoint_function(src_encoding, errmode);
    self->m_append_fn = get_append_unicode_codepoint_function(dst_encoding, errmode);
    self->m_dst_metadata = reinterpret_cast<const string_type_metadata *>(dst_metadata);
    return ckb_offset + sizeof(self_type);
}

/////////////////////////////////////////
// blockref string to fixedstring assignment

namespace {
    struct blockref_string_to_fixedstring_assign_ck : public kernels::assignment_ck<blockref_string_to_fixedstring_assign_ck> {
        next_unicode_codepoint_t m_next_fn;
        append_unicode_codepoint_t m_append_fn;
        intptr_t m_dst_data_size, m_src_element_size;
        bool m_overflow_check;

        inline void single(char *dst, const char *src)
        {
            char *dst_end = dst + m_dst_data_size;
            const string_type_data *src_d = reinterpret_cast<const string_type_data *>(src);
            const char *src_begin = src_d->begin;
            const char *src_end = src_d->end;
            next_unicode_codepoint_t next_fn = m_next_fn;
            append_unicode_codepoint_t append_fn = m_append_fn;
            uint32_t cp;

            while (src_begin < src_end && dst < dst_end) {
                cp = next_fn(src_begin, src_end);
                append_fn(cp, dst, dst_end);
            }
            if (src_begin < src_end) {
                if (m_overflow_check) {
                    throw std::runtime_error("Input string is too large to convert to destination fixed-size string");
                }
            } else if (dst < dst_end) {
                memset(dst, 0, dst_end - dst);
            }
        }
    };
} // anonymous namespace

size_t dynd::make_blockref_string_to_fixedstring_assignment_kernel(
                ckernel_builder *out_ckb, size_t ckb_offset,
                intptr_t dst_data_size, string_encoding_t dst_encoding,
                string_encoding_t src_encoding,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    typedef blockref_string_to_fixedstring_assign_ck self_type;
    self_type *self = self_type::create_leaf(out_ckb, ckb_offset, kernreq);
    self->m_next_fn = get_next_unicode_codepoint_function(src_encoding, errmode);
    self->m_append_fn = get_append_unicode_codepoint_function(dst_encoding, errmode);
    self->m_dst_data_size = dst_data_size;
    self->m_overflow_check = (errmode != assign_error_none);
    return ckb_offset + sizeof(self_type);
}
