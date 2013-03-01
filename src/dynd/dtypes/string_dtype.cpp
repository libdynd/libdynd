//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/string_comparison_kernels.hpp>
#include <dynd/kernels/string_numeric_assignment_kernels.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/exceptions.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

string_dtype::string_dtype(string_encoding_t encoding)
    : base_string_dtype(string_type_id, sizeof(string_dtype_data),
                    sizeof(const char *), dtype_flag_scalar|dtype_flag_zeroinit,
                    sizeof(string_dtype_metadata)),
            m_encoding(encoding)
{
    switch (encoding) {
        case string_encoding_ascii:
        case string_encoding_ucs_2:
        case string_encoding_utf_8:
        case string_encoding_utf_16:
        case string_encoding_utf_32:
            break;
        default:
            throw runtime_error("Unrecognized string encoding in string dtype constructor");
    }
}

string_dtype::~string_dtype()
{
}

void string_dtype::get_string_range(const char **out_begin, const char**out_end,
                const char *DYND_UNUSED(metadata), const char *data) const
{
    *out_begin = reinterpret_cast<const string_dtype_data *>(data)->begin;
    *out_end = reinterpret_cast<const string_dtype_data *>(data)->end;
}

void string_dtype::set_utf8_string(const char *data_metadata, char *data,
                assign_error_mode errmode, const char* utf8_begin, const char *utf8_end) const
{
    const string_dtype_metadata *data_md = reinterpret_cast<const string_dtype_metadata *>(data_metadata);
    const intptr_t src_charsize = 1;
    intptr_t dst_charsize = string_encoding_char_size_table[m_encoding];
    char *dst_begin = NULL, *dst_current, *dst_end = NULL;
    next_unicode_codepoint_t next_fn = get_next_unicode_codepoint_function(string_encoding_utf_8, errmode);
    append_unicode_codepoint_t append_fn = get_append_unicode_codepoint_function(m_encoding, errmode);
    uint32_t cp;

    memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(data_md->blockref);

    // Allocate the initial output as the src number of characters + some padding
    // TODO: Don't add padding if the output is not a multi-character encoding
    allocator->allocate(data_md->blockref, ((utf8_end - utf8_begin) / src_charsize + 16) * dst_charsize * 1124 / 1024,
                    dst_charsize, &dst_begin, &dst_end);

    dst_current = dst_begin;
    while (utf8_begin < utf8_end) {
        cp = next_fn(utf8_begin, utf8_end);
        // Append the codepoint, or increase the allocated memory as necessary
        if (dst_end - dst_current >= 8) {
            append_fn(cp, dst_current, dst_end);
        } else {
            char *dst_begin_saved = dst_begin;
            allocator->resize(data_md->blockref, 2 * (dst_end - dst_begin), &dst_begin, &dst_end);
            dst_current = dst_begin + (dst_current - dst_begin_saved);

            append_fn(cp, dst_current, dst_end);
        }
    }

    // Shrink-wrap the memory to just fit the string
    allocator->resize(data_md->blockref, dst_current - dst_begin, &dst_begin, &dst_end);

    // Set the output
    reinterpret_cast<string_dtype_data *>(data)->begin = dst_begin;
    reinterpret_cast<string_dtype_data*>(data)->end = dst_end;
}

void string_dtype::print_data(std::ostream& o, const char *DYND_UNUSED(metadata), const char *data) const
{
    uint32_t cp;
    next_unicode_codepoint_t next_fn;
    next_fn = get_next_unicode_codepoint_function(m_encoding, assign_error_none);
    const char *begin = reinterpret_cast<const string_dtype_data *>(data)->begin;
    const char *end = reinterpret_cast<const string_dtype_data *>(data)->end;

    // Print as an escaped string
    o << "\"";
    while (begin < end) {
        cp = next_fn(begin, end);
        print_escaped_unicode_codepoint(o, cp);
    }
    o << "\"";
}

void string_dtype::print_dtype(std::ostream& o) const {

    o << "string";
    if (m_encoding != string_encoding_utf_8) {
        o << "<'" << m_encoding << "'>";
    }
}

bool string_dtype::is_unique_data_owner(const char *metadata) const
{
    const string_dtype_metadata *md = reinterpret_cast<const string_dtype_metadata *>(metadata);
    if (md->blockref != NULL &&
            (md->blockref->m_use_count != 1 ||
             md->blockref->m_type != pod_memory_block_type)) {
        return false;
    }
    return true;
}

dtype string_dtype::get_canonical_dtype() const
{
    return dtype(this, true);
}

void string_dtype::get_shape(size_t DYND_UNUSED(i),
                intptr_t *DYND_UNUSED(out_shape)) const
{
}

void string_dtype::get_shape(size_t DYND_UNUSED(i),
                intptr_t *DYND_UNUSED(out_shape), const char *DYND_UNUSED(metadata)) const
{
}

bool string_dtype::is_lossless_assignment(
                const dtype& DYND_UNUSED(dst_dt),
                const dtype& DYND_UNUSED(src_dt)) const
{
    // Don't shortcut anything to 'none' error checking, so that
    // decoding errors get caught appropriately.
    return false;
}

bool string_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != string_type_id) {
        return false;
    } else {
        const string_dtype *dt = static_cast<const string_dtype*>(&rhs);
        return m_encoding == dt->m_encoding;
    }
}

void string_dtype::metadata_default_construct(char *metadata, size_t DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const
{
    // Simply allocate a POD memory block
    string_dtype_metadata *md = reinterpret_cast<string_dtype_metadata *>(metadata);
    md->blockref = make_pod_memory_block().release();
}

void string_dtype::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    // Copy the blockref, switching it to the embedded_reference if necessary
    const string_dtype_metadata *src_md = reinterpret_cast<const string_dtype_metadata *>(src_metadata);
    string_dtype_metadata *dst_md = reinterpret_cast<string_dtype_metadata *>(dst_metadata);
    dst_md->blockref = src_md->blockref ? src_md->blockref : embedded_reference;
    if (dst_md->blockref) {
        memory_block_incref(dst_md->blockref);
    }
}

void string_dtype::metadata_reset_buffers(char *metadata) const
{
    const string_dtype_metadata *md = reinterpret_cast<const string_dtype_metadata *>(metadata);
    if (md->blockref != NULL && md->blockref->m_type == pod_memory_block_type) {
        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        allocator->reset(md->blockref);
    } else {
        throw runtime_error("can only reset the buffers of a dynd string "
                        "dtype if the memory block reference was constructed by default");
    }
}

void string_dtype::metadata_finalize_buffers(char *metadata) const
{
    string_dtype_metadata *md = reinterpret_cast<string_dtype_metadata *>(metadata);
    if (md->blockref != NULL) {
        // Finalize the memory block
        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        if (allocator != NULL) {
            allocator->finalize(md->blockref);
        }
    }
}

void string_dtype::metadata_destruct(char *metadata) const
{
    string_dtype_metadata *md = reinterpret_cast<string_dtype_metadata *>(metadata);
    if (md->blockref) {
        memory_block_decref(md->blockref);
    }
}

void string_dtype::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    const string_dtype_metadata *md = reinterpret_cast<const string_dtype_metadata *>(metadata);
    o << indent << "string metadata\n";
    memory_block_debug_print(md->blockref, o, indent + " ");
}

size_t string_dtype::make_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_dt.extended()) {
        switch (src_dt.get_type_id()) {
            case string_type_id: {
                return make_blockref_string_assignment_kernel(out, offset_out,
                                dst_metadata, get_encoding(),
                                src_metadata, static_cast<const base_string_dtype *>(src_dt.extended())->get_encoding(),
                                kernreq, errmode, ectx);
            }
            case fixedstring_type_id: {
                return make_fixedstring_to_blockref_string_assignment_kernel(out, offset_out,
                                dst_metadata, get_encoding(),
                                src_dt.get_data_size(),
                                static_cast<const base_string_dtype *>(src_dt.extended())->get_encoding(),
                                kernreq, errmode, ectx);
            }
            default: {
                if (!src_dt.is_builtin()) {
                    return src_dt.extended()->make_assignment_kernel(out, offset_out,
                                    dst_dt, dst_metadata,
                                    src_dt, src_metadata,
                                    kernreq, errmode, ectx);
                } else {
                    return make_builtin_to_string_assignment_kernel(out, offset_out,
                                dst_dt, dst_metadata,
                                src_dt.get_type_id(),
                                kernreq, errmode, ectx);
                }
            }
        }
    } else {
        if (dst_dt.is_builtin()) {
            return make_string_to_builtin_assignment_kernel(out, offset_out,
                            dst_dt.get_type_id(),
                            src_dt, src_metadata,
                            kernreq, errmode, ectx);
        } else {
            stringstream ss;
            ss << "Cannot assign from " << src_dt << " to " << dst_dt;
            throw runtime_error(ss.str());
        }
    }
}

size_t string_dtype::make_comparison_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const dtype& src0_dt, const char *src0_metadata,
                const dtype& src1_dt, const char *src1_metadata,
                comparison_type_t comptype,
                const eval::eval_context *ectx) const
{
    if (this == src0_dt.extended()) {
        if (*this == *src1_dt.extended()) {
            return make_string_comparison_kernel(out, offset_out,
                            m_encoding,
                            comptype, ectx);
        } else if (src1_dt.get_kind() == string_kind) {
            return make_general_string_comparison_kernel(out, offset_out,
                            src0_dt, src0_metadata,
                            src1_dt, src1_metadata,
                            comptype, ectx);
        }
    }

    throw not_comparable_error(src0_dt, src1_dt, comptype);
}
