//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/string_comparison_kernels.hpp>
#include <dynd/kernels/string_numeric_assignment_kernels.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/gfunc/make_callable.hpp>

using namespace std;
using namespace dynd;

fixedstring_dtype::fixedstring_dtype(intptr_t stringsize, string_encoding_t encoding)
    : base_string_dtype(fixedstring_type_id, 0, 1, dtype_flag_scalar, 0),
            m_stringsize(stringsize), m_encoding(encoding)
{
    switch (encoding) {
        case string_encoding_ascii:
        case string_encoding_utf_8:
            m_members.data_size = m_stringsize;
            m_members.alignment = 1;
            break;
        case string_encoding_ucs_2:
        case string_encoding_utf_16:
            m_members.data_size = m_stringsize * 2;
            m_members.alignment = 2;
            break;
        case string_encoding_utf_32:
            m_members.data_size = m_stringsize * 4;
            m_members.alignment = 4;
            break;
        default:
            throw runtime_error("Unrecognized string encoding in fixedstring dtype constructor");
    }
}

fixedstring_dtype::~fixedstring_dtype()
{
}

void fixedstring_dtype::get_string_range(const char **out_begin, const char**out_end,
                const char *DYND_UNUSED(metadata), const char *data) const
{
    // Beginning of the string
    *out_begin = data;

    switch (string_encoding_char_size_table[m_encoding]) {
        case 1: {
            *out_end = data + strnlen(data, get_data_size());
            break;
        }
        case 2: {
            const uint16_t *ptr = reinterpret_cast<const uint16_t *>(data);
            const uint16_t *ptr_max = ptr + get_data_size() / sizeof(uint16_t);
            while (ptr < ptr_max && *ptr != 0) {
                ++ptr;
            }
            *out_end = reinterpret_cast<const char *>(ptr);
            break;
        }
        case 4: {
            const uint32_t *ptr = reinterpret_cast<const uint32_t *>(data);
            const uint32_t *ptr_max = ptr + get_data_size() / sizeof(uint32_t);
            while (ptr < ptr_max && *ptr != 0) {
                ++ptr;
            }
            *out_end = reinterpret_cast<const char *>(ptr);
            break;
        }
    }
}

void fixedstring_dtype::set_utf8_string(const char *DYND_UNUSED(metadata), char *dst,
                assign_error_mode errmode,
                const char* utf8_begin, const char *utf8_end) const
{
    char *dst_end = dst + get_data_size();
    next_unicode_codepoint_t next_fn = get_next_unicode_codepoint_function(string_encoding_utf_8, errmode);
    append_unicode_codepoint_t append_fn = get_append_unicode_codepoint_function(m_encoding, errmode);
    uint32_t cp;

    while (utf8_begin < utf8_end && dst < dst_end) {
        cp = next_fn(utf8_begin, utf8_end);
        append_fn(cp, dst, dst_end);
    }
    if (utf8_begin < utf8_end) {
        if (errmode != assign_error_none) {
            throw std::runtime_error("Input is too large to convert to destination fixed-size string");
        }
    } else if (dst < dst_end) {
        memset(dst, 0, dst_end - dst);
    }
}

void fixedstring_dtype::print_data(std::ostream& o, const char *DYND_UNUSED(metadata), const char *data) const
{
    uint32_t cp;
    next_unicode_codepoint_t next_fn;
    next_fn = get_next_unicode_codepoint_function(m_encoding, assign_error_none);
    const char *data_end = data + get_data_size();

    // Print as an escaped string
    o << "\"";
    while (data < data_end) {
        cp = next_fn(data, data_end);
        if (cp != 0) {
            print_escaped_unicode_codepoint(o, cp);
        } else {
            break;
        }
    }
    o << "\"";
}

void fixedstring_dtype::print_dtype(std::ostream& o) const
{
    o << "string<" << m_stringsize;
    if (m_encoding != string_encoding_utf_8) {
        o << ",'" << m_encoding << "'";
    }
    o << ">";
}

dtype fixedstring_dtype::get_canonical_dtype() const
{
    return dtype(this, true);
}

bool fixedstring_dtype::is_lossless_assignment(
                const dtype& DYND_UNUSED(dst_dt),
                const dtype& DYND_UNUSED(src_dt)) const
{
    // Don't shortcut anything to 'none' error checking, so that
    // decoding errors get caught appropriately.
    return false;
}

bool fixedstring_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != fixedstring_type_id) {
        return false;
    } else {
        const fixedstring_dtype *dt = static_cast<const fixedstring_dtype*>(&rhs);
        return m_encoding == dt->m_encoding && m_stringsize == dt->m_stringsize;
    }
}

size_t fixedstring_dtype::make_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_dt.extended()) {
        switch (src_dt.get_type_id()) {
            case fixedstring_type_id: {
                const fixedstring_dtype *src_fs = static_cast<const fixedstring_dtype *>(src_dt.extended());
                return make_fixedstring_assignment_kernel(out, offset_out,
                                get_data_size(), m_encoding, src_fs->get_data_size(), src_fs->m_encoding,
                                kernreq, errmode, ectx);
            }
            case string_type_id: {
                const base_string_dtype *src_fs = static_cast<const base_string_dtype *>(src_dt.extended());
                return make_blockref_string_to_fixedstring_assignment_kernel(out, offset_out,
                                get_data_size(), m_encoding, src_fs->get_encoding(),
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

size_t fixedstring_dtype::make_comparison_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const dtype& src0_dt, const char *src0_metadata,
                const dtype& src1_dt, const char *src1_metadata,
                comparison_type_t comptype,
                const eval::eval_context *ectx) const
{
    if (this == src0_dt.extended()) {
        if (*this == *src1_dt.extended()) {
            return make_fixedstring_comparison_kernel(out, offset_out,
                            m_stringsize, m_encoding,
                            comptype, ectx);
        } else if (src1_dt.get_kind() == string_kind) {
            return make_general_string_comparison_kernel(out, offset_out,
                            src0_dt, src0_metadata,
                            src1_dt, src1_metadata,
                            comptype, ectx);
        } else if (!src1_dt.is_builtin()) {
            return src1_dt.extended()->make_comparison_kernel(out, offset_out,
                            src0_dt, src0_metadata,
                            src1_dt, src1_metadata,
                            comptype, ectx);
        }
    }

    throw not_comparable_error(src0_dt, src1_dt, comptype);
}
