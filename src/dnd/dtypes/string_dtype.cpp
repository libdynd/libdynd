//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/dtypes/string_dtype.hpp>
#include <dnd/kernels/single_compare_kernel_instance.hpp>
#include <dnd/kernels/string_assignment_kernels.hpp>
#include <dnd/dtypes/fixedstring_dtype.hpp>
#include <dnd/exceptions.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

dynd::string_dtype::string_dtype(string_encoding_t encoding)
    : m_encoding(encoding)
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

void dynd::string_dtype::print_element(std::ostream& o, const char *data) const
{
    uint32_t cp;
    next_unicode_codepoint_t next_fn;
    next_fn = get_next_unicode_codepoint_function(m_encoding, assign_error_none);
    const char *begin = reinterpret_cast<const char * const *>(data)[0];
    const char *end = reinterpret_cast<const char * const *>(data)[1];

    // Print as an escaped string
    o << "\"";
    while (begin < end) {
        cp = next_fn(begin, end);
        print_escaped_unicode_codepoint(o, cp);
    }
    o << "\"";
}

void dynd::string_dtype::print_dtype(std::ostream& o) const {

    o << "string<" << m_encoding << ">";

}

dtype dynd::string_dtype::apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const
{
    if (nindices == 0) {
        return dtype(this);
    } else if (nindices == 1) {
        if (indices->step() == 0) {
            // Return a fixedstring dtype, since it's always one character.
            // If the string encoding is variable-length switch to UTF32 so that the result can always
            // store a single character.
            return make_fixedstring_dtype(is_variable_length_string_encoding(m_encoding) ? string_encoding_utf_32 : m_encoding, 1);
        } else {
            return dtype(this);
        }
    } else {
        throw too_many_indices(nindices, current_i + 1);
    }
}

void dynd::string_dtype::get_shape(int DND_UNUSED(i), std::vector<intptr_t>& DND_UNUSED(out_shape)) const
{
}

bool dynd::string_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (dst_dt.type_id() == string_type_id) {
            // If the source is a string, only the encoding matters because the dest is variable sized
            const extended_string_dtype *src_esd = static_cast<const extended_string_dtype*>(src_dt.extended());
            string_encoding_t src_encoding = src_esd->encoding();
            switch (m_encoding) {
                case string_encoding_ascii:
                    return src_encoding == string_encoding_ascii;
                case string_encoding_ucs_2:
                    return src_encoding == string_encoding_ascii ||
                            src_encoding == string_encoding_ucs_2;
                case string_encoding_utf_8:
                case string_encoding_utf_16:
                case string_encoding_utf_32:
                    return true;
                default:
                    return false;
            }
        } else {
            return false;
        }
    } else {
        return false;
    }
}

void dynd::string_dtype::get_single_compare_kernel(single_compare_kernel_instance& DND_UNUSED(out_kernel)) const {
    throw std::runtime_error("string_dtype::get_single_compare_kernel not supported yet");
}

void dynd::string_dtype::get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel) const
{
    if (this == dst_dt.extended()) {
        switch (src_dt.type_id()) {
            case string_type_id: {
                const string_dtype *src_fs = static_cast<const string_dtype *>(src_dt.extended());
                get_blockref_string_assignment_kernel(m_encoding, src_fs->m_encoding,
                                        errmode, out_kernel);
                break;
            }
            case fixedstring_type_id: {
                const extended_string_dtype *src_fs = static_cast<const extended_string_dtype *>(src_dt.extended());
                get_fixedstring_to_blockref_string_assignment_kernel(m_encoding,
                                        src_fs->element_size(), src_fs->encoding(),
                                        errmode, out_kernel);
                break;
            }
            default: {
                src_dt.extended()->get_dtype_assignment_kernel(dst_dt, src_dt, errmode, out_kernel);
                break;
            }
        }
    } else {
        throw runtime_error("conversions from string to non-string are not implemented");
    }
}


bool dynd::string_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.type_id() != string_type_id) {
        return false;
    } else {
        const string_dtype *dt = static_cast<const string_dtype*>(&rhs);
        return m_encoding == dt->m_encoding;
    }
}
