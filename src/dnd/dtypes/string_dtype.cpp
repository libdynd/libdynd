//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/dtypes/string_dtype.hpp>
#include <dnd/kernels/single_compare_kernel_instance.hpp>
#include <dnd/kernels/string_assignment_kernels.hpp>

#include <algorithm>

using namespace std;
using namespace dnd;

dnd::string_dtype::string_dtype(string_encoding_t encoding)
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

void dnd::string_dtype::print_element(std::ostream& o, const char *data) const
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
        if (cp == 0) {
            break;
        } else if (cp < 0x80) {
            switch (cp) {
                case '\n':
                    o << "\\n";
                    break;
                case '\r':
                    o << "\\r";
                    break;
                case '\t':
                    o << "\\t";
                    break;
                case '\\':
                    o << "\\\\";
                    break;
                case '\"':
                    o << "\\\"";
                    break;
                default:
                    if (cp < 32) {
                        o << "\\x";
                        hexadecimal_print(o, static_cast<char>(cp));
                    } else {
                        o << static_cast<char>(cp);
                    }
                    break;
            }
        } else if (cp < 0x10000) {
            o << "\\u";
            hexadecimal_print(o, static_cast<uint16_t>(cp));
        } else {
            o << "\\U";
            hexadecimal_print(o, static_cast<uint32_t>(cp));
        }
    }
    o << "\"";
}

void dnd::string_dtype::print_dtype(std::ostream& o) const {

    o << "string<" << m_encoding << ">";

}

bool dnd::string_dtype::is_lossless_assignment(const dtype& /*dst_dt*/, const dtype& /*src_dt*/) const {

    return false; // TODO

}

void dnd::string_dtype::get_single_compare_kernel(single_compare_kernel_instance& DND_UNUSED(out_kernel)) const {
    throw std::runtime_error("string_dtype::get_single_compare_kernel not supported yet");
}

void dnd::string_dtype::get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
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


bool dnd::string_dtype::operator==(const extended_dtype& rhs) const
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
