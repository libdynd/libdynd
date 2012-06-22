//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/dtypes/fixedstring_dtype.hpp>

using namespace std;
using namespace dnd;

dnd::fixedstring_dtype::fixedstring_dtype(string_encoding_t encoding, intptr_t stringsize)
    : m_stringsize(stringsize), m_encoding(encoding)
{
    switch (encoding) {
        case string_encoding_ascii:
        case string_encoding_utf8:
            m_element_size = m_stringsize;
            m_alignment = 1;
            break;
        case string_encoding_utf16:
            m_element_size = m_stringsize * 2;
            m_alignment = 2;
            break;
        case string_encoding_utf32:
            m_element_size = m_stringsize * 4;
            m_alignment = 4;
            break;
        default:
            throw runtime_error("Unrecognized string encoding in fixedstring dtype constructor");
    }
}

void dnd::fixedstring_dtype::print_element(std::ostream& o, const char *data) const
{
    o << "\"";
    o << "\"";
}

void dnd::fixedstring_dtype::print_dtype(std::ostream& o) const
{
    o << "fixedstring<" << m_stringsize << ",encoding=" << m_encoding << ">";
}

bool dnd::fixedstring_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (&dst_dt == &src_dt) {
        return true;
    } else if (dst_dt.type_id() != fixedstring_type_id || src_dt.type_id() != fixedstring_type_id) {
        return false;
    } else {
        const fixedstring_dtype *dst_fs = static_cast<const fixedstring_dtype*>(dst_dt.extended());
        const fixedstring_dtype *src_fs = static_cast<const fixedstring_dtype*>(src_dt.extended());
        if (dst_fs->m_encoding == src_fs->m_encoding) {
            return dst_fs->m_stringsize >= src_fs->m_stringsize;
        } else if (dst_fs->m_stringsize < src_fs->m_stringsize) {
            return false;
        } else {
            switch (src_fs->m_encoding) {
                case string_encoding_ascii:
                    return true;
                case string_encoding_utf8:
                    return dst_fs->m_encoding == string_encoding_utf16 ||
                            dst_fs->m_encoding == string_encoding_utf32;
                case string_encoding_utf16:
                    return dst_fs->m_encoding == string_encoding_utf32;
                default:
                    return false;
            }
        }
    }
}

bool dnd::fixedstring_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.type_id() != fixedstring_type_id) {
        return false;
    } else {
        const fixedstring_dtype *dt = static_cast<const fixedstring_dtype*>(&rhs);
        return m_encoding == dt->m_encoding && m_stringsize == dt->m_stringsize;
    }
}
