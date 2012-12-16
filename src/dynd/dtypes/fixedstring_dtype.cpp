//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/kernels/single_compare_kernel_instance.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/string_numeric_assignment_kernels.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/gfunc/make_callable.hpp>

using namespace std;
using namespace dynd;

fixedstring_dtype::fixedstring_dtype(string_encoding_t encoding, intptr_t stringsize)
    : extended_string_dtype(fixedstring_type_id, string_kind, 0, 1),
            m_stringsize(stringsize), m_encoding(encoding)
{
    switch (encoding) {
        case string_encoding_ascii:
        case string_encoding_utf_8:
            m_data_size = m_stringsize;
            m_alignment = 1;
            break;
        case string_encoding_ucs_2:
        case string_encoding_utf_16:
            m_data_size = m_stringsize * 2;
            m_alignment = 2;
            break;
        case string_encoding_utf_32:
            m_data_size = m_stringsize * 4;
            m_alignment = 4;
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
            *out_end = data + strnlen(data, m_data_size);
            break;
        }
        case 2: {
            const uint16_t *ptr = reinterpret_cast<const uint16_t *>(data);
            const uint16_t *ptr_max = ptr + m_data_size / sizeof(uint16_t);
            while (ptr < ptr_max && *ptr != 0) {
                ++ptr;
            }
            *out_end = reinterpret_cast<const char *>(ptr);
            break;
        }
        case 4: {
            const uint32_t *ptr = reinterpret_cast<const uint32_t *>(data);
            const uint32_t *ptr_max = ptr + m_data_size / sizeof(uint32_t);
            while (ptr < ptr_max && *ptr != 0) {
                ++ptr;
            }
            *out_end = reinterpret_cast<const char *>(ptr);
            break;
        }
    }
}

void fixedstring_dtype::set_utf8_string(const char *DYND_UNUSED(metadata), char *dst, assign_error_mode errmode, const std::string& utf8_str) const
{
    char *dst_end = dst + m_data_size;
    const char *src = utf8_str.data();
    const char *src_end = src + utf8_str.size();
    next_unicode_codepoint_t next_fn = get_next_unicode_codepoint_function(string_encoding_utf_8, errmode);
    append_unicode_codepoint_t append_fn = get_append_unicode_codepoint_function(m_encoding, errmode);
    uint32_t cp;

    while (src < src_end && dst < dst_end) {
        cp = next_fn(src, src_end);
        append_fn(cp, dst, dst_end);
    }
    if (src < src_end) {
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
    const char *data_end = data + m_data_size;

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
    o << "fixedstring<" << m_encoding << "," << m_stringsize << ">";
}

dtype fixedstring_dtype::apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& DYND_UNUSED(root_dt)) const
{
    if (nindices == 0) {
        return dtype(this, true);
    } else if (nindices == 1) {
        if (indices->step() == 0) {
            // If the string encoding is variable-length switch to UTF32 so that the result can always
            // store a single character.
            return make_fixedstring_dtype(is_variable_length_string_encoding(m_encoding) ? string_encoding_utf_32 : m_encoding, 1);
        } else {
            // Just use the same string width, no big reason to be "too smart" and shrink it
            return dtype(this, true);
        }
    } else {
        throw too_many_indices(nindices, current_i + 1);
    }
}

dtype fixedstring_dtype::get_canonical_dtype() const
{
    return dtype(this, true);
}

bool fixedstring_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.extended() == this) {
            return true;
        } else if (src_dt.get_type_id() == fixedstring_type_id) {
            const fixedstring_dtype *src_fs = static_cast<const fixedstring_dtype*>(src_dt.extended());
            if (m_encoding == src_fs->m_encoding) {
                return m_stringsize >= src_fs->m_stringsize;
            } else if (m_stringsize < src_fs->m_stringsize) {
                return false;
            } else {
                switch (src_fs->m_encoding) {
                    case string_encoding_ascii:
                        return true;
                    case string_encoding_utf_8:
                    case string_encoding_ucs_2:
                        return m_encoding == string_encoding_utf_16 ||
                                m_encoding == string_encoding_utf_32;
                    case string_encoding_utf_16:
                        return m_encoding == string_encoding_utf_32;
                    default:
                        return false;
                }
            }
        } else {
            return false;
        }
    } else {
        return false;
    }
}

namespace {

    struct ascii_utf8_compare_kernel {
        static bool less(const char *a, const char *b, const AuxDataBase *auxdata) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>(auxdata)>>1;
            return strncmp(a, b, stringsize) < 0;
        }

        static bool less_equal(const char *a, const char *b, const AuxDataBase *auxdata) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>(auxdata)>>1;
            return strncmp(a, b, stringsize) <= 0;
        }

        static bool equal(const char *a, const char *b, const AuxDataBase *auxdata) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>(auxdata)>>1;
            return strncmp(a, b, stringsize) == 0;
        }

        static bool not_equal(const char *a, const char *b, const AuxDataBase *auxdata) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>(auxdata)>>1;
            return strncmp(a, b, stringsize) != 0;
        }

        static bool greater_equal(const char *a, const char *b, const AuxDataBase *auxdata) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>(auxdata)>>1;
            return strncmp(a, b, stringsize) >= 0;
        }

        static bool greater(const char *a, const char *b, const AuxDataBase *auxdata) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>(auxdata)>>1;
            return strncmp(a, b, stringsize) > 0;
        }
    };

    struct utf16_compare_kernel {
        static bool less(const char *a, const char *b, const AuxDataBase *auxdata) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>(auxdata)>>1;
            return lexicographical_compare(
                reinterpret_cast<const uint16_t *>(a), reinterpret_cast<const uint16_t *>(a) + stringsize,
                reinterpret_cast<const uint16_t *>(b), reinterpret_cast<const uint16_t *>(b) + stringsize
            );
        }

        static bool less_equal(const char *a, const char *b, const AuxDataBase *auxdata) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>(auxdata)>>1;
            return !lexicographical_compare(
                reinterpret_cast<const uint16_t *>(b), reinterpret_cast<const uint16_t *>(b) + stringsize,
                reinterpret_cast<const uint16_t *>(a), reinterpret_cast<const uint16_t *>(a) + stringsize
            );
        }

        static bool equal(const char *a, const char *b, const AuxDataBase *auxdata) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>(auxdata)>>1;
            for (uint16_t *lhs = reinterpret_cast<uint16_t *>(const_cast<char *>(a)),
                    *rhs = reinterpret_cast<uint16_t *>(const_cast<char *>(b));
                    lhs < lhs + stringsize; ++lhs, ++rhs) {
                if (*lhs != *rhs) {
                    return false;
                }
            }
            return true;
        }

        static bool not_equal(const char *a, const char *b, const AuxDataBase *auxdata) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>(auxdata)>>1;
            for (uint16_t *lhs = reinterpret_cast<uint16_t *>(const_cast<char *>(a)),
                    *rhs = reinterpret_cast<uint16_t *>(const_cast<char *>(b));
                    lhs < lhs + stringsize; ++lhs, ++rhs) {
                if (*lhs != *rhs) {
                    return true;
                }
            }
            return false;
        }

        static bool greater_equal(const char *a, const char *b, const AuxDataBase *auxdata) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>(auxdata)>>1;
            return !lexicographical_compare(
                reinterpret_cast<const uint16_t *>(a), reinterpret_cast<const uint16_t *>(a) + stringsize,
                reinterpret_cast<const uint16_t *>(b), reinterpret_cast<const uint16_t *>(b) + stringsize
            );
        }

        static bool greater(const char *a, const char *b, const AuxDataBase *auxdata) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>(auxdata)>>1;
            return lexicographical_compare(
                reinterpret_cast<const uint16_t *>(b), reinterpret_cast<const uint16_t *>(b) + stringsize,
                reinterpret_cast<const uint16_t *>(a), reinterpret_cast<const uint16_t *>(a) + stringsize
            );
        }
    };

    struct utf32_compare_kernel {
        static bool less(const char *a, const char *b, const AuxDataBase *auxdata) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>(auxdata)>>1;
            return lexicographical_compare(
                reinterpret_cast<const uint32_t *>(a), reinterpret_cast<const uint32_t *>(a) + stringsize,
                reinterpret_cast<const uint32_t *>(b), reinterpret_cast<const uint32_t *>(b) + stringsize
            );
        }

        static bool less_equal(const char *a, const char *b, const AuxDataBase *auxdata) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>(auxdata)>>1;
            return !lexicographical_compare(
                reinterpret_cast<const uint32_t *>(b), reinterpret_cast<const uint32_t *>(b) + stringsize,
                reinterpret_cast<const uint32_t *>(a), reinterpret_cast<const uint32_t *>(a) + stringsize
            );
        }

        static bool equal(const char *a, const char *b, const AuxDataBase *auxdata) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>(auxdata)>>1;
            for (uint32_t *lhs = reinterpret_cast<uint32_t *>(const_cast<char *>(a)),
                    *rhs = reinterpret_cast<uint32_t *>(const_cast<char *>(b));
                    lhs < lhs + stringsize; ++lhs, ++rhs) {
                if (*lhs != *rhs) {
                    return false;
                }
            }
            return true;
        }

        static bool not_equal(const char *a, const char *b, const AuxDataBase *auxdata) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>(auxdata)>>1;
            for (uint32_t *lhs = reinterpret_cast<uint32_t *>(const_cast<char *>(a)),
                    *rhs = reinterpret_cast<uint32_t *>(const_cast<char *>(b));
                    lhs < lhs + stringsize; ++lhs, ++rhs) {
                if (*lhs != *rhs) {
                    return true;
                }
            }
            return false;
        }

        static bool greater_equal(const char *a, const char *b, const AuxDataBase *auxdata) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>(auxdata)>>1;
            return !lexicographical_compare(
                reinterpret_cast<const uint32_t *>(a), reinterpret_cast<const uint32_t *>(a) + stringsize,
                reinterpret_cast<const uint32_t *>(b), reinterpret_cast<const uint32_t *>(b) + stringsize
            );
        }

        static bool greater(const char *a, const char *b, const AuxDataBase *auxdata) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>(auxdata)>>1;
            return lexicographical_compare(
                reinterpret_cast<const uint32_t *>(b), reinterpret_cast<const uint32_t *>(b) + stringsize,
                reinterpret_cast<const uint32_t *>(a), reinterpret_cast<const uint32_t *>(a) + stringsize
            );
        }
    };

} // anonymous namespace

#define DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL(type) { \
    (single_compare_operation_t)type##_compare_kernel::less, \
    (single_compare_operation_t)type##_compare_kernel::less_equal, \
    (single_compare_operation_t)type##_compare_kernel::equal, \
    (single_compare_operation_t)type##_compare_kernel::not_equal, \
    (single_compare_operation_t)type##_compare_kernel::greater_equal, \
    (single_compare_operation_t)type##_compare_kernel::greater \
    }

void fixedstring_dtype::get_single_compare_kernel(single_compare_kernel_instance& out_kernel) const {
    static single_compare_operation_table_t fixedstring_comparisons_table[3] = {
        DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL(ascii_utf8),
        DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL(utf16),
        DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL(utf32)
    };
    static int lookup[5] = {0, 1, 0, 1, 2};
    out_kernel.comparisons = fixedstring_comparisons_table[lookup[m_encoding]];
    make_raw_auxiliary_data(out_kernel.auxdata, static_cast<uintptr_t>(m_stringsize)<<1);
}

#undef DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL

void fixedstring_dtype::get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel) const
{
    if (this == dst_dt.extended()) {
        switch (src_dt.get_type_id()) {
            case fixedstring_type_id: {
                const fixedstring_dtype *src_fs = static_cast<const fixedstring_dtype *>(src_dt.extended());
                get_fixedstring_assignment_kernel(m_data_size, m_encoding, src_fs->m_data_size, src_fs->m_encoding,
                                        errmode, out_kernel);
                break;
            }
            case string_type_id: {
                const extended_string_dtype *src_fs = static_cast<const extended_string_dtype *>(src_dt.extended());
                get_blockref_string_to_fixedstring_assignment_kernel(m_data_size, m_encoding, src_fs->get_encoding(),
                                        errmode, out_kernel);
                break;
            }
            default: {
                if (!src_dt.is_builtin()) {
                    src_dt.extended()->get_dtype_assignment_kernel(dst_dt, src_dt, errmode, out_kernel);
                } else {
                    get_builtin_to_string_assignment_kernel(dst_dt, src_dt.get_type_id(), errmode, out_kernel);
                }
                break;
            }
        }
    } else {
        if (dst_dt.is_builtin()) {
            get_string_to_builtin_assignment_kernel(dst_dt.get_type_id(), src_dt, errmode, out_kernel);
        } else {
            stringstream ss;
            ss << "assignment from " << src_dt << " to " << dst_dt << " is not implemented yet";
            throw runtime_error(ss.str());
        }
    }
}


bool fixedstring_dtype::operator==(const extended_dtype& rhs) const
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
