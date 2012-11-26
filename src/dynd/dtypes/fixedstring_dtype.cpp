//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/kernels/single_compare_kernel_instance.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/exceptions.hpp>

using namespace std;
using namespace dynd;

dynd::fixedstring_dtype::fixedstring_dtype(string_encoding_t encoding, intptr_t stringsize)
    : m_stringsize(stringsize), m_encoding(encoding)
{
    switch (encoding) {
        case string_encoding_ascii:
        case string_encoding_utf_8:
            m_element_size = m_stringsize;
            m_alignment = 1;
            break;
        case string_encoding_ucs_2:
        case string_encoding_utf_16:
            m_element_size = m_stringsize * 2;
            m_alignment = 2;
            break;
        case string_encoding_utf_32:
            m_element_size = m_stringsize * 4;
            m_alignment = 4;
            break;
        default:
            throw runtime_error("Unrecognized string encoding in fixedstring dtype constructor");
    }
}

void dynd::fixedstring_dtype::print_element(std::ostream& o, const char *data, const char *DYND_UNUSED(metadata)) const
{
    uint32_t cp;
    next_unicode_codepoint_t next_fn;
    next_fn = get_next_unicode_codepoint_function(m_encoding, assign_error_none);
    const char *data_end = data + m_element_size;

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

void dynd::fixedstring_dtype::print_dtype(std::ostream& o) const
{
    o << "fixedstring<" << m_encoding << "," << m_stringsize << ">";
}

dtype dynd::fixedstring_dtype::apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& DYND_UNUSED(root_dt)) const
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

dtype dynd::fixedstring_dtype::get_canonical_dtype() const
{
    return dtype(new string_dtype(string_encoding_utf_8));
}

bool dynd::fixedstring_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.extended() == this) {
            return true;
        } else if (src_dt.type_id() == fixedstring_type_id) {
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

void dynd::fixedstring_dtype::get_single_compare_kernel(single_compare_kernel_instance& out_kernel) const {
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

void dynd::fixedstring_dtype::get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel) const
{
    if (this == dst_dt.extended()) {
        switch (src_dt.type_id()) {
            case fixedstring_type_id: {
                const fixedstring_dtype *src_fs = static_cast<const fixedstring_dtype *>(src_dt.extended());
                get_fixedstring_assignment_kernel(m_element_size, m_encoding, src_fs->m_element_size, src_fs->m_encoding,
                                        errmode, out_kernel);
                break;
            }
            case string_type_id: {
                const extended_string_dtype *src_fs = static_cast<const extended_string_dtype *>(src_dt.extended());
                get_blockref_string_to_fixedstring_assignment_kernel(m_element_size, m_encoding, src_fs->encoding(),
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


bool dynd::fixedstring_dtype::operator==(const extended_dtype& rhs) const
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
