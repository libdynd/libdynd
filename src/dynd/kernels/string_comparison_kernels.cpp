//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/dtype.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/string_comparison_kernels.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>

using namespace std;
using namespace dynd;

/////////////////////////////////////////
// fixedstring comparison

namespace {
    struct ascii_utf8_fixedstring_compare_kernel {
        static bool less(const char *a, const char *b, single_compare_static_data *extra) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>((const AuxDataBase *)extra->auxdata)>>1;
            return strncmp(a, b, stringsize) < 0;
        }

        static bool less_equal(const char *a, const char *b, single_compare_static_data *extra) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>((const AuxDataBase *)extra->auxdata)>>1;
            return strncmp(a, b, stringsize) <= 0;
        }

        static bool equal(const char *a, const char *b, single_compare_static_data *extra) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>((const AuxDataBase *)extra->auxdata)>>1;
            return strncmp(a, b, stringsize) == 0;
        }

        static bool not_equal(const char *a, const char *b, single_compare_static_data *extra) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>((const AuxDataBase *)extra->auxdata)>>1;
            return strncmp(a, b, stringsize) != 0;
        }

        static bool greater_equal(const char *a, const char *b, single_compare_static_data *extra) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>((const AuxDataBase *)extra->auxdata)>>1;
            return strncmp(a, b, stringsize) >= 0;
        }

        static bool greater(const char *a, const char *b, single_compare_static_data *extra) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>((const AuxDataBase *)extra->auxdata)>>1;
            return strncmp(a, b, stringsize) > 0;
        }
    };

    struct utf16_fixedstring_compare_kernel {
        static bool less(const char *a, const char *b, single_compare_static_data *extra) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>((const AuxDataBase *)extra->auxdata)>>1;
            return lexicographical_compare(
                reinterpret_cast<const uint16_t *>(a), reinterpret_cast<const uint16_t *>(a) + stringsize,
                reinterpret_cast<const uint16_t *>(b), reinterpret_cast<const uint16_t *>(b) + stringsize
            );
        }

        static bool less_equal(const char *a, const char *b, single_compare_static_data *extra) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>((const AuxDataBase *)extra->auxdata)>>1;
            return !lexicographical_compare(
                reinterpret_cast<const uint16_t *>(b), reinterpret_cast<const uint16_t *>(b) + stringsize,
                reinterpret_cast<const uint16_t *>(a), reinterpret_cast<const uint16_t *>(a) + stringsize
            );
        }

        static bool equal(const char *a, const char *b, single_compare_static_data *extra) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>((const AuxDataBase *)extra->auxdata)>>1;
            for (uint16_t *lhs = reinterpret_cast<uint16_t *>(const_cast<char *>(a)),
                    *rhs = reinterpret_cast<uint16_t *>(const_cast<char *>(b));
                    lhs < lhs + stringsize; ++lhs, ++rhs) {
                if (*lhs != *rhs) {
                    return false;
                }
            }
            return true;
        }

        static bool not_equal(const char *a, const char *b, single_compare_static_data *extra) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>((const AuxDataBase *)extra->auxdata)>>1;
            for (uint16_t *lhs = reinterpret_cast<uint16_t *>(const_cast<char *>(a)),
                    *rhs = reinterpret_cast<uint16_t *>(const_cast<char *>(b));
                    lhs < lhs + stringsize; ++lhs, ++rhs) {
                if (*lhs != *rhs) {
                    return true;
                }
            }
            return false;
        }

        static bool greater_equal(const char *a, const char *b, single_compare_static_data *extra) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>((const AuxDataBase *)extra->auxdata)>>1;
            return !lexicographical_compare(
                reinterpret_cast<const uint16_t *>(a), reinterpret_cast<const uint16_t *>(a) + stringsize,
                reinterpret_cast<const uint16_t *>(b), reinterpret_cast<const uint16_t *>(b) + stringsize
            );
        }

        static bool greater(const char *a, const char *b, single_compare_static_data *extra) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>((const AuxDataBase *)extra->auxdata)>>1;
            return lexicographical_compare(
                reinterpret_cast<const uint16_t *>(b), reinterpret_cast<const uint16_t *>(b) + stringsize,
                reinterpret_cast<const uint16_t *>(a), reinterpret_cast<const uint16_t *>(a) + stringsize
            );
        }
    };

    struct utf32_fixedstring_compare_kernel {
        static bool less(const char *a, const char *b, single_compare_static_data *extra) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>((const AuxDataBase *)extra->auxdata)>>1;
            return lexicographical_compare(
                reinterpret_cast<const uint32_t *>(a), reinterpret_cast<const uint32_t *>(a) + stringsize,
                reinterpret_cast<const uint32_t *>(b), reinterpret_cast<const uint32_t *>(b) + stringsize
            );
        }

        static bool less_equal(const char *a, const char *b, single_compare_static_data *extra) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>((const AuxDataBase *)extra->auxdata)>>1;
            return !lexicographical_compare(
                reinterpret_cast<const uint32_t *>(b), reinterpret_cast<const uint32_t *>(b) + stringsize,
                reinterpret_cast<const uint32_t *>(a), reinterpret_cast<const uint32_t *>(a) + stringsize
            );
        }

        static bool equal(const char *a, const char *b, single_compare_static_data *extra) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>((const AuxDataBase *)extra->auxdata)>>1;
            for (uint32_t *lhs = reinterpret_cast<uint32_t *>(const_cast<char *>(a)),
                    *rhs = reinterpret_cast<uint32_t *>(const_cast<char *>(b));
                    lhs < lhs + stringsize; ++lhs, ++rhs) {
                if (*lhs != *rhs) {
                    return false;
                }
            }
            return true;
        }

        static bool not_equal(const char *a, const char *b, single_compare_static_data *extra) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>((const AuxDataBase *)extra->auxdata)>>1;
            for (uint32_t *lhs = reinterpret_cast<uint32_t *>(const_cast<char *>(a)),
                    *rhs = reinterpret_cast<uint32_t *>(const_cast<char *>(b));
                    lhs < lhs + stringsize; ++lhs, ++rhs) {
                if (*lhs != *rhs) {
                    return true;
                }
            }
            return false;
        }

        static bool greater_equal(const char *a, const char *b, single_compare_static_data *extra) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>((const AuxDataBase *)extra->auxdata)>>1;
            return !lexicographical_compare(
                reinterpret_cast<const uint32_t *>(a), reinterpret_cast<const uint32_t *>(a) + stringsize,
                reinterpret_cast<const uint32_t *>(b), reinterpret_cast<const uint32_t *>(b) + stringsize
            );
        }

        static bool greater(const char *a, const char *b, single_compare_static_data *extra) {
            intptr_t stringsize = reinterpret_cast<uintptr_t>((const AuxDataBase *)extra->auxdata)>>1;
            return lexicographical_compare(
                reinterpret_cast<const uint32_t *>(b), reinterpret_cast<const uint32_t *>(b) + stringsize,
                reinterpret_cast<const uint32_t *>(a), reinterpret_cast<const uint32_t *>(a) + stringsize
            );
        }
    };
} // anonymous namespace

#define DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL(type) { \
    type##_fixedstring_compare_kernel::less, \
    type##_fixedstring_compare_kernel::less_equal, \
    type##_fixedstring_compare_kernel::equal, \
    type##_fixedstring_compare_kernel::not_equal, \
    type##_fixedstring_compare_kernel::greater_equal, \
    type##_fixedstring_compare_kernel::greater \
    }

void dynd::get_fixedstring_comparison_kernel(intptr_t string_size, string_encoding_t encoding,
                kernel_instance<compare_operations_t>& out_kernel)
{
    static single_compare_operation_t fixedstring_comparisons_table[3][6] = {
        DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL(ascii_utf8),
        DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL(utf16),
        DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL(utf32)
    };
    static int lookup[5] = {0, 1, 0, 1, 2};
    memcpy(out_kernel.kernel.ops, fixedstring_comparisons_table[lookup[encoding]], sizeof(out_kernel.kernel.ops));
    make_raw_auxiliary_data(out_kernel.extra.auxdata, static_cast<uintptr_t>(string_size)<<1);
}

#undef DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL

/////////////////////////////////////////
// blockref string comparison

namespace {
    template<typename T>
    struct string_compare_kernel {
        static bool less(const char *a, const char *b, single_compare_static_data *DYND_UNUSED(extra)) {
            const string_dtype_data *da = reinterpret_cast<const string_dtype_data *>(a);
            const string_dtype_data *db = reinterpret_cast<const string_dtype_data *>(b);
            return lexicographical_compare(
                reinterpret_cast<const T *>(da->begin), reinterpret_cast<const T *>(da->end),
                reinterpret_cast<const T *>(db->begin), reinterpret_cast<const T *>(db->end));
        }

        static bool less_equal(const char *a, const char *b, single_compare_static_data *DYND_UNUSED(extra)) {
            const string_dtype_data *da = reinterpret_cast<const string_dtype_data *>(a);
            const string_dtype_data *db = reinterpret_cast<const string_dtype_data *>(b);
            return !lexicographical_compare(
                reinterpret_cast<const T *>(db->begin), reinterpret_cast<const T *>(db->end),
                reinterpret_cast<const T *>(da->begin), reinterpret_cast<const T *>(da->end));
        }

        static bool equal(const char *a, const char *b, single_compare_static_data *DYND_UNUSED(extra)) {
            const string_dtype_data *da = reinterpret_cast<const string_dtype_data *>(a);
            const string_dtype_data *db = reinterpret_cast<const string_dtype_data *>(b);
            return (da->end - da->begin == db->end - db->begin) &&
                    memcmp(da->begin, db->begin, da->end - da->begin) == 0;
        }

        static bool not_equal(const char *a, const char *b, single_compare_static_data *DYND_UNUSED(extra)) {
            const string_dtype_data *da = reinterpret_cast<const string_dtype_data *>(a);
            const string_dtype_data *db = reinterpret_cast<const string_dtype_data *>(b);
            return (da->end - da->begin != db->end - db->begin) ||
                    memcmp(da->begin, db->begin, da->end - da->begin) != 0;
        }

        static bool greater_equal(const char *a, const char *b, single_compare_static_data *DYND_UNUSED(extra)) {
            const string_dtype_data *da = reinterpret_cast<const string_dtype_data *>(a);
            const string_dtype_data *db = reinterpret_cast<const string_dtype_data *>(b);
            return !lexicographical_compare(
                reinterpret_cast<const T *>(da->begin), reinterpret_cast<const T *>(da->end),
                reinterpret_cast<const T *>(db->begin), reinterpret_cast<const T *>(db->end));
        }

        static bool greater(const char *a, const char *b, single_compare_static_data *DYND_UNUSED(extra)) {
            const string_dtype_data *da = reinterpret_cast<const string_dtype_data *>(a);
            const string_dtype_data *db = reinterpret_cast<const string_dtype_data *>(b);
            return lexicographical_compare(
                reinterpret_cast<const T *>(db->begin), reinterpret_cast<const T *>(db->end),
                reinterpret_cast<const T *>(da->begin), reinterpret_cast<const T *>(da->end));
        }
    };
} // anonymous namespace

#define DYND_STRING_COMPARISON_TABLE_TYPE_LEVEL(type) { \
    string_compare_kernel<type>::less, \
    string_compare_kernel<type>::less_equal, \
    string_compare_kernel<type>::equal, \
    string_compare_kernel<type>::not_equal, \
    string_compare_kernel<type>::greater_equal, \
    string_compare_kernel<type>::greater \
    }

void dynd::get_string_comparison_kernel(string_encoding_t encoding,
                kernel_instance<compare_operations_t>& out_kernel)
{
    static single_compare_operation_t fixedstring_comparisons_table[3][6] = {
        DYND_STRING_COMPARISON_TABLE_TYPE_LEVEL(uint8_t),
        DYND_STRING_COMPARISON_TABLE_TYPE_LEVEL(uint16_t),
        DYND_STRING_COMPARISON_TABLE_TYPE_LEVEL(uint32_t)
    };
    static int lookup[5] = {0, 1, 0, 1, 2};
    memcpy(out_kernel.kernel.ops, fixedstring_comparisons_table[lookup[encoding]], sizeof(out_kernel.kernel.ops));
    out_kernel.extra.auxdata.free();
}

#undef DYND_STRING_COMPARISON_TABLE_TYPE_LEVEL
