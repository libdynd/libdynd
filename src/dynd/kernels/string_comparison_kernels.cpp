//
// Copyright (C) 2011-13, DyND Developers
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
    struct fixedstring_compare_kernel_extra {
        kernel_data_prefix base;
        size_t string_size;
    };

    struct ascii_utf8_fixedstring_compare_kernel {
        static bool less(const char *a, const char *b, kernel_data_prefix *extra) {
            size_t stringsize = reinterpret_cast<fixedstring_compare_kernel_extra *>(extra)->string_size;
            return strncmp(a, b, stringsize) < 0;
        }

        static bool less_equal(const char *a, const char *b, kernel_data_prefix *extra) {
            size_t stringsize = reinterpret_cast<fixedstring_compare_kernel_extra *>(extra)->string_size;
            return strncmp(a, b, stringsize) <= 0;
        }

        static bool equal(const char *a, const char *b, kernel_data_prefix *extra) {
            size_t stringsize = reinterpret_cast<fixedstring_compare_kernel_extra *>(extra)->string_size;
            return strncmp(a, b, stringsize) == 0;
        }

        static bool not_equal(const char *a, const char *b, kernel_data_prefix *extra) {
            size_t stringsize = reinterpret_cast<fixedstring_compare_kernel_extra *>(extra)->string_size;
            return strncmp(a, b, stringsize) != 0;
        }

        static bool greater_equal(const char *a, const char *b, kernel_data_prefix *extra) {
            size_t stringsize = reinterpret_cast<fixedstring_compare_kernel_extra *>(extra)->string_size;
            return strncmp(a, b, stringsize) >= 0;
        }

        static bool greater(const char *a, const char *b, kernel_data_prefix *extra) {
            size_t stringsize = reinterpret_cast<fixedstring_compare_kernel_extra *>(extra)->string_size;
            return strncmp(a, b, stringsize) > 0;
        }
    };

    struct utf16_fixedstring_compare_kernel {
        static bool less(const char *a, const char *b, kernel_data_prefix *extra) {
            size_t stringsize = reinterpret_cast<fixedstring_compare_kernel_extra *>(extra)->string_size;
            return lexicographical_compare(
                reinterpret_cast<const uint16_t *>(a), reinterpret_cast<const uint16_t *>(a) + stringsize,
                reinterpret_cast<const uint16_t *>(b), reinterpret_cast<const uint16_t *>(b) + stringsize
            );
        }

        static bool less_equal(const char *a, const char *b, kernel_data_prefix *extra) {
            size_t stringsize = reinterpret_cast<fixedstring_compare_kernel_extra *>(extra)->string_size;
            return !lexicographical_compare(
                reinterpret_cast<const uint16_t *>(b), reinterpret_cast<const uint16_t *>(b) + stringsize,
                reinterpret_cast<const uint16_t *>(a), reinterpret_cast<const uint16_t *>(a) + stringsize
            );
        }

        static bool equal(const char *a, const char *b, kernel_data_prefix *extra)
        {
            size_t string_size = reinterpret_cast<fixedstring_compare_kernel_extra *>(extra)->string_size;
            const uint16_t *lhs = reinterpret_cast<const uint16_t *>(a);
            const uint16_t *rhs = reinterpret_cast<const uint16_t *>(b);
            for (size_t i = 0; i != string_size; ++i, ++lhs, ++rhs) {
                if (*lhs != *rhs) {
                    return false;
                }
            }
            return true;
        }

        static bool not_equal(const char *a, const char *b, kernel_data_prefix *extra)
        {
            size_t string_size = reinterpret_cast<fixedstring_compare_kernel_extra *>(extra)->string_size;
            const uint16_t *lhs = reinterpret_cast<const uint16_t *>(a);
            const uint16_t *rhs = reinterpret_cast<const uint16_t *>(b);
            for (size_t i = 0; i != string_size; ++i, ++lhs, ++rhs) {
                if (*lhs != *rhs) {
                    return true;
                }
            }
            return false;
        }

        static bool greater_equal(const char *a, const char *b, kernel_data_prefix *extra) {
            size_t stringsize = reinterpret_cast<fixedstring_compare_kernel_extra *>(extra)->string_size;
            return !lexicographical_compare(
                reinterpret_cast<const uint16_t *>(a), reinterpret_cast<const uint16_t *>(a) + stringsize,
                reinterpret_cast<const uint16_t *>(b), reinterpret_cast<const uint16_t *>(b) + stringsize
            );
        }

        static bool greater(const char *a, const char *b, kernel_data_prefix *extra) {
            size_t stringsize = reinterpret_cast<fixedstring_compare_kernel_extra *>(extra)->string_size;
            return lexicographical_compare(
                reinterpret_cast<const uint16_t *>(b), reinterpret_cast<const uint16_t *>(b) + stringsize,
                reinterpret_cast<const uint16_t *>(a), reinterpret_cast<const uint16_t *>(a) + stringsize
            );
        }
    };

    struct utf32_fixedstring_compare_kernel {
        static bool less(const char *a, const char *b, kernel_data_prefix *extra) {
            size_t stringsize = reinterpret_cast<fixedstring_compare_kernel_extra *>(extra)->string_size;
            return lexicographical_compare(
                reinterpret_cast<const uint32_t *>(a), reinterpret_cast<const uint32_t *>(a) + stringsize,
                reinterpret_cast<const uint32_t *>(b), reinterpret_cast<const uint32_t *>(b) + stringsize
            );
        }

        static bool less_equal(const char *a, const char *b, kernel_data_prefix *extra) {
            size_t stringsize = reinterpret_cast<fixedstring_compare_kernel_extra *>(extra)->string_size;
            return !lexicographical_compare(
                reinterpret_cast<const uint32_t *>(b), reinterpret_cast<const uint32_t *>(b) + stringsize,
                reinterpret_cast<const uint32_t *>(a), reinterpret_cast<const uint32_t *>(a) + stringsize
            );
        }

        static bool equal(const char *a, const char *b, kernel_data_prefix *extra)
        {
            size_t string_size = reinterpret_cast<fixedstring_compare_kernel_extra *>(extra)->string_size;
            const uint32_t *lhs = reinterpret_cast<const uint32_t *>(a);
            const uint32_t *rhs = reinterpret_cast<const uint32_t *>(b);
            for (size_t i = 0; i != string_size; ++i, ++lhs, ++rhs) {
                if (*lhs != *rhs) {
                    return false;
                }
            }
            return true;
        }

        static bool not_equal(const char *a, const char *b, kernel_data_prefix *extra)
        {
            size_t string_size = reinterpret_cast<fixedstring_compare_kernel_extra *>(extra)->string_size;
            const uint32_t *lhs = reinterpret_cast<const uint32_t *>(a);
            const uint32_t *rhs = reinterpret_cast<const uint32_t *>(b);
            for (size_t i = 0; i != string_size; ++i, ++lhs, ++rhs) {
                if (*lhs != *rhs) {
                    return true;
                }
            }
            return false;
        }

        static bool greater_equal(const char *a, const char *b, kernel_data_prefix *extra) {
            size_t stringsize = reinterpret_cast<fixedstring_compare_kernel_extra *>(extra)->string_size;
            return !lexicographical_compare(
                reinterpret_cast<const uint32_t *>(a), reinterpret_cast<const uint32_t *>(a) + stringsize,
                reinterpret_cast<const uint32_t *>(b), reinterpret_cast<const uint32_t *>(b) + stringsize
            );
        }

        static bool greater(const char *a, const char *b, kernel_data_prefix *extra) {
            size_t stringsize = reinterpret_cast<fixedstring_compare_kernel_extra *>(extra)->string_size;
            return lexicographical_compare(
                reinterpret_cast<const uint32_t *>(b), reinterpret_cast<const uint32_t *>(b) + stringsize,
                reinterpret_cast<const uint32_t *>(a), reinterpret_cast<const uint32_t *>(a) + stringsize
            );
        }
    };
} // anonymous namespace

#define DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL(type) { \
    type##_fixedstring_compare_kernel::less, \
    type##_fixedstring_compare_kernel::less, \
    type##_fixedstring_compare_kernel::less_equal, \
    type##_fixedstring_compare_kernel::equal, \
    type##_fixedstring_compare_kernel::not_equal, \
    type##_fixedstring_compare_kernel::greater_equal, \
    type##_fixedstring_compare_kernel::greater \
    }

size_t dynd::make_fixedstring_comparison_kernel(
                comparison_kernel *out, size_t offset_out,
                size_t string_size, string_encoding_t encoding,
                comparison_type_t comptype,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    static int lookup[5] = {0, 1, 0, 1, 2};
    static binary_single_predicate_t fixedstring_comparisons_table[3][7] = {
        DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL(ascii_utf8),
        DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL(utf16),
        DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL(utf32)
    };
    if (0 <= encoding && encoding < 5 && 0 <= comptype && comptype < 7) {
        out->ensure_capacity_leaf(offset_out + sizeof(fixedstring_compare_kernel_extra));
        fixedstring_compare_kernel_extra *e = out->get_at<fixedstring_compare_kernel_extra>(offset_out);
        e->base.set_function<binary_single_predicate_t>(fixedstring_comparisons_table[lookup[encoding]][comptype]);
        e->string_size = string_size;
        return offset_out + sizeof(fixedstring_compare_kernel_extra);
    } else {
        stringstream ss;
        ss << "make_fixedstring_comparison_kernel: Unexpected encoding (" << encoding;
        ss << ") or comparison type (" << comptype << ")";
        throw runtime_error(ss.str());
    }
}

#undef DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL

/////////////////////////////////////////
// blockref string comparison

namespace {
    template<typename T>
    struct string_compare_kernel {
        static bool less(const char *a, const char *b, kernel_data_prefix *DYND_UNUSED(extra)) {
            const string_dtype_data *da = reinterpret_cast<const string_dtype_data *>(a);
            const string_dtype_data *db = reinterpret_cast<const string_dtype_data *>(b);
            return lexicographical_compare(
                reinterpret_cast<const T *>(da->begin), reinterpret_cast<const T *>(da->end),
                reinterpret_cast<const T *>(db->begin), reinterpret_cast<const T *>(db->end));
        }

        static bool less_equal(const char *a, const char *b, kernel_data_prefix *DYND_UNUSED(extra)) {
            const string_dtype_data *da = reinterpret_cast<const string_dtype_data *>(a);
            const string_dtype_data *db = reinterpret_cast<const string_dtype_data *>(b);
            return !lexicographical_compare(
                reinterpret_cast<const T *>(db->begin), reinterpret_cast<const T *>(db->end),
                reinterpret_cast<const T *>(da->begin), reinterpret_cast<const T *>(da->end));
        }

        static bool equal(const char *a, const char *b, kernel_data_prefix *DYND_UNUSED(extra)) {
            const string_dtype_data *da = reinterpret_cast<const string_dtype_data *>(a);
            const string_dtype_data *db = reinterpret_cast<const string_dtype_data *>(b);
            return (da->end - da->begin == db->end - db->begin) &&
                    memcmp(da->begin, db->begin, da->end - da->begin) == 0;
        }

        static bool not_equal(const char *a, const char *b, kernel_data_prefix *DYND_UNUSED(extra)) {
            const string_dtype_data *da = reinterpret_cast<const string_dtype_data *>(a);
            const string_dtype_data *db = reinterpret_cast<const string_dtype_data *>(b);
            return (da->end - da->begin != db->end - db->begin) ||
                    memcmp(da->begin, db->begin, da->end - da->begin) != 0;
        }

        static bool greater_equal(const char *a, const char *b, kernel_data_prefix *DYND_UNUSED(extra)) {
            const string_dtype_data *da = reinterpret_cast<const string_dtype_data *>(a);
            const string_dtype_data *db = reinterpret_cast<const string_dtype_data *>(b);
            return !lexicographical_compare(
                reinterpret_cast<const T *>(da->begin), reinterpret_cast<const T *>(da->end),
                reinterpret_cast<const T *>(db->begin), reinterpret_cast<const T *>(db->end));
        }

        static bool greater(const char *a, const char *b, kernel_data_prefix *DYND_UNUSED(extra)) {
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
    string_compare_kernel<type>::less, \
    string_compare_kernel<type>::less_equal, \
    string_compare_kernel<type>::equal, \
    string_compare_kernel<type>::not_equal, \
    string_compare_kernel<type>::greater_equal, \
    string_compare_kernel<type>::greater \
    }

size_t dynd::make_string_comparison_kernel(
                comparison_kernel *out, size_t offset_out,
                string_encoding_t encoding,
                comparison_type_t comptype,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    static int lookup[5] = {0, 1, 0, 1, 2};
    static binary_single_predicate_t string_comparisons_table[3][7] = {
        DYND_STRING_COMPARISON_TABLE_TYPE_LEVEL(uint8_t),
        DYND_STRING_COMPARISON_TABLE_TYPE_LEVEL(uint16_t),
        DYND_STRING_COMPARISON_TABLE_TYPE_LEVEL(uint32_t)
    };
    if (0 <= encoding && encoding < 5 && 0 <= comptype && comptype < 7) {
        out->ensure_capacity_leaf(offset_out + sizeof(kernel_data_prefix));
        kernel_data_prefix *e = out->get_at<kernel_data_prefix>(offset_out);
        e->set_function<binary_single_predicate_t>(string_comparisons_table[lookup[encoding]][comptype]);
        return offset_out + sizeof(kernel_data_prefix);
    } else {
        stringstream ss;
        ss << "make_string_comparison_kernel: Unexpected encoding (" << encoding;
        ss << ") or comparison type (" << comptype << ")";
        throw runtime_error(ss.str());
    }
}

#undef DYND_STRING_COMPARISON_TABLE_TYPE_LEVEL

size_t dynd::make_general_string_comparison_kernel(
                comparison_kernel *DYND_UNUSED(out), size_t DYND_UNUSED(offset_out),
                const dtype& DYND_UNUSED(src0_dt), const char *DYND_UNUSED(src0_metadata),
                const dtype& DYND_UNUSED(src1_dt), const char *DYND_UNUSED(src1_metadata),
                comparison_type_t DYND_UNUSED(comptype),
                const eval::eval_context *DYND_UNUSED(ectx))
{
    throw runtime_error("TODO: implement make_general_string_comparison_kernel");
}
