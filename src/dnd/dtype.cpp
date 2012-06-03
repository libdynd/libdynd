//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#include <dnd/dtype.hpp>
#include <dnd/exceptions.hpp>

#include <sstream>
#include <cstring>

/** The maximum number of type ids which can be defined */
#define DND_MAX_NUM_TYPE_IDS 64

using namespace std;
using namespace dnd;

// Default destructor for the extended dtype does nothing
dnd::extended_dtype::~extended_dtype()
{
}

/**
 * A static look-up table structure which contains data about the type ids.
 * This must match up with the type id enumeration, and has space that
 * is intended to be filled up with more data when custom dtypes are added.
 */
static struct {
    unsigned char kind, alignment, itemsize;
} basic_type_id_info[DND_MAX_NUM_TYPE_IDS] = {
    {bool_kind, 1, 1},         // bool
    {int_kind, 1, 1},          // int8
    {int_kind, 2, 2},          // int16
    {int_kind, 4, 4},          // int32
    {int_kind, 8, 8},          // int64
    {uint_kind, 1, 1},         // uint8
    {uint_kind, 2, 2},         // uint16
    {uint_kind, 4, 4},         // uint32
    {uint_kind, 8, 8},         // uint64
    {float_kind, 4, 4},        // float32
    {float_kind, 8, 8},        // float64
    {composite_kind, 16, 16},  // sse128f
    {composite_kind, 16, 16},  // sse128d
    {string_kind, 1, 0},       // utf8
    {composite_kind, 1, 0},    // struct
    {composite_kind, 1, 0},    // subarray
    {pattern_kind, 1, 0}      // generic
};

/**
 * A static look-up table which contains the names of all the type ids.
 * This must match up with the type id enumeration, and has space that
 * is intended to be filled up with more data when custom dtypes are added.
 */
static char type_id_names[DND_MAX_NUM_TYPE_IDS][32] = {
    "generic",
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "sse128f",
    "sse128d",
    "utf8",
    "struct",
    "subarray"
};

/**
 * Validates that the given type ID is a proper ID. Throws
 * an exception if not.
 *
 * @param type_id  The type id to validate.
 */
static inline int validate_type_id(int type_id)
{
    // 0 <= type_id <= utf8_type_id
    if ((unsigned int)type_id <= utf8_type_id) {
        return type_id;
    } else {
        throw invalid_type_id(type_id);
    }
}

const char *dnd::get_type_id_basename(int type_id)
{
    return type_id_names[validate_type_id(type_id)];
}

dtype::dtype()
    : m_type_id(pattern_type_id), m_kind(pattern_kind), m_alignment(1),
      m_itemsize(0), m_data()
{
    // Default to a generic type with zero size
}

dtype::dtype(int type_id)
    : m_type_id(validate_type_id(type_id)),
      m_kind(basic_type_id_info[type_id].kind),
      m_alignment(basic_type_id_info[type_id].alignment),
      m_itemsize(basic_type_id_info[type_id].itemsize),
      m_data()
{
}

dtype::dtype(int type_id, uintptr_t size)
    : m_type_id(validate_type_id(type_id)),
      m_kind(basic_type_id_info[type_id].kind),
      m_alignment(basic_type_id_info[type_id].alignment),
      m_itemsize(basic_type_id_info[type_id].itemsize),
      m_data()
{
    if (m_itemsize != 0) {
        if (m_itemsize != size) {
            throw std::runtime_error(std::string() + "invalid itemsize for type id "
                                                    + get_type_id_basename(type_id));
        }
    } else {
        m_itemsize = size;
    }
}

/*
namespace {
    template<int size> struct sized_byteswapper;
    template<> struct sized_byteswapper<2> {
        static void byteswap(void *dst, const void *src, uintptr_t) {
            char *d = reinterpret_cast<char *>(dst);
            const char *s = reinterpret_cast<const char *>(src);
            char tmp[2];

            tmp[0] = s[0];
            tmp[1] = s[1];

            d[0] = tmp[1];
            d[1] = tmp[0];
        }
    };
    template<> struct sized_byteswapper<4> {
        static void byteswap(void *dst, const void *src, uintptr_t) {
            char *d = reinterpret_cast<char *>(dst);
            const char *s = reinterpret_cast<const char *>(src);
            char tmp[4];

            tmp[0] = s[3];
            tmp[1] = s[2];
            tmp[2] = s[1];
            tmp[3] = s[0];

            d[0] = tmp[3];
            d[1] = tmp[2];
            d[2] = tmp[1];
            d[3] = tmp[0];
        }
    };
    template<> struct sized_byteswapper<8> {
        static void byteswap(void *dst, const void *src, uintptr_t) {
            char *d = reinterpret_cast<char *>(dst);
            const char *s = reinterpret_cast<const char *>(src);
            char tmp[8];

            tmp[0] = s[7];
            tmp[1] = s[6];
            tmp[2] = s[5];
            tmp[3] = s[4];
            tmp[4] = s[3];
            tmp[5] = s[2];
            tmp[6] = s[1];
            tmp[7] = s[0];

            d[0] = tmp[7];
            d[1] = tmp[6];
            d[2] = tmp[5];
            d[3] = tmp[4];
            d[4] = tmp[3];
            d[5] = tmp[2];
            d[6] = tmp[1];
            d[7] = tmp[0];
        }
    };
}

byteswap_operation_t dnd::dtype::get_byteswap_operation() const
{
    if (m_data != NULL) {
        return m_data->get_byteswap_operation();
    }

    switch (m_type_id) {
        case int16_type_id:
        case uint16_type_id:
            return &sized_byteswapper<2>::byteswap;
        case int32_type_id:
        case uint32_type_id:
        case float32_type_id:
            return &sized_byteswapper<4>::byteswap;
        case int64_type_id:
        case uint64_type_id:
        case float64_type_id:
            return &sized_byteswapper<8>::byteswap;
    }

    stringstream ss;
    ss << "dtype " << *this << " does not support byte-swapping";
    throw std::runtime_error(ss.str());
}
*/


std::ostream& dnd::operator<<(std::ostream& o, const dtype& rhs)
{
    switch (rhs.type_id()) {
        case bool_type_id:
            o << "bool";
            break;
        case int8_type_id:
            o << "int8";
            break;
        case int16_type_id:
            o << "int16";
            break;
        case int32_type_id:
            o << "int32";
            break;
        case int64_type_id:
            o << "int64";
            break;
        case uint8_type_id:
            o << "uint8";
            break;
        case uint16_type_id:
            o << "uint16";
            break;
        case uint32_type_id:
            o << "uint32";
            break;
        case uint64_type_id:
            o << "uint64";
            break;
        case float32_type_id:
            o << "float32";
            break;
        case float64_type_id:
            o << "float64";
            break;
        case utf8_type_id:
            if (rhs.itemsize() == 0) {
                o << "utf8";
            } else {
                o << "utf8[" << rhs.itemsize() << "]";
            }
            break;
        case pattern_type_id:
            o << "pattern";
            break;
        default:
            if (rhs.extended()) {
                rhs.extended()->print(o);
            } else {
                o << "<dtype without formatting support>";
            }
            break;
    }

    return o;
}

template<class T>
static void strided_print(std::ostream& o, const char *data, intptr_t stride, intptr_t size, const char *separator)
{
    T value;
    memcpy(&value, data, sizeof(value));
    o << value;
    for (intptr_t i = 1; i < size; ++i) {
        data += stride;
        memcpy(&value, data, sizeof(value));
        o << separator << value;
    }
}

void dnd::dtype::print_data(std::ostream& o, const char *data, intptr_t stride, intptr_t size, const char *separator) const
{
    if (size > 0) {
        if (extended() != NULL) {
            extended()->print_data(o, data, stride, size, separator);
        } else {
            // TODO: Handle byte-swapped dtypes
            switch (type_id()) {
                case bool_type_id:
                    o << (*data ? "true" : "false");
                    for (intptr_t i = 1; i < size; ++i) {
                        data += stride;
                        o << separator << (*data ? "true" : "false");
                    }
                    break;
                case int8_type_id:
                    strided_print<int8_t>(o, data, stride, size, separator);
                    break;
                case int16_type_id:
                    strided_print<int16_t>(o, data, stride, size, separator);
                    break;
                case int32_type_id:
                    strided_print<int32_t>(o, data, stride, size, separator);
                    break;
                case int64_type_id:
                    strided_print<int64_t>(o, data, stride, size, separator);
                    break;
                case uint8_type_id:
                    strided_print<uint8_t>(o, data, stride, size, separator);
                    break;
                case uint16_type_id:
                    strided_print<uint16_t>(o, data, stride, size, separator);
                    break;
                case uint32_type_id:
                    strided_print<uint32_t>(o, data, stride, size, separator);
                    break;
                case uint64_type_id:
                    strided_print<uint64_t>(o, data, stride, size, separator);
                    break;
                case float32_type_id:
                    strided_print<float>(o, data, stride, size, separator);
                    break;
                case float64_type_id:
                    strided_print<double>(o, data, stride, size, separator);
                    break;
                default:
                    stringstream ss;
                    ss << "printing of dtype " << *this << " isn't supported yet";
                    throw std::runtime_error(ss.str());
            }
        }
    }
}
