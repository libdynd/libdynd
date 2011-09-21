//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#include <dnd/dtype.hpp>

#include <sstream>
#include <stdexcept>

#define DND_MAX_DTYPES (64)
// The number of type IDs that have a fixed size
#define DND_NUM_FIXEDSIZE_TYPE_IDS (sse128d_type_id + 1)
// A bitmask that the itemsize must fit in to allow a trivial dtype
#define DND_TRIVIAL_ITEMSIZE_MASK ((intptr_t)(((uintptr_t)(-1)) >> 18))

using namespace std;
using namespace dnd;

byteswap_operation_t dnd::extended_dtype::get_byteswap_operation() const {
    throw std::runtime_error("this dtype does not support byte-swapping");
}

// Default destructor for the extended dtype does nothing
dnd::extended_dtype::~extended_dtype()
{
}

bool dtype::set_to_type_id(int type_id)
{
    m_type_id = type_id;
    m_byteswapped = 0;
    switch (type_id) {
        case generic_type_id:
            m_kind = generic_kind;
            m_alignment = 1;
            m_itemsize = 0;
            return true;
        case bool_type_id:
            m_kind = bool_kind;
            m_alignment = 1;
            m_itemsize = 1;
            return true;
        case int8_type_id:
            m_kind = int_kind;
            m_alignment = 1;
            m_itemsize = 1;
            return true;
        case int16_type_id:
            m_kind = int_kind;
            m_alignment = 2;
            m_itemsize = 2;
            return true;
        case int32_type_id:
            m_kind = int_kind;
            m_alignment = 4;
            m_itemsize = 4;
            return true;
        case int64_type_id:
            m_kind = int_kind;
            m_alignment = 8;
            m_itemsize = 8;
            return true;
        case uint8_type_id:
            m_kind = uint_kind;
            m_alignment = 1;
            m_itemsize = 1;
            return true;
        case uint16_type_id:
            m_kind = uint_kind;
            m_alignment = 2;
            m_itemsize = 2;
            return true;
        case uint32_type_id:
            m_kind = uint_kind;
            m_alignment = 4;
            m_itemsize = 4;
            return true;
        case uint64_type_id:
            m_kind = uint_kind;
            m_alignment = 8;
            m_itemsize = 8;
            return true;
        case float32_type_id:
            m_kind = float_kind;
            m_alignment = 4;
            m_itemsize = 4;
            return true;
        case float64_type_id:
            m_kind = float_kind;
            m_alignment = 8;
            m_itemsize = 8;
            return true;
    }

    return false;
}

dtype::dtype()
{
    // Default to a generic type with zero size
    m_type_id = generic_type_id;
    m_kind = generic_kind;
    m_alignment = 1;
    m_byteswapped = 0;
    m_itemsize = 0;
}

dtype::dtype(int type_id)
{
    if (!set_to_type_id(type_id)) {
        throw std::runtime_error("custom dtypes not supported yet");
    }
}

dtype::dtype(int type_id, intptr_t size)
{
    if (set_to_type_id(type_id)) {
        if (m_itemsize != size) {
            throw std::runtime_error("invalid itemsize for given type ID");
        }
    } else if (type_id == utf8_type_id) {
        if (size < 0) {
            throw std::runtime_error("negative dtype itemsize is not allowed");
        }
        m_type_id = type_id;
        m_kind = string_kind;
        m_alignment = 1;
        m_byteswapped = 0;
        m_itemsize = size;
    }
    else {
        throw std::runtime_error("custom dtypes not supported yet");
    }
}

namespace {
    template<int size> struct sized_byteswapper;
    template<> struct sized_byteswapper<2> {
        static void byteswap(void *dst, const void *src, intptr_t) {
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
        static void byteswap(void *dst, const void *src, intptr_t) {
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
        static void byteswap(void *dst, const void *src, intptr_t) {
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


std::ostream& dnd::operator<<(std::ostream& o, const dtype& rhs)
{
    switch (rhs.type_id()) {
        case generic_type_id:
            o << "generic";
            break;
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
            o << "utf8[" << rhs.itemsize() << "]";
            break;
        default:
            o << "<dtype without formatting support>";
            break;
    }

    return o;
}

