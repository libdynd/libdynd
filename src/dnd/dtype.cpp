//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#include <dnd/dtype.hpp>

#include <stdexcept>

#define DND_MAX_DTYPES (64)
// The number of type IDs that have a fixed size
#define DND_NUM_FIXEDSIZE_TYPE_IDS (sse128d_type_id + 1)
// A bitmask that the itemsize must fit in to allow a trivial dtype
#define DND_TRIVIAL_ITEMSIZE_MASK ((intptr_t)(((uintptr_t)(-1)) >> 18))

using namespace dnd;

// Default destructor for the extended dtype does nothing
extended_dtype::~extended_dtype()
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

