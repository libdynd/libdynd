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

#define DND_TRIVIAL_DTYPE_DATA(type_id, kind, itemsize, \
                                log2_align, byteswapped) \
            (extended_dtype *)(1 | \
                               ((byteswapped) << 1) | \
                               ((type_id) << 2) | \
                               ((kind) << 11) | \
                               ((log2_align) << 15) | \
                               ((itemsize) << 18))

using namespace dnd;

// Default destructor for the extended dtype does nothing
extended_dtype::~extended_dtype()
{
}

static extended_dtype* static_fixedsize_dtype_data[] = {
    DND_TRIVIAL_DTYPE_DATA(generic_type_id, generic_kind, 0, 0, 0),
    DND_TRIVIAL_DTYPE_DATA(bool_type_id, bool_kind, 1, 0, 0),
    DND_TRIVIAL_DTYPE_DATA(int8_type_id, int_kind, 1, 0, 0),
    DND_TRIVIAL_DTYPE_DATA(int16_type_id, int_kind, 2, 1, 0),
    DND_TRIVIAL_DTYPE_DATA(int32_type_id, int_kind, 4, 2, 0),
    DND_TRIVIAL_DTYPE_DATA(int64_type_id, int_kind, 8, 3, 0),
    DND_TRIVIAL_DTYPE_DATA(uint8_type_id, uint_kind, 1, 0, 0),
    DND_TRIVIAL_DTYPE_DATA(uint16_type_id, uint_kind, 2, 1, 0),
    DND_TRIVIAL_DTYPE_DATA(uint32_type_id, uint_kind, 4, 2, 0),
    DND_TRIVIAL_DTYPE_DATA(uint64_type_id, uint_kind, 8, 3, 0),
    DND_TRIVIAL_DTYPE_DATA(float32_type_id, float_kind, 4, 2, 0),
    DND_TRIVIAL_DTYPE_DATA(float64_type_id, float_kind, 8, 3, 0),
    DND_TRIVIAL_DTYPE_DATA(sse128f_type_id, composite_kind, 16, 4, 0),
    DND_TRIVIAL_DTYPE_DATA(sse128d_type_id, composite_kind, 16, 4, 0),
};

dtype::dtype(const dtype& rhs)
    : m_data(rhs.m_data)
{
    // Increase the reference count to the extended data
    if (!is_trivial()) {
        ++m_data->m_refcount;
    }
}

dtype& dtype::operator=(const dtype& rhs)
{
    // Decrease the reference count to the old extended data
    if (!is_trivial()) {
        if (--m_data->m_refcount == 0) {
                delete m_data;
        }
    }

    m_data = rhs.m_data;

    // Increase the reference count to the new extended data
    if (!rhs.is_trivial()) {
        ++m_data->m_refcount;
    }

    return *this;
}

dtype::~dtype()
{
    // Decrease the reference count to the extended data
    if (!is_trivial()) {
        if (--m_data->m_refcount == 0) {
                delete m_data;
        }
    }
}

dtype::dtype()
{
    // Default to a generic type
    m_data = DND_TRIVIAL_DTYPE_DATA(generic_type_id, generic_kind, 0, 0, 0);
}

dtype::dtype(int type_id)
{
    if (type_id >= 0 && type_id < DND_NUM_FIXEDSIZE_TYPE_IDS) {
        m_data = static_fixedsize_dtype_data[type_id];
    }
    else {
        throw std::runtime_error("custom dtypes not supported yet");
    }
}

dtype::dtype(int type_id, intptr_t size)
{
    if (type_id >= 0 && type_id < DND_NUM_FIXEDSIZE_TYPE_IDS) {
        m_data = static_fixedsize_dtype_data[type_id];
        if (itemsize() != size) {
            throw std::runtime_error("invalid itemsize for given type ID");
        }
    }
    else if (type_id == utf8_type_id) {
        if (size < 0) {
            throw std::runtime_error("negative dtype itemsize is not allowed");
        }
        // If the size fits, use a trivial dtype
        if ((size & DND_TRIVIAL_ITEMSIZE_MASK) == size) {
            m_data = DND_TRIVIAL_DTYPE_DATA(utf8_type_id, string_kind,
                                                size, 0, 0);
        }
        // Otherwise allocate an extended_dtype
        else {
            m_data = new extended_dtype();
            m_data->m_refcount = 1;
            m_data->m_type_id = utf8_type_id;
            m_data->m_kind = string_kind;
            m_data->m_itemsize = size;
            m_data->m_alignment = 1;
            m_data->m_byteswapped = false;
        }
    }
    else {
        throw std::runtime_error("custom dtypes not supported yet");
    }
}

dtype::dtype(extended_dtype *exdata)
    : m_data(exdata)
{
}

const extended_dtype* dtype::extended() const
{
    if (is_trivial()) {
        return NULL;
    }
    else {
        return m_data;
    }
}
