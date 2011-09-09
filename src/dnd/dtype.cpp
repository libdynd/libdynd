#include <dnd/dtype.hpp>

#define DND_MAX_DTYPES (64)
// The number of type IDs that have a fixed size
#define DND_NUM_FIXEDSIZE_TYPE_IDS (float64_type_id + 1)
// A bitmask that the itemsize must fit in to allow a trivial dtype
#define DND_TRIVIAL_ITEMSIZE_MASK ((intptr_t)(((uintptr_t)(-1)) >> 18))

using namespace dnd;

static intptr_t static_trivial_dtype_data[] = {
    1 + (bool_type_id << 2) +    (bool_kind << 11) +  (0 << 15) + (1 << 18),
    3 + (bool_type_id << 2) +    (bool_kind << 11) +  (0 << 15) + (1 << 18),
    1 + (int8_type_id << 2) +    (int_kind << 11) +   (0 << 15) + (1 << 18),
    3 + (int8_type_id << 2) +    (int_kind << 11) +   (0 << 15) + (1 << 18),
    1 + (int16_type_id << 2) +   (int_kind << 11) +   (1 << 15) + (2 << 18),
    3 + (int16_type_id << 2) +   (int_kind << 11) +   (1 << 15) + (2 << 18),
    1 + (int32_type_id << 2) +   (int_kind << 11) +   (2 << 15) + (4 << 18),
    3 + (int32_type_id << 2) +   (int_kind << 11) +   (2 << 15) + (4 << 18),
    1 + (int64_type_id << 2) +   (int_kind << 11) +   (3 << 15) + (8 << 18),
    3 + (int64_type_id << 2) +   (int_kind << 11) +   (3 << 15) + (8 << 18),
    1 + (uint8_type_id << 2) +   (uint_kind << 11) +  (0 << 15) + (1 << 18),
    3 + (uint8_type_id << 2) +   (uint_kind << 11) +  (0 << 15) + (1 << 18),
    1 + (uint16_type_id << 2) +  (uint_kind << 11) +  (1 << 15) + (2 << 18),
    3 + (uint16_type_id << 2) +  (uint_kind << 11) +  (1 << 15) + (2 << 18),
    1 + (uint32_type_id << 2) +  (uint_kind << 11) +  (2 << 15) + (4 << 18),
    3 + (uint32_type_id << 2) +  (uint_kind << 11) +  (2 << 15) + (4 << 18),
    1 + (uint64_type_id << 2) +  (uint_kind << 11) +  (3 << 15) + (8 << 18),
    3 + (uint64_type_id << 2) +  (uint_kind << 11) +  (3 << 15) + (8 << 18),
    1 + (float32_type_id << 2) + (float_kind << 11) + (2 << 15) + (4 << 18),
    3 + (float32_type_id << 2) + (float_kind << 11) + (2 << 15) + (4 << 18),
    1 + (float64_type_id << 2) + (float_kind << 11) + (3 << 15) + (8 << 18),
    3 + (float64_type_id << 2) + (float_kind << 11) + (3 << 15) + (8 << 18),
};

dtype::dtype(int type_id)
{
    if (type_id >= 0 && type_id < DND_NUM_TRIVIAL_TYPE_IDS) {
        m_data = static_trivial_dtype_data[2*type_id];
    }
    else {
        throw std::runtime_error("custom dtypes not supported yet");
    }
}

dtype::dtype(int type_id, intptr_t size)
{
    if (type_id >= 0 && type_id < DND_NUM_TRIVIAL_TYPE_IDS) {
        m_data = static_trivial_dtype_data[2*type_id];
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
            m_data = (extended_dtype *)(1 + (utf8_type_id << 2) +
                        (string_kind << 11) + (0 << 15) + (size << 18));
        }
        // Otherwise allocate an extended_dtype
        else {
            m_data = new extended_dtype();
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
    : m_data(exdata);
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
