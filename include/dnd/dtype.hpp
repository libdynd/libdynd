#ifndef _DTYPE_HPP_
#define _DTYPE_HPP_

#include <stdint.h>

namespace dnd {

enum dtype_kind {
    bool_kind,
    int_kind,
    uint_kind,
    float_kind,
    complex_kind,
    string_kind,
    // For use when it becomes possible to register custom dtypes
    custom_kind
};

enum {
    bool_type_id,
    int8_type_id,
    int16_type_id,
    int32_type_id,
    int64_type_id,
    uint8_type_id,
    uint16_type_id,
    uint32_type_id,
    uint64_type_id,
    float32_type_id,
    float64_type_id,
    utf8_type_id,
};

// Type trait for the type numbers
template <typename T> struct type_id_of;

// This is dodgy - need sizeof(bool) == 1 consistently for this to be valid...
//template <> struct type_id_of<bool> {
//    enum {value = bool_type_id};
//};
template <> struct type_id_of<int8_t> {
    enum {value = int8_type_id};
};
template <> struct type_id_of<int16_t> {
    enum {value = int16_type_id};
};
template <> struct type_id_of<int32_t> {
    enum {value = int32_type_id};
};
template <> struct type_id_of<int64_t> {
    enum {value = int64_type_id};
};
template <> struct type_id_of<uint8_t> {
    enum {value = uint8_type_id};
};
template <> struct type_id_of<uint16_t> {
    enum {value = uint16_type_id};
};
template <> struct type_id_of<uint32_t> {
    enum {value = uint32_type_id};
};
template <> struct type_id_of<uint64_t> {
    enum {value = uint64_type_id};
};
template <> struct type_id_of<float> {
    enum {value = float32_type_id};
};
template <> struct type_id_of<double> {
    enum {value = float64_type_id};
};

// The extended_dtype class is for dtypes which require more data
// than a type_id, kind, and itemsize, and endianness.
class extended_dtype {
public:
    int m_type_id;
    dtype_kind m_kind;
    intptr_t m_itemsize;
    int m_alignment;
    bool m_byteswapped;

    virtual ~extended_dtype();
};

//
// The dtype class operates in two ways:
//   * Trivial mode, where the type info is encoded
//     in 'm_data' directly.
//   * An extended dtype mode, where m_data points to an object
//     of type 'extended_dtype', which contains extra data about
//     the dtype.
//     
class dtype {
private:
    extended_dtype *m_data;

public:
    // Construct from a type ID
    explicit dtype(int type_id);
    // Construct from a type ID and itemsize
    explicit dtype(int type_id, intptr_t size);
    // Construct from extended_dtype data. 'exdata' must have been
    // created with 'new', and the dtype assumes ownership of it.
    explicit dtype(extended_dtype *exdata);

    ~dtype() {
        if (!is_trivial()) {
            delete m_data;
        }
    }

    // Trivial mode is signaled by an odd pointer, something
    // which never occurs by default.
    bool is_trivial() const {
        return ((intptr_t)m_data & 1) == 1;
    }

    bool is_byteswapped() const {
        // In the trivial case, bit 1 indicates whether it's byte-swapped
        if (is_trivial()) {
            return (bool)(((intptr_t)m_data) >> 1);
        }
        else {
            return m_data->m_itemsize;
        }
    }

    // The type number is an enumeration of data types, starting
    // at 0, with one value for each unique data type. This is
    // inspired by the approach in NumPy, and the intention is
    // to have the default
    int type_id() const {
        // In the trivial case, bits 2 through 10 store the type number
        if (is_trivial()) {
            return (int)((((intptr_t)m_data) >> 2) & 0xff);
        }
        else {
            return m_data->m_type_id;
        }
    }

    dtype_kind kind() const {
        // In the trivial case, bits 11 through 14 store the kind
        if (is_trivial()) {
            return (dtype_kind)((((intptr_t)m_data) >> 11) & 0x0f);
        }
        else {
            return m_data->m_kind;
        }
    }

    int alignment() const {
        // In the trivial case, bits 15 through 17 store the alignment,
        // which may be 1, 2, 4, 8, or 16
        if (is_trivial()) {
            return 1 << ((((intptr_t)m_data) >> 15) & 0x07);
        }
        else {
            return m_data->m_alignment;
        }
    }

    intptr_t itemsize() const {
        // In the trivial case, bits 18 and up store the item size
        if (is_trivial()) {
            return (intptr_t)(((uintptr_t)m_data) >> 18);
        }
        else {
            return m_data->m_itemsize;
        }
    }

    /*
     * When the dtype isn't trivial, returns a const pointer
     * to an extended_dtype object which contains information
     * about the dtype.
     */
    const extended_dtype& extended() const;
};

}

#endif//_DTYPE_HPP_
