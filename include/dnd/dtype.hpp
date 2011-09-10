//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DTYPE_HPP_
#define _DTYPE_HPP_

#include <stdint.h>

namespace dnd {

enum dtype_kind {
    generic_kind,
    bool_kind,
    int_kind,
    uint_kind,
    float_kind,
    complex_kind,
    string_kind,
    // For struct_type_id and subarray_type_id
    composite_kind,
    // For use when it becomes possible to register custom dtypes
    custom_kind
};

enum {
    // A type which can convert into anything - cannot be instantiated
    generic_type_id,
    // A 1-byte boolean type
    bool_type_id,
    // Signed integer types
    int8_type_id,
    int16_type_id,
    int32_type_id,
    int64_type_id,
    // Unsigned integer types
    uint8_type_id,
    uint16_type_id,
    uint32_type_id,
    uint64_type_id,
    // Floating point types
    float32_type_id,
    float64_type_id,
    // SSE vector of 4 floats
    sse128f_type_id,
    // SSE2 vector of 2 doubles
    sse128d_type_id,
    // UTF8 strings
    utf8_type_id,
    // Composite dtypes
    struct_type_id,
    subarray_type_id
};

namespace detail {
    // Simple metaprogram taking log base 2 of 1, 2, 4, and 8
    template <int I> struct log2_x;
    template <> struct log2_x<1> {
        enum {value = 0};
    };
    template <> struct log2_x<2> {
        enum {value = 1};
    };
    template <> struct log2_x<4> {
        enum {value = 2};
    };
    template <> struct log2_x<8> {
        enum {value = 3};
    };
}

// Type trait for the type numbers
template <typename T> struct type_id_of;

// This is dodgy - need sizeof(bool) == 1 consistently for this to be valid...
//template <> struct type_id_of<bool> {
//    enum {value = bool_type_id};
//};
template <> struct type_id_of<signed char> {
    enum {value = int8_type_id};
};
template <> struct type_id_of<short> {
    enum {value = int16_type_id};
};
template <> struct type_id_of<int> {
    enum {value = int32_type_id};
};
template <> struct type_id_of<long> {
    enum {value = int8_type_id + detail::log2_x<sizeof(long)>::value};
};
template <> struct type_id_of<long long> {
    enum {value = int64_type_id};
};
template <> struct type_id_of<uint8_t> {
    enum {value = uint8_type_id};
};
template <> struct type_id_of<uint16_t> {
    enum {value = uint16_type_id};
};
template <> struct type_id_of<unsigned int> {
    enum {value = uint32_type_id};
};
template <> struct type_id_of<unsigned long> {
    enum {value = uint8_type_id + detail::log2_x<sizeof(unsigned long)>::value};
};
template <> struct type_id_of<unsigned long long> {
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
    // TODO: should be replaced by C++11 atomic<int>
    int m_refcount;

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
class dtype {
private:
    // NOTE: Could perhaps use an anonymous union of this pointer
    //       and a bitfield, but it seems that the bitfield doesn't
    //       guarantee the layout of its members, which is important here.
    extended_dtype *m_data;

public:
    // Default constructor
    dtype();
    // Construct from a type ID
    explicit dtype(int type_id);
    // Construct from a type ID and itemsize
    explicit dtype(int type_id, intptr_t size);
    // Construct from extended_dtype data. 'exdata' must have been
    // created with 'new', and the dtype assumes ownership of it.
    explicit dtype(extended_dtype *exdata);

    ~dtype();

    dtype(const dtype& rhs);
    dtype& operator=(const dtype& rhs);

    // Trivial mode is signaled by an odd pointer, something
    // which never occurs by default.
    bool is_trivial() const {
        return ((intptr_t)m_data & 1) == 1;
    }

    bool is_byteswapped() const {
        // In the trivial case, bit 1 indicates whether it's byte-swapped
        if (is_trivial()) {
            return (bool)((((intptr_t)m_data) >> 1) & 1);
        }
        else {
            return m_data->m_byteswapped;
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
     * to the extended_dtype object which contains information
     * about the dtype. This pointer is only valid during
     * the lifetime of the dtype.
     */
    const extended_dtype* extended() const;
};

// Convenience function which makes a dtype object from a template parameter
template<class T>
dtype mkdtype() {
    return dtype(type_id_of<T>::value);
}

} // namespace dnd

#endif//_DTYPE_HPP_
