//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DND__DTYPE_HPP_
#define _DND__DTYPE_HPP_

#include <stdint.h>
#include <iostream>
// For shared_ptr
#include <memory>

#include <boost/utility/enable_if.hpp>

namespace dnd {

// A boolean class for dynamicndarray which is one-byte big
class dnd_bool {
    char m_value;
public:
    dnd_bool() : m_value(0) {}
    dnd_bool(bool value) : m_value(value) {}
    operator bool() const {
        return (bool)m_value;
    }
};

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


// Type trait for the type id
template <typename T> struct type_id_of;

// Can't use bool, because it doesn't have a guaranteed sizeof
template <> struct type_id_of<dnd_bool> {enum {value = bool_type_id};};
template <> struct type_id_of<char> {enum {value = ((char)-1) < 0 ? int8_type_id : uint8_type_id};};
template <> struct type_id_of<signed char> {enum {value = int8_type_id};};
template <> struct type_id_of<short> {enum {value = int16_type_id};};
template <> struct type_id_of<int> {enum {value = int32_type_id};};
template <> struct type_id_of<long> {
    enum {value = int8_type_id + detail::log2_x<sizeof(long)>::value};
};
template <> struct type_id_of<long long> {enum {value = int64_type_id};};
template <> struct type_id_of<uint8_t> {enum {value = uint8_type_id};};
template <> struct type_id_of<uint16_t> {enum {value = uint16_type_id};};
template <> struct type_id_of<unsigned int> {enum {value = uint32_type_id};};
template <> struct type_id_of<unsigned long> {
    enum {value = uint8_type_id + detail::log2_x<sizeof(unsigned long)>::value};
};
template <> struct type_id_of<unsigned long long>{enum {value = uint64_type_id};};
template <> struct type_id_of<float> {enum {value = float32_type_id};};
template <> struct type_id_of<double> {enum {value = float64_type_id};};

// Type trait for the kind
template <typename T> struct dtype_kind_of;

// Can't use bool, because it doesn't have a guaranteed sizeof
template <> struct dtype_kind_of<dnd_bool> {static const dtype_kind value = bool_kind;};
template <> struct dtype_kind_of<char> {
    static const dtype_kind value = ((char)-1) < 0 ? int_kind : uint_kind;
};
template <> struct dtype_kind_of<signed char> {static const dtype_kind value = int_kind;};
template <> struct dtype_kind_of<short> {static const dtype_kind value = int_kind;};
template <> struct dtype_kind_of<int> {static const dtype_kind value = int_kind;};
template <> struct dtype_kind_of<long> {static const dtype_kind value = int_kind;};
template <> struct dtype_kind_of<long long> {static const dtype_kind value = int_kind;};
template <> struct dtype_kind_of<uint8_t> {static const dtype_kind value = uint_kind;};
template <> struct dtype_kind_of<uint16_t> {static const dtype_kind value = uint_kind;};
template <> struct dtype_kind_of<unsigned int> {static const dtype_kind value = uint_kind;};
template <> struct dtype_kind_of<unsigned long> {static const dtype_kind value = uint_kind;};
template <> struct dtype_kind_of<unsigned long long>{static const dtype_kind value = uint_kind;};
template <> struct dtype_kind_of<float> {static const dtype_kind value = float_kind;};
template <> struct dtype_kind_of<double> {static const dtype_kind value = float_kind;};

// Metaprogram for determining if a type is a valid C++ scalar
// of a particular dtype.
template<typename T> struct is_dtype_scalar {enum {value = false};};
template <> struct is_dtype_scalar<dnd_bool> {enum {value = true};};
template <> struct is_dtype_scalar<char> {enum {value = true};};
template <> struct is_dtype_scalar<signed char> {enum {value = true};};
template <> struct is_dtype_scalar<short> {enum {value = true};};
template <> struct is_dtype_scalar<int> {enum {value = true};};
template <> struct is_dtype_scalar<long> {enum {value = true};};
template <> struct is_dtype_scalar<long long> {enum {value = true};};
template <> struct is_dtype_scalar<unsigned char> {enum {value = true};};
template <> struct is_dtype_scalar<unsigned short> {enum {value = true};};
template <> struct is_dtype_scalar<unsigned int> {enum {value = true};};
template <> struct is_dtype_scalar<unsigned long> {enum {value = true};};
template <> struct is_dtype_scalar<unsigned long long> {enum {value = true};};
template <> struct is_dtype_scalar<float> {enum {value = true};};
template <> struct is_dtype_scalar<double> {enum {value = true};};

/** Typedef for dtype endian byte-swapping operation */
typedef void (*byteswap_operation_t)(void *dst, const void *src, uintptr_t itemsize);

class dtype;

// The extended_dtype class is for dtypes which require more data
// than a type_id, kind, and itemsize, and endianness.
class extended_dtype {
public:
    virtual ~extended_dtype();

    /**
     * This should make a clone of the extended_dtype information, with
     * everything in native byte-order.
     */
    virtual extended_dtype *clone_as_nbo() const = 0;

    /**
     * Should return true if everything in the dtype is native byte-order,
     * false if the dtype is partially or fully byte-swapped.
     */
    virtual bool is_nbo() const = 0;

    /**
     * For dtypes which are primitive and can support either native or swapped
     * byte endianness, this returns a function to do the swap. Byte-swapping of
     * more complex aggregate dtypes is only supported by assignment operations,
     * where the destination dtype has a different byte-order than the source.
     */
    virtual byteswap_operation_t get_byteswap_operation() const;

    virtual void print(std::ostream& o, const dtype& dt, const void *data, intptr_t stride, intptr_t size,
                        const char *separator) const = 0;

    /**
     * Tests that the two dtypes have identical binary layouts. This method
     * should only be called when this is the extended_dtype for one of the
     * two dtypes.
     */
    virtual bool can_cast_exact(const dtype& dst_dt, const dtype& src_dt) const = 0;
    /**
     * Tests that the two dtypes have identical binary layouts up to byte order.
     * This method should only be called when this is the extended_dtype for one
     * of the two dtypes.
     */
    virtual bool can_cast_equiv(const dtype& dst_dt, const dtype& src_dt) const = 0;
    /**
     * Tests that the 'src' values can be cast to 'dst' losslessly. This method
     * should only be called when this is the extended_dtype for one of the
     * two dtypes.
     */
    virtual bool can_cast_lossless(const dtype& dst_dt, const dtype& src_dt) const = 0;
    /**
     * Tests that the 'src' values can be cast to 'dst' without going to a lesser
     * dtype kind. This method should only be called when this is the extended_dtype
     * for one of the two dtypes.
     */
    virtual bool can_cast_same_kind(const dtype& dst_dt, const dtype& src_dt) const = 0;

    /** Should return true if the type has construct/copy/move/destruct semantics */
    virtual bool is_object_type() const = 0;
};

/**
 * This function returns a C-string of the base name for
 * the given type id. Raises an exception if the type id
 * is invalid.
 *
 * @param type_id  The type id for which to get the name.
 */
const char *get_type_id_basename(int type_id);

/**
 * This class represents a data type.
 *
 * The purpose of this data type is to describe the data layout
 * of elements in ndarrays. The class stores a number of common
 * properties, like a type id, a kind, an alignment, a byte-swapped
 * flag, and an itemsize. Some data types have additional data
 * which is stored as a dynamically allocated extended_dtype object.
 *
 * For the simple built-in dtypes, no extended data is needed, in
 * which case this is entirely a value type with no allocated memory.
 *
 */
class dtype {
private:
    unsigned char m_type_id, m_kind, m_alignment, m_byteswapped;
    uintptr_t m_itemsize;
    std::shared_ptr<extended_dtype> m_data;

public:
    /** Constructor */
    dtype();
    /** Copy constructor (should be "= default" in C++11) */
    dtype(const dtype& rhs)
        : m_type_id(rhs.m_type_id), m_kind(rhs.m_kind), m_alignment(rhs.m_alignment),
          m_byteswapped(rhs.m_byteswapped), m_itemsize(rhs.m_itemsize), m_data(rhs.m_data) {}
    /** Move constructor (should be "= default" in C++11) */
    dtype(dtype&& rhs)
        : m_type_id(rhs.m_type_id), m_kind(rhs.m_kind), m_alignment(rhs.m_alignment),
          m_byteswapped(rhs.m_byteswapped), m_itemsize(rhs.m_itemsize),
          m_data(std::move(rhs.m_data)) {}
    /** Assignment operator (should be "= default" in C++11) */
    dtype& operator=(const dtype& rhs) {
        m_type_id = rhs.m_type_id;
        m_kind = rhs.m_kind;
        m_alignment = rhs.m_alignment;
        m_byteswapped = rhs.m_byteswapped;
        m_itemsize = rhs.m_itemsize;
        m_data = rhs.m_data;
        return *this;
    }
    /** Move assignment operator (should be "= default" in C++11) */
    dtype& operator=(dtype&& rhs) {
        m_type_id = rhs.m_type_id;
        m_kind = rhs.m_kind;
        m_alignment = rhs.m_alignment;
        m_byteswapped = rhs.m_byteswapped;
        m_itemsize = rhs.m_itemsize;
        m_data = std::move(rhs.m_data);
        return *this;
    }

    /** Construct from a type ID */
    explicit dtype(int type_id);
    /** Construct from a type ID and itemsize */
    explicit dtype(int type_id, uintptr_t size);

    void swap(dtype& rhs) {
        std::swap(m_type_id, rhs.m_type_id);
        std::swap(m_kind, rhs.m_kind);
        std::swap(m_alignment, rhs.m_alignment);
        std::swap(m_byteswapped, rhs.m_byteswapped);
        std::swap(m_itemsize, rhs.m_itemsize);
        m_data.swap(rhs.m_data);
    }

    bool operator==(const dtype& rhs) const {
        return m_type_id == rhs.m_type_id &&
                m_itemsize == rhs.m_itemsize &&
                m_kind == rhs.m_kind &&
                m_alignment == rhs.m_alignment &&
                m_byteswapped == rhs.m_byteswapped &&
                (m_data != NULL ? m_data->can_cast_exact(*this, rhs)
                                : (rhs.m_data != NULL ? rhs.m_data->can_cast_exact(*this, rhs)
                                                      : true));
    }
    bool operator!=(const dtype& rhs) const {
        return !(operator==(rhs));
    }

    /**
     * Returns a version of the dtype in native byte order.
     */
    dtype as_nbo() const {
        if (m_data == NULL || m_data->is_nbo()) {
            // In this case, a simple copy with byteswapped set to false is enough
            if (m_byteswapped) {
                dtype result(*this);
                result.m_byteswapped = false;
                return std::move(result);
            } else {
                return *this;
            }
        } else {
            // Must copy m_data to be NBO as well
            dtype result(*this);
            result.m_byteswapped = false;
            result.m_data.reset(m_data->clone_as_nbo());
            return std::move(result);
        }
    }

    /** Whether the dtype is byte-swapped */
    bool is_byteswapped() const {
        return m_byteswapped;
    }

    /**
     * The type number is an enumeration of data types, starting
     * at 0, with one value for each unique data type. This is
     * inspired by the approach in NumPy, and the intention is
     * to have the default
     */
    int type_id() const {
        return m_type_id;
    }

    /** The 'kind' of the dtype (int, uint, float, etc) */
    dtype_kind kind() const {
        return (dtype_kind)m_kind;
    }

    /** The alignment of the dtype */
    int alignment() const {
        return m_alignment;
    }

    /** The item size of the dtype */
    uintptr_t itemsize() const {
        return m_itemsize;
    }

    bool is_object_type() const {
        return m_data != NULL && m_data->is_object_type();
    }

    /**
     * Returns true if the data pointer is aligned
     *
     * @param dataptr  The pointer to the data.
     */
    bool is_data_aligned(const void* dataptr) const {
        return ((m_alignment - 1) & reinterpret_cast<intptr_t>(dataptr)) == 0;
    }

    /**
     * Returns true if the data will always be aligned
     * for this data type.
     *
     * @param align_test  This value should be the bitwise-OR (|)
     *                    of the origin data pointer and all the strides
     *                    that may be added to the data.
     */
    bool is_data_aligned(char align_test) const {
        return ((char)(m_alignment - 1) & align_test) == 0;
    }

    /**
     * Returns a const pointer to the extended_dtype object which
     * contains information about the dtype, or NULL if no extended
     * dtype information exists. The returned pointer is only valid during
     * the lifetime of the dtype.
     */
    const extended_dtype* extended() const {
        return m_data.get();
    }

    byteswap_operation_t get_byteswap_operation() const;

    void print(std::ostream& o, const void *data, intptr_t stride, intptr_t size, const char *separator) const;

    friend std::ostream& operator<<(std::ostream& o, const dtype& rhs);
};

// Convenience function which makes a dtype object from a template parameter
template<class T>
dtype make_dtype() {
    return dtype(type_id_of<T>::value);
}

std::ostream& operator<<(std::ostream& o, const dtype& rhs);

} // namespace dnd

#endif // _DND__DTYPE_HPP_
