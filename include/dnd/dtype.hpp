//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// Copyright (C) 2012 Continuum Analytics
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DND__DTYPE_HPP_
#define _DND__DTYPE_HPP_

#include <stdint.h>
#include <iostream>
#include <complex>

#include <dnd/config.hpp>
#include <dnd/operations.hpp>

namespace dnd {

// A boolean class for dynamicndarray which is one-byte big
class dnd_bool {
    char m_value;
public:
    dnd_bool() : m_value(0) {}
    
    dnd_bool(bool value) : m_value(value) {}
    
    // Special case complex conversion to avoid ambiguous overload
    template<class T>
    dnd_bool(std::complex<T> value) : m_value(value != std::complex<T>(0)) {}

    operator bool() const {
        return (bool)m_value;
    }
};

enum dtype_kind_t {
    bool_kind,
    int_kind,
    uint_kind,
    real_kind,
    complex_kind,
    string_kind,
    // For struct_type_id and array_type_id
    composite_kind,
    // For dtypes whose value_dtype != the dtype, signals
    // that calculations should look at the value_dtype for
    // type promotion, etc.
    expression_kind,
    // For pattern-matching dtypes
    pattern_kind,
    // For use when it becomes possible to register custom dtypes
    custom_kind
};

enum type_id_t {
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
    // Complex floating-point types
    complex_float32_type_id,
    complex_float64_type_id,
    // UTF8 strings
    utf8_type_id,
    // Composite dtypes
    struct_type_id,
    tuple_type_id,
    array_type_id,

    // Adapter dtypes
    conversion_type_id,
    byteswap_type_id,

    // pattern matches against other types - cannot instantiate
    pattern_type_id,

    // The number of built-in, atomic types
    builtin_type_id_count = 13
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
template <> struct type_id_of<std::complex<float> > {enum {value = complex_float32_type_id};};
template <> struct type_id_of<std::complex<double> > {enum {value = complex_float64_type_id};};

// Type trait for the kind
template <typename T> struct dtype_kind_of;

// Can't use bool, because it doesn't have a guaranteed sizeof
template <> struct dtype_kind_of<dnd_bool> {static const dtype_kind_t value = bool_kind;};
template <> struct dtype_kind_of<char> {
    static const dtype_kind_t value = ((char)-1) < 0 ? int_kind : uint_kind;
};
template <> struct dtype_kind_of<signed char> {static const dtype_kind_t value = int_kind;};
template <> struct dtype_kind_of<short> {static const dtype_kind_t value = int_kind;};
template <> struct dtype_kind_of<int> {static const dtype_kind_t value = int_kind;};
template <> struct dtype_kind_of<long> {static const dtype_kind_t value = int_kind;};
template <> struct dtype_kind_of<long long> {static const dtype_kind_t value = int_kind;};
template <> struct dtype_kind_of<uint8_t> {static const dtype_kind_t value = uint_kind;};
template <> struct dtype_kind_of<uint16_t> {static const dtype_kind_t value = uint_kind;};
template <> struct dtype_kind_of<unsigned int> {static const dtype_kind_t value = uint_kind;};
template <> struct dtype_kind_of<unsigned long> {static const dtype_kind_t value = uint_kind;};
template <> struct dtype_kind_of<unsigned long long>{static const dtype_kind_t value = uint_kind;};
template <> struct dtype_kind_of<float> {static const dtype_kind_t value = real_kind;};
template <> struct dtype_kind_of<double> {static const dtype_kind_t value = real_kind;};
template <typename T> struct dtype_kind_of<std::complex<T> > {static const dtype_kind_t value = complex_kind;};

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
template <> struct is_dtype_scalar<std::complex<float> > {enum {value = true};};
template <> struct is_dtype_scalar<std::complex<double> > {enum {value = true};};

class dtype;

// The extended_dtype class is for dtypes which require more data
// than a type_id, kind, and itemsize, and endianness.
class extended_dtype {
public:
    virtual ~extended_dtype();

    virtual type_id_t type_id() const = 0;
    virtual dtype_kind_t kind() const = 0;
    virtual unsigned char alignment() const = 0;
    virtual uintptr_t itemsize() const = 0;

    /**
     * Should return a reference to the dtype representing the value which
     * is for calculation. This should never be an expression dtype.
     *
     * @param self    The dtype which holds the shared_ptr<extended_dtype> containing this.
     */
    virtual const dtype& value_dtype(const dtype& self) const = 0;
    /**
     * Should return a reference to a dtype representing the data this dtype
     * uses to produce the value. This should only be different from *this for expression_kind
     * dtypes.
     *
     * @param self    The dtype which holds the shared_ptr<extended_dtype> containing this.
     */
    virtual const dtype& operand_dtype(const dtype& self) const {
        return self;
    }

    virtual void print_data(std::ostream& o, const dtype& dt, const char *data, intptr_t stride, intptr_t size,
                        const char *separator) const = 0;

    virtual void print(std::ostream& o) const = 0;

    /** Should return true if the type has construct/copy/move/destruct semantics */
    virtual bool is_object_type() const = 0;

    virtual bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const = 0;

    virtual bool operator==(const extended_dtype& rhs) const = 0;

    // For expression_kind dtypes - converts from (operand_dtype().value_dtype()) to (value_dtype())
    virtual void get_operand_to_value_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride,
                                kernel_instance<unary_operation_t>& out_kernel) const;
    // For expression_kind dtypes - converts from (value_dtype()) to (operand_dtype().value_dtype())
    virtual void get_value_to_operand_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride,
                                kernel_instance<unary_operation_t>& out_kernel) const;
};

/**
 * This function returns a C-string of the base name for
 * the given type id. Raises an exception if the type id
 * is invalid.
 *
 * @param type_id  The type id for which to get the name.
 */
const char *get_type_id_basename(type_id_t type_id);

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
    unsigned char m_type_id, m_kind, m_alignment;
    uintptr_t m_itemsize;
    shared_ptr<extended_dtype> m_data;

public:
    /** Constructor */
    dtype();
    /** Constructor from an extended_dtype */
    dtype(const shared_ptr<extended_dtype>& data)
        : m_type_id(data->type_id()), m_kind(data->kind()), m_alignment(data->alignment()),
        m_itemsize(data->itemsize()), m_data(data) {}
    /** Copy constructor (should be "= default" in C++11) */
    dtype(const dtype& rhs)
        : m_type_id(rhs.m_type_id), m_kind(rhs.m_kind), m_alignment(rhs.m_alignment),
          m_itemsize(rhs.m_itemsize), m_data(rhs.m_data) {}
    /** Assignment operator (should be "= default" in C++11) */
    dtype& operator=(const dtype& rhs) {
        m_type_id = rhs.m_type_id;
        m_kind = rhs.m_kind;
        m_alignment = rhs.m_alignment;
        m_itemsize = rhs.m_itemsize;
        m_data = rhs.m_data;
        return *this;
    }
#ifdef DND_RVALUE_REFS
    /** Constructor from an rvalue extended_dtype */
    dtype(const shared_ptr<extended_dtype>&& data)
        : m_type_id(data->type_id(), m_kind(data->kind()), m_alignment(data->alignment()),
        m_itemsize(data->itemsize()), m_data(std::move(data)) {}
    /** Move constructor (should be "= default" in C++11) */
    dtype(dtype&& rhs)
        : m_type_id(rhs.m_type_id), m_kind(rhs.m_kind), m_alignment(rhs.m_alignment),
          m_itemsize(rhs.m_itemsize),
          m_data(std::move(rhs.m_data)) {}
    /** Move assignment operator (should be "= default" in C++11) */
    dtype& operator=(dtype&& rhs) {
        m_type_id = rhs.m_type_id;
        m_kind = rhs.m_kind;
        m_alignment = rhs.m_alignment;
        m_itemsize = rhs.m_itemsize;
        m_data = std::move(rhs.m_data);
        return *this;
    }
#endif // DND_RVALUE_REFS

    /** Construct from a type ID */
    explicit dtype(type_id_t type_id);
    explicit dtype(int type_id);
    /** Construct from a type ID and itemsize */
    explicit dtype(type_id_t type_id, uintptr_t size);
    explicit dtype(int type_id, uintptr_t size);

    /** Construct from a string representation */
    explicit dtype(const std::string& rep);

    void swap(dtype& rhs) {
        std::swap(m_type_id, rhs.m_type_id);
        std::swap(m_kind, rhs.m_kind);
        std::swap(m_alignment, rhs.m_alignment);
        std::swap(m_itemsize, rhs.m_itemsize);
        m_data.swap(rhs.m_data);
    }

    bool operator==(const dtype& rhs) const {
        if (m_data && rhs.m_data) {
            return *m_data == *rhs.m_data;
        }
        return m_type_id == rhs.m_type_id &&
                m_itemsize == rhs.m_itemsize &&
                m_kind == rhs.m_kind &&
                m_alignment == rhs.m_alignment &&
                m_data == rhs.m_data;
    }
    bool operator!=(const dtype& rhs) const {
        return !(operator==(rhs));
    }

    const dtype& value_dtype() const {
        // Only expression_kind dtypes have different value_dtype
        if (m_kind != expression_kind) {
            return *this;
        } else {
            return m_data->value_dtype(*this);
        }
    }

    const dtype& storage_dtype() const {
        // Only expression_kind dtypes have different storage_dtype
        if (m_kind != expression_kind) {
            return *this;
        } else {
            // Follow the operand dtype chain to get the storage dtype
            const dtype* sdt = &m_data->operand_dtype(*this);
            while (sdt->kind() == expression_kind) {
                sdt = &sdt->m_data->operand_dtype(*sdt);
            }
            return *sdt;
        }
    }

    /**
     * The type number is an enumeration of data types, starting
     * at 0, with one value for each unique data type. This is
     * inspired by the approach in NumPy, and the intention is
     * to have the default
     */
    type_id_t type_id() const {
        return (type_id_t)m_type_id;
    }

    /** The 'kind' of the dtype (int, uint, float, etc) */
    dtype_kind_t kind() const {
        return (dtype_kind_t)m_kind;
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

    void print_data(std::ostream& o, const char *data, intptr_t stride, intptr_t size,
                                                            const char *separator) const;

    // Converts from (storage_dtype()) to (value_dtype()). Only necessary for expression_kind dtypes
    void get_storage_to_value_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride,
                                kernel_instance<unary_operation_t>& out_kernel) const;
    // Converts from (value_dtype()) to (storage_dtype()). Only necessary for expression_kind dtypes
    void get_value_to_storage_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride,
                                kernel_instance<unary_operation_t>& out_kernel) const;

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
