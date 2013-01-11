//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DTYPE_HPP_
#define _DYND__DTYPE_HPP_

#include <iostream>
#include <complex>
#include <stdexcept>

#include <dynd/dtypes/base_dtype.hpp>
#include <dynd/dtypes/base_expression_dtype.hpp>
#include <dynd/dtypes/base_string_dtype.hpp>
#include <dynd/dtype_comparisons.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/exceptions.hpp>

namespace dynd {

// A boolean class for dynamicndarray which is one-byte big
class dynd_bool {
    char m_value;
public:
    dynd_bool() : m_value(0) {}

    dynd_bool(bool value) : m_value(value) {}

    // Special case complex conversion to avoid ambiguous overload
    template<class T>
    dynd_bool(std::complex<T> value) : m_value(value != std::complex<T>(0)) {}

    operator bool() const {
        return m_value != 0;
    }
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
template <> struct type_id_of<dynd_bool> {enum {value = bool_type_id};};
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
template <> struct type_id_of<void> {enum {value = void_type_id};};

// Type trait for the kind
template <typename T> struct dtype_kind_of;

template <> struct dtype_kind_of<void> {static const dtype_kind_t value = void_kind;};
// Can't use bool, because it doesn't have a guaranteed sizeof
template <> struct dtype_kind_of<dynd_bool> {static const dtype_kind_t value = bool_kind;};
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
template <> struct is_dtype_scalar<dynd_bool> {enum {value = true};};
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

// Metaprogram for determining scalar alignment
template<typename T> struct scalar_align_of;
template <> struct scalar_align_of<dynd_bool> {enum {value = 1};};
template <> struct scalar_align_of<char> {enum {value = 1};};
template <> struct scalar_align_of<signed char> {enum {value = 1};};
template <> struct scalar_align_of<short> {enum {value = sizeof(short)};};
template <> struct scalar_align_of<int> {enum {value = sizeof(int)};};
template <> struct scalar_align_of<long> {enum {value = sizeof(long)};};
template <> struct scalar_align_of<long long> {enum {value = sizeof(long long)};};
template <> struct scalar_align_of<unsigned char> {enum {value = 1};};
template <> struct scalar_align_of<unsigned short> {enum {value = sizeof(unsigned short)};};
template <> struct scalar_align_of<unsigned int> {enum {value = sizeof(unsigned int)};};
template <> struct scalar_align_of<unsigned long> {enum {value = sizeof(unsigned long)};};
template <> struct scalar_align_of<unsigned long long> {enum {value = sizeof(unsigned long long)};};
template <> struct scalar_align_of<float> {enum {value = sizeof(long)};};
template <> struct scalar_align_of<double> {enum {value = sizeof(double)};};
template <> struct scalar_align_of<std::complex<float> > {enum {value = sizeof(long)};};
template <> struct scalar_align_of<std::complex<double> > {enum {value = sizeof(double)};};

// Metaprogram for determining if a type is the C++ "bool" or not
template<typename T> struct is_type_bool {enum {value = false};};
template<> struct is_type_bool<bool> {enum {value = true};};

/**
 * Increments the offset value so that it is aligned to the requested alignment
 * NOTE: The alignment must be a power of two.
 */
inline size_t inc_to_alignment(size_t offset, size_t alignment) {
    return (offset + alignment - 1) & (std::size_t)(-(std::ptrdiff_t)alignment);
}

/**
 * Increments the pointer value so that it is aligned to the requested alignment
 * NOTE: The alignment must be a power of two.
 */
inline char *inc_to_alignment(char *ptr, size_t alignment) {
    return reinterpret_cast<char *>((reinterpret_cast<std::size_t>(ptr) + alignment - 1) &
                                    (std::size_t)(-(std::ptrdiff_t)alignment));
}

/**
 * Increments the pointer value so that it is aligned to the requested alignment
 * NOTE: The alignment must be a power of two.
 */
inline void *inc_to_alignment(void *ptr, size_t alignment) {
    return reinterpret_cast<char *>((reinterpret_cast<std::size_t>(ptr) + alignment - 1) &
                                    (size_t)(-(std::ptrdiff_t)alignment));
}

/**
 * \brief Tests whether the offset has the requested alignment.
 *
 * NOTE: The alignment must be a power of two.
 *
 * \param offset  The offset whose alignment is tested.
 * \param alignment  The required alignment, must be a power of two.
 *
 * \returns  True if the offset is divisible by the power of two alignment,
 *           False otherwise.
 */
inline bool offset_is_aligned(size_t offset, size_t alignment) {
    return (offset&(alignment - 1)) == 0;
}

/** Prints a single scalar of a builtin dtype to the stream */
void print_builtin_scalar(type_id_t type_id, std::ostream& o, const char *data);

/** Special iterdata which broadcasts to any number of additional dimensions */
struct iterdata_broadcasting_terminator {
    iterdata_common common;
    char *data;
};
char *iterdata_broadcasting_terminator_incr(iterdata_common *iterdata, int level);
char *iterdata_broadcasting_terminator_reset(iterdata_common *iterdata, char *data, int level);

/**
 * This class represents a data type.
 *
 * The purpose of this data type is to describe the data layout
 * of elements in ndarrays. The class stores a number of common
 * properties, like a type id, a kind, an alignment, a byte-swapped
 * flag, and an element_size. Some data types have additional data
 * which is stored as a dynamically allocated base_dtype object.
 *
 * For the simple built-in dtypes, no extended data is needed, in
 * which case this is entirely a value type with no allocated memory.
 *
 */
class dtype {
private:
    const base_dtype *m_extended;

    inline static bool is_builtin(const base_dtype *ext) {
        return (reinterpret_cast<uintptr_t>(ext)&(~builtin_type_id_mask)) == 0;
    }
    /**
     * Validates that the given type ID is a proper ID and casts to
     * an base_dtype pointer if it is. Throws
     * an exception if not.
     *
     * \param type_id  The type id to validate.
     */
    static inline const base_dtype *validate_builtin_type_id(type_id_t type_id)
    {
        // 0 <= type_id < (builtin_type_id_count + 1)
        // The + 1 is for void_type_id
        if ((unsigned int)type_id < builtin_type_id_count + 1) {
            return reinterpret_cast<const base_dtype *>(type_id);
        } else {
            throw invalid_type_id((int)type_id);
        }
    }

    static uint8_t builtin_kinds[builtin_type_id_count + 1];
    static uint8_t builtin_data_sizes[builtin_type_id_count + 1];
    static uint8_t builtin_data_alignments[builtin_type_id_count + 1];
public:
    /** Constructor */
    dtype()
        : m_extended(reinterpret_cast<const base_dtype *>(void_type_id))
    {}
    /** Constructor from an base_dtype. This claims ownership of the 'extended' reference by default, be careful! */
    inline explicit dtype(const base_dtype *extended, bool incref)
        : m_extended(extended)
    {
        if (incref && !dtype::is_builtin(extended)) {
            base_dtype_incref(m_extended);
        }
    }
    /** Copy constructor (should be "= default" in C++11) */
    dtype(const dtype& rhs)
        : m_extended(rhs.m_extended)
    {
        if (!dtype::is_builtin(m_extended)) {
            base_dtype_incref(m_extended);
        }
    }
    /** Assignment operator (should be "= default" in C++11) */
    dtype& operator=(const dtype& rhs) {
        if (!dtype::is_builtin(m_extended)) {
            base_dtype_decref(m_extended);
        }
        m_extended = rhs.m_extended;
        if (!dtype::is_builtin(m_extended)) {
            base_dtype_incref(m_extended);
        }
        return *this;
    }
#ifdef DYND_RVALUE_REFS
    /** Move constructor */
    dtype(dtype&& rhs)
        : m_extended(rhs.m_extended)
    {
        rhs.m_extended = reinterpret_cast<const base_dtype *>(void_type_id);
    }
    /** Move assignment operator */
    dtype& operator=(dtype&& rhs) {
        if (!dtype::is_builtin(m_extended)) {
            base_dtype_decref(m_extended);
        }
        m_extended = rhs.m_extended;
        rhs.m_extended = reinterpret_cast<const base_dtype *>(void_type_id);
        return *this;
    }
#endif // DYND_RVALUE_REFS

    /** Construct from a builtin type ID */
    explicit dtype(type_id_t type_id)
        : m_extended(dtype::validate_builtin_type_id(type_id))
    {}

    /** Construct from a string representation */
    explicit dtype(const std::string& rep);

    ~dtype() {
        if (!is_builtin()) {
            base_dtype_decref(m_extended);
        }
    }

    void swap(dtype& rhs) {
        std::swap(m_extended, rhs.m_extended);
    }

    inline bool operator==(const dtype& rhs) const {
        if (is_builtin() || rhs.is_builtin()) {
            return m_extended == rhs.m_extended;
        } else {
            return *m_extended == *rhs.m_extended;
        }
    }
    bool operator!=(const dtype& rhs) const {
        return !(operator==(rhs));
    }

    /**
     * Returns true if this dtype is of a builtin dtype, which
     * means the type id is encoded directly in the m_extended
     * pointer.
     */
    inline bool is_builtin() const {
        return dtype::is_builtin(m_extended);
    }

    /**
     * Indexes into the dtype. This function returns the dtype which results
     * from applying the same index to an ndarray of this dtype.
     *
     * \param ndim         The number of elements in the 'indices' array
     * \param indices      The indices to apply.
     */
    dtype at_array(int nindices, const irange *indices) const;

    /**
     * The 'at' function is used for indexing. Overloading operator[] isn't
     * practical for multidimensional objects. Indexing one dimension with
     * an integer index is special-cased, both for higher performance and
     * to provide a way to get a metadata pointer for the result dtype.
     *
     * \param i0  The index to apply.
     * \param inout_metadata  If non-NULL, points to a metadata pointer for
     *                        this dtype that is modified to point to the
     *                        result's metadata.
     * \param inout_data  If non-NULL, points to a data pointer that is modified
     *                    to point to the result's data. If `inout_data` is non-NULL,
     *                    `inout_metadata` must also be non-NULL.
     *
     * \returns  The dtype that results from the indexing operation.
     */
    inline dtype at_single(intptr_t i0, const char **inout_metadata = NULL, const char **inout_data = NULL) const {
        if (!is_builtin()) {
            return m_extended->at(i0, inout_metadata, inout_data);
        } else {
            throw too_many_indices(*this, 1, 0);
        }
    }

    inline dtype at(intptr_t i0) const {
        return at_single(i0);
    }

    /**
     * The 'at' function is used for indexing. Overloading operator[] isn't
     * practical for multidimensional objects.
     */
    inline dtype at(const irange& i0) const {
        return at_array(1, &i0);
    }

    /** Indexing with two index values */
    inline dtype at(const irange& i0, const irange& i1) const {
        irange i[2] = {i0, i1};
        return at_array(2, i);
    }

    /** Indexing with three index values */
    inline dtype at(const irange& i0, const irange& i1, const irange& i2) const {
        irange i[3] = {i0, i1, i2};
        return at_array(3, i);
    }
    /** Indexing with four index values */
    inline dtype at(const irange& i0, const irange& i1, const irange& i2, const irange& i3) const {
        irange i[4] = {i0, i1, i2, i3};
        return at_array(4, i);
    }

    /**
     * Indexes into the dtype, intended for recursive calls from the extended-dtype version. See
     * the function in base_dtype with the same name for more details.
     */
    dtype apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const;

    /**
     * Returns the non-expression dtype that this
     * dtype looks like for the purposes of calculation,
     * printing, etc.
     */
    const dtype& value_dtype() const {
        // Only expression_kind dtypes have different value_dtype
        if (is_builtin() || m_extended->get_kind() != expression_kind) {
            return *this;
        } else {
            // All chaining happens in the operand_dtype
            return static_cast<const base_expression_dtype *>(m_extended)->get_value_dtype();
        }
    }

    /**
     * For expression dtypes, returns the operand dtype,
     * which is the source dtype of this dtype's expression.
     * This is one link down the expression chain.
     */
    const dtype& operand_dtype() const {
        // Only expression_kind dtypes have different operand_dtype
        if (is_builtin() || m_extended->get_kind() != expression_kind) {
            return *this;
        } else {
            return static_cast<const base_expression_dtype *>(m_extended)->get_operand_dtype();
        }
    }

    /**
     * For expression dtypes, returns the storage dtype,
     * which is the dtype of the underlying input data.
     * This is the bottom of the expression chain.
     */
    const dtype& storage_dtype() const {
        // Only expression_kind dtypes have different storage_dtype
        if (is_builtin() || m_extended->get_kind() != expression_kind) {
            return *this;
        } else {
            // Follow the operand dtype chain to get the storage dtype
            const dtype* dt = &static_cast<const base_expression_dtype *>(m_extended)->get_operand_dtype();
            while (dt->get_kind() == expression_kind) {
                dt = &static_cast<const base_expression_dtype *>(dt->m_extended)->get_operand_dtype();
            }
            return *dt;
        }
    }

    /**
     * The type number is an enumeration of data types, starting
     * at 0, with one value for each unique data type. This is
     * inspired by the approach in NumPy, and the intention is
     * to have the default
     */
    type_id_t get_type_id() const {
        if (is_builtin()) {
            return static_cast<type_id_t>(reinterpret_cast<intptr_t>(m_extended));
        } else {
            return m_extended->get_type_id();
        }
    }

    /** The 'kind' of the dtype (int, uint, float, etc) */
    dtype_kind_t get_kind() const {
        if (is_builtin()) {
            return static_cast<dtype_kind_t>(dtype::builtin_kinds[reinterpret_cast<intptr_t>(m_extended)]);
        } else {
            return m_extended->get_kind();
        }
    }

    /** The alignment of the dtype */
    size_t get_alignment() const {
        if (is_builtin()) {
            return static_cast<size_t>(dtype::builtin_data_alignments[reinterpret_cast<intptr_t>(m_extended)]);
        } else {
            return m_extended->get_alignment();
        }
    }

    /** The element size of the dtype */
    size_t get_data_size() const {
        if (is_builtin()) {
            return static_cast<size_t>(dtype::builtin_data_sizes[reinterpret_cast<intptr_t>(m_extended)]);
        } else {
            return m_extended->get_data_size();
        }
    }

    inline dtype_memory_management_t get_memory_management() const {
        if (is_builtin()) {
            return pod_memory_management;
        } else {
            return m_extended->get_memory_management();
        }
    }

    /*
     * Return a comparison kernel that can perform the requested single comparison on
     * data of this dtype
     *
     * \param compare_id the identifier of the comparison
     */
    void get_single_compare_kernel(kernel_instance<compare_operations_t>& out_kernel) const;

    inline bool is_scalar() const {
        if (is_builtin()) {
            return true;
        } else {
            return m_extended->is_scalar();
        }
    }

    /**
     * Returns true if the dtype contains any expression
     * dtype within it somewhere.
     */
    inline bool is_expression() const {
        if (is_builtin()) {
            return false;
        } else {
            return m_extended->is_expression();
        }
    }

    /**
     * For array types, recursively applies to each child type, and for
     * scalar types converts to the provided one.
     *
     * \param scalar_dtype  The scalar dtype to convert all scalars to.
     * \param errmode       The error mode for the conversion.
     */
    dtype with_replaced_scalar_types(const dtype& scalar_dtype, assign_error_mode errmode = assign_error_default) const;

    /**
     * Returns a modified dtype with all expression dtypes replaced with
     * their value dtypes, and dtypes replaced with "standard versions"
     * whereever appropriate. For example, an offset-based uniform array
     * would be replaced by a strided uniform array.
     */
    inline dtype get_canonical_dtype() const {
        if (is_builtin()) {
            return *this;
        } else {
            return m_extended->get_canonical_dtype();
        }
    }

    /**
     * Gets the number of uniform dimensions in the dtype.
     */
    inline size_t get_undim() const {
        if (is_builtin()) {
            return 0;
        } else {
            return m_extended->get_undim();
        }
    }

    /**
     * Gets the dtype with all the unifom dimensions stripped away.
     */
    inline dtype get_udtype() const {
        size_t undim = get_undim();
        if (get_undim() == 0) {
            return *this;
        } else {
            return m_extended->get_dtype_at_dimension(NULL, undim);
        }
    }

    intptr_t get_dim_size(const char *data, const char *metadata) const {
        if (!is_builtin()) {
            return m_extended->get_dim_size(data, metadata);
        } else {
            std::stringstream ss;
            ss << "Cannot get the leading dimension size of ndobject with scalar dtype " << *this;
            throw std::runtime_error(ss.str());
        }
    }

    inline dtype get_dtype_at_dimension(char **inout_metadata, size_t i, size_t total_ndim = 0) const {
        if (!is_builtin()) {
            return m_extended->get_dtype_at_dimension(inout_metadata, i, total_ndim);
        } else if (i == 0) {
            return *this;
        } else {
            throw too_many_indices(*this, total_ndim + i, total_ndim);
        }
    }


    /**
     * Returns a const pointer to the base_dtype object which
     * contains information about the dtype, or NULL if no extended
     * dtype information exists. The returned pointer is only valid during
     * the lifetime of the dtype.
     */
    inline const base_dtype* extended() const {
        return m_extended;
    }

    /** The size of the data required for uniform iteration */
    inline size_t get_iterdata_size(int ndim) const {
        if (is_builtin()) {
            return 0;
        } else {
            return m_extended->get_iterdata_size(ndim);
        }
    }
    /**
     * \brief Constructs the iterdata for processing iteration of the specified shape.
     *
     * \param iterdata  The allocated iterdata to construct.
     * \param inout_metadata  The metadata corresponding to the dtype for the iterdata construction.
     *                        This is modified in place to become the metadata for the uniform dtype.
     * \param ndim      Number of iteration dimensions.
     * \param shape     The iteration shape.
     * \param out_uniform_dtype  This is populated with the dtype of each iterated element
     */
    inline void iterdata_construct(iterdata_common *iterdata, const char **inout_metadata,
                    int ndim, const intptr_t* shape, dtype& out_uniform_dtype) const
    {
        if (!is_builtin()) {
            m_extended->iterdata_construct(iterdata, inout_metadata, ndim, shape, out_uniform_dtype);
        }
    }

    /** Destructs any references or other state contained in the iterdata */
    inline void iterdata_destruct(iterdata_common *iterdata, int ndim) const
    {
        if (!is_builtin()) {
            m_extended->iterdata_destruct(iterdata, ndim);
        }
    }

    inline size_t get_broadcasted_iterdata_size(int ndim) const {
        if (is_builtin()) {
            return sizeof(iterdata_broadcasting_terminator);
        } else {
            return m_extended->get_iterdata_size(ndim) + sizeof(iterdata_broadcasting_terminator);
        }
    }

    /**
     * Constructs an iterdata which can be broadcast to the left indefinitely, by capping
     * off the iterdata with a iterdata_broadcasting_terminator.
     * \param iterdata  The allocated iterdata to construct.
     * \param inout_metadata  The metadata corresponding to the dtype for the iterdata construction.
     *                        This is modified in place to become the metadata for the uniform dtype.
     * \param ndim      Number of iteration dimensions.
     * \param shape     The iteration shape.
     * \param out_uniform_dtype  This is populated with the dtype of each iterated element
     */
    inline void broadcasted_iterdata_construct(iterdata_common *iterdata, const char **inout_metadata,
                    int ndim, const intptr_t* shape, dtype& out_uniform_dtype) const
    {
        size_t size;
        if (is_builtin()) {
            size = 0;
        } else {
            size = m_extended->iterdata_construct(iterdata, inout_metadata, ndim, shape, out_uniform_dtype);
        }
        iterdata_broadcasting_terminator *id = reinterpret_cast<iterdata_broadcasting_terminator *>(
                        reinterpret_cast<char *>(iterdata) + size);
        id->common.incr = &iterdata_broadcasting_terminator_incr;
        id->common.reset = &iterdata_broadcasting_terminator_reset;
    }

    /**
     * print data interpreted as a single value of this dtype
     *
     * \param o         the std::ostream to print to
     * \param data      pointer to the data element to print
     * \param metadata  pointer to the ndobject metadata for the data element
     */
    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    friend std::ostream& operator<<(std::ostream& o, const dtype& rhs);
};

// Convenience function which makes a dtype object from a template parameter
template<class T>
dtype make_dtype()
{
    return dtype(static_cast<type_id_t>(type_id_of<T>::value));
}

/**
 * A static array of the builtin dtypes and void.
 * If code is specialized just for a builtin type, like int, it can use
 * static_builtin_dtypes[type_id_of<int>::value] as a fast
 * way to get a const reference to its dtype.
 */
extern const dtype static_builtin_dtypes[builtin_type_id_count + 1];

std::ostream& operator<<(std::ostream& o, const dtype& rhs);
/** Prints raw bytes as hexadecimal */
void hexadecimal_print(std::ostream& o, char value);
void hexadecimal_print(std::ostream& o, unsigned char value);
void hexadecimal_print(std::ostream& o, unsigned short value);
void hexadecimal_print(std::ostream& o, unsigned int value);
void hexadecimal_print(std::ostream& o, unsigned long value);
void hexadecimal_print(std::ostream& o, unsigned long long value);
void hexadecimal_print(std::ostream& o, const char *data, intptr_t element_size);

} // namespace dynd

#endif // _DYND__DTYPE_HPP_
