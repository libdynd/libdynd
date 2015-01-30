//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <stdexcept>

#include <dynd/types/base_type.hpp>
#include <dynd/types/base_expr_type.hpp>
#include <dynd/types/base_string_type.hpp>
#include <dynd/types/dynd_float16.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/exceptions.hpp>

namespace dynd {

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

/** Prints a single scalar of a builtin type to the stream */
void print_builtin_scalar(type_id_t type_id, std::ostream& o, const char *data);

/** Special iterdata which broadcasts to any number of additional dimensions */
struct iterdata_broadcasting_terminator {
    iterdata_common common;
    char *data;
};
char *iterdata_broadcasting_terminator_incr(iterdata_common *iterdata, intptr_t level);
char *iterdata_broadcasting_terminator_adv(iterdata_common *iterdata, intptr_t level, intptr_t i);
char *iterdata_broadcasting_terminator_reset(iterdata_common *iterdata, char *data, intptr_t level);

// Forward declaration of nd::array and nd::strided_vals
namespace nd {
    class array;

    template <typename T, int N>
    class strided_vals;
} // namespace nd

namespace ndt {

template <typename I>
struct index_proxy;

template <size_t... I>
struct index_proxy<index_sequence<I...>> {
  enum { size = index_sequence<I...>::size };

#if !(defined(_MSC_VER) && _MSC_VER == 1800)
  template <typename... A>
  static void get_types(A &&... a);
#else
  static void get_types();
  template <typename A0>
  static void get_types(A0 &&a0);
  template <typename A0, typename A1>
  static void get_types(A0 &&a0, A1 &&a1);
  template <typename A0, typename A1, typename A2>
  static void get_types(A0 &&a0, A1 &&a1, A2 &&a2);
  template <typename A0, typename A1, typename A2, typename A3>
  static void get_types(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3);
  template <typename A0, typename A1, typename A2, typename A3, typename A4>
  static void get_types(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4);
  template <typename A0, typename A1, typename A2, typename A3, typename A4,
            typename A5>
  static void get_types(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5);
#endif

  template <typename... T>
  static void get_types(type *tp, const std::tuple<T...> &vals,
                        const intptr_t *perm = NULL);

//  template <typename... T>
  //static void get_data(char **data, const std::tuple<T...> &vals);

#if !(defined(_MSC_VER) && _MSC_VER == 1800)
  template <typename... A>
  static void get_forward_types(A &&... a);
#else
  static void get_forward_types();
  template <typename A0>
  static void get_forward_types(A0 &&a0);
  template <typename A0, typename A1>
  static void get_forward_types(A0 &&a0, A1 &&a1);
  template <typename A0, typename A1, typename A2>
  static void get_forward_types(A0 &&a0, A1 &&a1, A2 &&a2);
  template <typename A0, typename A1, typename A2, typename A3>
  static void get_forward_types(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3);
  template <typename A0, typename A1, typename A2, typename A3, typename A4>
  static void get_forward_types(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4);
  template <typename A0, typename A1, typename A2, typename A3, typename A4,
            typename A5>
  static void get_forward_types(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5);
#endif

  template <typename... T>
  static void get_forward_types(type *tp, const std::tuple<T...> &vals,
                                const intptr_t *perm = NULL);
};

/**
 * This class represents a data type.
 *
 * The purpose of this data type is to describe the data layout
 * of elements in ndarrays. The class stores a number of common
 * properties, like a type id, a kind, an alignment, a byte-swapped
 * flag, and an element_size. Some data types have additional data
 * which is stored as a dynamically allocated base_type object.
 *
 * For the simple built-in types, no extended data is needed, in
 * which case this is entirely a value type with no allocated memory.
 *
 */
class type {
    const base_type *m_extended;

    /**
     * Validates that the given type ID is a proper ID and casts to
     * a base_type pointer if it is. Throws
     * an exception if not.
     *
     * \param type_id  The type id to validate.
     */
    static const base_type *validate_builtin_type_id(type_id_t type_id)
    {
        // 0 <= type_id < builtin_type_id_count
        if ((unsigned int)type_id < builtin_type_id_count) {
            return reinterpret_cast<const base_type *>(type_id);
        } else {
            throw invalid_type_id((int)type_id);
        }
    }

public:
    /** Constructor */
    type()
        : m_extended(reinterpret_cast<const base_type *>(uninitialized_type_id))
    {}
    /**
     * Constructor from a base_type. This claims ownership of the 'extended'
     * reference if incref is false, be careful!
     */
    explicit type(const base_type *extended, bool incref)
        : m_extended(extended)
    {
        if (incref && !is_builtin_type(extended)) {
            base_type_incref(m_extended);
        }
    }
    /** Copy constructor (should be "= default" in C++11) */
    type(const type& rhs)
        : m_extended(rhs.m_extended)
    {
        if (!is_builtin_type(m_extended)) {
            base_type_incref(m_extended);
        }
    }
    /** Assignment operator (should be "= default" in C++11) */
    type& operator=(const type& rhs) {
        if (!is_builtin_type(m_extended)) {
            base_type_decref(m_extended);
        }
        m_extended = rhs.m_extended;
        if (!is_builtin_type(m_extended)) {
            base_type_incref(m_extended);
        }
        return *this;
    }
#ifdef DYND_RVALUE_REFS
    /** Move constructor */
    type(type&& rhs)
        : m_extended(rhs.m_extended)
    {
        rhs.m_extended = reinterpret_cast<const base_type *>(uninitialized_type_id);
    }
    /** Move assignment operator */
    type& operator=(type&& rhs) {
        if (!is_builtin_type(m_extended)) {
            base_type_decref(m_extended);
        }
        m_extended = rhs.m_extended;
        rhs.m_extended = reinterpret_cast<const base_type *>(uninitialized_type_id);
        return *this;
    }
#endif // DYND_RVALUE_REFS

    /** Construct from a builtin type ID */
    explicit type(type_id_t type_id)
        : m_extended(type::validate_builtin_type_id(type_id))
    {}

    /** Construct from a string representation */
    explicit type(const std::string& rep);

    /** Construct from a string representation */
    type(const char *rep_begin, const char *rep_end);

    ~type() {
        if (!is_builtin()) {
            base_type_decref(m_extended);
        }
    }

    /**
     * The type class operates as a smart pointer for dynamically
     * allocated base_type instances, with raw storage of type id
     * for the built-in types. This function gives away the held
     * reference, leaving behind a void type.
     */
    const base_type *release() {
        const base_type *result = m_extended;
        m_extended = reinterpret_cast<const base_type *>(uninitialized_type_id);
        return result;
    }

    void swap(type& rhs) {
        std::swap(m_extended, rhs.m_extended);
    }

    void swap(const base_type *&rhs) {
        std::swap(m_extended, rhs);
    }

    bool operator==(const type& rhs) const {
        return m_extended == rhs.m_extended ||
               (!is_builtin() && !rhs.is_builtin() &&
                *m_extended == *rhs.m_extended);
    }
    bool operator!=(const type& rhs) const {
        return !(operator==(rhs));
    }

    bool is_null() const {
        return m_extended == NULL;
    }

    /**
     * Returns true if this type is built in, which
     * means the type id is encoded directly in the m_extended
     * pointer.
     */
    bool is_builtin() const {
        return is_builtin_type(m_extended);
    }

    /**
     * Indexes into the type. This function returns the type which results
     * from applying the same index to an ndarray of this type.
     *
     * \param nindices     The number of elements in the 'indices' array
     * \param indices      The indices to apply.
     */
    type at_array(int nindices, const irange *indices) const;

    /**
     * The 'at_single' function is used for indexing by a single dimension, without
     * touching any leading dimensions after the first, in contrast to the 'at'
     * function. Overloading operator[] isn't
     * practical for multidimensional objects. Indexing one dimension with
     * an integer index is special-cased, both for higher performance and
     * to provide a way to get a arrmeta pointer for the result type.
     *
     * \param i0  The index to apply.
     * \param inout_arrmeta  If non-NULL, points to an arrmeta pointer for
     *                        this type that is modified to point to the
     *                        result's arrmeta.
     * \param inout_data  If non-NULL, points to a data pointer that is modified
     *                    to point to the result's data. If `inout_data` is non-NULL,
     *                    `inout_arrmeta` must also be non-NULL.
     *
     * \returns  The type that results from the indexing operation.
     */
    type at_single(intptr_t i0, const char **inout_arrmeta = NULL, const char **inout_data = NULL) const {
        if (!is_builtin()) {
            return m_extended->at_single(i0, inout_arrmeta, inout_data);
        } else {
            throw too_many_indices(*this, 1, 0);
        }
    }

    /**
     * The 'at' function is used for indexing. Overloading operator[] isn't
     * practical for multidimensional objects.
     *
     * NOTE: Calling 'at' may simplify the leading dimension after the indices,
     *       e.g. convert a var_dim to a strided_dim, or collapsing pointers.
     *       If you do not want this collapsing behavior, use the 'at_single' function.
     */
    type at(const irange& i0) const {
        return at_array(1, &i0);
    }

    /** Indexing with two index values */
    type at(const irange& i0, const irange& i1) const {
        irange i[2] = {i0, i1};
        return at_array(2, i);
    }

    /** Indexing with three index values */
    type at(const irange& i0, const irange& i1, const irange& i2) const {
        irange i[3] = {i0, i1, i2};
        return at_array(3, i);
    }
    /** Indexing with four index values */
    type at(const irange& i0, const irange& i1, const irange& i2, const irange& i3) const {
        irange i[4] = {i0, i1, i2, i3};
        return at_array(4, i);
    }

    /** Returns true if this type matches the pattern type provided */
    bool matches(const ndt::type &pattern) const;

    /**
     * Accesses a dynamic property of the type.
     *
     * \param property_name  The property to access.
     */
    nd::array p(const char *property_name) const;
    /**
     * Accesses a dynamic property of the type.
     *
     * \param property_name  The property to access.
     */
    nd::array p(const std::string& property_name) const;

    /**
     * Indexes into the type, intended for recursive calls from the extended-type version. See
     * the function in base_type with the same name for more details.
     */
    type apply_linear_index(intptr_t nindices, const irange *indices,
                size_t current_i, const type& root_tp, bool leading_dimension) const;

    /**
     * Returns the non-expression type that this
     * type looks like for the purposes of calculation,
     * printing, etc.
     */
    const type& value_type() const {
        // Only expr_kind types have different value_type
        if (is_builtin() || m_extended->get_kind() != expr_kind) {
            return *this;
        } else {
            // All chaining happens in the operand_type
            return static_cast<const base_expr_type *>(m_extended)->get_value_type();
        }
    }

    /**
     * For expression types, returns the operand type,
     * which is the source type of this type's expression.
     * This is one link down the expression chain.
     */
    const type& operand_type() const {
        // Only expr_kind types have different operand_type
        if (is_builtin() || m_extended->get_kind() != expr_kind) {
            return *this;
        } else {
            return static_cast<const base_expr_type *>(m_extended)->get_operand_type();
        }
    }

    /**
     * For expression types, returns the storage type,
     * which is the type of the underlying input data.
     * This is the bottom of the expression chain.
     */
    const type& storage_type() const {
        // Only expr_kind types have different storage_type
        if (is_builtin() || m_extended->get_kind() != expr_kind) {
            return *this;
        } else {
            // Follow the operand type chain to get the storage type
            const type* dt = &static_cast<const base_expr_type *>(m_extended)->get_operand_type();
            while (dt->get_kind() == expr_kind) {
                dt = &static_cast<const base_expr_type *>(dt->m_extended)->get_operand_type();
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

    /**
     * For when it is known that the type is a builtin type,
     * to simply retrieve that type id.
     *
     * WARNING: Normally just use get_type_id().
     */
    type_id_t unchecked_get_builtin_type_id() const {
        return static_cast<type_id_t>(reinterpret_cast<intptr_t>(m_extended));
    }

    /** The 'kind' of the type (int, uint, float, etc) */
    type_kind_t get_kind() const {
        return get_base_type_kind(m_extended);
    }

    /** The alignment of the type */
    size_t get_data_alignment() const {
        return get_base_type_alignment(m_extended);
    }

    /** The element size of the type */
    size_t get_data_size() const {
        return get_base_type_data_size(m_extended);
    }

    /** The element size of the type when default-constructed */
    size_t get_default_data_size() const
    {
      if (is_builtin_type(m_extended)) {
        return static_cast<intptr_t>(
            detail::builtin_data_sizes[reinterpret_cast<uintptr_t>(
                m_extended)]);
      } else {
        return m_extended->get_default_data_size();
      }
    }

    size_t get_arrmeta_size() const {
        if (is_builtin()) {
            return 0;
        } else {
            return m_extended->get_arrmeta_size();
        }
    }

    /**
     * Returns true if the data layout (both data and arrmeta)
     * is compatible with that of 'rhs'. If this returns true,
     * the types can be substituted for each other in an nd::array.
     */
    bool data_layout_compatible_with(const type& rhs) const;

    
    /**
     * Returns true if the given type is a subarray of this type.
     * For example, "int" is a subarray of "strided, int". This
     * relationship may exist for unequal types with the same number
     * of dimensions, for example "int" is a subarray of "pointer(int)".
     *
     * \param subarray_tp  Testing if it is a subarray of 'this'.
     */
    bool is_type_subarray(const ndt::type& subarray_tp) const
    {
        if (is_builtin()) {
            return *this == subarray_tp;
        } else {
            return m_extended->is_type_subarray(subarray_tp);
        }
    }

    /**
     * Returns true if the type represents a chunk of
     * consecutive memory of raw data.
     */
    bool is_pod() const {
        if (is_builtin()) {
            return true;
        } else {
            return m_extended->get_data_size() > 0 &&
                            (m_extended->get_flags() & (type_flag_blockref|
                                            type_flag_destructor)) == 0;
        }
    }

    bool is_scalar() const {
        return is_builtin() || m_extended->is_scalar();
    }

    /**
     * Returns true if the type contains any expression
     * type within it somewhere.
     */
    bool is_expression() const {
        if (is_builtin()) {
            return false;
        } else {
            return m_extended->is_expression();
        }
    }

    /**
     * Returns true if the type constains a symbolic construct
     * like a type var.
     */
    bool is_symbolic() const
    {
      return !is_builtin() &&
             (m_extended->get_flags() & type_flag_symbolic) != 0;
    }

    /**
     * Returns true if the type constains a symbolic dimension
     * which matches a variadic number of dimensions.
     */
    bool is_dim_variadic() const
    {
      return !is_builtin() &&
             (m_extended->get_flags() & type_flag_dim_variadic) != 0;
    }

    /**
     * For array types, recursively applies to each child type, and for
     * scalar types converts to the provided one.
     *
     * \param scalar_type  The scalar type to convert all scalars to.
     */
    type with_replaced_scalar_types(const type &scalar_type) const;

    /**
     * Replaces the data type of the this type with the provided one.
     *
     * \param replacement_tp  The type to substitute for the existing one.
     * \param replace_ndim  The number of array dimensions to include in
     *                      the data type which is replaced.
     */
    type with_replaced_dtype(const type &replacement_tp,
                             intptr_t replace_ndim = 0) const;

    /**
     * Returns this type without the leading memory type, if there is one.
     */
    type without_memory_type() const;

    /**
     * Returns this type with a new strided dimension.
     *
     * \param i  The axis of the new strided dimension.
     */
    type with_new_axis(intptr_t i, intptr_t new_ndim = 1) const;

    /**
     * Returns a modified type with all expression types replaced with
     * their value types, and types replaced with "standard versions"
     * whereever appropriate. For example, an offset-based uniform array
     * would be replaced by a strided uniform array.
     */
    type get_canonical_type() const {
        if (is_builtin()) {
            return *this;
        } else {
            return m_extended->get_canonical_type();
        }
    }

    base_type::flags_type get_flags() const
    {
      if (is_builtin()) {
        return type_flag_scalar;
      } else {
        return m_extended->get_flags();
      }
    }

    /**
     * Gets the number of array dimensions in the type.
     */
    intptr_t get_ndim() const
    {
      if (is_builtin()) {
        return 0;
      } else {
        return m_extended->get_ndim();
      }
    }

    /**
     * Gets the number of outer strided dimensions this type has in a row.
     * The initial arrmeta for this type begins with this many
     * strided_dim_type_arrmeta instances.
     */
    intptr_t get_strided_ndim() const
    {
      if (is_builtin()) {
        return 0;
      } else {
        return m_extended->get_strided_ndim();
      }
    }

    /**
     * Gets the type with array dimensions stripped away.
     *
     * \param include_ndim  The number of array dimensions to keep.
     * \param inout_arrmeta  If non-NULL, is a pointer to arrmeta to advance
     *                       in place.
     */
    type get_dtype(size_t include_ndim = 0,
                          char **inout_arrmeta = NULL) const
    {
      size_t ndim = get_ndim();
      if (ndim == include_ndim) {
        return *this;
      } else if (ndim > include_ndim) {
        return m_extended->get_type_at_dimension(inout_arrmeta,
                                                 ndim - include_ndim);
      } else {
        std::stringstream ss;
        ss << "Cannot use " << include_ndim << " array ";
        ss << "dimensions from dynd type " << *this;
        ss << ", it only has " << ndim;
        throw dynd::type_error(ss.str());
      }
    }

    type get_dtype(size_t include_ndim, const char **inout_arrmeta) const
    {
      return get_dtype(include_ndim, const_cast<char **>(inout_arrmeta));
    }

    intptr_t get_dim_size(const char *arrmeta, const char *data) const;

    type get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim = 0) const {
        if (!is_builtin()) {
            return m_extended->get_type_at_dimension(inout_arrmeta, i, total_ndim);
        } else if (i == 0) {
            return *this;
        } else {
            throw too_many_indices(*this, total_ndim + i, total_ndim);
        }
    }

    /**
     * Returns a const pointer to the base_type object which
     * contains information about the type, or NULL if no extended
     * type information exists. The returned pointer is only valid during
     * the lifetime of the type.
     */
    const base_type* extended() const {
        return m_extended;
    }

    /**
     * Casts to the specified <x>_type class using static_cast.
     * This does not validate the type id to make sure this is
     * a valid cast, the caller MUST check this itself.
     */
    template <class T>
    const T *extended() const {
        // TODO: In debug mode, assert the type id
        return static_cast<const T *>(m_extended);
    }

    /**
     * If the type is a strided dimension type, where the dimension has a fixed size
     * and the data is at addresses `dst`, `dst + stride`, etc, this extracts those
     * values and returns true.
     *
     * \param arrmeta  The arrmeta for the type.
     * \param out_el_tp  Is filled with the element type.
     * \param out_el_arrmeta  Is filled with the arrmeta of the element type.
     *
     * \returns  True if it is a strided array type, false otherwise.
     */
    bool get_as_strided(const char *arrmeta, intptr_t *out_dim_size,
                        intptr_t *out_stride, ndt::type *out_el_tp,
                        const char **out_el_arrmeta) const;

    /**
     * If the type is a multidimensional strided dimension type, where the
     * dimension has a fixed size and the data is at addresses `dst`, `dst +
     * stride`, etc, this extracts those values and returns true.
     *
     * \param arrmeta  The arrmeta for the type.
     * \param ndim  The number of strided dimensions desired.
     * \param out_size_stride  Is filled with a pointer to an array of
     *                         size_stride_t of length ``ndim``.
     * \param out_el_tp  Is filled with the element type.
     * \param out_el_arrmeta  Is filled with the arrmeta of the element type.
     *
     * \returns  True if it is a strided array type, false otherwise.
     */
    bool get_as_strided(const char *arrmeta, intptr_t ndim,
                        const size_stride_t **out_size_stride, ndt::type *out_el_tp,
                        const char **out_el_arrmeta) const;

    /** The size of the data required for uniform iteration */
    size_t get_iterdata_size(intptr_t ndim) const {
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
     * \param inout_arrmeta  The arrmeta corresponding to the type for the iterdata construction.
     *                        This is modified in place to become the arrmeta for the array data type.
     * \param ndim      Number of iteration dimensions.
     * \param shape     The iteration shape.
     * \param out_uniform_type  This is populated with the type of each iterated element
     */
    void iterdata_construct(iterdata_common *iterdata, const char **inout_arrmeta,
                    intptr_t ndim, const intptr_t* shape, type& out_uniform_type) const
    {
        if (!is_builtin()) {
            m_extended->iterdata_construct(iterdata, inout_arrmeta, ndim, shape, out_uniform_type);
        }
    }

    /** Destructs any references or other state contained in the iterdata */
    void iterdata_destruct(iterdata_common *iterdata, intptr_t ndim) const
    {
        if (!is_builtin()) {
            m_extended->iterdata_destruct(iterdata, ndim);
        }
    }

    size_t get_broadcasted_iterdata_size(intptr_t ndim) const {
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
     * \param inout_arrmeta  The arrmeta corresponding to the type for the iterdata construction.
     *                        This is modified in place to become the arrmeta for the array data type.
     * \param ndim      Number of iteration dimensions.
     * \param shape     The iteration shape.
     * \param out_uniform_tp  This is populated with the type of each iterated element
     */
    void broadcasted_iterdata_construct(iterdata_common *iterdata, const char **inout_arrmeta,
                    intptr_t ndim, const intptr_t* shape, type& out_uniform_tp) const
    {
        size_t size;
        if (is_builtin()) {
            size = 0;
        } else {
            size = m_extended->iterdata_construct(iterdata, inout_arrmeta, ndim, shape, out_uniform_tp);
        }
        iterdata_broadcasting_terminator *id = reinterpret_cast<iterdata_broadcasting_terminator *>(
                        reinterpret_cast<char *>(iterdata) + size);
        id->common.incr = &iterdata_broadcasting_terminator_incr;
        id->common.adv = &iterdata_broadcasting_terminator_adv;
        id->common.reset = &iterdata_broadcasting_terminator_reset;
    }

    /**
     * print data interpreted as a single value of this type
     *
     * \param o         the std::ostream to print to
     * \param data      pointer to the data element to print
     * \param arrmeta  pointer to the nd::array arrmeta for the data element
     */
    void print_data(std::ostream& o, const char *arrmeta, const char *data) const;

    std::string str() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

    friend std::ostream& operator<<(std::ostream& o, const type& rhs);
};

// Forward declarations
type make_pointer(const type &target_tp);
type make_cfixed_dim(size_t size, const type &element_tp);

namespace detail {
  template <typename T>
  struct exact_type_from;

  template <typename T>
  struct exact_type_from {
    static type make()
    {
      return type(static_cast<type_id_t>(type_id_of<T>::value));
    }
  };

  template <>
  struct exact_type_from<ndt::type> {
    static type make()
    {
      return ndt::type("type");
    }
  };

  template <typename T>
  struct exact_type_from<T *> {
    static type make() { return make_pointer(exact_type_from<T>::make()); }
  };

  template <typename T, int N>
  struct exact_type_from<T[N]> {
    static type make()
    {
      return make_cfixed_dim(N, exact_type_from<T>::make());
    }
  };

  // The default implementation of type_from is exact_type_from, after
  // which we add additional overloads
  template <typename T>
  struct type_from : public exact_type_from<T> {
  };

  template <>
  struct type_from<bool> {
    static type make() { return type(bool_type_id); }
  };

  template <typename T, int N>
  struct type_from<nd::strided_vals<T, N> > {
    static type make() { return make_fixed_dimsym(type_from<T>::make(), N); }
  };

  template <typename T, int N>
  struct type_from<T[N]> {
    static type make()
    {
      return make_cfixed_dim(N, type_from<T>::make());
    }
  };

} // namespace detail

/**
 * Convenience function which makes an ndt::type
 * object from a template parameter. This includes
 * convenience cases, where the memory layout of the given
 * type may not precisely match that of T.
 */
template<class T>
type make_type()
{
  return detail::type_from<T>::make();
}

template<class T>
type make_exact_type()
{
  return detail::exact_type_from<T>::make();
}

/**
 * Constructs an array type from a shape and
 * a data type. Each dimension >= 0 is made
 * using a fixed_dim type, and each dimension == -1
 * is made using a var_dim type.
 *
 * \param ndim   The number of dimensions in the shape
 * \param shape  The shape of the array type to create.
 * \param dtype  The data type of each array element.
 */
type make_type(intptr_t ndim, const intptr_t *shape, const ndt::type &dtype);

/**
 * Constructs an array type from a shape and
 * a data type specified as a string. Each dimension >= 0 is made
 * using a fixed_dim type, and each dimension == -1
 * is made using a var_dim type.
 *
 * \param ndim   The number of dimensions in the shape
 * \param shape  The shape of the array type to create.
 * \param dtype  The data type of each array element.
 */
template <int N>
inline type make_type(intptr_t ndim, const intptr_t *shape,
                      const char (&dtype)[N])
{
  return make_type(ndim, shape, ndt::type(dtype));
}

/**
 * Constructs an array type from a shape and
 * a data type. Each dimension >= 0 is made
 * using a fixed_dim type, and each dimension == -1
 * is made using a var_dim type.
 *
 * \param ndim   The number of dimensions in the shape
 * \param shape  The shape of the array type to create.
 * \param dtype  The data type of each array element.
 * \param out_any_var  This output variable is set to true if any var
 *                     dimension is in the shape. If no var dimension
 *                     is encountered, it is untouched, so the caller
 *                     should initialize it to false.
 */
type make_type(intptr_t ndim, const intptr_t *shape, const ndt::type &dtype,
               bool &out_any_var);

/**
 * Returns the type of an array constructed from a value.
 */
template <typename T>
type as_type(const T &DYND_UNUSED(value))
{
  return make_type<T>();
}

template <typename T>
type as_type(const std::vector<T> &value)
{
  return make_fixed_dim(value.size(), make_type<T>());
}

type as_type(const nd::array &val);

type as_type(const nd::arrfunc &val);

template <typename T>
void get_types(type &tp, const T &val)
{
  tp = ndt::as_type(val);
}

template <typename T, typename... A>
void get_types(type &tp, const T &val, A &&... a)
{
  get_types(tp, val);
  get_types(std::forward<A>(a)...);
}

#if !(defined(_MSC_VER) && _MSC_VER == 1800)
template <size_t... I>
template <typename... A>
void index_proxy<index_sequence<I...>>::get_types(A &&... a)
{
  ndt::get_types(get<I>(std::forward<A>(a)...)...);
}
#else
// Workaround for MSVC 2013 compiler bug reported here:
// https://connect.microsoft.com/VisualStudio/feedback/details/1045260/unpacking-std-forward-a-a-fails-when-nested-with-another-unpacking
template <size_t... I>
void index_proxy<index_sequence<I...>>::get_types()
{
  ndt::get_types(get<I>()...);
}
template <size_t... I>
template <typename A0>
void index_proxy<index_sequence<I...>>::get_types(A0 &&a0)
{
  ndt::get_types(get<I>(std::forward<A0>(a0))...);
}
template <size_t... I>
template <typename A0, typename A1>
void index_proxy<index_sequence<I...>>::get_types(A0 &&a0, A1 &&a1)
{
  ndt::get_types(get<I>(std::forward<A0>(a0), std::forward<A1>(a1))...);
}
template <size_t... I>
template <typename A0, typename A1, typename A2>
void index_proxy<index_sequence<I...>>::get_types(A0 &&a0, A1 &&a1, A2 &&a2)
{
  ndt::get_types(get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                        std::forward<A2>(a2))...);
}
template <size_t... I>
template <typename A0, typename A1, typename A2, typename A3>
void index_proxy<index_sequence<I...>>::get_types(A0 &&a0, A1 &&a1, A2 &&a2,
                                                  A3 &&a3)
{
  ndt::get_types(get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                        std::forward<A2>(a2), std::forward<A3>(a3))...);
}
template <size_t... I>
template <typename A0, typename A1, typename A2, typename A3, typename A4>
void index_proxy<index_sequence<I...>>::get_types(A0 &&a0, A1 &&a1, A2 &&a2,
                                                  A3 &&a3, A4 &&a4)
{
  ndt::get_types(get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                        std::forward<A2>(a2), std::forward<A3>(a3),
                        std::forward<A4>(a4))...);
}
template <size_t... I>
template <typename A0, typename A1, typename A2, typename A3, typename A4,
          typename A5>
void index_proxy<index_sequence<I...>>::get_types(A0 &&a0, A1 &&a1, A2 &&a2,
                                                  A3 &&a3, A4 &&a4, A5 &&a5)
{
  ndt::get_types(get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                        std::forward<A2>(a2), std::forward<A3>(a3),
                        std::forward<A4>(a4), std::forward<A5>(a5))...);
}
#endif

template <size_t... I>
template <typename... T>
void index_proxy<index_sequence<I...>>::get_types(type *tp,
                                                  const std::tuple<T...> &vals,
                                                  const intptr_t *perm)
{
  typedef make_index_sequence<size, 2 * size> J;

  if (perm == NULL) {
    index_proxy<typename alternate<index_sequence<I...>, J>::type>::
        template get_types(tp[I]..., std::get<I>(vals)...);
  } else {
    index_proxy<typename alternate<index_sequence<I...>, J>::type>::
        template get_types(tp[perm[I]]..., std::get<I>(vals)...);
  }
}

/**
 * Returns the type to use for packing this specific value. The value
 * is allowed to affect the type, e.g. for packing a std::vector
 */
template <typename T>
type get_forward_type(const T &DYND_UNUSED(val))
{
  // Default case is for when T and the ndt::type have identical
  // memory layout, which is guaranteed by make_exact_type<T>().
  return make_exact_type<T>();
}

template <typename T>
type get_forward_type(const std::vector<T> &val)
{
  // Depending on the data size, store the data by value or as a pointer
  // to an nd::array
  if (sizeof(T) * val.size() > 32) {
    return make_pointer(make_fixed_dim(val.size(), make_exact_type<T>()));
  } else {
    return make_fixed_dim(val.size(), make_exact_type<T>());
  }
}

type get_forward_type(const nd::array &val);

type get_forward_type(const nd::arrfunc &val);

template <typename T>
void get_forward_types(type &tp, const T &val)
{
    tp = get_forward_type(val);
}

template <typename T, typename... A>
void get_forward_types(type &tp, const T &val, A &&... a)
{
    get_forward_types(tp, val);
    get_forward_types(std::forward<A>(a)...);
}

#if !(defined(_MSC_VER) && _MSC_VER == 1800)
template <size_t... I>
template <typename... A>
void index_proxy<index_sequence<I...>>::get_forward_types(A &&... a)
{
  ndt::get_forward_types(get<I>(std::forward<A>(a)...)...);
}
#else
// Workaround for MSVC 2013 compiler bug reported here:
// https://connect.microsoft.com/VisualStudio/feedback/details/1045260/unpacking-std-forward-a-a-fails-when-nested-with-another-unpacking
template <size_t... I>
void index_proxy<index_sequence<I...>>::get_forward_types()
{
  ndt::get_forward_types(get<I>()...);
}
template <size_t... I>
template <typename A0>
void index_proxy<index_sequence<I...>>::get_forward_types(A0 &&a0)
{
  ndt::get_forward_types(get<I>(std::forward<A0>(a0))...);
}
template <size_t... I>
template <typename A0, typename A1>
void index_proxy<index_sequence<I...>>::get_forward_types(A0 &&a0, A1 &&a1)
{
  ndt::get_forward_types(get<I>(std::forward<A0>(a0), std::forward<A1>(a1))...);
}
template <size_t... I>
template <typename A0, typename A1, typename A2>
void index_proxy<index_sequence<I...>>::get_forward_types(A0 &&a0, A1 &&a1, A2 &&a2)
{
  ndt::get_forward_types(get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                        std::forward<A2>(a2))...);
}
template <size_t... I>
template <typename A0, typename A1, typename A2, typename A3>
void index_proxy<index_sequence<I...>>::get_forward_types(A0 &&a0, A1 &&a1, A2 &&a2,
                                                  A3 &&a3)
{
  ndt::get_forward_types(get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                        std::forward<A2>(a2), std::forward<A3>(a3))...);
}
template <size_t... I>
template <typename A0, typename A1, typename A2, typename A3, typename A4>
void index_proxy<index_sequence<I...>>::get_forward_types(A0 &&a0, A1 &&a1, A2 &&a2,
                                                  A3 &&a3, A4 &&a4)
{
  ndt::get_forward_types(get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                        std::forward<A2>(a2), std::forward<A3>(a3),
                        std::forward<A4>(a4))...);
}
template <size_t... I>
template <typename A0, typename A1, typename A2, typename A3, typename A4,
          typename A5>
void index_proxy<index_sequence<I...>>::get_forward_types(A0 &&a0, A1 &&a1, A2 &&a2,
                                                 A3 &&a3, A4 &&a4, A5 &&a5)
{
  ndt::get_forward_types(get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                        std::forward<A2>(a2), std::forward<A3>(a3),
                        std::forward<A4>(a4), std::forward<A5>(a5))...);
}
#endif

template <size_t... I>
template <typename... T>
void index_proxy<index_sequence<I...>>::get_forward_types(
    type *tp, const std::tuple<T...> &vals, const intptr_t *perm)
{
  typedef make_index_sequence<size, 2 * size> J;

  if (perm == NULL) {
    index_proxy<typename alternate<index_sequence<I...>, J>::type>::
        template get_forward_types(tp[I]..., std::get<I>(vals)...);
  } else {
    index_proxy<typename alternate<index_sequence<I...>, J>::type>::
        template get_forward_types(tp[perm[I]]..., std::get<I>(vals)...);
  }
}

/**
 * A static array of the builtin types and void.
 * If code is specialized just for a builtin type, like int, it can use
 * static_builtin_types[type_id_of<int>::value] as a fast
 * way to get a const reference to its type.
 */
extern const type static_builtin_types[builtin_type_id_count];

std::ostream& operator<<(std::ostream& o, const type& rhs);

} // namespace ndt

/** Prints raw bytes as hexadecimal */
void hexadecimal_print(std::ostream& o, char value);
void hexadecimal_print(std::ostream& o, unsigned char value);
void hexadecimal_print(std::ostream& o, unsigned short value);
void hexadecimal_print(std::ostream& o, unsigned int value);
void hexadecimal_print(std::ostream& o, unsigned long value);
void hexadecimal_print(std::ostream& o, unsigned long long value);
void hexadecimal_print(std::ostream& o, const char *data, intptr_t element_size);
void hexadecimal_print_summarized(std::ostream &o, const char *data,
                                  intptr_t element_size, intptr_t summary_size);

void strided_array_summarized(std::ostream &o, const ndt::type &tp,
                              const char *arrmeta, const char *data,
                              intptr_t dim_size, intptr_t stride);
void print_indented(std::ostream &o, const std::string &indent,
                    const std::string &s, bool skipfirstline = false);

} // namespace dynd
