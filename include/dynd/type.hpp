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
#include <dynd/eval/eval_context.hpp>
#include <dynd/exceptions.hpp>

namespace dynd {
namespace detail {

  template <typename ValueType, int NDim>
  class scalar_wrapper_iterator;

  template <typename ValueType>
  class scalar_wrapper {
  protected:
    const char *m_metadata;
    char *m_data;

  public:
    typedef ValueType data_type;
    static const intptr_t ndim = 0;

    template <int NDim>
    class iterator_type : public scalar_wrapper_iterator<ValueType, NDim> {
    public:
      iterator_type(const char *metadata, char *data) : scalar_wrapper_iterator<ValueType, NDim>(metadata, data)
      {
      }
    };

    scalar_wrapper(const char *metadata, char *data) : m_metadata(metadata), m_data(data)
    {
    }

    data_type &operator()(const char *DYND_UNUSED(metadata), char *data)
    {
      return *reinterpret_cast<data_type *>(data);
    }
  };

  template <typename ValueType>
  class scalar_wrapper_iterator<ValueType, 0> {
  protected:
    char *m_data;

  public:
    scalar_wrapper_iterator(const char *DYND_UNUSED(metadata), char *data) : m_data(data)
    {
    }

    ValueType &operator*()
    {
      return *reinterpret_cast<ValueType *>(m_data);
    }

    bool operator==(const scalar_wrapper_iterator &rhs) const
    {
      return m_data == rhs.m_data;
    }

    bool operator!=(const scalar_wrapper_iterator &rhs) const
    {
      return m_data != rhs.m_data;
    }
  };

} // namespace dynd::detail

template <typename T>
using identity_t = T;

template <typename T>
using as_t = typename conditional_make<!std::is_fundamental<typename std::remove_cv<T>::type>::value &&
                                           !std::is_same<typename std::remove_cv<T>::type, ndt::type>::value,
                                       identity_t, detail::scalar_wrapper, T>::type;

/**
 * Increments the offset value so that it is aligned to the requested alignment
 * NOTE: The alignment must be a power of two.
 */
inline size_t inc_to_alignment(size_t offset, size_t alignment)
{
  return (offset + alignment - 1) & (std::size_t)(-(std::ptrdiff_t)alignment);
}

/**
 * Increments the pointer value so that it is aligned to the requested alignment
 * NOTE: The alignment must be a power of two.
 */
inline char *inc_to_alignment(char *ptr, size_t alignment)
{
  return reinterpret_cast<char *>((reinterpret_cast<std::size_t>(ptr) + alignment - 1) &
                                  (std::size_t)(-(std::ptrdiff_t)alignment));
}

/**
 * Increments the pointer value so that it is aligned to the requested alignment
 * NOTE: The alignment must be a power of two.
 */
inline void *inc_to_alignment(void *ptr, size_t alignment)
{
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
inline bool offset_is_aligned(size_t offset, size_t alignment)
{
  return (offset & (alignment - 1)) == 0;
}

/** Prints a single scalar of a builtin type to the stream */
void DYND_API print_builtin_scalar(type_id_t type_id, std::ostream &o, const char *data);

/** Special iterdata which broadcasts to any number of additional dimensions */
struct DYND_API iterdata_broadcasting_terminator {
  iterdata_common common;
  char *data;
};
DYND_API char *iterdata_broadcasting_terminator_incr(iterdata_common *iterdata, intptr_t level);
DYND_API char *iterdata_broadcasting_terminator_adv(iterdata_common *iterdata, intptr_t level, intptr_t i);
DYND_API char *iterdata_broadcasting_terminator_reset(iterdata_common *iterdata, char *data, intptr_t level);

// Forward declaration of nd::array and nd::strided_vals
namespace nd {
  class DYND_API array;
} // namespace dynd::nd

namespace ndt {
  typedef type (*type_make_t)(type_id_t tp_id, const nd::array &args);

  DYND_API type make_fixed_dim(size_t dim_size, const type &element_tp);

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
  class DYND_API type : public intrusive_ptr<const base_type> {
    static type instances[DYND_TYPE_ID_MAX + 1];

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
      }

      return NULL;
      //      throw invalid_type_id((int)type_id);
    }

  public:
    using intrusive_ptr<const base_type>::intrusive_ptr;

    /**
      * Default constructor.
      */
    type() = default;

    /** Construct from a type ID */
    type(type_id_t tp_id) : type((validate_type_id(tp_id), instances[tp_id]))
    {
    }

    /** Construct from a string representation */
    explicit type(const std::string &rep);

    /** Construct from a string representation */
    type(const char *rep_begin, const char *rep_end);

    bool operator==(const type &rhs) const
    {
      return m_ptr == rhs.m_ptr || (!is_builtin() && !rhs.is_builtin() && *m_ptr == *rhs.m_ptr);
    }

    bool operator!=(const type &rhs) const
    {
      return !(operator==(rhs));
    }

    bool is_null() const
    {
      return m_ptr == NULL;
    }

    /**
     * Returns true if this type is built in, which
     * means the type id is encoded directly in the m_ptr
     * pointer.
     */
    bool is_builtin() const
    {
      return is_builtin_type(m_ptr);
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
     * The 'at_single' function is used for indexing by a single dimension,
     *without
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
     *                    to point to the result's data. If `inout_data` is
     *non-NULL,
     *                    `inout_arrmeta` must also be non-NULL.
     *
     * \returns  The type that results from the indexing operation.
     */
    type at_single(intptr_t i0, const char **inout_arrmeta = NULL, const char **inout_data = NULL) const
    {
      if (!is_builtin()) {
        return m_ptr->at_single(i0, inout_arrmeta, inout_data);
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
     *       If you do not want this collapsing behavior, use the 'at_single'
     *function.
     */
    type at(const irange &i0) const
    {
      return at_array(1, &i0);
    }

    /** Indexing with two index values */
    type at(const irange &i0, const irange &i1) const
    {
      irange i[2] = {i0, i1};
      return at_array(2, i);
    }

    /** Indexing with three index values */
    type at(const irange &i0, const irange &i1, const irange &i2) const
    {
      irange i[3] = {i0, i1, i2};
      return at_array(3, i);
    }
    /** Indexing with four index values */
    type at(const irange &i0, const irange &i1, const irange &i2, const irange &i3) const
    {
      irange i[4] = {i0, i1, i2, i3};
      return at_array(4, i);
    }

    /**
     * Matches the provided candidate type against the current type. The
     * 'this' type is the pattern to match against, and may be symbolic
     * or concrete. If it is concrete, the candidate type must be equal
     * for the match to succeed.
     *
     * The candidate type may also be symbolic.
     *
     * Returns true if it matches, false otherwise.
     *
     * This function may be called multiple times in a row, building up the
     * typevars dictionary which is used to enforce consistent usage of
     * type vars.
     *
     * \param arrmeta     The arrmeta for this type, maybe NULL.
     * \param candidate_tp    A type to match against this one.
     * \param candidate_arrmeta   The arrmeta for the candidate type,
     *                            may be NULL.
     * \param tp_vars     A map of names to matched type vars.
     */
    bool match(const char *arrmeta, const ndt::type &candidate_tp, const char *candidate_arrmeta,
               std::map<std::string, ndt::type> &tp_vars) const;

    bool match(const char *arrmeta, const ndt::type &candidate_tp, const char *candidate_arrmeta) const;

    bool match(const ndt::type &candidate_tp, std::map<std::string, ndt::type> &tp_vars) const;

    bool match(const ndt::type &candidate_tp) const;

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
    nd::array p(const std::string &property_name) const;

    /**
     * Indexes into the type, intended for recursive calls from the
     * extended-type version. See
     * the function in base_type with the same name for more details.
     */
    type apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i, const type &root_tp,
                            bool leading_dimension) const;

    /**
     * Returns the non-expression type that this
     * type looks like for the purposes of calculation,
     * printing, etc.
     */
    const type &value_type() const
    {
      // Only expr_kind types have different value_type
      if (is_builtin() || m_ptr->get_kind() != expr_kind) {
        return *this;
      } else {
        // All chaining happens in the operand_type
        return static_cast<const base_expr_type *>(m_ptr)->get_value_type();
      }
    }

    /**
     * For expression types, returns the operand type,
     * which is the source type of this type's expression.
     * This is one link down the expression chain.
     */
    const type &operand_type() const
    {
      // Only expr_kind types have different operand_type
      if (is_builtin() || m_ptr->get_kind() != expr_kind) {
        return *this;
      } else {
        return static_cast<const base_expr_type *>(m_ptr)->get_operand_type();
      }
    }

    /**
     * For expression types, returns the storage type,
     * which is the type of the underlying input data.
     * This is the bottom of the expression chain.
     */
    const type &storage_type() const
    {
      // Only expr_kind types have different storage_type
      if (is_builtin() || m_ptr->get_kind() != expr_kind) {
        return *this;
      } else {
        // Follow the operand type chain to get the storage type
        const type *dt = &static_cast<const base_expr_type *>(m_ptr)->get_operand_type();
        while (dt->get_kind() == expr_kind) {
          dt = &static_cast<const base_expr_type *>(dt->m_ptr)->get_operand_type();
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
    type_id_t get_type_id() const
    {
      if (is_builtin()) {
        return static_cast<type_id_t>(reinterpret_cast<intptr_t>(m_ptr));
      } else {
        return m_ptr->get_type_id();
      }
    }

    /**
     * For when it is known that the type is a builtin type,
     * to simply retrieve that type id.
     *
     * WARNING: Normally just use get_type_id().
     */
    type_id_t unchecked_get_builtin_type_id() const
    {
      return static_cast<type_id_t>(reinterpret_cast<intptr_t>(m_ptr));
    }

    /** The 'kind' of the type (int, uint, float, etc) */
    type_kind_t get_kind() const
    {
      return get_base_type_kind(m_ptr);
    }

    /** The alignment of the type */
    size_t get_data_alignment() const
    {
      return get_base_type_alignment(m_ptr);
    }

    /** The element size of the type */
    size_t get_data_size() const
    {
      return get_base_type_data_size(m_ptr);
    }

    /** The element size of the type when default-constructed */
    size_t get_default_data_size() const
    {
      if (is_builtin_type(m_ptr)) {
        return static_cast<intptr_t>(detail::builtin_data_sizes[reinterpret_cast<uintptr_t>(m_ptr)]);
      } else {
        return m_ptr->get_default_data_size();
      }
    }

    size_t get_arrmeta_size() const
    {
      if (is_builtin()) {
        return 0;
      } else {
        return m_ptr->get_arrmeta_size();
      }
    }

    /**
     * Returns true if the data layout (both data and arrmeta)
     * is compatible with that of 'rhs'. If this returns true,
     * the types can be substituted for each other in an nd::array.
     */
    bool data_layout_compatible_with(const type &rhs) const;

    /**
     * Returns true if the given type is a subarray of this type.
     * For example, "int" is a subarray of "strided, int". This
     * relationship may exist for unequal types with the same number
     * of dimensions, for example "int" is a subarray of "pointer(int)".
     *
     * \param subarray_tp  Testing if it is a subarray of 'this'.
     */
    bool is_type_subarray(const ndt::type &subarray_tp) const
    {
      if (is_builtin()) {
        return *this == subarray_tp;
      } else {
        return m_ptr->is_type_subarray(subarray_tp);
      }
    }

    /**
     * Returns true if the type represents a chunk of
     * consecutive memory of raw data.
     */
    bool is_pod() const
    {
      if (is_builtin()) {
        return true;
      } else {
        return m_ptr->get_data_size() > 0 && (m_ptr->get_flags() & (type_flag_blockref | type_flag_destructor)) == 0;
      }
    }

    bool is_c_contiguous(const char *arrmeta) const
    {
      if (is_builtin()) {
        return true;
      }

      return m_ptr->is_c_contiguous(arrmeta);
    }

    bool is_indexable() const
    {
      return !is_builtin() && m_ptr->is_indexable();
    }

    bool is_scalar() const
    {
      return is_builtin() || m_ptr->is_scalar();
    }

#ifdef DYND_CUDA

    bool is_cuda_device_readable() const
    {
      if (is_builtin()) {
        return get_kind() == void_kind;
      }

      return m_ptr->get_type_id() == cuda_device_type_id;
    }

#endif

    /**
     * Returns true if the type contains any expression
     * type within it somewhere.
     */
    bool is_expression() const
    {
      if (is_builtin()) {
        return false;
      } else {
        return m_ptr->is_expression();
      }
    }

    /**
     * Returns true if the type contains a symbolic construct
     * like a type var.
     */
    bool is_symbolic() const
    {
      return !is_builtin() && (m_ptr->get_flags() & type_flag_symbolic);
    }

    /**
     * Returns true if the type constains a symbolic dimension
     * which matches a variadic number of dimensions.
     */
    bool is_variadic() const
    {
      return !is_builtin() && (m_ptr->get_flags() & type_flag_variadic);
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
    type with_replaced_dtype(const type &replacement_tp, intptr_t replace_ndim = 0) const;

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
    type get_canonical_type() const
    {
      if (is_builtin()) {
        return *this;
      } else {
        return m_ptr->get_canonical_type();
      }
    }

    base_type::flags_type get_flags() const
    {
      if (is_builtin()) {
        return type_flag_none;
      } else {
        return m_ptr->get_flags();
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
        return m_ptr->get_ndim();
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
        return m_ptr->get_strided_ndim();
      }
    }

    /**
     * Gets the type with array dimensions stripped away.
     *
     * \param include_ndim  The number of array dimensions to keep.
     * \param inout_arrmeta  If non-NULL, is a pointer to arrmeta to advance
     *                       in place.
     */
    type get_dtype(size_t include_ndim = 0, char **inout_arrmeta = NULL) const
    {
      size_t ndim = get_ndim();
      if (ndim == include_ndim) {
        return *this;
      } else if (ndim > include_ndim) {
        return m_ptr->get_type_at_dimension(inout_arrmeta, ndim - include_ndim);
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

    intptr_t get_size(const char *arrmeta) const;

    type get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim = 0) const
    {
      if (!is_builtin()) {
        return m_ptr->get_type_at_dimension(inout_arrmeta, i, total_ndim);
      } else if (i == 0) {
        return *this;
      } else {
        throw too_many_indices(*this, total_ndim + i, total_ndim);
      }
    }

    void get_vars(std::unordered_set<std::string> &vars) const
    {
      if (!is_builtin()) {
        m_ptr->get_vars(vars);
      }
    }

    std::unordered_set<std::string> get_vars() const
    {
      std::unordered_set<std::string> vars;
      get_vars(vars);

      return vars;
    }

    /**
     * Returns a const pointer to the base_type object which
     * contains information about the type, or NULL if no extended
     * type information exists. The returned pointer is only valid during
     * the lifetime of the type.
     */
    const base_type *extended() const
    {
      return m_ptr;
    }

    /**
     * Casts to the specified <x>_type class using static_cast.
     * This does not validate the type id to make sure this is
     * a valid cast, the caller MUST check this itself.
     */
    template <class T>
    const T *extended() const
    {
      // TODO: In debug mode, assert the type id
      return static_cast<const T *>(m_ptr);
    }

    /**
     * If the type is a strided dimension type, where the dimension has a fixed
     * size and the data is at addresses `dst`, `dst + stride`, etc, this
     * extracts those values and returns true.
     *
     * \param arrmeta  The arrmeta for the type.
     * \param out_el_tp  Is filled with the element type.
     * \param out_el_arrmeta  Is filled with the arrmeta of the element type.
     *
     * \returns  True if it is a strided array type, false otherwise.
     */
    bool get_as_strided(const char *arrmeta, intptr_t *out_dim_size, intptr_t *out_stride, ndt::type *out_el_tp,
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
    bool get_as_strided(const char *arrmeta, intptr_t ndim, const size_stride_t **out_size_stride, ndt::type *out_el_tp,
                        const char **out_el_arrmeta) const;

    /** The size of the data required for uniform iteration */
    size_t get_iterdata_size(intptr_t ndim) const
    {
      if (is_builtin()) {
        return 0;
      } else {
        return m_ptr->get_iterdata_size(ndim);
      }
    }
    /**
     * \brief Constructs the iterdata for processing iteration of the specified
     *        shape.
     *
     * \param iterdata  The allocated iterdata to construct.
     * \param inout_arrmeta  The arrmeta corresponding to the type for the
     *                       iterdata construction. This is modified in place to
     *                       become the arrmeta for the array data type.
     * \param ndim      Number of iteration dimensions.
     * \param shape     The iteration shape.
     * \param out_uniform_type  This is populated with the type of each iterated
     *                          element
     */
    void iterdata_construct(iterdata_common *iterdata, const char **inout_arrmeta, intptr_t ndim, const intptr_t *shape,
                            type &out_uniform_type) const
    {
      if (!is_builtin()) {
        m_ptr->iterdata_construct(iterdata, inout_arrmeta, ndim, shape, out_uniform_type);
      }
    }

    /** Destructs any references or other state contained in the iterdata */
    void iterdata_destruct(iterdata_common *iterdata, intptr_t ndim) const
    {
      if (!is_builtin()) {
        m_ptr->iterdata_destruct(iterdata, ndim);
      }
    }

    size_t get_broadcasted_iterdata_size(intptr_t ndim) const
    {
      if (is_builtin()) {
        return sizeof(iterdata_broadcasting_terminator);
      } else {
        return m_ptr->get_iterdata_size(ndim) + sizeof(iterdata_broadcasting_terminator);
      }
    }

    /**
     * Constructs an iterdata which can be broadcast to the left indefinitely,
     * by capping off the iterdata with a iterdata_broadcasting_terminator.
     *
     * \param iterdata  The allocated iterdata to construct.
     * \param inout_arrmeta  The arrmeta corresponding to the type for the
     *                       iterdata construction. This is modified in place to
     *                       become the arrmeta for the array data type.
     * \param ndim      Number of iteration dimensions.
     * \param shape     The iteration shape.
     * \param out_uniform_tp  This is populated with the type of each iterated
     *                        element
     */
    void broadcasted_iterdata_construct(iterdata_common *iterdata, const char **inout_arrmeta, intptr_t ndim,
                                        const intptr_t *shape, type &out_uniform_tp) const
    {
      size_t size;
      if (is_builtin()) {
        size = 0;
      } else {
        size = m_ptr->iterdata_construct(iterdata, inout_arrmeta, ndim, shape, out_uniform_tp);
      }
      iterdata_broadcasting_terminator *id =
          reinterpret_cast<iterdata_broadcasting_terminator *>(reinterpret_cast<char *>(iterdata) + size);
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
    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    std::string str() const
    {
      std::stringstream ss;
      ss << *this;
      return ss.str();
    }

    template <typename T>
    struct equivalent {
      ~equivalent() = delete;
    };

    template <typename T>
    struct has_equivalent {
      static const bool value = std::is_destructible<equivalent<T>>::value;
    };

    /**
     * ``value'' should be true for an ndt::type object whose memory layout
     * matches that of C++.
     */
    template <typename T>
    struct is_layout_compatible {
      static const bool value = false;
    };

    /**
     * Convenience function which makes an ndt::type
     * object from a template parameter. This includes
     * convenience cases, where the memory layout of the given
     * type may not precisely match that of T.
     */
    template <typename T, typename... A>
    static type make(A &&... a)
    {
      return equivalent<T>::make(std::forward<A>(a)...);
    }

    static type make(type_id_t tp_id, const nd::array &args);

    friend DYND_API std::ostream &operator<<(std::ostream &o, const type &rhs);
  };

  template <>
  struct type::equivalent<bool1> {
    static type make()
    {
      return type(type_id_of<bool1>::value);
    }
  };

  template <>
  struct type::equivalent<bool> {
    static type make()
    {
      return type::make<bool1>();
    }
  };

  template <>
  struct type::equivalent<signed char> {
    static type make()
    {
      return type(type_id_of<signed char>::value);
    }
  };

  template <>
  struct type::equivalent<short> {
    static type make()
    {
      return type(type_id_of<short>::value);
    }
  };

  template <>
  struct type::equivalent<int> {
    static type make()
    {
      return type(type_id_of<int>::value);
    }
  };

  template <>
  struct type::equivalent<long> {
    static type make()
    {
      return type(type_id_of<long>::value);
    }
  };

  template <>
  struct type::equivalent<long long> {
    static type make()
    {
      return type(type_id_of<long long>::value);
    }
  };

  template <>
  struct type::equivalent<int128> {
    static type make()
    {
      return type(type_id_of<int128>::value);
    }
  };

  template <>
  struct type::equivalent<unsigned char> {
    static type make()
    {
      return type(type_id_of<unsigned char>::value);
    }
  };

  template <>
  struct type::equivalent<unsigned short> {
    static type make()
    {
      return type(type_id_of<unsigned short>::value);
    }
  };

  template <>
  struct type::equivalent<unsigned int> {
    static type make()
    {
      return type(type_id_of<unsigned int>::value);
    }
  };

  template <>
  struct type::equivalent<unsigned long> {
    static type make()
    {
      return type(type_id_of<unsigned long>::value);
    }
  };

  template <>
  struct type::equivalent<unsigned long long> {
    static type make()
    {
      return type(type_id_of<unsigned long long>::value);
    }
  };

  template <>
  struct type::equivalent<uint128> {
    static type make()
    {
      return type(type_id_of<uint128>::value);
    }
  };

  template <>
  struct type::equivalent<char> {
    static type make()
    {
      return type(type_id_of<char>::value);
    }
  };

  template <>
  struct type::equivalent<float16> {
    static type make()
    {
      return type(type_id_of<float16>::value);
    }
  };

  template <>
  struct type::equivalent<float> {
    static type make()
    {
      return type(type_id_of<float>::value);
    }
  };

  template <>
  struct type::equivalent<double> {
    static type make()
    {
      return type(type_id_of<double>::value);
    }
  };

  /*
    template <>
    struct type::equivalent<long double> {
      static type make() { return type(type_id_of<long double>::value); }
    };
  */

  template <>
  struct type::equivalent<float128> {
    static type make()
    {
      return type(type_id_of<float128>::value);
    }
  };

  template <typename T>
  struct type::equivalent<complex<T>> {
    static type make()
    {
      return type(type_id_of<complex<T>>::value);
    }
  };

  template <typename T>
  struct type::equivalent<std::complex<T>> {
    static type make()
    {
      return type::make<complex<T>>();
    }
  };

  template <>
  struct type::equivalent<void> {
    static type make()
    {
      return type(type_id_of<void>::value);
    }
  };

  template <>
  struct type::equivalent<type> {
    static type make()
    {
      return type(type_type_id);
    }
  };

  // The removal of const is a temporary solution until we decide if and how
  // types should support const
  template <typename T>
  struct type::equivalent<const T> {
    template <typename... A>
    static type make(A &&... a)
    {
      return type::make<T>(std::forward<A>(a)...);
    }
  };

  // Same as for const
  template <typename T>
  struct type::equivalent<T &> {
    template <typename... A>
    static type make(A &&... a)
    {
      return type::make<T>(std::forward<A>(a)...);
    }
  };

  // Same as for const
  template <typename T>
  struct type::equivalent<T &&> {
    template <typename... A>
    static type make(A &&... a)
    {
      return type::make<T>(std::forward<A>(a)...);
    }
  };

  /**
    * Returns the common type of two types. For built-in types, this is analogous to
    * std::common_type.
    */
  DYND_API extern class common_type {
    typedef type (*child_type)(const type &, const type &);

    struct init;

    static child_type children[DYND_TYPE_ID_MAX][DYND_TYPE_ID_MAX];

  public:
    common_type();

    DYND_API ndt::type operator()(const ndt::type &tp0, const ndt::type &tp1) const;
  } common_type;

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
  DYND_API type make_type(intptr_t ndim, const intptr_t *shape, const ndt::type &dtype);

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
  inline type make_type(intptr_t ndim, const intptr_t *shape, const char (&dtype)[N])
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
  DYND_API type make_type(intptr_t ndim, const intptr_t *shape, const ndt::type &dtype, bool &out_any_var);

  DYND_API type_id_t register_type(const std::string &name, type_make_t make);

  template <typename TypeType>
  type_id_t register_type(const std::string &name)
  {
    return register_type(name,
                         [](type_id_t tp_id, const nd::array &args) { return type(new TypeType(tp_id, args), false); });
  }

  /**
   * Returns the type of an array constructed from a value.
   */
  template <typename T>
  type type_of(const T &DYND_UNUSED(value))
  {
    return type::make<T>();
  }

  template <typename T>
  type type_of(const std::vector<T> &value)
  {
    return make_fixed_dim(value.size(), type::make<T>());
  }

  DYND_API type type_of(const nd::array &val);

  DYND_API type type_of(const nd::callable &val);

  /**
   * Returns the type to use for packing this specific value. The value
   * is allowed to affect the type, e.g. for packing a std::vector
   */
  template <typename T>
  type get_forward_type(const T &DYND_UNUSED(val))
  {
    // check for exact type

    // Default case is for when T and the ndt::type have identical
    // memory layout, which is guaranteed by make_exact_type<T>().
    return type::make<T>();
  }

  template <typename T>
  type get_forward_type(const std::vector<T> &val)
  {
    // check for exact type

    // Depending on the data size, store the data by value or as a pointer
    // to an nd::array
    if (sizeof(T) * val.size() > 32) {
      return make_pointer_type(make_fixed_dim(val.size(), type::make<T>()));
    } else {
      return make_fixed_dim(val.size(), type::make<T>());
    }
  }

  DYND_API type get_forward_type(const nd::array &val);

  DYND_API type get_forward_type(const nd::callable &val);

  /**
   * A static array of the builtin types and void.
   * If code is specialized just for a builtin type, like int, it can use
   * static_builtin_types[type_id_of<int>::value] as a fast
   * way to get a const reference to its type.
   */
  extern DYND_API const type static_builtin_types[builtin_type_id_count];

  DYND_API std::ostream &operator<<(std::ostream &o, const type &rhs);

} // namespace dynd::ndt

/** Prints raw bytes as hexadecimal */
DYND_API void hexadecimal_print(std::ostream &o, char value);
DYND_API void hexadecimal_print(std::ostream &o, unsigned char value);
DYND_API void hexadecimal_print(std::ostream &o, unsigned short value);
DYND_API void hexadecimal_print(std::ostream &o, unsigned int value);
DYND_API void hexadecimal_print(std::ostream &o, unsigned long value);
DYND_API void hexadecimal_print(std::ostream &o, unsigned long long value);
DYND_API void hexadecimal_print(std::ostream &o, const char *data, intptr_t element_size);
DYND_API void hexadecimal_print_summarized(std::ostream &o, const char *data, intptr_t element_size,
                                           intptr_t summary_size);

DYND_API void strided_array_summarized(std::ostream &o, const ndt::type &tp, const char *arrmeta, const char *data,
                                       intptr_t dim_size, intptr_t stride);
DYND_API void print_indented(std::ostream &o, const std::string &indent, const std::string &s,
                             bool skipfirstline = false);

} // namespace dynd
