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
#include <dynd/types/type_id.hpp>
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
      iterator_type(const char *metadata, char *data) : scalar_wrapper_iterator<ValueType, NDim>(metadata, data) {}
    };

    scalar_wrapper(const char *metadata, char *data) : m_metadata(metadata), m_data(data) {}

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
    scalar_wrapper_iterator(const char *DYND_UNUSED(metadata), char *data) : m_data(data) {}

    ValueType &operator*() { return *reinterpret_cast<ValueType *>(m_data); }

    bool operator==(const scalar_wrapper_iterator &rhs) const { return m_data == rhs.m_data; }

    bool operator!=(const scalar_wrapper_iterator &rhs) const { return m_data != rhs.m_data; }
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
inline bool offset_is_aligned(size_t offset, size_t alignment) { return (offset & (alignment - 1)) == 0; }

/** Prints a single scalar of a builtin type to the stream */
void DYNDT_API print_builtin_scalar(type_id_t type_id, std::ostream &o, const char *data);

/** Special iterdata which broadcasts to any number of additional dimensions */
struct DYNDT_API iterdata_broadcasting_terminator {
  iterdata_common common;
  char *data;
};
DYNDT_API char *iterdata_broadcasting_terminator_incr(iterdata_common *iterdata, intptr_t level);
DYNDT_API char *iterdata_broadcasting_terminator_adv(iterdata_common *iterdata, intptr_t level, intptr_t i);
DYNDT_API char *iterdata_broadcasting_terminator_reset(iterdata_common *iterdata, char *data, intptr_t level);

namespace ndt {
  typedef type (*type_make_t)(type_id_t tp_id, const nd::array &args);

  DYNDT_API type make_fixed_dim(size_t dim_size, const type &element_tp);
  inline type make_var_dim(const type &element_tp);

  template <typename T>
  struct traits {
    ~traits() = delete;
  };

  template <typename T>
  struct has_traits {
    static const bool value = std::is_destructible<traits<T>>::value;
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
  class DYNDT_API type : public intrusive_ptr<const base_type> {
  private:
    /**
     * Valid type properties can have type scalar, std::string, ndt::type
     * or a vector of any of these.
     */
    template <typename T, typename C = typename T::value_type>
    std::enable_if_t<is_vector<T>::value, const T> &property(const char *name) const
    {
      const std::pair<ndt::type, const char *> pair = get_properties()[name];
      const ndt::type &dt = pair.first.get_dtype();

      if (pair.first.get_id() != fixed_dim_id) {
        throw std::runtime_error("unsupported type for property access");
      }

      if (dt.get_id() == property_type_id_of<C>::value) {
        return *reinterpret_cast<const T *>(pair.second);
      }

      throw std::runtime_error("type mismatch or unsupported type in property access");
    }

    template <typename T>
    std::enable_if_t<!is_vector<T>::value, const T> &property(const char *name) const
    {
      const std::pair<ndt::type, const char *> pair = get_properties()[name];

      if (pair.first.get_id() == property_type_id_of<T>::value) {
        return *reinterpret_cast<const T *>(pair.second);
      }

      throw std::runtime_error("type mismatch in property access");
    }

  public:
    using intrusive_ptr<const base_type>::intrusive_ptr;

    /**
      * Default constructor.
      */
    type() = default;

    /** Construct from a type ID */
    type(type_id_t tp_id);

    /** Construct from a string representation */
    explicit type(const std::string &rep);

    /** Construct from a string representation */
    type(const char *rep_begin, const char *rep_end);

    bool operator==(const type &rhs) const
    {
      return m_ptr == rhs.m_ptr || (!is_builtin() && !rhs.is_builtin() && *m_ptr == *rhs.m_ptr);
    }

    bool operator!=(const type &rhs) const { return !(operator==(rhs)); }

    bool is_null() const { return m_ptr == NULL; }

    /**
     * Returns true if this type is built in, which
     * means the type id is encoded directly in the m_ptr
     * pointer.
     */
    bool is_builtin() const { return is_builtin_type(m_ptr); }

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
      }
      else {
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
    type at(const irange &i0) const { return at_array(1, &i0); }

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
     * \param candidate_tp    A type to match against this one.
     * \param tp_vars     A map of names to matched type vars.
     */
    bool match(const ndt::type &candidate_tp, std::map<std::string, ndt::type> &tp_vars) const;

    bool match(const type &other) const
    {
      std::map<std::string, type> tp_vars;
      return match(other, tp_vars);
    }

    /**
     * Accesses a dynamic property of the type.
     *
     * \param name  The property to access.
     */
    template <typename T>
    const T &p(const char *name) const
    {
      return property<T>(name);
    }
    template <typename T>
    const T &p(const std::string &name) const
    {
      return property<T>(name.c_str());
    }

    /**
     * Indexes into the type, intended for recursive calls from the
     * extended-type version. See
     * the function in base_type with the same name for more details.
     */
    type apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i, const type &root_tp,
                            bool leading_dimension) const;

    /**
     * Returns the non-expression type that this type looks like for the purposes
     * of calculation, printing, etc.
     */
    const type &value_type() const;

    /**
     * For expression types, returns the storage type, which is the type of the
     * underlying input data. This is the bottom of the expression chain.
     */
    const type &storage_type() const;

    /**
     * The type number is an enumeration of data types, starting
     * at 0, with one value for each unique data type. This is
     * inspired by the approach in NumPy, and the intention is
     * to have the default
     */
    type_id_t get_id() const
    {
      if (is_builtin()) {
        return static_cast<type_id_t>(reinterpret_cast<intptr_t>(m_ptr));
      }
      else {
        return m_ptr->get_id();
      }
    }

    /**
     * For when it is known that the type is a builtin type,
     * to simply retrieve that type id.
     *
     * WARNING: Normally just use get_id().
     */
    type_id_t unchecked_get_builtin_id() const { return static_cast<type_id_t>(reinterpret_cast<intptr_t>(m_ptr)); }

    type_id_t get_base_id() const;

    /** The alignment of the type */
    size_t get_data_alignment() const;

    /** The element size of the type */
    size_t get_data_size() const;

    /** The element size of the type when default-constructed */
    size_t get_default_data_size() const;

    size_t get_arrmeta_size() const
    {
      if (is_builtin()) {
        return 0;
      }
      else {
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
      }
      else {
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
      }
      else {
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

    bool is_indexable() const { return !is_builtin() && m_ptr->is_indexable(); }

    bool is_scalar() const { return is_builtin() || m_ptr->is_scalar(); }

    /**
     * Returns true if the type contains any expression
     * type within it somewhere.
     */
    bool is_expression() const
    {
      if (is_builtin()) {
        return false;
      }
      else {
        return m_ptr->is_expression();
      }
    }

    /**
     * Returns true if the type contains a symbolic construct
     * like a type var.
     */
    bool is_symbolic() const { return !is_builtin() && m_ptr->is_symbolic(); }

    /**
     * Returns true if the type constains a symbolic dimension
     * which matches a variadic number of dimensions.
     */
    bool is_variadic() const { return !is_builtin() && (m_ptr->get_flags() & type_flag_variadic); }

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
      }
      else {
        return m_ptr->get_canonical_type();
      }
    }

    uint32_t get_flags() const
    {
      if (is_builtin()) {
        return type_flag_none;
      }
      else {
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
      }
      else {
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
      }
      else {
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
      }
      else if (ndim > include_ndim) {
        return m_ptr->get_type_at_dimension(inout_arrmeta, ndim - include_ndim);
      }
      else {
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
      }
      else if (i == 0) {
        return *this;
      }
      else {
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

    std::map<std::string, std::pair<ndt::type, const char *>> get_properties() const;

    /**
     * Returns a const pointer to the base_type object which
     * contains information about the type, or NULL if no extended
     * type information exists. The returned pointer is only valid during
     * the lifetime of the type.
     */
    const base_type *extended() const { return m_ptr; }

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
      }
      else {
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
      }
      else {
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
      }
      else {
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

    static type make(type_id_t tp_id, const nd::array &args);

    friend DYNDT_API std::ostream &operator<<(std::ostream &o, const type &rhs);
  };

  template <>
  struct traits<void> {
    static const size_t ndim = 0;

    static const bool is_same_layout = false;

    static type equivalent() { return type(type_id_of<void>::value); }
  };

  namespace detail {

    /**
     * Returns the equivalent type.
     */
    template <typename T, typename... ArgTypes>
    auto make_type(int, ArgTypes &&... args) -> decltype(traits<T>::equivalent(std::forward<ArgTypes>(args)...))
    {
      return traits<T>::equivalent(std::forward<ArgTypes>(args)...);
    }

    /**
     * Returns the equivalent type.
     */
    template <typename T, typename... ArgTypes>
    auto make_type(char, ArgTypes &&... DYND_UNUSED(args)) -> decltype(traits<T>::equivalent())
    {
      return traits<T>::equivalent();
    }

  } // namespace dynd::ndt::detail

  /**
   * Returns the equivalent type.
   */
  template <typename T, typename... ArgTypes>
  auto make_type(ArgTypes &&... args) -> decltype(detail::make_type<T>(0, std::forward<ArgTypes>(args)...))
  {
    return detail::make_type<T>(0, std::forward<ArgTypes>(args)...);
  }

  template <typename ValueType>
  type type_for(const ValueType &value)
  {
    return make_type<ValueType>(value);
  }

  template <typename ValueType>
  type type_for(const std::initializer_list<ValueType> &values)
  {
    return make_type<std::initializer_list<ValueType>>(values);
  }

  template <typename ValueType>
  type type_for(const std::initializer_list<std::initializer_list<ValueType>> &values)
  {
    return make_type<std::initializer_list<std::initializer_list<ValueType>>>(values);
  }

  template <typename ValueType>
  type type_for(const std::initializer_list<std::initializer_list<std::initializer_list<ValueType>>> &values)
  {
    return make_type<std::initializer_list<std::initializer_list<std::initializer_list<ValueType>>>>(values);
  }

  /**
   * Allocates and constructs a type with a use count of 1.
   */
  template <typename T, typename... ArgTypes>
  std::enable_if_t<std::is_base_of<base_type, T>::value, type> make_type(ArgTypes &&... args)
  {
    return type(new T(std::forward<ArgTypes>(args)...), false);
  }

  /*
  #define DYND_BOOL_NA (2)
  #define DYND_INT8_NA (std::numeric_limits<int8_t>::min())
  #define DYND_INT16_NA (std::numeric_limits<int16_t>::min())
  #define DYND_INT32_NA (std::numeric_limits<int32_t>::min())
  #define DYND_INT64_NA (std::numeric_limits<int64_t>::min())
  #define DYND_INT128_NA (std::numeric_limits<int128>::min())
  #define DYND_FLOAT16_NA_AS_UINT (0x7e0au)
  #define DYND_FLOAT32_NA_AS_UINT (0x7f8007a2U)
  #define DYND_FLOAT64_NA_AS_UINT (0x7ff00000000007a2ULL)
  */

  template <>
  struct traits<bool1> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<bool1>::value); }
  };

  template <>
  struct traits<bool> {
    static const size_t ndim = 0;

    static const bool is_same_layout = false;

    static type equivalent() { return traits<bool1>::equivalent(); }
  };

  template <>
  struct traits<signed char> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<signed char>::value); }

    static signed char na() { return std::numeric_limits<signed char>::min(); }
  };

  template <>
  struct traits<short> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<short>::value); }
  };

  template <>
  struct traits<int> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<int>::value); }

    static int na() { return std::numeric_limits<int>::min(); }
  };

  template <>
  struct traits<long> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<long>::value); }
  };

  template <>
  struct traits<long long> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<long long>::value); }
  };

  template <>
  struct traits<int128> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<int128>::value); }
  };

  template <>
  struct traits<unsigned char> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<unsigned char>::value); }
  };

  template <>
  struct traits<unsigned short> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<unsigned short>::value); }
  };

  template <>
  struct traits<unsigned int> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<unsigned int>::value); }

    static unsigned int na() { return std::numeric_limits<unsigned int>::max(); }
  };

  template <>
  struct traits<unsigned long> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<unsigned long>::value); }
  };

  template <>
  struct traits<unsigned long long> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<unsigned long long>::value); }
  };

  template <>
  struct traits<uint128> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<uint128>::value); }
  };

  template <>
  struct traits<char> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<char>::value); }
  };

  template <>
  struct traits<float16> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<float16>::value); }
  };

  template <>
  struct traits<float> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<float>::value); }
  };

  template <>
  struct traits<double> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<double>::value); }
  };

  template <>
  struct traits<float128> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<float128>::value); }
  };

  template <typename T>
  struct traits<complex<T>> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<complex<T>>::value); }
  };

  template <typename T>
  struct traits<std::complex<T>> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id_of<std::complex<T>>::value); }
  };

  template <>
  struct traits<const char *> {
    static const size_t ndim = 0;

    static const bool is_same_layout = false;

    static type equivalent() { return type(string_id); }
  };

  template <size_t N>
  struct traits<char[N]> {
    static const size_t ndim = 0;

    static const bool is_same_layout = false;

    static type equivalent() { return type(string_id); }
  };

  template <size_t N>
  struct traits<const char[N]> {
    static const size_t ndim = 0;

    static const bool is_same_layout = false;

    static type equivalent() { return type(string_id); }
  };

  template <>
  struct traits<type> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(type_id); }
  };

  template <typename T>
  struct traits<const T> {
    static const size_t ndim = traits<T>::ndim;

    static const bool is_same_layout = traits<T>::is_same_layout;

    static type equivalent() { return traits<T>::equivalent(); }
  };

  // Same as for const
  template <typename T>
  struct traits<T &> {
    static const size_t ndim = traits<T>::ndim;

    static const bool is_same_layout = traits<T>::is_same_layout;

    static type equivalent() { return traits<T>::equivalent(); }
  };

  // Same as for const
  template <typename T>
  struct traits<T &&> {
    static const size_t ndim = traits<T>::ndim;

    static const bool is_same_layout = traits<T>::is_same_layout;

    static type equivalent() { return traits<T>::equivalent(); }
  };

  template <typename T, size_t N>
  struct traits<T[N]> {
    static const size_t ndim = traits<T>::ndim + 1;

    static const bool is_same_layout = traits<T>::is_same_layout;

    static type equivalent() { return make_fixed_dim(N, make_type<T>()); }
  };

  template <>
  struct traits<assign_error_mode> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return make_type<typename std::underlying_type<assign_error_mode>::type>(); }

    static assign_error_mode na()
    {
      return static_cast<assign_error_mode>(traits<typename std::underlying_type<assign_error_mode>::type>::na());
    }
  };

  // Need to handle const properly
  template <typename T, size_t N>
  struct traits<const T[N]> {
    static const size_t ndim = traits<T[N]>::ndim;

    static const bool is_same_layout = traits<T[N]>::is_same_layout;

    static type equivalent() { return make_type<T[N]>(); }
  };

  template <typename ContainerType, size_t NDim>
  struct container_traits {
    static const size_t ndim = NDim;

    static const bool is_same_layout = false;

    static type equivalent(const ContainerType &values)
    {
      intptr_t shape[ndim];
      container_traits::shape(shape, values);

      type tp = value_type();
      for (intptr_t i = ndim - 1; i >= 0; --i) {
        if (shape[i] == -1) {
          tp = make_var_dim(tp);
        }
        else {
          tp = make_fixed_dim(shape[i], tp);
        }
      }

      return tp;
    }

    static void shape(intptr_t *res, const ContainerType &values)
    {
      res[0] = values.size();

      auto iter = values.begin();
      ndt::traits<typename ContainerType::value_type>::shape(res + 1, *iter);

      while (++iter != values.end()) {
        intptr_t next_shape[ndim - 1];
        ndt::traits<typename ContainerType::value_type>::shape(next_shape, *iter);

        for (size_t i = 1; i < ndim; ++i) {
          if (res[i] != next_shape[i - 1]) {
            res[i] = -1;
          }
        }
      }
    }

    static type value_type() { return container_traits<typename ContainerType::value_type, ndim - 1>::value_type(); }
  };

  template <typename ContainerType>
  struct container_traits<ContainerType, 1> {
    static const size_t ndim = 1;

    static const bool is_same_layout = false;

    static type equivalent(const ContainerType &values)
    {
      intptr_t size;
      shape(&size, values);

      return make_fixed_dim(size, value_type());
    }

    static void shape(intptr_t *res, const ContainerType &values) { res[0] = values.size(); }

    static type value_type() { return traits<typename ContainerType::value_type>::equivalent(); }
  };

  template <typename ValueType>
  struct traits<std::initializer_list<ValueType>>
      : container_traits<std::initializer_list<ValueType>, traits<ValueType>::ndim + 1> {
  };

  template <typename ValueType>
  struct traits<std::vector<ValueType>> : container_traits<std::vector<ValueType>, traits<ValueType>::ndim + 1> {
  };

  /**
    * Returns the common type of two types. For built-in types, this is analogous to
    * std::common_type.
    */
  DYNDT_API extern class common_type {
    typedef type (*child_type)(const type &, const type &);

    struct init;

    static std::map<std::array<type_id_t, 2>, child_type> children;

  public:
    common_type();

    DYNDT_API ndt::type operator()(const ndt::type &tp0, const ndt::type &tp1) const;
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
  DYNDT_API type make_type(intptr_t ndim, const intptr_t *shape, const ndt::type &dtype);

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
  inline type make_type(intptr_t ndim, const intptr_t *shape, const char(&dtype)[N])
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
  DYNDT_API type make_type(intptr_t ndim, const intptr_t *shape, const ndt::type &dtype, bool &out_any_var);

  DYNDT_API type_id_t register_type(const std::string &name, type_make_t make);

  template <typename TypeType>
  type_id_t register_type(const std::string &name)
  {
    return register_type(name,
                         [](type_id_t tp_id, const nd::array &args) { return type(new TypeType(tp_id, args), false); });
  }

  DYNDT_API std::ostream &operator<<(std::ostream &o, const type &rhs);

} // namespace dynd::ndt

/** Prints raw bytes as hexadecimal */
DYNDT_API void hexadecimal_print(std::ostream &o, char value);
DYNDT_API void hexadecimal_print(std::ostream &o, unsigned char value);
DYNDT_API void hexadecimal_print(std::ostream &o, unsigned short value);
DYNDT_API void hexadecimal_print(std::ostream &o, unsigned int value);
DYNDT_API void hexadecimal_print(std::ostream &o, unsigned long value);
DYNDT_API void hexadecimal_print(std::ostream &o, unsigned long long value);
DYNDT_API void hexadecimal_print(std::ostream &o, const char *data, intptr_t element_size);
DYNDT_API void hexadecimal_print_summarized(std::ostream &o, const char *data, intptr_t element_size,
                                           intptr_t summary_size);

DYNDT_API void strided_array_summarized(std::ostream &o, const ndt::type &tp, const char *arrmeta, const char *data,
                                       intptr_t dim_size, intptr_t stride);
DYNDT_API void print_indented(std::ostream &o, const std::string &indent, const std::string &s,
                             bool skipfirstline = false);

/** If 'src' can always be cast to 'dst' with no loss of information */
DYNDT_API bool is_lossless_assignment(const ndt::type &dst_tp, const ndt::type &src_tp);

} // namespace dynd
