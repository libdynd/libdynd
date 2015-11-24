//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream> // FOR DEBUG
#include <stdexcept>
#include <string>

#include <dynd/config.hpp>

#include <dynd/type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/shortvector.hpp>
#include <dynd/irange.hpp>
#include <dynd/memblock/array_memory_block.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/var_dim_type.hpp>

namespace dynd {

namespace ndt {
  DYND_API type make_fixed_dim(size_t dim_size, const type &element_tp);
} // namespace ndt;

namespace nd {

  class DYND_API array;

  enum array_access_flags {
    /** If an array is readable */
    read_access_flag = 0x01,
    /** If an array is writable */
    write_access_flag = 0x02,
    /** If an array will not be written to by anyone else either */
    immutable_access_flag = 0x04
  };

  // Some additional access flags combinations for convenience
  enum {
    readwrite_access_flags = read_access_flag | write_access_flag,
    default_access_flags = read_access_flag | write_access_flag,
  };

  /** Stream printing function */
  DYND_API std::ostream &operator<<(std::ostream &o, const array &rhs);

  class array_vals;
  class array_vals_at;

  /**
   * This is the primary multi-dimensional array class.
   */
  class DYND_API array : public intrusive_ptr<memory_block_data> {
    // Don't allow implicit construction from a raw pointer
    array(const void *);

  public:
    /**
      * Constructs an array with no data.
      */
    array() = default;

    /**
      * Copy constructs an array.
      */
    array(const array &other) = default;

    /**
     * Move constructs an array.
     */
    array(array &&other) = default;

    /**
     * Constructs a zero-dimensional scalar from a C++ scalar.
     *
     */
    array(bool1 value);
    array(bool value);
    array(signed char value);
    array(short value);
    array(int value);
    array(long value);
    array(long long value);
    array(const int128 &value);
    array(unsigned char value);
    array(unsigned short value);
    array(unsigned int value);
    array(unsigned long value);
    array(unsigned long long value);
    array(const uint128 &value);
    array(float16 value);
    array(float value);
    array(double value);
    array(const float128 &value);
    array(complex<float> value);
    array(complex<double> value);
    array(std::complex<float> value);
    array(std::complex<double> value);
    array(const std::string &value);
    /** Construct a string from a NULL-terminated UTF8 string */
    array(const char *cstr);
    /** Construct a string from a UTF8 buffer and specified buffer size */
    array(const char *str, size_t size);
    /**
     * Constructs a scalar with the 'type' type.
     * NOTE: Does NOT create a scalar of the provided type,
     *       use nd::empty(type) for that!
     */
    array(const ndt::type &dt);

    /**
     * Constructs an array from a multi-dimensional C-style array.
     */
    template <class T, int N>
    array(const T (&rhs)[N]);
    /** Specialize to treat char arrays as strings */
    template <int N>
    array(const char (&rhs)[N]);
    /** Specialize to create 1D arrays of strings */
    template <int N>
    array(const char *(&rhs)[N]);
    template <int N>
    array(const std::string *(&rhs)[N]);
    /** Specialize to create 1D arrays of ndt::types */
    template <int N>
    array(const ndt::type (&rhs)[N]);

    /**
     * Constructs a 1D array from a pointer and a size.
     */
    template <class T>
    array(const T *rhs, intptr_t dim_size);
    array(const ndt::type *rhs, intptr_t dim_size);

    /** Constructs an array from a 1D initializer list */
    template <class T>
    array(const std::initializer_list<T> &il);
    /** Constructs an array from a 2D initializer list */
    template <class T>
    array(const std::initializer_list<std::initializer_list<T>> &il);
    /** Constructs an array from a 3D initializer list */
    template <class T>
    array(const std::initializer_list<std::initializer_list<std::initializer_list<T>>> &il);

    /** Constructs an array from a 1D const char * (string) initializer list */
    array(const std::initializer_list<const char *> &il);

    /** Constructs an array from a 1D ndt::type initializer list */
    array(const std::initializer_list<ndt::type> &il);

    /** Constructs an array from a 1D bool initializer list */
    array(const std::initializer_list<bool> &il);

    /** Assigns an array from a 1D initializer list */
    template <class T>
    inline array operator=(const std::initializer_list<T> &il)
    {
      array(il).swap(*this);
      return *this;
    }
    /** Assigns an array from a 2D initializer list */
    template <class T>
    inline array operator=(const std::initializer_list<std::initializer_list<T>> &il)
    {
      array(il).swap(*this);
      return *this;
    }
    /** Assigns an array from a 3D initializer list */
    template <class T>
    inline array operator=(const std::initializer_list<std::initializer_list<std::initializer_list<T>>> &il)
    {
      array(il).swap(*this);
      return *this;
    }

    /**
     * Constructs an array from a std::vector.
     */
    template <class T>
    array(const std::vector<T> &vec);

    explicit array(const intrusive_ptr<memory_block_data> &ndobj_memblock)
        : intrusive_ptr<memory_block_data>(ndobj_memblock)
    {
      if (intrusive_ptr<memory_block_data>::get()->m_type != array_memory_block_type) {
        throw std::runtime_error("array can only be constructed from a memblock with array type");
      }
    }

    explicit array(array_preamble *ndo, bool add_ref) : intrusive_ptr<memory_block_data>(ndo, add_ref)
    {
    }

    void set(const intrusive_ptr<memory_block_data> &ndobj_memblock)
    {
      if (ndobj_memblock.get()->m_type != array_memory_block_type) {
        throw std::runtime_error("array can only be constructed from a memblock with array type");
      }
      intrusive_ptr<memory_block_data>::operator=(ndobj_memblock);
    }

    void set(intrusive_ptr<memory_block_data> &&ndobj_memblock)
    {
      if (ndobj_memblock.get()->m_type != array_memory_block_type) {
        throw std::runtime_error("array can only be constructed from a memblock with array type");
      }
      intrusive_ptr<memory_block_data>::operator=(std::move(ndobj_memblock));
    }

    /**
     * This function releases the memory block reference, setting the
     * array to NULL. The caller takes explicit ownership of the
     * reference.
     */
    array_preamble *release()
    {
      return reinterpret_cast<array_preamble *>(intrusive_ptr<memory_block_data>::release());
    }

    /**
     * Assignment operator.
     */
    array &operator=(const array &rhs) = default;

    /**
     * Move assignment operator.
     */
    array &operator=(array &&rhs) = default;

    /** Low level access to the array preamble */
    array_preamble *get() const
    {
      return reinterpret_cast<array_preamble *>(intrusive_ptr<memory_block_data>::get());
    }

    /** Returns true if the array is NULL */
    inline bool is_null() const
    {
      return intrusive_ptr<memory_block_data>::get() == NULL;
    }

    char *data() const
    {
      if (get()->flags & write_access_flag) {
        return get()->data;
      }

      throw std::runtime_error("tried to write to a dynd array that is not writable");
    }

    const char *cdata() const
    {
      return get()->data;
    }

    inline uint32_t get_access_flags() const
    {
      return get()->flags & (immutable_access_flag | read_access_flag | write_access_flag);
    }

    inline bool is_immutable() const
    {
      return (get()->flags & immutable_access_flag) != 0;
    }

    /** Returns true if the object is a scalar */
    inline bool is_scalar() const
    {
      return get_type().is_scalar();
    }

    /** The type */
    const ndt::type &get_type() const
    {
      return *reinterpret_cast<const ndt::type *>(&get()->tp);
    }

    inline intptr_t get_ndim() const
    {
      if (get()->tp.is_builtin()) {
        return 0;
      } else {
        return get()->tp->get_ndim();
      }
    }

    /**
     * The data type of the array. This is similar to numpy's
     * ndarray.dtype property
     */
    inline ndt::type get_dtype() const
    {
      size_t ndim = get()->tp.get_ndim();
      if (ndim == 0) {
        return get()->tp;
      }

      return get()->tp->get_type_at_dimension(NULL, ndim);
    }

    /**
     * The data type of the array. This is similar to numpy's
     * ndarray.dtype property, but may include some array dimensions
     * if requested.
     *
     * \param include_ndim  The number of array dimensions to include
     *                   in the data type.
     */
    inline ndt::type get_dtype(size_t include_ndim) const
    {
      if (get()->tp.is_builtin()) {
        if (include_ndim > 0) {
          throw too_many_indices(get_type(), include_ndim, 0);
        }
        return ndt::type(get()->tp.get_type_id());
      } else {
        size_t ndim = get()->tp->get_ndim();
        if (ndim < include_ndim) {
          throw too_many_indices(get_type(), include_ndim, ndim);
        }
        ndim -= include_ndim;
        if (ndim == 0) {
          return get()->tp;
        } else {
          return get()->tp->get_type_at_dimension(NULL, ndim);
        }
      }
    }

    /**
     * If the caller has the only reference to this array and its data,
     * makes the access flags into read-only and immutable.
     */
    void flag_as_immutable();

    /** The flags, including access permissions. */
    inline uint64_t get_flags() const
    {
      return get()->flags;
    }

    inline std::vector<intptr_t> get_shape() const
    {
      std::vector<intptr_t> result(get_ndim());
      get_shape(&result[0]);
      return result;
    }
    inline void get_shape(intptr_t *out_shape) const
    {
      if (!get()->tp.is_builtin() && get()->tp->get_ndim() > 0) {
        get()->tp->get_shape(get()->tp->get_ndim(), 0, out_shape, get()->metadata(), get()->data);
      }
    }

    /**
     * Returns the size of the leading (leftmost) dimension.
     */
    inline intptr_t get_dim_size() const
    {
      return get_type().get_dim_size(get()->metadata(), get()->data);
    }

    /**
     * Returns the size of the requested dimension.
     */
    inline intptr_t get_dim_size(intptr_t i) const
    {
      if (0 <= i && i < get_type().get_strided_ndim()) {
        const size_stride_t *ss = reinterpret_cast<const size_stride_t *>(get()->metadata());
        return ss[i].dim_size;
      } else if (0 <= i && i < get_ndim()) {
        dimvector shape(i + 1);
        get()->tp->get_shape(i + 1, 0, shape.get(), get()->metadata(), get()->data);
        return shape[i];
      } else {
        std::stringstream ss;
        ss << "Not enough dimensions in array, tried to access axis " << i << " for type " << get_type();
        throw std::invalid_argument(ss.str());
      }
    }

    std::vector<intptr_t> get_strides() const
    {
      std::vector<intptr_t> result(get_ndim());
      get_strides(&result[0]);
      return result;
    }
    inline void get_strides(intptr_t *out_strides) const
    {
      if (!get()->tp.is_builtin()) {
        get()->tp->get_strides(0, out_strides, get()->metadata());
      }
    }

    inline intrusive_ptr<memory_block_data> get_data_memblock() const
    {
      if (get()->owner) {
        return get()->owner;
      } else {
        return *this;
      }
    }

    /**
     * Accesses a dynamic property of the array.
     *
     * \param property_name  The property to access.
     */
    array p(const char *property_name) const;
    /**
     * Accesses a dynamic property of the array.
     *
     * \param property_name  The property to access.
     */
    array p(const std::string &property_name) const;
    /**
     * Finds the dynamic function of the array. Throws an
     * exception if it does not exist. To call the function,
     * use ndobj.f("funcname").call(ndobj, ...). The reason
     * ndobj.f("funcname", ...) isn't used is due to a circular
     * dependency between callable and array. A resolution
     * to this will make calling these functions much more
     * convenient.
     *
     * \param function_name  The name of the function.
     */
    const gfunc::callable &find_dynamic_function(const char *function_name) const;

    /** Calls the dynamic function - #include <dynd/func/call_callable.hpp> to
     * use it */
    array f(const char *function_name);

    /** Calls the dynamic function - #include <dynd/func/call_callable.hpp> to
     * use it */
    array f(const char *function_name) const;

    /** Calls the dynamic function - #include <dynd/func/call_callable.hpp> to
     * use it */
    template <class T0>
    array f(const char *function_name, const T0 &p0);

    /** Calls the dynamic function - #include <dynd/func/call_callable.hpp> to
     * use it */
    template <class T0, class T1>
    array f(const char *function_name, const T0 &p0, const T1 &p1);

    /** Calls the dynamic function - #include <dynd/func/call_callable.hpp> to
     * use it */
    template <class T0, class T1, class T2>
    array f(const char *function_name, const T0 &p0, const T1 &p1, const T2 &p2);

    /** Calls the dynamic function - #include <dynd/func/call_callable.hpp> to
     * use it */
    template <class T0, class T1, class T2, class T3>
    array f(const char *function_name, const T0 &p0, const T1 &p1, const T2 &p2, const T3 &p3);

    array &operator+=(const array &rhs);
    array &operator-=(const array &rhs);
    array &operator*=(const array &rhs);
    array &operator/=(const array &rhs);

    /**
     * A helper for assigning to the values in 'this'. Normal assignment to
     * an array variable has reference semantics, the reference gets
     * overwritten to point to the new array. The 'vals' function provides
     * syntactic sugar for the 'val_assign' function, allowing for more
     * natural looking assignments.
     *
     * Example:
     *      array a(ndt::make_type<float>());
     *      a.vals() = 100;
     */
    array_vals vals() const;

    /**
     * A helper for assigning to the values indexed in an array.
     */
    array_vals_at vals_at(const irange &i0) const;

    /**
     * A helper for assigning to the values indexed in an array.
     */
    array_vals_at vals_at(const irange &i0, const irange &i1) const;

    /**
     * A helper for assigning to the values indexed in an array.
     */
    array_vals_at vals_at(const irange &i0, const irange &i1, const irange &i2) const;

    /**
     * A helper for assigning to the values indexed in an array.
     */
    array_vals_at vals_at(const irange &i0, const irange &i1, const irange &i2, const irange &i3) const;

    /**
     * Evaluates the array, attempting to do the minimum work
     * required. If the array is not ane expression, simply
     * returns it as is, otherwise evaluates into a new copy.
     */
    array eval(const eval::eval_context *ectx = &eval::default_eval_context) const;

    /**
     * Evaluates the array into an immutable strided array, or
     * returns it untouched if it is already both immutable and strided.
     */
    array eval_immutable(const eval::eval_context *ectx = &eval::default_eval_context) const;

    /**
     * Evaluates the array node into a newly allocated strided array,
     * with the requested access flags.
     *
     * \param access_flags  The access flags for the result, default immutable.
     * \param ectx  The evaluation context
     */
    array eval_copy(uint32_t access_flags = 0, const eval::eval_context *ectx = &eval::default_eval_context) const;

    /**
     * Returns a view of the array as bytes (for POD) or the storage type,
     * peeling away any expression types or encodings.
     */
    array storage() const;

    /**
     * Returns either this array or the array stored in the reference if the
     * type is a reference tpye.
     */
    array &underlying();

    const array &underlying() const;

    /**
     * General irange-based indexing operation.
     *
     * \param nindices  The number of 'irange' indices.
     * \param indices  The array of indices to apply.
     * \param collapse_leading  If true, collapses the leading dimension
     *                          to simpler types where possible. If false,
     *                          does not. If you want to read values, typically
     *                          use true, if you want to write values, typically
     *                          use false.
     */
    array at_array(intptr_t nindices, const irange *indices, bool collapse_leading = true) const;

    /**
     * The function call operator is used for indexing. Overloading
     * operator[] isn't practical for multidimensional objects.
     */
    array operator()(const irange &i0) const
    {
      return at_array(1, &i0);
    }

    /** Indexing with two index values */
    array operator()(const irange &i0, const irange &i1) const
    {
      irange i[2] = {i0, i1};
      return at_array(2, i);
    }

    /** Indexing with three index values */
    array operator()(const irange &i0, const irange &i1, const irange &i2) const
    {
      irange i[3] = {i0, i1, i2};
      return at_array(3, i);
    }
    /** Indexing with four index values */
    array operator()(const irange &i0, const irange &i1, const irange &i2, const irange &i3) const
    {
      irange i[4] = {i0, i1, i2, i3};
      return at_array(4, i);
    }
    /** Indexing with five index values */
    array operator()(const irange &i0, const irange &i1, const irange &i2, const irange &i3, const irange &i4) const
    {
      irange i[5] = {i0, i1, i2, i3, i4};
      return at_array(5, i);
    }

    array at(const irange &i0) const
    {
      return at_array(1, &i0);
    }

    /** Does a value-assignment from the rhs array. */
    void val_assign(const array &rhs, const eval::eval_context *ectx = &eval::default_eval_context) const;
    /** Does a value-assignment from the rhs raw scalar */
    void val_assign(const ndt::type &rhs_dt, const char *rhs_arrmeta, const char *rhs_data,
                    const eval::eval_context *ectx = &eval::default_eval_context) const;

    /**
     * Casts the type of the array into the specified type.
     * This casts the entire type. If you want to cast the
     * array data type, use 'ucast' instead.
     *
     * \param tp  The type into which the array should be cast.
     */
    array cast(const ndt::type &tp) const;

    /**
     * Casts the array data type of the array into the specified type.
     *
     * \param uniform_dt  The type into which the array's
     *                    dtype should be cast.
     * \param replace_ndim  The number of array dimensions of
     *                       this type which should be replaced.
     *                       E.g. the value 1 could cast the last
     *                       array dimension and the array data type
     *                       to the replacement uniform_dt.
     */
    array ucast(const ndt::type &uniform_dt, intptr_t replace_ndim = 0) const;

    /**
     * Casts the array data type of the array into the type specified
     * as the template parameter.
     */
    template <class T>
    inline array ucast(intptr_t replace_ndim = 0) const
    {
      return ucast(ndt::type::make<T>(), replace_ndim);
    }

    /**
     * Attempts to view the data of the array as a new dynd type,
     * raising an error if it cannot be done.
     *
     * \param tp  The dynd type to view the entire data as.
     */
    array view(const ndt::type &tp) const;

    template <int N>
    inline array view(const char (&rhs)[N])
    {
      return view(ndt::type(rhs));
    }

    template <typename Type>
    Type view()
    {
      return Type(get()->metadata(), data());
    }

    /**
     * Attempts to view the uniform type-level of the array as a
     * new dynd type, raising an error if it cannot be done.
     *
     * \param uniform_dt  The dynd type to view the uniform data as.
     * \param replace_ndim  The number of array dimensions to swallow
     *                       into the viewed uniform_dt.
     */
    array uview(const ndt::type &uniform_dt, intptr_t replace_ndim) const;

    /**
     * Adapts the array into the destination type, using the provided
     * adaption operator. This creates an adapt[] type.
     *
     * Example:
     * nd::array({3, 5, 10}).adapt(ndt::date_type::make(), "days since
     *2001-1-1");
     */
    array adapt(const ndt::type &tp, const std::string &adapt_op);

    /**
     * Permutes the dimensions of the array, returning a view to the result.
     * Only strided dimensions can be permuted and no dimension can be permuted
     * across a variable dimension. At present, there is no error checking.
     *
     * \param ndim The number of dimensions to permute. If `ndim' is less than
     *that
     *             of this array, the other dimensions will not be modified.
     * \param axes The permutation. It must be of length `ndim' and not contain
     *             a value greater or equal to `ndim'.
     */
    array permute(intptr_t ndim, const intptr_t *axes) const;

    /**
     * Rotates the dimensions of the array so the axis `from' becomes the axis
     * `to'.
     * At present, there cannot be any variable dimensions.
     */
    array rotate(intptr_t to, intptr_t from = 0) const;

    array rotate(intptr_t from = 0) const
    {
      return rotate(get_ndim() - 1, from);
    }

    /**
     * Transposes (reverses) the dimensions of the array.
     * At present, there cannot be any variable dimensions.
     */
    array transpose() const;

    /**
     * DEPRECATED
     * Views the array's memory as another type, where such an operation
     * makes sense. This is analogous to reinterpret_cast<>.
     */
    array view_scalars(const ndt::type &scalar_tp) const;

    /**
     * Replaces the array data type with a new one, returning a view to
     * the result. The new type must have the same storage as the
     * existing type.
     *
     * \param replacement_tp  The replacement type.
     * \param replace_ndim  The number of array dimensions to replace
     *                       in addition to the array data type.
     */
    array replace_dtype(const ndt::type &replacement_tp, intptr_t replace_ndim = 0) const;

    /**
     * Inserts new fixed dimensions into the array.
     *
     * \param i  The axis at which the new fixed dimensions start.
     * \param new_ndim  The number of array dimensions to insert.
     */
    array new_axis(intptr_t i, intptr_t new_ndim = 1) const;

    /**
     * Views the array's memory as another type, where such an operation
     * makes sense. This is analogous to reinterpret_case<>.
     */
    template <class T>
    array view_scalars() const
    {
      return view_scalars(ndt::type::make<T>());
    }

    /**
     * When this is a zero-dimensional array, converts it to a C++ scalar of the
     * requested template type. This function may be extended in the future for
     * 1D vectors (as<std::vector<T>>), matrices, etc.
     *
     * \param errmode  The assignment error mode to use.
     */
    template <class T>
    T as(assign_error_mode errmode = assign_error_default) const;

    /** Returns a copy of this array in default memory. */
    array to_host() const;

#ifdef DYND_CUDA
    /** Returns a copy of this array in CUDA host memory. */
    array to_cuda_host(unsigned int cuda_host_flags = cudaHostAllocDefault) const;

    /** Returns a copy of this array in CUDA global memory. */
    array to_cuda_device() const;
#endif // DYND_CUDA

    bool is_missing() const;

    void assign_na();

    /** Sorting comparison between two arrays. (Returns a bool, does not
     * broadcast) */
    //   bool op_sorting_less(const array &rhs) const;

    bool equals_exact(const array &rhs) const;

    void debug_print(std::ostream &o, const std::string &indent = "") const;

    template <typename T>
    struct convert {
      static_assert(ndt::type::is_layout_compatible<T>::value, "must be layout compatible");

      static void from(char *DYND_UNUSED(metadata), char *data, const T &value)
      {
        *reinterpret_cast<const T *>(data) = value;
      }

      static const T &to(const char *DYND_UNUSED(metadata), const char *data)
      {
        return *reinterpret_cast<const T *>(data);
      }
    };

    friend DYND_API std::ostream &operator<<(std::ostream &o, const array &rhs);
    friend class array_vals;
    friend class array_vals_at;
  };

} // namespace dynd::nd

namespace ndt {

  template <>
  struct type::equivalent<nd::array> {
    static const type &make(const nd::array &val)
    {
      return val.get_type();
    }
  };

} // namespace dynd::ndt

namespace nd {

  DYND_API array as_struct();
  DYND_API array as_struct(std::size_t size, const char **names, const array *values);

  DYND_API array operator+(const array &a0);
  DYND_API array operator-(const array &a0);
  DYND_API array operator!(const array & a0);
  DYND_API array operator~(const array &a0);

  DYND_API array operator+(const array &op0, const array &op1);
  DYND_API array operator-(const array &op0, const array &op1);
  DYND_API array operator/(const array &op0, const array &op1);
  DYND_API array operator*(const array &op0, const array &op1);

  DYND_API array operator&&(const array &a0, const array &a1);
  DYND_API array operator||(const array &a0, const array &a1);

  DYND_API array operator<(const array &a0, const array &a1);
  DYND_API array operator<=(const array &a0, const array &a1);
  DYND_API array operator==(const array &a0, const array &a1);
  DYND_API array operator!=(const array &a0, const array &a1);
  DYND_API array operator>=(const array &a0, const array &a1);
  DYND_API array operator>(const array &a0, const array &a1);

  DYND_API nd::array array_rw(bool1 value);
  DYND_API nd::array array_rw(bool value);
  DYND_API nd::array array_rw(signed char value);
  DYND_API nd::array array_rw(short value);
  DYND_API nd::array array_rw(int value);
  DYND_API nd::array array_rw(long value);
  DYND_API nd::array array_rw(long long value);
  DYND_API nd::array array_rw(const int128 &value);
  DYND_API nd::array array_rw(unsigned char value);
  DYND_API nd::array array_rw(unsigned short value);
  DYND_API nd::array array_rw(unsigned int value);
  DYND_API nd::array array_rw(unsigned long value);
  DYND_API nd::array array_rw(unsigned long long value);
  DYND_API nd::array array_rw(const uint128 &value);
  DYND_API nd::array array_rw(float16 value);
  DYND_API nd::array array_rw(float value);
  DYND_API nd::array array_rw(double value);
  DYND_API nd::array array_rw(const float128 &value);
  DYND_API nd::array array_rw(dynd::complex<float> value);
  DYND_API nd::array array_rw(dynd::complex<double> value);
  DYND_API nd::array array_rw(std::complex<float> value);
  DYND_API nd::array array_rw(std::complex<double> value);
  DYND_API nd::array array_rw(const std::string &value);
  /** Construct a string from a NULL-terminated UTF8 string */
  DYND_API nd::array array_rw(const char *cstr);
  /** Construct a string from a UTF8 buffer and specified buffer size */
  DYND_API nd::array array_rw(const char *str, size_t size);
  /**
   * Constructs a scalar with the 'type' type.
   * NOTE: Does NOT create a scalar of the provided type,
   *       use dynd::empty(type) for that!
   */
  DYND_API nd::array array_rw(const ndt::type &tp);
  /**
   * Constructs a readwrite array from a C-style array.
   */
  template <class T, int N>
  nd::array array_rw(const T (&rhs)[N]);

  /**
   * This is a helper class for dealing with value assignment and collapsing
   * a view-based array into a strided array. Only the array class itself
   * is permitted to construct this helper object, and it is non-copyable.
   *
   * All that can be done is assigning the values of the referenced array
   * to another array, or assigning values from another array into the elements
   * the referenced array.
   */
  class DYND_API array_vals {
    const array &m_arr;
    array_vals(const array &arr) : m_arr(arr)
    {
    }

    // Non-copyable, not default-constructable
    array_vals(const array_vals &);
    array_vals &operator=(const array_vals &);

  public:
    /**
     * Assigns values from an array to the internally referenced array.
     * this does a val_assign with the default assignment error mode.
     */
    array_vals &operator=(const array &rhs)
    {
      m_arr.val_assign(rhs);
      return *this;
    }

    /** Does a value-assignment from the rhs C++ scalar. */
    template <class T>
    typename std::enable_if<is_dynd_scalar<T>::value, array_vals &>::type operator=(const T &rhs)
    {
      m_arr.val_assign(ndt::type::make<T>(), NULL, (const char *)&rhs);
      return *this;
    }

    // TODO: Could also do +=, -=, *=, etc.

    friend class array;
    friend array_vals array::vals() const;
  };

  /**
   * This is a helper class like array_vals, but it holds a reference
   * to the temporary array. This is needed by vals_at.
   *
   */
  class DYND_API array_vals_at {
    array m_arr;
    array_vals_at(const array &arr) : m_arr(arr)
    {
    }

    array_vals_at(array &&arr) : m_arr(std::move(arr))
    {
    }

    // Non-copyable, not default-constructable
    array_vals_at(const array_vals &);
    array_vals_at &operator=(const array_vals_at &);

  public:
    /**
     * Assigns values from an array to the internally referenced array.
     * this does a val_assign with the default assignment error mode.
     */
    array_vals_at &operator=(const array &rhs)
    {
      m_arr.val_assign(rhs);
      return *this;
    }

    /** Does a value-assignment from the rhs C++ scalar. */
    template <class T>
    typename std::enable_if<is_dynd_scalar<T>::value, array_vals_at &>::type operator=(const T &rhs)
    {
      m_arr.val_assign(ndt::type::make<T>(), NULL, (const char *)&rhs);
      return *this;
    }

    // TODO: Could also do +=, -=, *=, etc.

    friend class array;
    friend array_vals_at array::vals_at(const irange &) const;
    friend array_vals_at array::vals_at(const irange &, const irange &) const;
    friend array_vals_at array::vals_at(const irange &, const irange &, const irange &) const;
    friend array_vals_at array::vals_at(const irange &, const irange &, const irange &, const irange &) const;
  };

  /** Makes a strided array with uninitialized data. If axis_perm is NULL, it is
   * C-order */
  DYND_API array make_strided_array(const ndt::type &uniform_dtype, intptr_t ndim, const intptr_t *shape,
                                    int64_t access_flags = read_access_flag | write_access_flag,
                                    const int *axis_perm = NULL);

  /**
   * \brief Makes a strided array pointing to existing data
   *
   * \param uniform_dtype  The type of each element in the strided array.
   * \param ndim  The number of strided dimensions.
   * \param shape  The shape of the strided dimensions.
   * \param strides  The strides of the strided dimensions.
   * \param access_flags Read/write/immutable flags.
   * \param data_ptr  Pointer to the element at index 0.
   * \param data_reference  A memory block which holds a reference to the data.
   * \param out_uniform_arrmeta  If the uniform_dtype has arrmeta
   *(get_arrmeta_size() > 0),
   *                              this must be non-NULL, and is populated with a
   *pointer to the
   *                              arrmeta for the uniform_dtype. The caller must
   *populate it
   *                              with valid data.
   *
   * \returns  The created array.
   */
  DYND_API array make_strided_array_from_data(const ndt::type &uniform_dtype, intptr_t ndim, const intptr_t *shape,
                                              const intptr_t *strides, int64_t access_flags, char *data_ptr,
                                              const intrusive_ptr<memory_block_data> &data_reference,
                                              char **out_uniform_arrmeta = NULL);

  /** Makes a POD (plain old data) array with data initialized by the provided
   * pointer */
  DYND_API array make_pod_array(const ndt::type &pod_dt, const void *data);

  /** Makes an array of 'bytes' type from the data */
  DYND_API array make_bytes_array(const char *data, size_t len, size_t alignment = 1);

  DYND_API array make_string_array(const char *str, size_t len, string_encoding_t encoding, uint64_t access_flags);
  inline array make_ascii_array(const char *str, size_t len)
  {
    return make_string_array(str, len, string_encoding_ascii, nd::default_access_flags);
  }
  inline array make_utf8_array(const char *str, size_t len)
  {
    return make_string_array(str, len, string_encoding_utf_8, nd::default_access_flags);
  }
  inline array make_utf16_array(const uint16_t *str, size_t len)
  {
    return make_string_array(reinterpret_cast<const char *>(str), len * sizeof(uint16_t), string_encoding_utf_16,
                             nd::default_access_flags);
  }
  inline array make_utf32_array(const uint32_t *str, size_t len)
  {
    return make_string_array(reinterpret_cast<const char *>(str), len * sizeof(uint32_t), string_encoding_utf_32,
                             nd::default_access_flags);
  }

  template <int N>
  inline array make_ascii_array(const char (&static_string)[N])
  {
    return make_ascii_array(&static_string[0], N);
  }
  template <int N>
  inline array make_utf8_array(const char (&static_string)[N])
  {
    return make_utf8_array(&static_string[0], N);
  }
  template <int N>
  inline array make_utf8_array(const unsigned char (&static_string)[N])
  {
    return make_utf8_array(reinterpret_cast<const char *>(&static_string[0]), N);
  }
  template <int N>
  inline array make_utf16_array(const uint16_t (&static_string)[N])
  {
    return make_utf16_array(&static_string[0], N);
  }
  template <int N>
  inline array make_utf32_array(const uint32_t (&static_string)[N])
  {
    return make_utf32_array(&static_string[0], N);
  }

  /**
   * \brief Creates a strided array of strings.
   *
   * \param cstr_array  An array of NULL-terminated UTF8 strings.
   * \param array_size  The number of elements in `cstr_array`.
   *
   * \returns  An array of type "N * string".
   */
  DYND_API array make_strided_string_array(const char *const *cstr_array, size_t array_size);
  DYND_API array make_strided_string_array(const std::string **str_array, size_t array_size);

  inline array_vals array::vals() const
  {
    return array_vals(*this);
  }

  inline array_vals_at array::vals_at(const irange &i0) const
  {
    return array_vals_at(at_array(1, &i0, false));
  }

  inline array_vals_at array::vals_at(const irange &i0, const irange &i1) const
  {
    irange i[2] = {i0, i1};
    return array_vals_at(at_array(2, i, false));
  }

  inline array_vals_at array::vals_at(const irange &i0, const irange &i1, const irange &i2) const
  {
    irange i[3] = {i0, i1, i2};
    return array_vals_at(at_array(3, i, false));
  }

  inline array_vals_at array::vals_at(const irange &i0, const irange &i1, const irange &i2, const irange &i3) const
  {
    irange i[4] = {i0, i1, i2, i3};
    return array_vals_at(at_array(4, i, false));
  }

  // Some C array metaprogramming used by empty<> as well as assignment
  // from a C array
  namespace detail {
    template <class T>
    struct dtype_from_array {
      typedef T type;
      enum {
        element_size = sizeof(T)
      };
      enum {
        type_id = type_id_of<T>::value
      };
    };
    template <class T, int N>
    struct dtype_from_array<T[N]> {
      typedef typename dtype_from_array<T>::type type;
      enum {
        element_size = dtype_from_array<T>::element_size
      };
      enum {
        type_id = dtype_from_array<T>::type_id
      };
    };

    template <class T>
    struct ndim_from_array {
      enum {
        value = 0
      };
    };
    template <class T, int N>
    struct ndim_from_array<T[N]> {
      enum {
        value = ndim_from_array<T>::value + 1
      };
    };

    template <class T>
    struct fill_shape {
      inline static size_t fill(intptr_t *)
      {
        return sizeof(T);
      }
    };
    template <class T, int N>
    struct fill_shape<T[N]> {
      inline static size_t fill(intptr_t *out_shape)
      {
        out_shape[0] = N;
        return N * fill_shape<T>::fill(out_shape + 1);
      }
    };
  } // namespace detail

  /**
   * Constructs an uninitialized array of the given dtype. This is
   * the usual function to use for allocating such an array.
   */
  DYND_API array empty(const ndt::type &tp);

  /**
   * Constructs an uninitialized array with uninitialized arrmeta of the
   * given dtype. Default-sized space for data is allocated.
   *
   * IMPORTANT: You should use nd::empty normally. If you use this function,
   *            you must manually initialize the arrmeta as well.
   */
  DYND_API array empty_shell(const ndt::type &tp);

  /**
   * Constructs an uninitialized array of the given dtype, with ndim/shape
   * pointer. This function is not named ``empty`` because (intptr_t, intptr_t,
   * type) and (intptr_t, const intptr_t *, type) can sometimes result in
   * unexpected overloads.
   */
  inline array dtyped_empty(intptr_t ndim, const intptr_t *shape, const ndt::type &tp)
  {
    if (ndim > 0) {
      intptr_t i = ndim - 1;
      ndt::type rtp = shape[i] >= 0 ? ndt::make_fixed_dim(shape[i], tp) : ndt::var_dim_type::make(tp);
      while (i-- > 0) {
        rtp = shape[i] >= 0 ? ndt::make_fixed_dim(shape[i], rtp) : ndt::var_dim_type::make(rtp);
      }
      return empty(rtp);
    } else {
      return empty(tp);
    }
  }

  /**
   * A version of dtyped_empty that accepts a std::vector as the shape.
   */
  inline array dtyped_empty(const std::vector<intptr_t> &shape, const ndt::type &tp)
  {
    return dtyped_empty(shape.size(), shape.empty() ? NULL : &shape[0], tp);
  }

  /**
   * Constructs an uninitialized array of the given dtype,
   * specified as a string literal. This is a shortcut for expressions
   * like
   *
   * nd::array a = nd::empty("10 * int32");
   */
  template <int N>
  inline array empty(const char (&dshape)[N])
  {
    return nd::empty(ndt::type(dshape, dshape + N - 1));
  }

  /**
   * Constructs a writable uninitialized array of the specified shape
   * and dtype. Prefixes the dtype with ``strided`` or ``var`` dimensions.
   */
  inline array empty(intptr_t dim0, const ndt::type &tp)
  {
    return nd::empty(dim0 >= 0 ? ndt::make_fixed_dim(dim0, tp) : ndt::var_dim_type::make(tp));
  }

  /**
   * Constructs an uninitialized array of the given type,
   * specified as a string. This is a shortcut for expressions
   * like
   *
   *      array a = nd::empty(10, "int32");
   */
  template <int N>
  inline array empty(intptr_t dim0, const char (&dshape)[N])
  {
    return empty(dim0, ndt::type(dshape, dshape + N - 1));
  }

  /**
   * Constructs a writable uninitialized array of the specified shape
   * and dtype. Prefixes the dtype with ``strided`` or ``var`` dimensions.
   */
  inline array empty(intptr_t dim0, intptr_t dim1, const ndt::type &tp)
  {
    ndt::type rtp = (dim1 >= 0) ? ndt::make_fixed_dim(dim1, tp) : ndt::var_dim_type::make(tp);
    rtp = (dim0 >= 0) ? ndt::make_fixed_dim(dim0, rtp) : ndt::var_dim_type::make(rtp);
    return nd::empty(rtp);
  }

  /**
   * Constructs an uninitialized array of the given type,
   * specified as a string. This is a shortcut for expressions
   * like
   *
   *      array a = nd::empty(10, 10, "int32");
   */
  template <int N>
  inline array empty(intptr_t dim0, intptr_t dim1, const char (&dshape)[N])
  {
    return empty(dim0, dim1, ndt::type(dshape, dshape + N - 1));
  }

  /**
   * Constructs a writable uninitialized array of the specified type.
   * This type should be at least three dimensional, and is initialized
   * using the specified dimension sizes.
   */
  inline array empty(intptr_t dim0, intptr_t dim1, intptr_t dim2, const ndt::type &tp)
  {
    ndt::type rtp = (dim2 >= 0) ? ndt::make_fixed_dim(dim2, tp) : ndt::var_dim_type::make(tp);
    rtp = (dim1 >= 0) ? ndt::make_fixed_dim(dim1, rtp) : ndt::var_dim_type::make(rtp);
    rtp = (dim0 >= 0) ? ndt::make_fixed_dim(dim0, rtp) : ndt::var_dim_type::make(rtp);
    return empty(rtp);
  }

  /**
   * Constructs an uninitialized array of the given type,
   * specified as a string. This is a shortcut for expressions
   * like
   *
   *      array a = nd::empty(10, 10, 10, "int32");
   */
  template <int N>
  inline array empty(intptr_t dim0, intptr_t dim1, intptr_t dim2, const char (&dshape)[N])
  {
    return empty(dim0, dim1, dim2, ndt::type(dshape, dshape + N - 1));
  }

  /**
   * Constructs an uninitialized array of the given C++ type, as a strided
   *array.
   *
   * // a has type "float64"
   * array a = nd::empty<double>();
   * // b has type "3 * 4 * float64", and is in C order
   * array b = nd::empty<double[3][4]>();
   */
  template <typename T>
  array empty()
  {
    return empty(ndt::type::make<T>());
  }

  /**
   * Constructs an array with the same shape and memory layout
   * of the one given, but replacing the
   *
   * \param rhs  The array whose shape and memory layout to emulate.
   * \param uniform_dtype   The array data type of the new array.
   */
  DYND_API array empty_like(const array &rhs, const ndt::type &uniform_dtype);

  /**
   * Constructs an empty array matching the parameters of 'rhs'
   *
   * \param rhs  The array whose shape, memory layout, and dtype to emulate.
   */
  DYND_API array empty_like(const array &rhs);

  /**
   * Constructs an array, with each element initialized to 0, of the given
   * dtype,
   * with ndim/shape pointer.
   */
  inline array dtyped_zeros(intptr_t ndim, const intptr_t *shape, const ndt::type &tp)
  {
    nd::array res = dtyped_empty(ndim, shape, tp);
    res.val_assign(0);

    return res;
  }

  inline array zeros(intptr_t dim0, const ndt::type &tp)
  {
    intptr_t shape[1] = {dim0};

    return dtyped_zeros(1, shape, tp);
  }
  inline array zeros(intptr_t dim0, intptr_t dim1, const ndt::type &tp)
  {
    intptr_t shape[2] = {dim0, dim1};

    return dtyped_zeros(2, shape, tp);
  }

  /**
   * Primitive function to construct an nd::array with each element initialized
   * to 1.
   * In this function, the type provided is the complete type of the array
   * result, not just its dtype.
   */
  DYND_API array typed_ones(intptr_t ndim, const intptr_t *shape, const ndt::type &tp);

  /**
   * A version of typed_ones that accepts a std::vector as the shape.
   */
  inline array ones(const std::vector<intptr_t> &shape, const ndt::type &tp)
  {
    return typed_ones(shape.size(), shape.empty() ? NULL : &shape[0], tp);
  }

  /**
   * Constructs an array, with each element initialized to 1, of the given
   * dtype,
   * with ndim/shape pointer.
   */
  inline array dtyped_ones(intptr_t ndim, const intptr_t *shape, const ndt::type &tp)
  {
    nd::array res = dtyped_empty(ndim, shape, tp);
    res.val_assign(1);

    return res;
  }

  inline array ones(intptr_t dim0, const ndt::type &tp)
  {
    intptr_t shape[1] = {dim0};

    return dtyped_ones(1, shape, tp);
  }
  inline array ones(intptr_t dim0, intptr_t dim1, const ndt::type &tp)
  {
    intptr_t shape[2] = {dim0, dim1};

    return dtyped_ones(2, shape, tp);
  }

  /**
   * This concatenates two 1D arrays. It is really just a placeholder until a
   * proper
   * concatenate is written. It shouldn't be used unless you absolutely know
   * what you
   * are doing. It needs to be implemented properly.
   */
  DYND_API array concatenate(const nd::array &x, const nd::array &y);

  /**
   * Reshapes an array into the new shape. This is currently a prototype and
   * should only be used
   * with contiguous arrays of built-in dtypes.
   */
  DYND_API array reshape(const array &a, const array &shape);

  /**
   * Reshapes an array into the new shape. This is currently a prototype and
   * should only be used
   * with contiguous arrays of built-in dtypes.
   */
  inline array reshape(const array &a, intptr_t ndim, const intptr_t *shape)
  {
    return reshape(a, nd::array(shape, ndim));
  }

  ///////////// Initializer list constructor implementation
  ////////////////////////////
  namespace detail {
    // Computes the number of dimensions in a nested initializer list
    // constructor
    template <class T>
    struct initializer_list_ndim {
      static const int value = 0;
    };
    template <class T>
    struct initializer_list_ndim<std::initializer_list<T>> {
      static const int value = initializer_list_ndim<T>::value + 1;
    };

    // Computes the array type of a nested initializer list constructor
    template <class T>
    struct initializer_list_type {
      typedef T type;
    };
    template <class T>
    struct initializer_list_type<std::initializer_list<T>> {
      typedef typename initializer_list_type<T>::type type;
    };

    // Gets the shape of the nested initializer list constructor, and validates
    // that
    // it isn't ragged
    template <class T>
    struct initializer_list_shape;
    // Base case, an initializer list parameterized by a non-initializer list
    template <class T>
    struct initializer_list_shape<std::initializer_list<T>> {
      static void compute(intptr_t *out_shape, const std::initializer_list<T> &il)
      {
        out_shape[0] = il.size();
      }
      static void validate(const intptr_t *shape, const std::initializer_list<T> &il)
      {
        if ((intptr_t)il.size() != shape[0]) {
          throw std::runtime_error("initializer list for array is ragged, must be "
                                   "nested in a regular fashion");
        }
      }
      static void copy_data(T **dataptr, const std::initializer_list<T> &il)
      {
        DYND_MEMCPY(*dataptr, il.begin(), il.size() * sizeof(T));
        *dataptr += il.size();
      }
    };
    // Recursive case, an initializer list parameterized by an initializer list
    template <class T>
    struct initializer_list_shape<std::initializer_list<std::initializer_list<T>>> {
      static void compute(intptr_t *out_shape, const std::initializer_list<std::initializer_list<T>> &il)
      {
        out_shape[0] = il.size();
        if (out_shape[0] > 0) {
          // Recursively compute the rest of the shape
          initializer_list_shape<std::initializer_list<T>>::compute(out_shape + 1, *il.begin());
          // Validate the shape for the nested initializer lists
          for (auto i = il.begin() + 1; i != il.end(); ++i) {
            initializer_list_shape<std::initializer_list<T>>::validate(out_shape + 1, *i);
          }
        }
      }
      static void validate(const intptr_t *shape, const std::initializer_list<std::initializer_list<T>> &il)
      {
        if ((intptr_t)il.size() != shape[0]) {
          throw std::runtime_error("initializer list for array is ragged, must be "
                                   "nested in a regular fashion");
        }
        // Validate the shape for the nested initializer lists
        for (auto i = il.begin(); i != il.end(); ++i) {
          initializer_list_shape<std::initializer_list<T>>::validate(shape + 1, *i);
        }
      }
      static void copy_data(typename initializer_list_type<T>::type **dataptr,
                            const std::initializer_list<std::initializer_list<T>> &il)
      {
        for (auto i = il.begin(); i != il.end(); ++i) {
          initializer_list_shape<std::initializer_list<T>>::copy_data(dataptr, *i);
        }
      }
    };
  } // namespace detail

  // Implementation of initializer list construction
  template <class T>
  dynd::nd::array::array(const std::initializer_list<T> &il)
  {
    intptr_t dim0 = il.size();
    make_strided_array(ndt::type::make<T>(), 1, &dim0, nd::default_access_flags, NULL).swap(*this);
    DYND_MEMCPY(get()->data, il.begin(), sizeof(T) * dim0);
  }
  template <class T>
  dynd::nd::array::array(const std::initializer_list<std::initializer_list<T>> &il)
  {
    typedef std::initializer_list<std::initializer_list<T>> S;
    intptr_t shape[2];

    // Get and validate that the shape is regular
    detail::initializer_list_shape<S>::compute(shape, il);
    make_strided_array(ndt::type::make<T>(), 2, shape, nd::default_access_flags, NULL).swap(*this);
    T *dataptr = reinterpret_cast<T *>(get()->data);
    detail::initializer_list_shape<S>::copy_data(&dataptr, il);
  }
  template <class T>
  dynd::nd::array::array(const std::initializer_list<std::initializer_list<std::initializer_list<T>>> &il)
  {
    typedef std::initializer_list<std::initializer_list<std::initializer_list<T>>> S;
    intptr_t shape[3];

    // Get and validate that the shape is regular
    detail::initializer_list_shape<S>::compute(shape, il);
    make_strided_array(ndt::type::make<T>(), 3, shape, nd::default_access_flags, NULL).swap(*this);
    T *dataptr = reinterpret_cast<T *>(get()->data);
    detail::initializer_list_shape<S>::copy_data(&dataptr, il);
  }

  inline dynd::nd::array::array(const std::initializer_list<const char *> &il)
  {
    make_strided_string_array(il.begin(), il.size()).swap(*this);
  }

  inline dynd::nd::array::array(const std::initializer_list<ndt::type> &il)
  {
    intptr_t dim0 = il.size();
    make_strided_array(ndt::make_type(), 1, &dim0, nd::default_access_flags, NULL).swap(*this);
    auto data_ptr = reinterpret_cast<ndt::type *>(get()->data);
    for (intptr_t i = 0; i < dim0; ++i) {
      data_ptr[i] = *(il.begin() + i);
    }
  }

  inline dynd::nd::array::array(const std::initializer_list<bool> &il)
  {
    intptr_t dim0 = il.size();
    make_strided_array(ndt::type::make<bool>(), 1, &dim0, nd::default_access_flags, NULL).swap(*this);
    auto data_ptr = reinterpret_cast<bool1 *>(get()->data);
    for (intptr_t i = 0; i < dim0; ++i) {
      data_ptr[i] = *(il.begin() + i);
    }
  }

  ///////////// C-style array constructor implementation
  ////////////////////////////

  template <class T, int N>
  nd::array::array(const T (&rhs)[N])
  {
    const int ndim = detail::ndim_from_array<T[N]>::value;
    intptr_t shape[ndim];
    size_t size = detail::fill_shape<T[N]>::fill(shape);

    make_strided_array(ndt::type(static_cast<type_id_t>(detail::dtype_from_array<T>::type_id)), ndim, shape,
                       default_access_flags, NULL).swap(*this);
    DYND_MEMCPY(get()->data, reinterpret_cast<const void *>(&rhs), size);
  }

  // Temporarily removed due to conflicting dll linkage with earlier versions of this function.
  template <class T, int N>
  nd::array array_rw(const T (&rhs)[N])
  {
    const int ndim = detail::ndim_from_array<T[N]>::value;
    intptr_t shape[ndim];
    size_t size = detail::fill_shape<T[N]>::fill(shape);

    nd::array result = make_strided_array(ndt::type(static_cast<type_id_t>(detail::dtype_from_array<T>::type_id)), ndim,
                                          shape, readwrite_access_flags, NULL);
    DYND_MEMCPY(result.get()->data, reinterpret_cast<const void *>(&rhs), size);
    return result;
  }

  template <int N>
  nd::array::array(const ndt::type (&rhs)[N])
  {
    nd::empty(N, ndt::make_type()).swap(*this);
    ndt::type *out = reinterpret_cast<ndt::type *>(get()->data);
    for (int i = 0; i < N; ++i) {
      out[i] = rhs[i];
    }
    flag_as_immutable();
  }

  template <int N>
  inline nd::array::array(const char (&rhs)[N])
  {
    make_string_array(rhs, N, string_encoding_utf_8, nd::default_access_flags).swap(*this);
  }

  template <int N>
  inline nd::array::array(const char *(&rhs)[N])
  {
    make_strided_string_array(rhs, N).swap(*this);
  }

  template <int N>
  inline nd::array::array(const std::string *(&rhs)[N])
  {
    make_strided_string_array(rhs, N).swap(*this);
  }

  template <class T>
  inline nd::array::array(const T *rhs, intptr_t dim_size)
  {
    nd::empty(dim_size, ndt::type::make<T>()).swap(*this);
    DYND_MEMCPY(get()->data, reinterpret_cast<const void *>(&rhs), dim_size * sizeof(T));
  }

  inline nd::array::array(const ndt::type *rhs, intptr_t dim_size)
  {
    nd::empty(dim_size, ndt::make_type()).swap(*this);
    auto lhs = reinterpret_cast<ndt::type *>(get()->data);
    for (intptr_t i = 0; i < dim_size; ++i) {
      lhs[i] = rhs[i];
    }
  }

  ///////////// std::vector constructor implementation /////////////////////////
  namespace detail {
    template <class T>
    struct make_from_vec {
      inline static typename std::enable_if<is_dynd_scalar<T>::value, array>::type make(const std::vector<T> &vec)
      {
        array result = nd::empty(vec.size(), ndt::type::make<T>());
        if (!vec.empty()) {
          DYND_MEMCPY(result.data(), &vec[0], vec.size() * sizeof(T));
        }
        return result;
      }
    };

    template <>
    struct make_from_vec<ndt::type> {
      static DYND_API array make(const std::vector<ndt::type> &vec);
    };

    template <>
    struct make_from_vec<std::string> {
      static DYND_API array make(const std::vector<std::string> &vec);
    };
  } // namespace detail

  template <class T>
  array::array(const std::vector<T> &vec)
  {
    detail::make_from_vec<T>::make(vec).swap(*this);
  }

  ///////////// The array.as<type>() templated function
  ////////////////////////////
  namespace detail {
    template <class T>
    struct array_as_helper {
      inline static typename std::enable_if < is_dynd_scalar<T>::value || is_dynd_scalar_pointer<T>::value,
          T > ::type as(const array &lhs, const eval::eval_context *ectx)
      {
        T result;
        if (!lhs.is_scalar()) {
          throw std::runtime_error("can only convert arrays with 0 dimensions to scalars");
        }
        typed_data_assign(ndt::type::make<T>(), NULL, (char *)&result, lhs.get_type(), lhs.get()->metadata(),
                          lhs.get()->data, ectx);
        return result;
      }
    };

    template <>
    struct array_as_helper<bool> {
      inline static bool as(const array &lhs, const eval::eval_context *ectx)
      {
        return static_cast<bool>(array_as_helper<bool1>::as(lhs, ectx));
      }
    };

    DYND_API std::string array_as_string(const array &lhs, assign_error_mode errmode);
    DYND_API ndt::type array_as_type(const array &lhs);

    template <>
    struct array_as_helper<std::string> {
      static std::string as(const array &lhs, const eval::eval_context *ectx)
      {
        return array_as_string(lhs, ectx->errmode);
      }
    };

    template <>
    struct array_as_helper<ndt::type> {
      static ndt::type as(const array &lhs, const eval::eval_context *DYND_UNUSED(ectx))
      {
        return array_as_type(lhs);
      }
    };

    // Could do as<std::vector<T>> for 1D arrays, and other similiar conversions
  } // namespace detail;

  template <class T>
  T array::as(assign_error_mode errmode) const
  {
    if (errmode == assign_error_default || errmode == eval::default_eval_context.errmode) {
      return detail::array_as_helper<T>::as(*this, &eval::default_eval_context);
    } else {
      eval::eval_context tmp_ectx(eval::default_eval_context);
      tmp_ectx.errmode = errmode;
      return detail::array_as_helper<T>::as(*this, &tmp_ectx);
    }
  }

  /**
   * Given the type/arrmeta/data of an array (or sub-component of an array),
   * evaluates a new copy of it as the canonical type.
   */
  DYND_API array eval_raw_copy(const ndt::type &dt, const char *arrmeta, const char *data);

  /**
   * Memory-maps a file with dynd type 'bytes'.
   *
   * \param filename  The name of the file to memory map.
   * \param begin  If provided, the start of where to memory map. Uses
   *               Python semantics for out of bounds and negative values.
   * \param end  If provided, the end of where to memory map. Uses
   *             Python semantics for out of bounds and negative values.
   * \param access  The access permissions with which to open the file.
   */
  DYND_API array memmap(const std::string &filename, intptr_t begin = 0,
                        intptr_t end = std::numeric_limits<intptr_t>::max(), uint32_t access = default_access_flags);

  DYND_API bool is_scalar_avail(const ndt::type &tp, const char *arrmeta, const char *data,
                                const eval::eval_context *ectx);

  /**
   * Returns true if the array is a scalar whose value is available (not NA).
   */
  inline bool is_scalar_avail(const array &arr, const eval::eval_context *ectx = &eval::default_eval_context)
  {
    return is_scalar_avail(arr.get_type(), arr.get()->metadata(), arr.cdata(), ectx);
  }

  DYND_API void assign_na(const ndt::type &tp, const char *arrmeta, char *data, const eval::eval_context *ectx);

  inline void assign_na(array &out, const eval::eval_context *ectx = &eval::default_eval_context)
  {
    assign_na(out.get_type(), out.get()->metadata(), out.data(), ectx);
  }

  /**
   * Creates a ctuple nd::array with the given field names and
   * pointers to the provided field values.
   *
   * \param  field_count  The number of fields.
   * \param  field_values  The values of the fields.
   */
  DYND_API array combine_into_tuple(size_t field_count, const array *field_values);

  /**
   * Packs a value into memory allocated to store it via the ``make_type(val)``
   * call. Because the destination arrmeta is guaranteed to be for only one
   * data element
   */
  template <typename T>
  void forward_as_array(const ndt::type &DYND_UNUSED(tp), char *DYND_UNUSED(arrmeta), char *data, const T &val)
  {
    *reinterpret_cast<T *>(data) = val;
  }

  DYND_API void forward_as_array(const ndt::type &tp, char *arrmeta, char *out_data, const nd::array &val);

  template <typename T>
  void forward_as_array(const ndt::type &tp, char *arrmeta, char *data, const std::vector<T> &val)
  {
    if (tp.get_type_id() == pointer_type_id) {
      forward_as_array(tp, arrmeta, data, array(val));
    } else {
      if (!tp.is_builtin()) {
        tp.extended()->arrmeta_default_construct(arrmeta, true);
      }
      if (!val.empty()) {
        memcpy(data, &val[0], sizeof(T) * val.size());
      }
    }
  }

} // namespace dynd::nd
} // namespace dynd
