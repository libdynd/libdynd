//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__NDOBJECT_HPP_
#define _DYND__NDOBJECT_HPP_

#include <iostream> // FOR DEBUG
#include <stdexcept>
#include <string>

#include <dynd/config.hpp>

#include <dynd/dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/shortvector.hpp>
#include <dynd/irange.hpp>
#include <dynd/eval/eval_engine.hpp>
#include <dynd/memblock/ndobject_memory_block.hpp>

namespace dynd {

class ndobject;

/** Stream printing function */
std::ostream& operator<<(std::ostream& o, const ndobject& rhs);

class ndobject_vals;

/**
 * This is the primary multi-dimensional array class.
 */
class ndobject {
    /**
     * The ndobject class is a wrapper around an ndobject_memory_block, which
     * contains metadata as described by the dtype.
     */
    memory_block_ptr m_memblock;

    // Don't allow implicit construction from a raw pointer
    ndobject(const void *);
public:
    /** Constructs an array with no buffer (NULL state) */
    ndobject();
    /**
     * Constructs a zero-dimensional scalar array from a C++ scalar.
     *
     * TODO: Figure out why enable_if with is_dtype_scalar didn't work for this constructor
     *       in g++ 4.6.0.
     */
    ndobject(dynd_bool value);
    ndobject(bool value);
    ndobject(signed char value);
    ndobject(short value);
    ndobject(int value);
    ndobject(long value);
    ndobject(long long value);
    ndobject(unsigned char value);
    ndobject(unsigned short value);
    ndobject(unsigned int value);
    ndobject(unsigned long value);
    ndobject(unsigned long long value);
    ndobject(float value);
    ndobject(double value);
    ndobject(std::complex<float> value);
    ndobject(std::complex<double> value);
    ndobject(const std::string& value);

    /**
     * Constructs an array from a multi-dimensional C-style array.
     */
    template<class T, int N>
    ndobject(const T (&rhs)[N]);

    explicit ndobject(const memory_block_ptr& ndobj_memblock)
        : m_memblock(ndobj_memblock)
    {
        if (m_memblock.get()->m_type != ndobject_memory_block_type) {
            throw std::runtime_error("ndobject can only be constructed from a memblock with ndobject type");
        }
    }

    void set(const memory_block_ptr& ndobj_memblock)
    {
        if (ndobj_memblock.get()->m_type != ndobject_memory_block_type) {
            throw std::runtime_error("ndobject can only be constructed from a memblock with ndobject type");
        }
        m_memblock = ndobj_memblock;
    }

    /**
     * Constructs a writeable uninitialized ndobject of the specified dtype.
     * This dtype should be scalar or already fully specify the datashape.
     */
    explicit ndobject(const dtype& dt);
    /**
     * Constructs a writeable uninitialized ndobject of the specified dtype.
     * This dtype should be at least one dimensional, and is initialized
     * using the specified dimension size.
     */
    ndobject(const dtype& dt, intptr_t dim0);
    /**
     * Constructs a writeable uninitialized ndobject of the specified dtype.
     * This dtype should be at least two dimensional, and is initialized
     * using the specified dimension size.
     */
    ndobject(const dtype& dt, intptr_t dim0, intptr_t dim1);
    /**
     * Constructs a writeable uninitialized ndobject of the specified dtype.
     * This dtype should be at least three dimensional, and is initialized
     * using the specified dimension size.
     */
    ndobject(const dtype& dt, intptr_t dim0, intptr_t dim1, intptr_t dim2);

    // TODO: Copy the initializer list and C array constructor mechanisms from ndarray

    /** Swap operation (should be "noexcept" in C++11) */
    void swap(ndobject& rhs);

    /**
     * Assignment operator (should be just "= default" in C++11).
     * Copies with reference semantics.
     */
    inline ndobject& operator=(const ndobject& rhs) {
        m_memblock = rhs.m_memblock;
        return *this;
    }
#ifdef DYND_RVALUE_REFS
    /** Move assignment operator (should be just "= default" in C++11) */
    inline ndobject& operator=(ndobject&& rhs) {
        m_memblock = DYND_MOVE(rhs.m_memblock);

        return *this;
    }
#endif // DYND_RVALUE_REFS

    /**
     * Assignment operator from an ndobject_vals object. This converts the
     * array 'rhs' has a reference to into a strided array, evaluating
     * it from the expression tree if necessary. If 'rhs' contains a
     * strided array, this copies it by reference, use the function 'copy'
     * when a copy is required.
     */
    ndobject& operator=(const ndobject_vals& rhs);

    /** Low level access to the reference-counted memory */
    inline memory_block_ptr get_memblock() const {
        return m_memblock;
    }

    /** Low level access to the ndobject preamble */
    inline const ndobject_preamble *get_ndo() const {
        return reinterpret_cast<const ndobject_preamble *>(m_memblock.get());
    }

    /** Low level access to the ndobject preamble */
    inline ndobject_preamble *get_ndo() {
        return reinterpret_cast<ndobject_preamble *>(m_memblock.get());
    }

    /** Low level access to the ndobject metadata */
    inline const char *get_ndo_meta() const {
        return reinterpret_cast<const char *>(get_ndo() + 1);
    }

    /** Low level access to the ndobject metadata */
    inline char *get_ndo_meta() {
        return reinterpret_cast<char *>(get_ndo() + 1);
    }

    inline char *get_readwrite_originptr() const {
        if (get_ndo()->m_flags & write_access_flag) {
            return get_ndo()->m_data_pointer;
        } else {
            throw std::runtime_error("dynd::ndarray node is not writeable");
        }
    }

    inline const char *get_readonly_originptr() const {
        return get_ndo()->m_data_pointer;
    }

    /** Returns true if the object is a scalar */
    inline bool is_scalar() const {
        return get_ndo()->is_builtin_dtype() ||
            get_ndo()->m_dtype->is_scalar(get_ndo()->m_data_pointer, get_ndo_meta());
    }

    /** The number of uniform dimensions */
    inline int get_uniform_ndim() const {
        if (get_ndo()->is_builtin_dtype()) {
            return 0;
        } else {
            return get_ndo()->m_dtype->get_uniform_ndim();
        }
    }

    /** The dtype */
    inline dtype get_dtype() const {
        if (get_ndo()->is_builtin_dtype()) {
            return dtype(get_ndo()->get_builtin_type_id());
        } else {
            return dtype(get_ndo()->m_dtype, true);
        }
    }

    /** The uniform dtype, which has all initial uniform dimensions stripped away */
    inline dtype get_uniform_dtype() const {
        if (get_ndo()->is_builtin_dtype()) {
            return dtype(get_ndo()->get_builtin_type_id());
        } else {
            return get_ndo()->m_dtype->get_uniform_dtype();
        }
    }

    /** The flags, including access permissions. */
    inline int64_t get_flags() const {
        return get_ndo()->m_flags;
    }

    inline std::vector<intptr_t> get_shape() const {
        std::vector<intptr_t> result(get_uniform_ndim());
        get_shape(&result[0]);
        return result;
    }
    inline void get_shape(intptr_t *out_shape) const {
        if (!get_ndo()->is_builtin_dtype()) {
            get_ndo()->m_dtype->get_shape(0, out_shape, get_ndo()->m_data_pointer, get_ndo_meta());
        }
    }

    std::vector<intptr_t> get_strides() const {
        std::vector<intptr_t> result(get_uniform_ndim());
        get_strides(&result[0]);
        return result;
    }
    inline void get_strides(intptr_t *out_strides) const {
        if (!get_ndo()->is_builtin_dtype()) {
            get_ndo()->m_dtype->get_strides(0, out_strides, get_ndo()->m_data_pointer, get_ndo_meta());
        }
    }

    /**
     * Returns a value-exposing helper object, which allows one to assign to
     * the values of the ndobject, or collapse the expression tree of the
     * ndobject into a strided array.
     */
    ndobject_vals vals() const;

    /**
     * Evaluates the ndobject node into an immutable strided array, or
     * returns it untouched if it is already both immutable and strided.
     */
    ndobject eval_immutable(const eval::eval_context *ectx = &eval::default_eval_context) const;

    /**
     * Evaluates the ndobject node into a newly allocated strided array,
     * with the requested access flags.
     *
     * @param access_flags  The access flags for the result, default read and write.
     */
    ndobject eval_copy(const eval::eval_context *ectx = &eval::default_eval_context,
                        uint32_t access_flags=read_access_flag|write_access_flag) const;

    /**
     * Returnas a view of the array as the dtype's storage_dtype, peeling
     * away any expression dtypes or encodings.
     */
    ndobject storage() const;

    /**
     * General irange-based indexing operation.
     */
    ndobject at_array(int nindices, const irange *indices) const;

    /**
     * The 'at' function is used for indexing. Overloading operator[] isn't
     * practical for multidimensional objects.
     */
    const ndobject at(const irange& i0) const {
        return at_array(1, &i0);
    }

    /** Indexing with two index values */
    const ndobject at(const irange& i0, const irange& i1) const {
        irange i[2] = {i0, i1};
        return at_array(2, i);
    }

    /** Indexing with three index values */
    const ndobject at(const irange& i0, const irange& i1, const irange& i2) const {
        irange i[3] = {i0, i1, i2};
        return at_array(3, i);
    }
    /** Indexing with four index values */
    const ndobject at(const irange& i0, const irange& i1, const irange& i2, const irange& i3) const {
        irange i[4] = {i0, i1, i2, i3};
        return at_array(4, i);
    }

    /** Does a value-assignment from the rhs array. */
    void val_assign(const ndobject& rhs, assign_error_mode errmode = assign_error_default,
                        const eval::eval_context *ectx = &eval::default_eval_context) const;
    /** Does a value-assignment from the rhs raw scalar */
    void val_assign(const dtype& dt, const char *data, assign_error_mode errmode = assign_error_default,
                        const eval::eval_context *ectx = &eval::default_eval_context) const;

    /**
     * Converts all the scalar dtypes of the array into the specified scalar dtype.
     */
    ndobject cast_scalars(const dtype& scalar_dtype, assign_error_mode errmode = assign_error_default) const;

    /**
     * Converts the array into the specified explicit scalar dtype.
     * For example, arr.cast_scalar<float>().
     */
    template<class T>
    ndobject cast_scalars(assign_error_mode errmode = assign_error_default) const {
        return cast_scalars(make_dtype<T>(), errmode);
    }

    /**
     * Views the array's memory as another dtype, where such an operation
     * makes sense. This is analogous to reinterpret_cast<>.
     */
    ndobject view_as_scalar(const dtype& scalar_dtype) const;

    /**
     * Views the array's memory as another dtype, where such an operation
     * makes sense. This is analogous to reinterpret_case<>.
     */
    template<class T>
    ndobject view_as_scalar() const {
        return view_as_scalar(make_dtype<T>());
    }

    /**
     * When this is a zero-dimensional array, converts it to a C++ scalar of the
     * requested template type. This function may be extended in the future for
     * 1D vectors (as<std::vector<T>>), matrices, etc.
     *
     * @param errmode  The assignment error mode to use.
     */
    template<class T>
    T as(assign_error_mode errmode = assign_error_default) const;

    bool equals_exact(const ndobject& rhs) const;

    void debug_dump(std::ostream& o, const std::string& indent = "") const;

    friend std::ostream& operator<<(std::ostream& o, const ndobject& rhs);
    friend class ndobject_vals;
};

ndobject operator+(const ndobject& op0, const ndobject& op1);
ndobject operator-(const ndobject& op0, const ndobject& op1);
ndobject operator/(const ndobject& op0, const ndobject& op1);
ndobject operator*(const ndobject& op0, const ndobject& op1);

/**
 * This is a helper class for dealing with value assignment and collapsing
 * a view-based ndobject into a strided array. Only the ndobject class itself
 * is permitted to construct this helper object, and it is non-copyable.
 *
 * All that can be done is assigning the values of the referenced array
 * to another array, or assigning values from another array into the elements
 * the referenced array.
 */
class ndobject_vals {
    const ndobject& m_arr;
    ndobject_vals(const ndobject& arr)
        : m_arr(arr) {
    }

    // Non-copyable, not default-constructable
    ndobject_vals(const ndobject_vals&);
    ndobject_vals& operator=(const ndobject_vals&);
public:
    /**
     * Assigns values from an array to the internally referenced array.
     * this does a val_assign with the default assignment error mode.
     */
    ndobject_vals& operator=(const ndobject& rhs) {
        m_arr.val_assign(rhs);
        return *this;
    }

    /** Does a value-assignment from the rhs C++ scalar. */
    template<class T>
    typename enable_if<is_dtype_scalar<T>::value, ndobject_vals&>::type operator=(const T& rhs) {
        m_arr.val_assign(make_dtype<T>(), (const char *)&rhs);
        return *this;
    }
    /**
     * Does a value-assignment from the rhs C++ boolean scalar.
     *
     * By default, many things are convertible to bool, and this will cause
     * screwed up assignments if we accept any such thing. Thus, we use
     * enable_if to only allow bools here instead of just accepting "const bool&"
     * as would seem obvious.
     */
    template<class T>
    typename enable_if<is_type_bool<T>::value, ndobject_vals&>::type  operator=(const T& rhs) {
        dynd_bool v = rhs;
        m_arr.val_assign(make_dtype<dynd_bool>(), (const char *)&v);
        return *this;
    }

    // TODO: Could also do +=, -=, *=, etc.

    // Can implicitly convert to an ndobject, by collapsing to a strided array
    operator ndobject() const {
        return ndobject();//TODO ndobject(eval::evaluate(m_arr.m_memblock.get()));
    }

    friend class ndobject;
    friend ndobject_vals ndobject::vals() const;
};

/** Makes a strided ndobject with uninitialized data. If axis_perm is NULL, it is C-order */
ndobject make_strided_ndobject(const dtype& uniform_dtype, int ndim, const intptr_t *shape, const int *axis_perm = NULL);

inline ndobject make_strided_ndobject(const dtype& uniform_dtype, intptr_t shape0) {
    return make_strided_ndobject(uniform_dtype, 1, &shape0, NULL);
}
inline ndobject make_strided_ndobject(const dtype& uniform_dtype, intptr_t shape0, intptr_t shape1) {
    intptr_t shape[2] = {shape0, shape1};
    return make_strided_ndobject(uniform_dtype, 2, shape, NULL);
}
inline ndobject make_strided_ndobject(const dtype& uniform_dtype, intptr_t shape0, intptr_t shape1, intptr_t shape2) {
    intptr_t shape[3] = {shape0, shape1, shape2};
    return make_strided_ndobject(uniform_dtype, 3, shape, NULL);
}

inline ndobject_vals ndobject::vals() const {
    return ndobject_vals(*this);
}

inline ndobject& ndobject::operator=(const ndobject_vals& rhs) {
    //TODO m_memblock = eval::evaluate(rhs.m_arr.m_memblock.get_memblock());
    return *this;
}

///////////// Initializer list constructor implementation /////////////////////////
#ifdef DYND_INIT_LIST
namespace detail {
    // Computes the number of dimensions in a nested initializer list constructor
    template<class T>
    struct initializer_list_ndim {static const int value = 0;};
    template<class T>
    struct initializer_list_ndim<std::initializer_list<T> > {
        static const int value = initializer_list_ndim<T>::value + 1;
    };

    // Computes the array dtype of a nested initializer list constructor
    template<class T>
    struct initializer_list_dtype {typedef T type;};
    template<class T>
    struct initializer_list_dtype<std::initializer_list<T> > {
        typedef typename initializer_list_dtype<T>::type type;
    };

    // Gets the shape of the nested initializer list constructor, and validates that
    // it isn't ragged
    template <class T>
    struct initializer_list_shape;
    // Base case, an initializer list parameterized by a non-initializer list
    template<class T>
    struct initializer_list_shape<std::initializer_list<T> > {
        static void compute(intptr_t *out_shape, const std::initializer_list<T>& il) {
            out_shape[0] = il.size();
        }
        static void validate(const intptr_t *shape, const std::initializer_list<T>& il) {
            if ((intptr_t)il.size() != shape[0]) {
                throw std::runtime_error("initializer list for ndobject is ragged, must be "
                                        "nested in a regular fashion");
            }
        }
        static void copy_data(T **dataptr, const std::initializer_list<T>& il) {
            DYND_MEMCPY(*dataptr, il.begin(), il.size() * sizeof(T));
            *dataptr += il.size();
        }
    };
    // Recursive case, an initializer list parameterized by an initializer list
    template<class T>
    struct initializer_list_shape<std::initializer_list<std::initializer_list<T> > > {
        static void compute(intptr_t *out_shape,
                        const std::initializer_list<std::initializer_list<T> >& il) {
            out_shape[0] = il.size();
            if (out_shape[0] > 0) {
                // Recursively compute the rest of the shape
                initializer_list_shape<std::initializer_list<T> >::
                                                compute(out_shape + 1, *il.begin());
                // Validate the shape for the nested initializer lists
                for (auto i = il.begin() + 1; i != il.end(); ++i) {
                    initializer_list_shape<std::initializer_list<T> >::
                                                        validate(out_shape + 1, *i);
                }
            }
        }
        static void validate(const intptr_t *shape,
                        const std::initializer_list<std::initializer_list<T> >& il) {
            if ((intptr_t)il.size() != shape[0]) {
                throw std::runtime_error("initializer list for ndobject is ragged, must be "
                                        "nested in a regular fashion");
            }
            // Validate the shape for the nested initializer lists
            for (auto i = il.begin(); i != il.end(); ++i) {
                initializer_list_shape<std::initializer_list<T> >::validate(shape + 1, *i);
            }
        }
        static void copy_data(typename initializer_list_dtype<T>::type **dataptr,
                        const std::initializer_list<std::initializer_list<T> >& il) {
            for (auto i = il.begin(); i != il.end(); ++i) {
                initializer_list_shape<std::initializer_list<T> >::copy_data(dataptr, *i);
            }
        }
    };
} // namespace detail

// Implementation of initializer list construction
template<class T>
dynd::ndobject::ndobject(std::initializer_list<T> il)
    : m_memblock()
{
    intptr_t dim0 = il.size();
    intptr_t stride = (dim0 == 1) ? 0 : sizeof(T);
    char *originptr = 0;
    memory_block_ptr memblock = make_fixed_size_pod_memory_block(sizeof(T) * dim0, sizeof(T), &originptr);
    DYND_MEMCPY(originptr, il.begin(), sizeof(T) * dim0);
    make_strided_ndobject_node(make_dtype<T>(), 1, &dim0, &stride,
                            originptr, read_access_flag | write_access_flag, DYND_MOVE(memblock)).swap(m_memblock);
}
template<class T>
dynd::ndobject::ndobject(std::initializer_list<std::initializer_list<T> > il)
    : m_memblock()
{
    typedef std::initializer_list<std::initializer_list<T> > S;
    intptr_t shape[2], strides[2];

    // Get and validate that the shape is regular
    detail::initializer_list_shape<S>::compute(shape, il);
    // Compute the number of elements in the array, and the strides at the same time
    intptr_t num_elements = 1, stride = sizeof(T);
    for (int i = 1; i >= 0; --i) {
        strides[i] = (shape[i] == 1) ? 0 : stride;
        num_elements *= shape[i];
        stride *= shape[i];
    }
    char *originptr = 0;
    memory_block_ptr memblock = make_fixed_size_pod_memory_block(sizeof(T) * num_elements, sizeof(T), &originptr);
    T *dataptr = reinterpret_cast<T *>(originptr);
    detail::initializer_list_shape<S>::copy_data(&dataptr, il);
    make_strided_ndobject_node(make_dtype<T>(), 2, shape, strides,
                        originptr, read_access_flag | write_access_flag, DYND_MOVE(memblock)).swap(m_memblock);
}
template<class T>
dynd::ndobject::ndobject(std::initializer_list<std::initializer_list<std::initializer_list<T> > > il)
    : m_memblock()
{
    typedef std::initializer_list<std::initializer_list<std::initializer_list<T> > > S;
    intptr_t shape[3], strides[3];

    // Get and validate that the shape is regular
    detail::initializer_list_shape<S>::compute(shape, il);
    // Compute the number of elements in the array, and the strides at the same time
    intptr_t num_elements = 1, stride = sizeof(T);
    for (int i = 2; i >= 0; --i) {
        strides[i] = (shape[i] == 1) ? 0 : stride;
        num_elements *= shape[i];
        stride *= shape[i];
    }
    char *originptr = 0;
    memory_block_ptr memblock = make_fixed_size_pod_memory_block(sizeof(T) * num_elements, sizeof(T), &originptr);
    T *dataptr = reinterpret_cast<T *>(originptr);
    detail::initializer_list_shape<S>::copy_data(&dataptr, il);
    make_strided_ndobject_node(make_dtype<T>(), 3, shape, strides,
                    originptr, read_access_flag | write_access_flag, DYND_MOVE(memblock)).swap(m_memblock);
}
#endif // DYND_INIT_LIST

///////////// C-style array constructor implementation /////////////////////////
namespace detail {
    template<class T> struct uniform_type_from_array {
        typedef T type;
        static const size_t element_size = sizeof(T);
        static const int type_id = type_id_of<T>::value;
    };
    template<class T, int N> struct uniform_type_from_array<T[N]> {
        typedef typename uniform_type_from_array<T>::type type;
        static const size_t element_size = uniform_type_from_array<T>::element_size;
        static const int type_id = uniform_type_from_array<T>::type_id;
    };

    template<class T> struct ndim_from_array {static const int value = 0;};
    template<class T, int N> struct ndim_from_array<T[N]> {
        static const int value = ndim_from_array<T>::value + 1;
    };

    template<class T> struct fill_shape {
        static size_t fill(intptr_t *) {
            return sizeof(T);
        }
    };
    template<class T, int N> struct fill_shape<T[N]> {
        static size_t fill(intptr_t *out_shape) {
            out_shape[0] = N;
            return N * fill_shape<T>::fill(out_shape + 1);
        }
    };
};

template<class T, int N>
dynd::ndobject::ndobject(const T (&rhs)[N])
    : m_memblock()
{
    const int ndim = detail::ndim_from_array<T[N]>::value;
    intptr_t shape[ndim];
    size_t size = detail::fill_shape<T[N]>::fill(shape);

    *this = make_strided_ndobject(dtype(detail::uniform_type_from_array<T>::type_id), ndim, shape);
    DYND_MEMCPY(get_ndo()->m_data_pointer, reinterpret_cast<const void *>(&rhs), size);
}

///////////// The ndobject.as<type>() templated function /////////////////////////
namespace detail {
    template <class T>
    struct ndobject_as_helper {
        static typename enable_if<is_dtype_scalar<T>::value, T>::type as(const ndobject& lhs,
                                                                    assign_error_mode errmode) {
            T result;
            if (!lhs.is_scalar()) {
                throw std::runtime_error("can only convert ndobjects with 0 dimensions to scalars");
            }
            dtype_assign(make_dtype<T>(), (char *)&result, lhs.get_dtype(), lhs.get_ndo()->m_data_pointer, errmode);
            return result;
        }
    };

    template <>
    struct ndobject_as_helper<bool> {
        static bool as(const ndobject& lhs, assign_error_mode errmode) {
            return ndobject_as_helper<dynd_bool>::as(lhs, errmode);
        }
    };

    std::string ndobject_as_string(const ndobject& lhs, assign_error_mode errmode);

    template <>
    struct ndobject_as_helper<std::string> {
        static std::string as(const ndobject& lhs, assign_error_mode errmode) {
            return ndobject_as_string(lhs, errmode);
        }
    };

    // Could do as<std::vector<T>> for 1D arrays, and other similiar conversions
} // namespace detail;

template<class T>
T dynd::ndobject::as(assign_error_mode errmode) const {
    return detail::ndobject_as_helper<T>::as(*this, errmode);
}

/**
 * Constructs an array with the same shape and memory layout
 * of the one given, but replacing the
 *
 * @param rhs  The array whose shape and memory layout to emulate.
 * @param dt   The uniform dtype of the new array.
 */
ndobject empty_like(const ndobject& rhs, const dtype& uniform_dtype);

/**
 * Constructs an empty array matching the parameters of 'rhs'
 *
 * @param rhs  The array whose shape, memory layout, and dtype to emulate.
 */
ndobject empty_like(const ndobject& rhs);

} // namespace dynd

#endif // _DYND__NDOBJECT_HPP_
