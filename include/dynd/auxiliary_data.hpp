//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__AUXILIARY_DATA_HPP_
#define _DYND__AUXILIARY_DATA_HPP_

#include <new>
#include <algorithm>

#include <dynd/memblock/memory_block.hpp>

namespace dynd {

// AuxDataBase is the same as NpyAuxData, see the numpy doc link
// http://docs.scipy.org/doc/numpy/reference/c-api.array.html#auxiliary-data-with-object-semantics
struct AuxDataBase;

// Function pointers for freeing or cloning auxiliary data
// IMPORTANT NOTE: These are C-ABI functions, they should
//                 not throw exceptions!
typedef void (*auxdata_free_function_t) (AuxDataBase *);
typedef AuxDataBase *(*auxdata_clone_function_t) (const AuxDataBase *);

struct AuxDataBase {
    /** Mandatory free and clone functions */
    auxdata_free_function_t free;
    auxdata_clone_function_t clone;
    void *reserved0, *reserved1; /* for future expansion */
};

class auxiliary_data;

template<typename T>
void make_auxiliary_data(auxiliary_data& out_created);

namespace detail {

    template<typename T>
    void auxiliary_data_holder_free(AuxDataBase *auxdata);

    template<typename T>
    AuxDataBase *auxiliary_data_holder_clone(const AuxDataBase *auxdata);

    /**
     * The auxiliary data class encapsulates data lifetime-managed
     * data which pairs with kernel functions. This object is oriented
     * to be C-ABI compatible, in particular it is binary compatible with
     * the NpyAuxData structure in Numpy. The difference with Numpy's version
     * is that we use C++ templates to wrap it in an easier-to-use class
     * object instead of having some macros.
     *
     */
    template<typename Taux>
    class auxiliary_data_holder {
        AuxDataBase m_base;
        Taux m_auxdata;

        // Only make_auxiliary_data<T> can default-construct one of these
        auxiliary_data_holder()
            : m_base(), m_auxdata()
        {
        }

        // Only auxiliary_data_holder_free<T> can delete one of these
        ~auxiliary_data_holder() {
        }

        // Only auxiliary_data_holder_clone<T> can copy one of these
        auxiliary_data_holder(const auxiliary_data_holder& rhs)
            : m_base(rhs.m_base), m_auxdata(rhs.m_auxdata)
        {}

        // Non-copyable
        auxiliary_data_holder& operator=(const auxiliary_data_holder&);

        template<typename T>
        friend void ::dynd::make_auxiliary_data(auxiliary_data& out_created);
        template<typename T>
        friend void auxiliary_data_holder_free(AuxDataBase *auxdata);
        template<typename T>
        friend AuxDataBase *auxiliary_data_holder_clone(const AuxDataBase *auxdata);

        friend class ::dynd::auxiliary_data;
    public:
        inline const Taux& get() const {
            return m_auxdata;
        }

        inline Taux& get() {
            return m_auxdata;
        }
    };

    template<typename T>
    void auxiliary_data_holder_free(AuxDataBase *auxdata)
    {
        auxiliary_data_holder<T> *adh = reinterpret_cast<auxiliary_data_holder<T> *>(auxdata);
        delete adh;
    }

    template<typename T>
    AuxDataBase *auxiliary_data_holder_clone(const AuxDataBase *auxdata)
    {
        const auxiliary_data_holder<T> *adh = reinterpret_cast<const auxiliary_data_holder<T> *>(auxdata);
        try {
            return reinterpret_cast<AuxDataBase *>(new auxiliary_data_holder<T>(*adh));
        } catch(const std::exception&) {
            return 0;
        }
    }

} // namespace detail

/**
 * Holds an auxiliary data object, and manages its lifetime.
 * It's non-copyable, as no reference counting has been added
 * for performance reasons, and we want to support pre-C++11.
 * With C++11, we would follow the example of unique_ptr<T>.
 *
 * The invariant that m_auxdata satisfies:
 *  - If the 0th bit is set, it is by-value data or a borrowed reference
 *  - If the 0th bit is not set, it is managed with the AuxDataBase
 *    free and clone functions.
 */
class auxiliary_data {
    uintptr_t m_auxdata;

    // Non-copyable
    auxiliary_data(const auxiliary_data&);
    auxiliary_data& operator=(const auxiliary_data&);
    // Would be nice if we could rely on it being movable, like std::unique_ptr,
    // but we probably need to support pre C++11...
public:
    // Initialize as statically stored zero.
    auxiliary_data()
        : m_auxdata(1)
    {
    }
    ~auxiliary_data() {
        free();
    }

    /**
     * If this is true, the auxiliary data is by-value, signaled
     * by its 0th bit being set to 1.
     */
    bool is_byvalue() const {
        return (m_auxdata&1) == 1;
    }

    // Frees the auxdata memory, makes this auxdata statically
    // stored zero.
    void free() {
        if ((m_auxdata&1) == 0) {
            reinterpret_cast<AuxDataBase *>(m_auxdata)->free(
                            reinterpret_cast<AuxDataBase *>(m_auxdata));
            m_auxdata = 1;
        }
    }

    /**
     * Clones the auxiliary data, using the AuxDataBase
     * stored clone function.
     */
    void clone_from(const auxiliary_data& rhs) {
        free();
        if ((rhs.m_auxdata&1) == 1) {
            // Bit zero is set, copy by value
            m_auxdata = rhs.m_auxdata;
        } else {
            // Bit zero is not set, clone the data
            m_auxdata = reinterpret_cast<uintptr_t>(
                            reinterpret_cast<AuxDataBase *>(rhs.m_auxdata)->clone(
                                    reinterpret_cast<AuxDataBase *>(rhs.m_auxdata)));

            if (m_auxdata == 0) {
                m_auxdata = 1;
                throw std::bad_alloc();
            }
        }
    }

    /**
     * Creates a borrowed reference to the auxiliary data,
     * by setting the "is_byvalue" bit of the m_auxdata member.
     * The caller is responsible to ensure the lifetime of the borrowed
     * reference is shorter than the lifetime of the original,
     * no automatic tracking is done.
     */
    void borrow_from(const auxiliary_data& rhs) {
        m_auxdata = rhs.m_auxdata|1;
    }

    void swap(auxiliary_data& rhs) {
        std::swap(m_auxdata, rhs.m_auxdata);
    }

    /**
     * When the auxiliary_data was created with make_auxiliary_data<T>, this
     * returns a reference to the T member. This should only be called when
     * the auxiliary data is known to have been created with the
     * make_auxiliary_data<T> template, and works with both tracked and
     * borrowed auxdatas of this type.
     */
    template<typename T>
    T& get() {
        return reinterpret_cast<detail::auxiliary_data_holder<T> *>(m_auxdata&~1)->m_auxdata;
    }

    // Allow implicit conversion to const AuxDataBase *, so that this
    // can be passed as a parameter to kernel functions.
    operator const AuxDataBase *() const {
        return reinterpret_cast<const AuxDataBase *>(m_auxdata);
    }

    operator AuxDataBase *() {
        return reinterpret_cast<AuxDataBase *>(m_auxdata);
    }

    template<typename T>
    friend void make_auxiliary_data(auxiliary_data& out_created);

    friend void make_raw_auxiliary_data(auxiliary_data& out_created, uintptr_t raw_value);
};

/**
 * Default-constructs an auxiliary data object, with a payload of the requested type.
 * This will typically be paired with a kernel function which uses a
 * const AuxDataBase* for auxiliary data.
 */
template<typename T>
void make_auxiliary_data(auxiliary_data& out_created)
{
    out_created.free();
    out_created.m_auxdata = reinterpret_cast<uintptr_t>(new detail::auxiliary_data_holder<T>());
    reinterpret_cast<AuxDataBase *>(out_created.m_auxdata)->free = detail::auxiliary_data_holder_free<T>;
    reinterpret_cast<AuxDataBase *>(out_created.m_auxdata)->clone = detail::auxiliary_data_holder_clone<T>;
}

/**
 * Default-constructs an auxiliary data object, with a payload of the requested type.
 * This will typically be paired with a kernel function which uses a
 * const AuxDataBase* for auxiliary data.
 *
 * After construction, assigns the given value into the auxdata.
 */
template<typename T>
void make_auxiliary_data(auxiliary_data& out_created, const T& value)
{
    make_auxiliary_data<T>(out_created);
    out_created.get<T>() = value;
}

/**
 * Creates a by-value auxiliary data object. This sets the 0th bit of the value,
 * so the caller must be sure to avoid that value. To set a by-value auxiliary aligned
 * pointer, use make_raw_auxiliary_data(output, reinterpret_cast<uintptr_t>(aligned_ptr),
 * and to set an integer, use make_raw_auxiliary_data(output, static_cast<uintptr_t>(value) << 1).
 */
inline void make_raw_auxiliary_data(auxiliary_data& out_created, uintptr_t raw_value)
{
    out_created.m_auxdata = raw_value|1;
}


/**
 * When auxdata points to an object created with make_auxiliary_data<T>,
 * this returns a reference to the T object it contains. Should not be
 * called when auxdata is NULL. This works with either tracked auxiliary
 * data or borrowed auxiliary data.
 */
template<typename T>
inline const T& get_auxiliary_data(const AuxDataBase *auxdata)
{
    return reinterpret_cast<const detail::auxiliary_data_holder<T>*>(reinterpret_cast<uintptr_t>(auxdata)&~1)->get();
}

/**
 * When auxdata points to an object created with make_auxiliary_data<T>,
 * this returns a reference to the T object it contains. Should not be
 * called when auxdata is NULL. This works with either tracked auxiliary
 * data or borrowed auxiliary data.
 */
template<typename T>
inline T& get_auxiliary_data(AuxDataBase *auxdata)
{
    return reinterpret_cast<detail::auxiliary_data_holder<T>*>(reinterpret_cast<uintptr_t>(auxdata)&~1)->get();
}

/**
 * When auxdata points to by-value auxiliary data, this gets the value, including the
 * 0th bit being set. To use this with an aligned pointer, use get_raw_auxiliary_data(auxdata)&~1,
 * and to use this as an integer, use raw_auxiliary_data(auxdata)>>1.
 */
inline uintptr_t get_raw_auxiliary_data(const AuxDataBase *auxdata)
{
    return reinterpret_cast<uintptr_t>(auxdata);
}


} // namespace dynd

#endif // _DYND__AUXILIARY_DATA_HPP_
