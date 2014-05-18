//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CKERNEL_BUILDER_HPP_
#define _DYND__CKERNEL_BUILDER_HPP_

#include <new>
#include <algorithm>

#include <dynd/config.hpp>
#include <dynd/kernels/ckernel_prefix.hpp>
#include <dynd/types/type_id.hpp>

namespace dynd {

/**
 * Function pointers + data for a hierarchical
 * kernel which operates on type/metadata in
 * some configuration. Individual kernel types
 * are handled by the classes assignment_ckernel_builder, etc.
 *
 * The data placed in the kernel's data must
 * be relocatable with a memcpy, it must not rely on its
 * own address.
 */
class ckernel_builder {
    // Pointer to the kernel function pointers + data
    intptr_t *m_data;
    intptr_t m_capacity;
    // When the amount of data is small, this static data is used,
    // otherwise dynamic memory is allocated when it gets too big
    intptr_t m_static_data[16];

    inline bool using_static_data() const {
        return m_data == &m_static_data[0];
    }

    inline void destroy() {
        if (m_data != NULL) {
            ckernel_prefix *self = reinterpret_cast<ckernel_prefix *>(m_data);
            // Destroy whatever was created
            self->destroy();
            if (!using_static_data()) {
                // Free the memory
                free(self);
            }
        }
    }
protected:
public:
    ckernel_builder() {
        m_data = &m_static_data[0];
        m_capacity = sizeof(m_static_data);
        memset(m_static_data, 0, sizeof(m_static_data));
    }

    ~ckernel_builder() {
        destroy();
    }

    void reset() {
        destroy();
        m_data = &m_static_data[0];
        m_capacity = sizeof(m_static_data);
        memset(m_static_data, 0, sizeof(m_static_data));
    }

    /**
     * This function ensures that the ckernel's data
     * is at least the required number of bytes. It
     * should only be called during the construction phase
     * of the kernel.
     *
     * NOTE: This function ensures that there is room for
     *       another base at the end, so if you are sure
     *       that you're a leaf kernel, use ensure_capacity_leaf
     *       instead.
     */
    inline void ensure_capacity(intptr_t requested_capacity) {
        ensure_capacity_leaf(requested_capacity +
                        sizeof(ckernel_prefix));
    }

    /**
     * This function ensures that the ckernel's data
     * is at least the required number of bytes. It
     * should only be called during the construction phase
     * of the kernel when constructing a leaf kernel.
     */
    void ensure_capacity_leaf(intptr_t requested_capacity);

    /**
     * For use during construction, gets the ckernel component
     * at the requested offset.
     */
    template<class T>
    T *get_at(size_t offset) {
        return reinterpret_cast<T *>(
                        reinterpret_cast<char *>(m_data) + offset);
    }

    ckernel_prefix *get() const {
        return reinterpret_cast<ckernel_prefix *>(m_data);
    }

    void swap(ckernel_builder& rhs) {
        if (using_static_data()) {
            if (rhs.using_static_data()) {
                intptr_t tmp_static_data[16];
                memcpy(tmp_static_data, m_static_data, 16 * sizeof(intptr_t));
                memcpy(m_static_data, rhs.m_static_data, 16 * sizeof(intptr_t));
                memcpy(rhs.m_static_data, tmp_static_data, 16 * sizeof(intptr_t));
            } else {
                memcpy(rhs.m_static_data, m_static_data, 16 * sizeof(intptr_t));
                m_data = rhs.m_data;
                m_capacity = rhs.m_capacity;
                rhs.m_data = &rhs.m_static_data[0];
                rhs.m_capacity = 16 * sizeof(intptr_t);
            }
        } else {
            if (rhs.using_static_data()) {
                memcpy(m_static_data, rhs.m_static_data, 16 * sizeof(intptr_t));
                rhs.m_data = m_data;
                rhs.m_capacity = m_capacity;
                m_data = &m_static_data[0];
                m_capacity = 16 * sizeof(intptr_t);
            } else {
                (std::swap)(m_data, rhs.m_data);
                (std::swap)(m_capacity, rhs.m_capacity);
            }
        }
    }

    friend int ckernel_builder_ensure_capacity_leaf(void *ckb, intptr_t requested_capacity);
};

/**
 * C API function for constructing a ckernel_builder object
 * in place. The `ckb` pointer must point to memory which
 * has sizeof(ckernel_builder) bytes (== 18 * sizeof(void *)),
 * and is aligned appropriately (to sizeof(void *)).
 *
 * After a ckernel_builder instance is initialized this way,
 * all the other ckernel_builder functions can be used on it,
 * and when it is no longer needed, it must be destructed
 * by calling `ckernel_builder_destruct`.
 *
 * \param ckb  Pointer to the ckernel_builder instance. Must have
 *             size 18 * sizeof(void *), and alignment sizeof(void *).
 */
inline void ckernel_builder_construct(void *ckb)
{
    // Use the placement new operator to initialize in-place
    new (ckb) ckernel_builder();
}

/**
 * C API function for destroying a valid ckernel_builder object
 * in place. The `ckb` pointer must point to memory which
 * was previously initialized with `ckernel_buidler_construct`
 *
 * \param ckb  Pointer to the ckernel_builder instance.
 */
inline void ckernel_builder_destruct(void *ckb)
{
    // Call the destructor
    ckernel_builder *ckb_ptr = reinterpret_cast<ckernel_builder *>(ckb);
    ckb_ptr->~ckernel_builder();
}

/**
 * C API function for resetting a valid ckernel_builder object
 * to an uninitialized state.
 *
 * \param ckb  Pointer to the ckernel_builder instance.
 */
inline void ckernel_builder_reset(void *ckb)
{
    ckernel_builder *ckb_ptr = reinterpret_cast<ckernel_builder *>(ckb);
    ckb_ptr->reset();
}

/**
 * C API function for ensuring that the kernel's data
 * is at least the required number of bytes. It
 * should only be called during the construction phase
 * of the kernel when constructing a leaf kernel.
 *
 * If the created kernel has a child kernel, use
 * the function `ckernel_builder_ensure_capacity` instead.
 *
 * \param ckb  Pointer to the ckernel_builder instance.
 * \param requested_capacity  The number of bytes required by the ckernel.
 *
 * \returns  0 on success, -1 on memory allocation failure.
 */
inline int ckernel_builder_ensure_capacity_leaf(void *ckb, intptr_t requested_capacity)
{
    ckernel_builder *ckb_ptr = reinterpret_cast<ckernel_builder *>(ckb);
    if (ckb_ptr->m_capacity < requested_capacity) {
        // Grow by a factor of 1.5
        // https://github.com/facebook/folly/blob/master/folly/docs/FBVector.md
        intptr_t grown_capacity = ckb_ptr->m_capacity * 3 / 2;
        if (requested_capacity < grown_capacity) {
            requested_capacity = grown_capacity;
        }
        intptr_t *new_data;
        if (ckb_ptr->using_static_data()) {
            // If we were previously using the static data, do a malloc
            new_data = reinterpret_cast<intptr_t *>(malloc(requested_capacity));
            // If the allocation succeeded, copy the old data as the realloc would
            if (new_data != NULL) {
                memcpy(new_data, ckb_ptr->m_data, ckb_ptr->m_capacity);
            }
        } else {
            // Otherwise do a realloc
            new_data = reinterpret_cast<intptr_t *>(realloc(
                            ckb_ptr->m_data, requested_capacity));
        }
        if (new_data == NULL) {
            ckb_ptr->destroy();
            ckb_ptr->m_data = NULL;
            return -1;
        }
        // Zero out the newly allocated capacity
        memset(reinterpret_cast<char *>(new_data) + ckb_ptr->m_capacity,
                        0, requested_capacity - ckb_ptr->m_capacity);
        ckb_ptr->m_data = new_data;
        ckb_ptr->m_capacity = requested_capacity;
    }
    return 0;
}

/**
 * C API function for ensuring that the kernel's data
 * is at least the required number of bytes. It
 * should only be called during the construction phase
 * of the kernel when constructing a leaf kernel.
 *
 * This function allocates the requested capacity, plus enough
 * space for an empty child kernel to ensure safe destruction
 * during error handling. If a leaf kernel is being constructed,
 * use `ckernel_builder_ensure_capacity_leaf` instead.
 *
 * \param ckb  Pointer to the ckernel_builder instance.
 * \param requested_capacity  The number of bytes required by the ckernel.
 *
 * \returns  0 on success, -1 on memory allocation failure.
 */
inline int ckernel_builder_ensure_capacity(void *ckb, intptr_t requested_capacity)
{
    return ckernel_builder_ensure_capacity_leaf(ckb,
                    requested_capacity + sizeof(ckernel_prefix));
}

inline void ckernel_builder::ensure_capacity_leaf(intptr_t requested_capacity)
{
    if (ckernel_builder_ensure_capacity_leaf(this, requested_capacity) < 0) {
        throw std::bad_alloc();
    }
}

namespace kernels {
    /**
     * Some common shared implementation details of a CRTP
     * (curiously recurring template pattern) base class to help
     * create ckernels.
     */
    template<class CKT>
    struct general_ck {
        typedef CKT self_type;

        ckernel_prefix base;

        static self_type *get_self(ckernel_prefix *rawself) {
            return reinterpret_cast<self_type *>(rawself);
        }

        static const self_type *get_self(const ckernel_prefix *rawself) {
            return reinterpret_cast<const self_type *>(rawself);
        }

        static self_type *get_self(ckernel_builder *ckb, intptr_t ckb_offset) {
            return ckb->get_at<self_type>(ckb_offset);
        }

        static inline self_type *create(ckernel_builder *ckb,
                                             intptr_t ckb_offset,
                                             kernel_request_t kernreq)
        {
            ckb->ensure_capacity(ckb_offset + sizeof(self_type));
            ckernel_prefix *rawself = ckb->get_at<ckernel_prefix>(ckb_offset);
            return self_type::init(rawself, kernreq);
        }

        static inline self_type *create_leaf(ckernel_builder *ckb,
                                             intptr_t ckb_offset,
                                             kernel_request_t kernreq)
        {
            ckb->ensure_capacity_leaf(ckb_offset + sizeof(self_type));
            ckernel_prefix *rawself = ckb->get_at<ckernel_prefix>(ckb_offset);
            return self_type::init(rawself, kernreq);
        }

        /**
         * Initializes an instance of this ckernel in-place according to the
         * kernel request. This calls the constructor in-place, and initializes
         * the base function and destructor
         */
        static inline self_type *init(ckernel_prefix *rawself,
                                      kernel_request_t kernreq)
        {
            // Alignment requirement of the type
            DYND_STATIC_ASSERT(
                (size_t)scalar_align_of<self_type>::value ==
                    (size_t)scalar_align_of<void *>::value,
                "ckernel types require alignment matching that of pointers");

            // Call the constructor in-place
            self_type *self = new (rawself) self_type();
            // Double check that the C++ struct layout is as we expect
            if (self != get_self(rawself)) {
                throw std::runtime_error("internal ckernel error: struct layout is not valid");
            }
            self->base.destructor = &self_type::destruct;
            // A child class must implement this to fill in self->base.function
            self->init_kernfunc(kernreq);
            return self;
        }

        /**
         * The ckernel destructor function, which is placed in
         * base.destructor.
         */
        static void destruct(ckernel_prefix *rawself) {
            self_type *self = get_self(rawself);
            // If there are any child kernels, a child class
            // must implement this to destroy them.
            self->destruct_children();
            self->~self_type();
        }

        /**
         * Default implementation of destruct_children does nothing.
         */
        inline void destruct_children()
        {
        }

        /**
         * Returns the child ckernel immediately following this one.
         */
        inline ckernel_prefix *get_child_ckernel() {
            return get_child_ckernel(sizeof(self_type));
        }

        /**
         * Returns the child ckernel at the specified offset.
         */
        inline ckernel_prefix *get_child_ckernel(intptr_t offset) {
            return base.get_child_ckernel(offset);
        }
    };
} // namespace kernels

} // namespace dynd


#endif // _DYND__CKERNEL_BUILDER_HPP_
