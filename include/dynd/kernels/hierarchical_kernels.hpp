//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__HIERARCHICAL_KERNELS_HPP_
#define _DYND__HIERARCHICAL_KERNELS_HPP_

#include <dynd/config.hpp>

namespace dynd {

struct kernel_data_prefix {
    typedef void (*destructor_fn_t)(kernel_data_prefix *);

    void *function;
    destructor_fn_t destructor;

    /**
     * Call to get the kernel function pointer, whose type
     * must be known by the context.
     *
     *      kdp->get<unary_single_operation_t>()
     */
    template<typename T>
    T get_function() const {
        return reinterpret_cast<T>(function);
    }

    template<typename T>
    void set_function(T fnptr) {
        function = reinterpret_cast<void *>(fnptr);
    }
};

/**
 * Function pointers + data for a hierarchical
 * kernel which operates on dtype/metadata in
 * some configuration. Individual kernel types
 * are handled by the classes assignment_kernel, etc.
 *
 * The data placed in the kernel's data must
 * be relocatable with a memcpy, it must not rely on its
 * own address.
 */
class hierarchical_kernel {
    // Pointer to the kernel function pointers + data
    intptr_t *m_data;
    size_t m_capacity, m_size;
    // When the amount of data is small, this static data is used,
    // otherwise dynamic memory is allocated when it gets too big
    intptr_t m_static_data[16];

    inline bool using_static_data() const {
        return m_data == &m_static_data[0];
    }

    inline void destroy() {
        if (m_data != NULL) {
            kernel_data_prefix *data;
            data = reinterpret_cast<kernel_data_prefix *>(m_data);
            // Destroy whatever was created
            if (data->destructor != NULL) {
                data->destructor(data);
            }
             if (!using_static_data()) {
                // Free the memory
                free(data);
             }
        }
    }
protected:
public:
    hierarchical_kernel() {
        m_data = &m_static_data[0];
        m_capacity = sizeof(m_static_data);
        m_size = 0;
        memset(m_static_data, 0, sizeof(m_static_data));
    }

    ~hierarchical_kernel() {
        destroy();
    }

    void reset() {
        destroy();
        m_data = &m_static_data[0];
        m_capacity = sizeof(m_static_data);
        m_size = 0;
        memset(m_static_data, 0, sizeof(m_static_data));
    }

    /**
     * This function ensures that the kernel's data
     * is at least the required number of bytes. It
     * should only be called during the construction phase
     * of the kernel.
     *
     * NOTE: This function ensures that there is room for
     *       another base at the end, so if you are sure
     *       that you're a leaf kernel, use ensure_capacity_leaf
     *       instead.
     */
    inline void ensure_capacity(size_t requested_capacity) {
        ensure_capacity_leaf(requested_capacity +
                        sizeof(kernel_data_prefix));
    }

    /**
     * This function ensures that the kernel's data
     * is at least the required number of bytes. It
     * should only be called during the construction phase
     * of the kernel when constructing a leaf kernel.
     */
    void ensure_capacity_leaf(size_t requested_capacity) {
        if (m_capacity < requested_capacity) {
            // Grow by a factor of 1.5
            // https://github.com/facebook/folly/blob/master/folly/docs/FBVector.md
            size_t grown_capacity = m_capacity * 3 / 2;
            if (requested_capacity < grown_capacity) {
                requested_capacity = grown_capacity;
            }
            intptr_t *new_data;
            if (using_static_data()) {
                // If we were previously using the static data, do a malloc
                new_data = reinterpret_cast<intptr_t *>(malloc(requested_capacity));
                // If the allocation succeeded, copy the old data as the realloc would
                if (new_data != NULL) {
                    memcpy(new_data, m_data, m_capacity);
                }
            } else {
                // Otherwise do a realloc
                new_data = reinterpret_cast<intptr_t *>(realloc(
                                m_data, requested_capacity));
            }
            if (new_data == NULL) {
                destroy();
                m_data = NULL;
                throw std::bad_alloc();
            }
            // Zero out the newly allocated capacity
            memset(reinterpret_cast<char *>(new_data) + m_capacity,
                            0, requested_capacity - m_capacity);
            m_data = new_data;
            m_capacity = requested_capacity;
        }
    }

    /**
     * For use during construction, get's the kernel component
     * at the requested offset.
     */
    template<class T>
    T *get_at(size_t offset) {
        return reinterpret_cast<T *>(
                        reinterpret_cast<char *>(m_data) + offset);
    }

    kernel_data_prefix *get() const {
        return reinterpret_cast<kernel_data_prefix *>(m_data);
    }
};

} // namespace dynd

#endif // _DYND__HIERARCHICAL_KERNELS_HPP_
