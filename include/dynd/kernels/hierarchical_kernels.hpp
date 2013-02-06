//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__HIERARCHICAL_KERNELS_HPP_
#define _DYND__HIERARCHICAL_KERNELS_HPP_

#include <stdexcept>

#include <dynd/config.hpp>

namespace dynd {

struct hierarchical_kernel_common_base {
    typedef void (*hierarchical_kernel_data_destructor)(hierarchical_kernel_common_base *);

    void *function;
    hierarchical_kernel_data_destructor destructor;

    /**
     * Call to get the kernel function pointer, whose type
     * must be known by the context.
     *
     *      hkcm.get<unary_single_operation_t>()
     */
    template<typename T>
    T get_function() const {
        return reinterpret_cast<T>(function);
    }
};

/**
 * Function pointers + data for a hierarchical kernel for
 * processing ndobjects.
 *
 * The data placed in the hierarchical kernel's data must
 * be relocatable with a memcpy, it must not rely on its
 * own address.
 */
template<typename FT>
class hierarchical_kernel {
    // Pointer to the kernel function pointers + data
    hierarchical_kernel_common_base *m_data;
    size_t m_capacity;
    // When the amount of data is small, this static data is used,
    // otherwise dynamic memory is allocated when it gets too big
    intptr_t m_static_data[16];

    inline bool using_static_data() const {
        return m_data == reinterpret_cast<const hierarchical_kernel_common_base *>(&m_static_data[0]);
    }

    inline void destroy() {
        if (m_data != NULL) {
            // Destroy whatever was created
            if (m_data->destructor != NULL) {
                m_data->destructor(m_data);
            }
             if (!using_static_data()) {
                // Free the memory
                free(m_data);
             }
        }
    }
public:
    hierarchical_kernel() {
        m_data = reinterpret_cast<hierarchical_kernel_common_base *>(&m_static_data[0]);
        m_capacity = sizeof(m_static_data);
        memset(m_static_data, 0, sizeof(m_static_data));
    }

    ~hierarchical_kernel() {
        destroy();
    }

    void reset() {
        destroy();
        m_data = reinterpret_cast<hierarchical_kernel_common_base *>(&m_static_data[0]);
        m_capacity = sizeof(m_static_data);
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
    inline void ensure_capacity(size_t size) {
        ensure_capacity_leaf(size + sizeof(hierarchical_kernel_common_base));
    }

    /**
     * This function ensures that the kernel's data
     * is at least the required number of bytes. It
     * should only be called during the construction phase
     * of the kernel when constructing a leaf kernel.
     */
    void ensure_capacity_leaf(size_t size) {
        if (m_capacity < size) {
            // At least double the capacity
            if (size < 2 * m_capacity) {
                size = 2 * m_capacity;
            }
            hierarchical_kernel_common_base *new_data;
            if (using_static_data()) {
                // If we were previously using the static data, do a malloc
                new_data = reinterpret_cast<hierarchical_kernel_common_base *>(malloc(size));
                // If the allocation succeeded, copy the old data as the realloc would
                if (new_data != NULL) {
                    memcpy(new_data, m_data, m_capacity);
                }
            } else {
                // Otherwise do a realloc
                new_data = reinterpret_cast<hierarchical_kernel_common_base *>(realloc(m_data, size));
            }
            if (new_data == NULL) {
                // Destroy whatever was created
                if (m_data->destructor != NULL) {
                    m_data->destructor(m_data);
                }
                // Free the memory
                if (!using_static_data()) {
                    free(m_data);
                }
                m_data = NULL;
                throw std::bad_alloc();
            }
            // Zero out the newly allocated capacity
            memset(reinterpret_cast<char *>(new_data) + m_capacity, 0, size - m_capacity);
            m_data = new_data;
            m_capacity = size;
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

    hierarchical_kernel_common_base *get() const {
        return m_data;
    }

    FT get_function() const {
        return m_data->get_function<FT>();
    }
};

/** Typedef for a unary operation on a single element */
typedef void (*unary_single_operation_t)(char *dst, const char *src,
                hierarchical_kernel_common_base *extra);
/** Typedef for a unary operation on a strided segment of elements */
typedef void (*unary_strided_operation_t)(char *dst, intptr_t dst_stride,
                const char *src, intptr_t src_stride, size_t count,
                hierarchical_kernel_common_base *extra);


} // namespace dynd

#endif // _DYND__HIERARCHICAL_KERNELS_HPP_