//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__HIERARCHICAL_KERNELS_HPP_
#define _DYND__HIERARCHICAL_KERNELS_HPP_

#include <dynd/config.hpp>

namespace dynd {

struct ckernel_data_prefix {
    typedef void (*destructor_fn_t)(ckernel_data_prefix *);

    void *function;
    destructor_fn_t destructor;

    /**
     * To help with generic code a bit, structs which
     * begin with a ckernel_data_prefix can define this
     * base() function which returns that ckernel_data_prefix.
     */
    ckernel_data_prefix& base() {
        return *this;
    }

    /**
     * Call to get the kernel function pointer, whose type
     * must be known by the context.
     *
     *      kdp->get_function<unary_single_operation_t>()
     */
    template<typename T>
    T get_function() const {
        return reinterpret_cast<T>(function);
    }

    template<typename T>
    void set_function(T fnptr) {
        function = reinterpret_cast<void *>(fnptr);
    }

    template<typename T, typename U, typename V>
    void init(const T&, const U&, const V&) {
    }
};

/**
 * This is a struct designed for interoperability at
 * the C ABI level. It contains enough information
 * to pass kernels from one library to another with
 * no dependencies between them.
 *
 * To free a dynamic kernel instance, one must
 * first call the destructor in the ckernel_data_prefix,
 * then call the free_func to deallocate the memory.
 * The function free_dynamic_kernel_instance is provided
 * here for this purpose.
 */
struct dynamic_kernel_instance {
    /** Pointer to dynamically allocated kernel data */
    ckernel_data_prefix *kernel;
    /**
     * How many bytes in the kernel data. Because the
     * kernel data must be movable, one may move the
     * data somewhere else, and then free the kernel memory
     * without calling the kernel destructor.
     *
     * This allows for kernels which are not complete,
     * e.g. a kernel which handles some leading dimensions,
     * and is expecting a child kernel immediately after
     * it to handle the elements.
     */
    size_t kernel_size;
    /** Pointer to a function for freeing 'kernel'. */
    void (*free_func)(void *);
};

inline void free_dynamic_kernel_instance(dynamic_kernel_instance& dki)
{
    if (dki.kernel != NULL) {
        if (dki.kernel->destructor != NULL) {
            dki.kernel->destructor(dki.kernel);
        }
        dki.free_func(dki.kernel);
        dki.kernel = NULL;
    }
}

/**
 * Function pointers + data for a hierarchical
 * kernel which operates on type/metadata in
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
    size_t m_capacity;
    // When the amount of data is small, this static data is used,
    // otherwise dynamic memory is allocated when it gets too big
    intptr_t m_static_data[16];

    inline bool using_static_data() const {
        return m_data == &m_static_data[0];
    }

    inline void destroy() {
        if (m_data != NULL) {
            ckernel_data_prefix *data;
            data = reinterpret_cast<ckernel_data_prefix *>(m_data);
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
        memset(m_static_data, 0, sizeof(m_static_data));
    }

    ~hierarchical_kernel() {
        destroy();
    }

    void reset() {
        destroy();
        m_data = &m_static_data[0];
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
    inline void ensure_capacity(size_t requested_capacity) {
        ensure_capacity_leaf(requested_capacity +
                        sizeof(ckernel_data_prefix));
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

    ckernel_data_prefix *get() const {
        return reinterpret_cast<ckernel_data_prefix *>(m_data);
    }

    /**
     * Moves the kernel data held by this hierarchical kernel
     * into the provide dynamic_kernel_instance struct. Ownership
     * is transferred to 'dki'.
     *
     * Because the kernel size is not tracked by the hierarchical_kernel
     * object, but rather produced by the factory functions, it
     * is required as a parameter here.
     *
     * \param out  The dynamic_kernel_instance to populate.
     * \param kernel_size  The size, in bytes, of the hierarchical_kernel.
     */
    void move_into_dki(dynamic_kernel_instance *out, size_t kernel_size) {
        if (using_static_data()) {
            // Allocate some memory and move the kernel data into it
            out->kernel = reinterpret_cast<ckernel_data_prefix *>(malloc(kernel_size));
            if (out->kernel == NULL) {
                out->free_func = NULL;
                throw std::bad_alloc();
            }
            memcpy(out->kernel, m_data, kernel_size);
            memset(m_static_data, 0, sizeof(m_static_data));
        } else {
            // Use the existing kernel data memory
            out->kernel = reinterpret_cast<ckernel_data_prefix *>(m_data);
            // Switch this kernel back to an empty static data kernel
            m_data = &m_static_data[0];
            m_capacity = sizeof(m_static_data);
            memset(m_static_data, 0, sizeof(m_static_data));
        }
        out->kernel_size = kernel_size;
        out->free_func = free;
    }
};

} // namespace dynd

#endif // _DYND__HIERARCHICAL_KERNELS_HPP_
