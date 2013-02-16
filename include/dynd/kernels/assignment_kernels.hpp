//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ASSIGNMENT_KERNELS_HPP_
#define _DYND__ASSIGNMENT_KERNELS_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/kernels/kernel_instance.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/type_id.hpp>

namespace dynd {

struct kernel_data_prefix {
    typedef void (*destructor_fn_t)(kernel_data_prefix *);

    void *function;
    destructor_fn_t destructor;

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

    template<typename T>
    void set_function(T fnptr) {
        function = reinterpret_cast<void *>(fnptr);
    }
};

/** Typedef for a unary operation on a single element */
typedef void (*unary_single_operation_t)(char *dst, const char *src,
                kernel_data_prefix *extra);
/** Typedef for a unary operation on a strided segment of elements */
typedef void (*unary_strided_operation_t)(
                char *dst, intptr_t dst_stride,
                const char *src, intptr_t src_stride,
                size_t count, kernel_data_prefix *extra);

/**
 * Function pointers + data for a hierarchical assignment
 * kernel which assigns values from a source dtype/metadata
 * to a destination dtype/metadata.
 *
 * The data placed in the kernel's data must
 * be relocatable with a memcpy, it must not rely on its
 * own address.
 */
class assignment_kernel {
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
public:
    assignment_kernel() {
        m_data = &m_static_data[0];
        m_capacity = sizeof(m_static_data);
        m_size = 0;
        memset(m_static_data, 0, sizeof(m_static_data));
    }

    ~assignment_kernel() {
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

    /** Calls the function to do the assignment */
    inline void operator()(char *dst, const char *src) {
        unary_single_operation_t fn = reinterpret_cast<kernel_data_prefix *>(
                        m_data)->get_function<unary_single_operation_t>();
        fn(dst, src, get());
    }
};

/**
 * Creates an assignment kernel for one data value from the
 * src dtype/metadata to the dst dtype/metadata. This adds the
 * kernel at the 'out_offset' position in 'out's data, as part
 * of a hierarchy matching the dtype's hierarchy.
 *
 * This function should always be called with this == dst_dt first,
 * and dtypes which don't support the particular assignment should
 * then call the corresponding function with this == src_dt.
 *
 * \param out  The hierarchical assignment kernel being constructed.
 * \param offset_out  The offset within 'out'.
 * \param dst_dt  The destination dynd type.
 * \param dst_metadata  Metadata for the destination data.
 * \param src_dt  The source dynd type.
 * \param src_metadata  Metadata for the cource data
 * \param kernreq  What kind of kernel must be placed in 'out'.
 * \param errmode  The error mode to use for assignments.
 * \param ectx  DyND evaluation context.
 *
 * \returns  The offset within 'out' immediately after the
 *           created kernel.
 */
size_t make_assignment_kernel(
                assignment_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx);

/**
 * Creates an assignment kernel when the src and the dst are the same,
 * and are POD (plain old data).
 *
 * \param out  The hierarchical assignment kernel being constructed.
 * \param offset_out  The offset within 'out'.
 * \param data_size  The size of the data being assigned.
 * \param data_alignment  The alignment of the data being assigned.
 * \param kernreq  What kind of kernel must be placed in 'out'.
 */
size_t make_pod_dtype_assignment_kernel(
                assignment_kernel *out, size_t offset_out,
                size_t data_size, size_t data_alignment,
                kernel_request_t kernreq);

/**
 * Creates an assignment kernel from the src to the dst built in
 * type ids.
 *
 * \param out  The hierarchical assignment kernel being constructed.
 * \param offset_out  The offset within 'out'.
 * \param dst_dt  The destination dynd type id.
 * \param src_dt  The srouce dynd type id.
 * \param kernreq  What kind of kernel must be placed in 'out'.
 * \param errmode  The error mode to use for assignments.
 */
size_t make_builtin_dtype_assignment_function(
                assignment_kernel *out, size_t offset_out,
                type_id_t dst_type_id, type_id_t src_type_id,
                kernel_request_t kernreq, assign_error_mode errmode);

/**
 * When kernreq != kernel_request_single, adds an adapter to
 * the kernel which provides the requested kernel, and uses
 * a single kernel to fulfill the assignments. The
 * caller can use it like:
 *
 *  {
 *      offset_out = make_kernreq_to_single_kernel_adapter(
 *                      out, offset_out, kernreq);
 *      // Proceed to create 'single' kernel...
 */
size_t make_kernreq_to_single_kernel_adapter(
                assignment_kernel *out, size_t offset_out,
                kernel_request_t kernreq);

/**
 * Generic assignment kernel + destructor for a strided dimension.
 * This requires that the child kernel be created with the
 * kernel_request_strided type of kernel.
 */
struct strided_assign_kernel_extra {
    typedef strided_assign_kernel_extra extra_type;

    kernel_data_prefix base;
    intptr_t size;
    intptr_t dst_stride, src_stride;

    static void single(char *dst, const char *src,
                    kernel_data_prefix *extra);
    static void strided(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    size_t count, kernel_data_prefix *extra);
    static void destruct(kernel_data_prefix *extra);
};

} // namespace dynd

#endif // _DYND__ASSIGNMENT_KERNELS_HPP_
