//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRIDED_NDARRAY_NODE_HPP_
#define _DYND__STRIDED_NDARRAY_NODE_HPP_

#include <vector>

#include <dynd/nodes/ndarray_node.hpp>

namespace dynd {

/**
 * NDArray node which holds a raw strided array.
 */
class strided_ndarray_node : public ndarray_node {
    /* The number of dimensions in the result array */
    int m_ndim;
    int m_access_flags;
    /* The shape of the result array */
    dimvector m_shape;
    /* The data type of this node's result */
    dtype m_dtype;
    char *m_originptr;
    dimvector m_strides;
    memory_block_ptr m_memblock;

    strided_ndarray_node() {
    }

    // Non-copyable
    strided_ndarray_node(const strided_ndarray_node&);
    strided_ndarray_node& operator=(const strided_ndarray_node&);

public:
    /**
     * Creates a strided array node from the raw values. Does not validate them.
     */
    strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, char *originptr, int access_flags, const memory_block_ptr& memblock);

    /**
     * Creates a strided array node from the raw values. Does not validate them.
     */
    strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, const char *originptr, int access_flags, const memory_block_ptr& memblock);

#ifdef DYND_RVALUE_REFS
    /**
     * Creates a strided array node from the raw values. Does not validate them.
     */
    strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, char *originptr, int access_flags, memory_block_ptr&& memblock);

    /**
     * Creates a strided array node from the raw values. Does not validate them.
     */
    strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, const char *originptr, int access_flags, memory_block_ptr&& memblock);
#endif

    /**
     * Constructs a strided array node with the given dtype, shape, and axis_perm (for memory layout)
     */
    strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape, const int *axis_perm);

    strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape, const int *axis_perm,
                    int access_flags, const memory_block_ptr *blockrefs_begin, const memory_block_ptr *blockrefs_end);

    virtual ~strided_ndarray_node() {
    }

    memory_block_ptr get_data_memory_block()
    {
        return m_memblock;
    }

    ndarray_node_category get_category() const
    {
        return strided_array_node_category;
    }

    const dtype& get_dtype() const {
        return m_dtype;
    }

    int get_ndim() const
    {
        return m_ndim;
    }

    const intptr_t *get_shape() const
    {
        return m_shape.get();
    }

    const intptr_t *get_strides() const {
        return m_strides.get();
    }

    uint32_t get_access_flags() const
    {
        return m_access_flags;
    }
    
    char *get_readwrite_originptr() const
    {
        if (m_access_flags & write_access_flag) {
            return m_originptr;
        } else {
            throw std::runtime_error("dynd::ndarray node is not writeable");
        }
    }

    const char *get_readonly_originptr() const
    {
        return m_originptr;
    }

    ndarray_node_ptr as_dtype(const dtype& dt,
                        assign_error_mode errmode, bool allow_in_place);

    ndarray_node_ptr apply_linear_index(
                    int ndim, const bool *remove_axis,
                    const intptr_t *start_index, const intptr_t *index_strides,
                    const intptr_t *shape,
                    bool allow_in_place);

    const char *node_name() const {
        return "strided_array";
    }

    void debug_print_extra(std::ostream& o, const std::string& indent) const;

    friend ndarray_node_ptr make_strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape,
                const intptr_t *strides, char *originptr, int access_flags, const memory_block_ptr& memblock);

    friend ndarray_node_ptr make_strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape,
                const intptr_t *strides, const char *originptr, int access_flags, const memory_block_ptr& memblock);

    #ifdef DYND_RVALUE_REFS
    friend ndarray_node_ptr make_strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape,
                const intptr_t *strides, char *originptr, int access_flags, memory_block_ptr&& memblock);

    friend ndarray_node_ptr make_strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape,
                const intptr_t *strides, const char *originptr, int access_flags, memory_block_ptr&& memblock);
    #endif
};

ndarray_node_ptr make_strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, char *originptr, int access_flags, const memory_block_ptr& memblock);

ndarray_node_ptr make_strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, const char *originptr, int access_flags, const memory_block_ptr& memblock);

#ifdef DYND_RVALUE_REFS
ndarray_node_ptr make_strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, char *originptr, int access_flags, memory_block_ptr&& memblock);

ndarray_node_ptr make_strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, const char *originptr, int access_flags, memory_block_ptr&& memblock);
#endif

ndarray_node_ptr make_strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape, const int *axis_perm);

ndarray_node_ptr make_strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape, const int *axis_perm,
                int access_flags, const memory_block_ptr *blockrefs_begin, const memory_block_ptr *blockrefs_end);

} // namespace dynd

#endif // _DYND__STRIDED_NDARRAY_NODE_HPP_
