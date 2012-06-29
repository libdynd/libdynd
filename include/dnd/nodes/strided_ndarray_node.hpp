//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__STRIDED_NDARRAY_NODE_HPP_
#define _DND__STRIDED_NDARRAY_NODE_HPP_

#include <dnd/nodes/ndarray_node.hpp>

namespace dnd {

/**
 * NDArray node which holds a raw strided array.
 */
class strided_ndarray_node : public ndarray_node {
    /* The data type of this node's result */
    dtype m_dtype;
    char *m_originptr;
    dimvector m_strides;
    memory_block_ref m_memblock;

    // Non-copyable
    strided_ndarray_node(const strided_ndarray_node&);
    strided_ndarray_node& operator=(const strided_ndarray_node&);

public:
    /**
     * Creates a strided array node from the raw values.
     *
     * It's prefereable to use the function make_strided_ndarray_node function,
     * as it does parameter validation.
     */
    strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, char *originptr, const memory_block_ref& memblock);

    /**
     * Constructs a strided array node with the given dtype, shape, and axis_perm (for memory layout)
     */
    strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape, const int *axis_perm);

    virtual ~strided_ndarray_node() {
    }

    ndarray_node_category get_category() const
    {
        return strided_array_node_category;
    }

    const dtype& get_dtype() const {
        return m_dtype;
    }

    char *get_readwrite_originptr() const {
        return m_originptr;
    }

    const char *get_readonly_originptr() const {
        return m_originptr;
    }

    const intptr_t *get_strides() const {
        return m_strides.get();
    }

    memory_block_ref get_memory_block() const {
        return m_memblock;
    }

    /** Provides the data pointer and strides array for the tree evaluation code */
    void as_readwrite_data_and_strides(int ndim, char **out_originptr, intptr_t *out_strides) const;

    void as_readonly_data_and_strides(int ndim, char const **out_originptr, intptr_t *out_strides) const;

    ndarray_node_ref as_dtype(const dtype& dt,
                        assign_error_mode errmode, bool allow_in_place);

    ndarray_node_ref apply_linear_index(
                    int ndim, const bool *remove_axis,
                    const intptr_t *start_index, const intptr_t *index_strides,
                    const intptr_t *shape,
                    bool allow_in_place);

    const char *node_name() const {
        return "strided_array";
    }

    void debug_dump_extra(std::ostream& o, const std::string& indent) const;

    friend ndarray_node_ref make_strided_ndarray_node(
                    const dtype& dt, int ndim, const intptr_t *shape,
                    const intptr_t *strides, char *originptr,
                    const memory_block_ref& memblock);
};

/**
 * Creates an expr node out of the raw data for a strided array. This will create
 * a strided_ndarray_node, the caller should ensure that the data is aligned,
 * and use an unaligned<> dtype if not.
 *
 * @param dt        The data type of the raw elements.
 * @param ndim      The number of dimensions in the array.
 * @param shape     The shape of the array (has 'ndim' elements)
 * @param strides   The strides of the array (has 'ndim' elements)
 * @param originptr The pointer to the element whose multi-index is all zeros.
 * @param buffer_owner  A reference-counted pointer to the owner of the buffer.
 */
ndarray_node_ref make_strided_ndarray_node(
            const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, char *originptr,
            const dnd::shared_ptr<void>& buffer_owner);

} // namespace dnd

#endif // _DND__STRIDED_NDARRAY_NODE_HPP_
