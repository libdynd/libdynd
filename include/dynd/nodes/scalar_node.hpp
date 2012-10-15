//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__SCALAR_NODE_HPP_
#define _DND__SCALAR_NODE_HPP_

#include <dynd/nodes/ndarray_node.hpp>

namespace dynd {

/**
 * NDArray expression node which holds a scalar.
 */
class scalar_node : public ndarray_node {
    char *m_originptr;
    /** If m_dtype is a blockref dtype, this holds its memory */
    memory_block_ptr m_blockref_memblock;
    dtype m_dtype;
    int m_access_flags;

    // Non-copyable
    scalar_node(const scalar_node&);
    scalar_node& operator=(const scalar_node&);

    // Use make_scalar_node to actually create one of these
    scalar_node(const dtype& dt, char* originptr, int access_flags)
        : m_originptr(originptr), m_blockref_memblock(), m_dtype(dt),
            m_access_flags(access_flags)
    {
    }

    // Use make_scalar_node
    scalar_node(const dtype& dt, char* originptr, int access_flags, const memory_block_ptr& blockref_memblock)
        : m_originptr(originptr), m_blockref_memblock(blockref_memblock), m_dtype(dt),
            m_access_flags(access_flags)
    {
    }

public:
    virtual ~scalar_node() {
    }

    memory_block_ptr get_data_memory_block()
    {
        // The data is always stored in the node itself
        return as_memory_block_ptr();
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
        return 0;
    }

    const intptr_t *get_shape() const
    {
        return NULL;
    }

    const intptr_t *get_strides() const
    {
        return NULL;
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
                        dynd::assign_error_mode errmode, bool allow_in_place);

    ndarray_node_ptr apply_linear_index(
                    int ndim, const bool *remove_axis,
                    const intptr_t *start_index, const intptr_t *index_strides,
                    const intptr_t *shape,
                    bool allow_in_place);

    const char *node_name() const {
        return "scalar";
    }

    void debug_dump_extra(std::ostream& o, const std::string& indent) const;

    friend ndarray_node_ptr make_scalar_node(const dtype& dt, const char* data, int access_flags);
    friend ndarray_node_ptr make_scalar_node(const dtype& dt, const char* data, int access_flags,
                const memory_block_ptr& blockref_memblock);
};

ndarray_node_ptr make_scalar_node(const dtype& dt, const char* data, int access_flags);

ndarray_node_ptr make_scalar_node(const dtype& dt, const char* data, int access_flags,
                const memory_block_ptr& blockref_memblock);

} // namespace dynd

#endif // _DND__SCALAR_NODE_HPP_
