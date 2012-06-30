//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__IMMUTABLE_SCALAR_NODE_HPP_
#define _DND__IMMUTABLE_SCALAR_NODE_HPP_

#include <dnd/nodes/ndarray_node.hpp>

namespace dnd {

/**
 * NDArray expression node which holds an immutable scalar.
 */
class immutable_scalar_node : public ndarray_node {
    /* The data type of this node's result */
    dtype m_dtype;
    char *m_originptr;

    // Non-copyable
    immutable_scalar_node(const immutable_scalar_node&);
    immutable_scalar_node& operator=(const immutable_scalar_node&);

    // Use make_immutable_scalar_node
    immutable_scalar_node(const dtype& dt, char* originptr)
        : m_dtype(dt), m_originptr(originptr)
    {
    }

public:
    virtual ~immutable_scalar_node() {
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
        return read_access_flag | immutable_access_flag;
    }
        
    const char *get_readonly_originptr() const
    {
        return m_originptr;
    }

    memory_block_ptr get_memory_block() const
    {
        return memory_block_ptr();
    }

    ndarray_node_ptr as_dtype(const dtype& dt,
                        dnd::assign_error_mode errmode, bool allow_in_place);

    ndarray_node_ptr apply_linear_index(
                    int ndim, const bool *remove_axis,
                    const intptr_t *start_index, const intptr_t *index_strides,
                    const intptr_t *shape,
                    bool allow_in_place);

    const char *node_name() const {
        return "immutable_scalar";
    }

    void debug_dump_extra(std::ostream& o, const std::string& indent) const;

    friend ndarray_node_ptr make_immutable_scalar_node(const dtype& dt, const char* data);
};

ndarray_node_ptr make_immutable_scalar_node(const dtype& dt, const char* data);

template<class T>
typename enable_if<is_dtype_scalar<T>::value, ndarray_node_ptr>::type make_immutable_scalar_node(const T& value)
{
    return ndarray_node_ptr(make_immutable_scalar_node(make_dtype<T>(), reinterpret_cast<const char *>(&value)));
}

} // namespace dnd

#endif // _DND__IMMUTABLE_SCALAR_NODE_HPP_
