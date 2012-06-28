//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__IMMUTABLE_SCALAR_NODE_HPP_
#define _DND__IMMUTABLE_SCALAR_NODE_HPP_

#include <dnd/nodes/ndarray_expr_node.hpp>

namespace dnd {

/**
 * NDArray expression node which holds an immutable scalar.
 */
class immutable_scalar_node : public ndarray_expr_node {
    char *m_data;
    /** Builtin storage for small immutable scalars */
    int64_t m_storage[2];

    // Non-copyable
    immutable_scalar_node(const immutable_scalar_node&);
    immutable_scalar_node& operator=(const immutable_scalar_node&);

public:
    immutable_scalar_node(const dtype& dt, const char* data);
    immutable_scalar_node(const dtype& dt, const char* data, int ndim, const intptr_t *shape);

    virtual ~immutable_scalar_node();

    const char *get_readonly_originptr() const {
        return m_data;
    }

    /** Raises an exception, since this node is not writeable */
    void as_readwrite_data_and_strides(int ndim, char **out_originptr, intptr_t *out_strides) const;

    /** Provides the data pointer and strides array for the tree evaluation code */
    void as_readonly_data_and_strides(int ndim, char const **out_originptr, intptr_t *out_strides) const;

    ndarray_expr_node_ptr as_dtype(const dtype& dt,
                        dnd::assign_error_mode errmode, bool allow_in_place);

    ndarray_expr_node_ptr apply_linear_index(
                    int ndim, const bool *remove_axis,
                    const intptr_t *start_index, const intptr_t *index_strides,
                    const intptr_t *shape,
                    bool allow_in_place);

    const char *node_name() const {
        return "immutable_scalar";
    }

    void debug_dump_extra(std::ostream& o, const std::string& indent) const;
};

template<class T>
typename enable_if<is_dtype_scalar<T>::value, ndarray_expr_node_ptr>::type make_immutable_scalar_node(const T& value)
{
    return ndarray_expr_node_ptr(new immutable_scalar_node(make_dtype<T>(), reinterpret_cast<const char *>(&value)));
}


} // namespace dnd

#endif // _DND__IMMUTABLE_SCALAR_NODE_HPP_
