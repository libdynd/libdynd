//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__IMMUTABLE_BUILTIN_SCALAR_NODE_HPP_
#define _DND__IMMUTABLE_BUILTIN_SCALAR_NODE_HPP_

#include <dnd/nodes/ndarray_node.hpp>
#include <dnd/nodes/immutable_scalar_node.hpp>

namespace dnd {

template<class T>
typename enable_if<is_dtype_scalar<T>::value, ndarray_node_ptr>::type make_immutable_builtin_scalar_node(const T& value);

/**
 * NDArray expression node which holds an immutable scalar
 * of a builtin type.
 */
template <class T>
class immutable_builtin_scalar_node : public ndarray_node {
    /* The data of this node */
    T m_value;

    // Non-copyable
    immutable_builtin_scalar_node(const immutable_builtin_scalar_node&);
    immutable_builtin_scalar_node& operator=(const immutable_builtin_scalar_node&);

    // Use make_immutable_builtin_scalar_node
    immutable_builtin_scalar_node(const T& value)
        : m_value(value)
    {
    }

public:
    virtual ~immutable_builtin_scalar_node() {
    }

    ndarray_node_category get_category() const
    {
        return strided_array_node_category;
    }

    const dtype& get_dtype() const {
        return static_builtin_dtypes[type_id_of<T>::value];
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
        return reinterpret_cast<const char *>(&m_value);
    }

    memory_block_ptr get_memory_block() const
    {
        return memory_block_ptr();
    }

    ndarray_node_ptr as_dtype(const dtype& dt,
                        dnd::assign_error_mode errmode, bool DND_UNUSED(allow_in_place))
    {
        return make_immutable_scalar_node(
                        make_conversion_dtype(dt, static_builtin_dtypes[type_id_of<T>::value], errmode),
                        reinterpret_cast<const char *>(&m_value));
    }

    ndarray_node_ptr apply_linear_index(
                int DND_UNUSED(ndim), const bool *DND_UNUSED(remove_axis),
                const intptr_t *DND_UNUSED(start_index), const intptr_t *DND_UNUSED(index_strides),
                const intptr_t *DND_UNUSED(shape),
                bool DND_UNUSED(allow_in_place))
    {
        return as_ndarray_node_ptr();
    }

    const char *node_name() const {
        return "immutable_builtin_scalar";
    }

    void debug_dump_extra(std::ostream& o, const std::string& indent) const
    {
        o << indent << " data: ";
        hexadecimal_print(o, reinterpret_cast<const char *>(&m_value), sizeof(m_value));
        o << "\n";
        o << indent << " value: " << m_value << "\n";
    }

    template<class U>
    friend typename enable_if<is_dtype_scalar<U>::value, ndarray_node_ptr>::type make_immutable_builtin_scalar_node(const U& value);
};

template<class T>
inline typename enable_if<is_dtype_scalar<T>::value, ndarray_node_ptr>::type make_immutable_builtin_scalar_node(const T& value)
{
    // Allocate the memory_block
    char *result = reinterpret_cast<char *>(malloc(sizeof(memory_block_data) + sizeof(immutable_builtin_scalar_node<T>)));
    if (result == NULL) {
        throw std::bad_alloc();
    }
    // Placement new
    new (result + sizeof(memory_block_data))
            immutable_builtin_scalar_node<T>(value);
    return ndarray_node_ptr(new (result) memory_block_data(1, ndarray_node_memory_block_type), false);
}

} // namespace dnd

#endif // _DND__IMMUTABLE_BUILTIN_SCALAR_NODE_HPP_
