//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// DEPRECATED

#ifndef _DYND__IMMUTABLE_SCALAR_NODE_HPP_
#define _DYND__IMMUTABLE_SCALAR_NODE_HPP_

#include <dynd/nodes/ndarray_node.hpp>
#include <dynd/dtypes/string_dtype.hpp>

namespace dynd {

template<int N>
ndarray_node_ptr make_static_utf8_string_immutable_scalar_node(const char (&static_string)[N]);

namespace detail {
    ndarray_node_ptr unchecked_make_immutable_scalar_node(const dtype& dt, const char* data);
    ndarray_node_ptr unchecked_make_immutable_scalar_node(const dtype& dt);
} // namespace detail

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
        return read_access_flag | immutable_access_flag;
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
        return "immutable_scalar";
    }

    void debug_print_extra(std::ostream& o, const std::string& indent) const;

    friend ndarray_node_ptr detail::unchecked_make_immutable_scalar_node(const dtype& dt, const char* data);
    friend ndarray_node_ptr detail::unchecked_make_immutable_scalar_node(const dtype& dt);
    template<int N>
    friend ndarray_node_ptr make_static_utf8_string_immutable_scalar_node(const char (&static_string)[N]);
};

/**
 * Makes an immutable scalar node from a POD dtype and its
 * corresponding data, which is coppied into the node.
 */
inline ndarray_node_ptr make_immutable_scalar_node(const dtype& dt, const char* data)
{
    if (dt.get_memory_management() != pod_memory_management) {
        throw std::runtime_error("immutable_scalar_node only supports pod dtypes presently");
    }

    return detail::unchecked_make_immutable_scalar_node(dt, data);
}

/**
 * Makes an immutable scalar node from a POD dtype. To initialize the data,
 * which must only be done immediately after creation, use const_cast<char *> on
 * the get_readonly_originptr() value.
 */
inline ndarray_node_ptr make_immutable_scalar_node(const dtype& dt)
{
    if (dt.get_memory_management() != pod_memory_management) {
        throw std::runtime_error("immutable_scalar_node only supports pod dtypes presently");
    }

    return detail::unchecked_make_immutable_scalar_node(dt);
}

/**
 * Special case function, intended only for static string data like strings
 * in quotes or static char arrays. This creates a blockref dtype, but because
 * the string data is static, there is no memory_block which owns that data.
 */
template<int N>
ndarray_node_ptr make_static_utf8_string_immutable_scalar_node(const int8_t (&static_string)[N])
{
    int8_t *refs[2] = {static_string, static_string + N};
    return detail::unchecked_make_immutable_scalar_node(make_string_dtype(string_encoding_utf_8),
                    reinterpret_cast<char *>(&refs));
}

/**
 * Special case function, intended only for static string data like strings
 * in quotes or static char arrays. This creates a blockref dtype, but because
 * the string data is static, there is no memory_block which owns that data.
 */
template<int N>
ndarray_node_ptr make_static_ascii_string_immutable_scalar_node(const char (&static_string)[N])
{
    const char *refs[2] = {static_string, static_string + N};
    return detail::unchecked_make_immutable_scalar_node(make_string_dtype(string_encoding_ascii),
                    reinterpret_cast<char *>(&refs));
}

/**
 * Special case function, intended only for static string data like strings
 * in quotes or static char arrays. This creates a blockref dtype, but because
 * the string data is static, there is no memory_block which owns that data.
 */
template<int N>
ndarray_node_ptr make_static_utf8_string_immutable_scalar_node(const uint8_t (&static_string)[N])
{
    const uint8_t *refs[2] = {static_string, static_string + N};
    return detail::unchecked_make_immutable_scalar_node(make_string_dtype(string_encoding_utf_8),
                    reinterpret_cast<char *>(&refs));
}

/**
 * Special case function, intended only for static string data like strings
 * in quotes or static char arrays. This creates a blockref dtype, but because
 * the string data is static, there is no memory_block which owns that data.
 */
template<int N>
ndarray_node_ptr make_static_utf16_string_immutable_scalar_node(const uint16_t (&static_string)[N])
{
    const uint16_t *refs[2] = {static_string, static_string + N};
    return detail::unchecked_make_immutable_scalar_node(make_string_dtype(string_encoding_utf_16),
                    reinterpret_cast<char *>(&refs));
}

/**
 * Special case function, intended only for static string data like strings
 * in quotes or static char arrays. This creates a blockref dtype, but because
 * the string data is static, there is no memory_block which owns that data.
 */
template<int N>
ndarray_node_ptr make_static_utf32_string_immutable_scalar_node(const uint32_t (&static_string)[N])
{
    const uint32_t *refs[2] = {static_string, static_string + N};
    return detail::unchecked_make_immutable_scalar_node(make_string_dtype(string_encoding_utf_32),
                    reinterpret_cast<char *>(&refs));
}

} // namespace dynd

#endif // _DYND__IMMUTABLE_SCALAR_NODE_HPP_
