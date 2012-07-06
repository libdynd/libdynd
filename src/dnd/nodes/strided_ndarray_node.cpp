//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>
#include <string>

#include <dnd/nodes/strided_ndarray_node.hpp>
#include <dnd/memblock/fixed_size_pod_memory_block.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>

using namespace std;
using namespace dnd;

dnd::strided_ndarray_node::strided_ndarray_node(const dtype& dt, int ndim,
                                const intptr_t *shape, const intptr_t *strides,
                                char *originptr, int access_flags, const memory_block_ptr& memblock)
    : m_ndim(ndim), m_access_flags(access_flags), m_shape(ndim, shape), m_dtype(dt),
      m_originptr(originptr), m_strides(ndim, strides), m_memblock(memblock)
{
}

dnd::strided_ndarray_node::strided_ndarray_node(const dtype& dt, int ndim,
                                const intptr_t *shape, const intptr_t *strides,
                                const char *originptr, int access_flags, const memory_block_ptr& memblock)
    : m_ndim(ndim), m_access_flags(access_flags), m_shape(ndim, shape), m_dtype(dt),
      m_originptr(const_cast<char *>(originptr)), m_strides(ndim, strides), m_memblock(memblock)
{
    if (access_flags&write_access_flag) {
        throw runtime_error("Cannot create a writeable strided ndarray node from readonly data");
    }
}

#ifdef DND_RVALUE_REFS
dnd::strided_ndarray_node::strided_ndarray_node(const dtype& dt, int ndim,
                                const intptr_t *shape, const intptr_t *strides,
                                char *originptr, int access_flags, memory_block_ptr&& memblock)
    : m_ndim(ndim), m_access_flags(access_flags), m_shape(ndim, shape), m_dtype(dt),
      m_originptr(originptr), m_strides(ndim, strides), m_memblock(DND_MOVE(memblock))
{
}

dnd::strided_ndarray_node::strided_ndarray_node(const dtype& dt, int ndim,
                                const intptr_t *shape, const intptr_t *strides,
                                const char *originptr, int access_flags, memory_block_ptr&& memblock)
    : m_ndim(ndim), m_access_flags(access_flags), m_shape(ndim, shape), m_dtype(dt),
      m_originptr(const_cast<char *>(originptr)), m_strides(ndim, strides), m_memblock(DND_MOVE(memblock))
{
    if (access_flags&write_access_flag) {
        throw runtime_error("Cannot create a writeable strided ndarray node from readonly data");
    }
}
#endif

dnd::strided_ndarray_node::strided_ndarray_node(const dtype& dt, int ndim,
                                const intptr_t *shape, const int *axis_perm)
    : m_ndim(ndim), m_access_flags(read_access_flag | write_access_flag), m_shape(ndim, shape), m_dtype(dt),
      m_originptr(NULL), m_strides(ndim), m_memblock()
{
    // Build the strides using the ordering and shape
    intptr_t num_elements = 1;
    intptr_t stride = dt.element_size();
    for (int i = 0; i < ndim; ++i) {
        int p = axis_perm[i];
        intptr_t size = shape[p];
        if (size == 1) {
            m_strides[p] = 0;
        } else {
            m_strides[p] = stride;
            stride *= size;
            num_elements *= size;
        }
    }

    m_memblock = make_fixed_size_pod_memory_block(dt.alignment(), dt.element_size() * num_elements, &m_originptr);
}

ndarray_node_ptr dnd::strided_ndarray_node::as_dtype(const dtype& dt,
                    dnd::assign_error_mode errmode, bool allow_in_place)
{
    if (allow_in_place) {
        m_dtype = make_conversion_dtype(dt, m_dtype, errmode);
        return as_ndarray_node_ptr();
    } else {
        return make_strided_ndarray_node(
                        make_conversion_dtype(dt, m_dtype, errmode),
                        m_ndim, m_shape.get(), m_strides.get(),
                        m_originptr, m_access_flags, m_memblock);
    }
}

ndarray_node_ptr dnd::strided_ndarray_node::apply_linear_index(
                int ndim, const bool *remove_axis,
                const intptr_t *start_index, const intptr_t *index_strides,
                const intptr_t *shape,
                bool allow_in_place)
{
    /*
    cout << "Applying linear index:\n";
    cout << "ndim: " << ndim << "\n";
    cout << "remove_axis: ";
    for (int i = 0; i < ndim; ++i) {
        cout << remove_axis[i] << " ";
    }
    cout << "\n";
    cout << "start_index: ";
    for (int i = 0; i < ndim; ++i) {
        cout << start_index[i] << " ";
    }
    cout << "\n";
    cout << "index_strides: ";
    for (int i = 0; i < ndim; ++i) {
        cout << index_strides[i] << " ";
    }
    cout << "\n";
    cout << "shape: ";
    for (int i = 0; i < ndim; ++i) {
        cout << shape[i] << " ";
    }
    cout << "\n";
    */

    // Ignore the leftmost dimensions to which this node would broadcast
    if (ndim > m_ndim) {
        remove_axis += (ndim - m_ndim);
        start_index += (ndim - m_ndim);
        index_strides += (ndim - m_ndim);
        shape += (ndim - m_ndim);
        ndim = m_ndim;
    }

    // For each axis not being removed, apply the start_index and index_strides
    // to originptr and the node's strides, respectively. At the same time,
    // apply the remove_axis compression to the strides and shape.
    if (allow_in_place) {
        int j = 0;
        for (int i = 0; i < m_ndim; ++i) {
            m_originptr += m_strides[i] * start_index[i];
            if (!remove_axis[i]) {
                if (m_shape[i] != 1) {
                    m_strides[j] = m_strides[i] * index_strides[i];
                    m_shape[j] = shape[i];
                } else {
                    m_strides[j] = 0;
                    m_shape[j] = 1;
                }
                ++j;
            }
        }
        m_ndim = j;

        return as_ndarray_node_ptr();
    } else {
        // Apply the start_index to m_originptr
        char *new_originptr = m_originptr;
        dimvector new_strides(m_ndim);
        dimvector new_shape(m_ndim);

        int j = 0;
        for (int i = 0; i < m_ndim; ++i) {
            new_originptr += m_strides[i] * start_index[i];
            if (!remove_axis[i]) {
                if (m_shape[i] != 1) {
                    new_strides[j] = m_strides[i] * index_strides[i];
                    new_shape[j] = shape[i];
                } else {
                    new_strides[j] = 0;
                    new_shape[j] = 1;
                }
                ++j;
            }
        }
        ndim = j;

        return make_strided_ndarray_node(m_dtype, ndim, new_shape.get(), new_strides.get(),
                                        new_originptr, m_access_flags, m_memblock);
    }
}

void dnd::strided_ndarray_node::debug_dump_extra(ostream& o, const string& indent) const
{
    o << indent << " strides: (";
    for (int i = 0; i < m_ndim; ++i) {
        o << m_strides[i];
        if (i != m_ndim - 1) {
            o << ", ";
        }
    }
    o << ")\n";
    o << indent << " originptr: " << (void *)m_originptr << "\n";
    o << indent << " memoryblock owning the data:\n";
    memory_block_debug_dump(m_memblock.get(), o, indent + " ");
}

ndarray_node_ptr dnd::make_strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, char *originptr, int access_flags, const memory_block_ptr& memblock)
{
    // Allocate the memory_block
    char *result = reinterpret_cast<char *>(malloc(sizeof(memory_block_data) + sizeof(strided_ndarray_node)));
    if (result == NULL) {
        throw bad_alloc();
    }
    // Placement new
    new (result + sizeof(memory_block_data))
            strided_ndarray_node(dt, ndim, shape, strides, originptr, access_flags, memblock);
    return ndarray_node_ptr(new (result) memory_block_data(1, ndarray_node_memory_block_type), false);
}

ndarray_node_ptr dnd::make_strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, const char *originptr, int access_flags, const memory_block_ptr& memblock)
{
    // Allocate the memory_block
    char *result = reinterpret_cast<char *>(malloc(sizeof(memory_block_data) + sizeof(strided_ndarray_node)));
    if (result == NULL) {
        throw bad_alloc();
    }
    // Placement new
    new (result + sizeof(memory_block_data))
            strided_ndarray_node(dt, ndim, shape, strides, originptr, access_flags, memblock);
    return ndarray_node_ptr(new (result) memory_block_data(1, ndarray_node_memory_block_type), false);
}

#ifdef DND_RVALUE_REFS
ndarray_node_ptr dnd::make_strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, char *originptr, int access_flags, memory_block_ptr&& memblock)
{
    // Allocate the memory_block
    char *result = reinterpret_cast<char *>(malloc(sizeof(memory_block_data) + sizeof(strided_ndarray_node)));
    if (result == NULL) {
        throw bad_alloc();
    }
    // Placement new
    new (result + sizeof(memory_block_data))
            strided_ndarray_node(dt, ndim, shape, strides, originptr, access_flags, DND_MOVE(memblock));
    return ndarray_node_ptr(new (result) memory_block_data(1, ndarray_node_memory_block_type), false);
}

ndarray_node_ptr dnd::make_strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, const char *originptr, int access_flags, memory_block_ptr&& memblock)
{
    // Allocate the memory_block
    char *result = reinterpret_cast<char *>(malloc(sizeof(memory_block_data) + sizeof(strided_ndarray_node)));
    if (result == NULL) {
        throw bad_alloc();
    }
    // Placement new
    new (result + sizeof(memory_block_data))
            strided_ndarray_node(dt, ndim, shape, strides, originptr, access_flags, DND_MOVE(memblock));
    return ndarray_node_ptr(new (result) memory_block_data(1, ndarray_node_memory_block_type), false);
}
#endif

ndarray_node_ptr dnd::make_strided_ndarray_node(const dtype& dt, int ndim, const intptr_t *shape, const int *axis_perm)
{
    // Allocate the memory_block
    char *result = reinterpret_cast<char *>(malloc(sizeof(memory_block_data) + sizeof(strided_ndarray_node)));
    if (result == NULL) {
        throw bad_alloc();
    }
    // Placement new
    new (result + sizeof(memory_block_data))
            strided_ndarray_node(dt, ndim, shape, axis_perm);
    return ndarray_node_ptr(new (result) memory_block_data(1, ndarray_node_memory_block_type), false);
}
