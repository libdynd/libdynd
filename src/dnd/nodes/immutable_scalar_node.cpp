//
// Copyright (C) 2011-2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/nodes/immutable_scalar_node.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>

using namespace std;
using namespace dnd;

dnd::immutable_scalar_node::immutable_scalar_node(const dtype& dt, const char* data)
    : ndarray_node(immutable_scalar_node_type), m_dtype(dt)
{
    if (dt.is_object_type()) {
        throw runtime_error("immutable_scalar_node doesn't support object dtypes yet");
    }

    if (dt.element_size() <= (intptr_t)sizeof(m_storage)) {
        m_data = reinterpret_cast<char *>(m_storage);
    } else {
        m_data = new char[dt.element_size()];
    }

    memcpy(m_data, data, dt.element_size());
}

dnd::immutable_scalar_node::immutable_scalar_node(const dtype& dt, const char* data, int ndim, const intptr_t *shape)
    : ndarray_node(immutable_scalar_node_type), m_dtype(dt)
{
    if (dt.is_object_type()) {
        throw runtime_error("immutable_scalar_node doesn't support object dtypes yet");
    }

    if (dt.element_size() <= (intptr_t)sizeof(m_storage)) {
        m_data = reinterpret_cast<char *>(m_storage);
    } else {
        m_data = new char[dt.element_size()];
    }

    memcpy(m_data, data, dt.element_size());
}

dnd::immutable_scalar_node::~immutable_scalar_node()
{
    if (m_data != reinterpret_cast<const char *>(m_storage)) {
        delete[] m_data;
    }
}

void dnd::immutable_scalar_node::as_readwrite_data_and_strides(int ndim, char **DND_UNUSED(out_originptr),
                                                    intptr_t *DND_UNUSED(out_strides)) const
{
    throw std::runtime_error("cannot write to an immutable scalar dnd::ndarray node");
}

void dnd::immutable_scalar_node::as_readonly_data_and_strides(int ndim, char const **out_originptr,
                                                    intptr_t *out_strides) const
{
    *out_originptr = reinterpret_cast<const char *>(&m_data[0]);
    memset(out_strides, 0, ndim * sizeof(intptr_t));
}

ndarray_node_ref dnd::immutable_scalar_node::as_dtype(const dtype& dt,
                    dnd::assign_error_mode errmode, bool allow_in_place)
{
    if (allow_in_place) {
        m_dtype = make_conversion_dtype(dt, m_dtype, errmode);
        return ndarray_node_ref(this);
    } else {
        return ndarray_node_ref(new immutable_scalar_node(
                        make_conversion_dtype(dt, m_dtype, errmode),
                        reinterpret_cast<const char *>(&m_data[0])));
    }
}

ndarray_node_ref dnd::immutable_scalar_node::apply_linear_index(
                int DND_UNUSED(ndim), const bool *DND_UNUSED(remove_axis),
                const intptr_t *DND_UNUSED(start_index), const intptr_t *DND_UNUSED(index_strides),
                const intptr_t *DND_UNUSED(shape),
                bool DND_UNUSED(allow_in_place))
{
    return ndarray_node_ref(this);
}

void dnd::immutable_scalar_node::debug_dump_extra(std::ostream& o, const std::string& indent) const
{
    o << indent << " data: ";
    hexadecimal_print(o, reinterpret_cast<const char *>(&m_data[0]), m_dtype.element_size());
    o << "\n";
}
