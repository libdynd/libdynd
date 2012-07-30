//
// Copyright (C) 2011-2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/nodes/groupby_node.hpp>
#include <dnd/dtypes/categorical_dtype.hpp>

using namespace std;
using namespace dnd;

dnd::groupby_node::groupby_node(const ndarray_node_ptr& data_node, const ndarray_node_ptr& by_node, const dtype& groups)
    : m_data_node(data_node), m_by_node(by_node), m_groups(groups)
{
    if (groups.type_id() != categorical_type_id) {
        throw runtime_error("The groups dtype for a groupby_node must be categorical");
    }
    const categorical_dtype *cdt = static_cast<const categorical_dtype*>(groups.extended());
    m_shape[0] = cdt->get_category_count();
    m_shape[1] = -1;
}

ndarray_node_ptr dnd::groupby_node::as_dtype(const dtype& dt,
                    dnd::assign_error_mode errmode, bool allow_in_place)
{
    throw std::runtime_error("TODO: groupby_node::as_dtype");
}

ndarray_node_ptr dnd::groupby_node::apply_linear_index(
                int ndim, const bool *remove_axis,
                const intptr_t *start_index, const intptr_t *index_strides,
                const intptr_t *shape,
                bool allow_in_place)
{
    throw std::runtime_error("TODO: groupby_node::apply_linear_index");
}

