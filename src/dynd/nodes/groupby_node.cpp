//
// Copyright (C) 2011-2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/nodes/groupby_node.hpp>
#include <dynd/dtypes/categorical_dtype.hpp>
#include <dynd/memblock/ndarray_node_memory_block.hpp>

using namespace std;
using namespace dynd;

dynd::groupby_node::groupby_node(const ndarray_node_ptr& data_node, const ndarray_node_ptr& by_node, const dtype& groups)
    : m_data_node(data_node), m_by_node(by_node->as_dtype(groups)), m_groups(groups)
{
    if (groups.type_id() != categorical_type_id) {
        throw runtime_error("The groups dtype for a groupby_node must be categorical");
    }
    const categorical_dtype *cdt = static_cast<const categorical_dtype*>(groups.extended());
    m_shape[0] = cdt->get_category_count();
    m_shape[1] = -1;
}

ndarray_node_ptr dynd::groupby_node::as_dtype(const dtype& /*dt*/,
                    dynd::assign_error_mode /*errmode*/, bool /*allow_in_place*/)
{
    throw std::runtime_error("TODO: groupby_node::as_dtype");
}

ndarray_node_ptr dynd::groupby_node::apply_linear_index(
                int /*ndim*/, const bool * /*remove_axis*/,
                const intptr_t * /*start_index*/, const intptr_t * /*index_strides*/,
                const intptr_t * /*shape*/,
                bool /*allow_in_place*/)
{
    throw std::runtime_error("TODO: groupby_node::apply_linear_index");
}

void dynd::groupby_node::debug_print_extra(std::ostream& o, const std::string& indent) const
{
    o << indent << " groups dtype: " << m_groups << "\n";
}

ndarray_node_ptr dynd::make_groupby_node(const ndarray_node_ptr& data_node,
                        const ndarray_node_ptr& by_node, const dtype& groups)
{
    char *node_memory = NULL;
    ndarray_node_ptr result(make_uninitialized_ndarray_node_memory_block(sizeof(groupby_node), &node_memory));

    // Placement new
    new (node_memory) groupby_node(data_node, by_node, groups);

    return DYND_MOVE(result);
}

