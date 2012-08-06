//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__GROUPBY_NODE_HPP_
#define _DND__GROUPBY_NODE_HPP_

#include <dnd/nodes/ndarray_node.hpp>

namespace dnd {

/**
 * NDArray expression node which represents the
 * result of a groupby operation.
 */
class groupby_node : public ndarray_node {
    /** The node containing the data to be grouped */
    ndarray_node_ptr m_data_node;
    /** The node containing the group labels */
    ndarray_node_ptr m_by_node;
    /** The categorical dtype defining the groups */
    dtype m_groups;
    /**
     * The shape of the node. The second entry of this
     * is always -1, since the result can have varying sizes
     */
    intptr_t m_shape[2];

    // Non-copyable
    groupby_node(const groupby_node&);
    groupby_node& operator=(const groupby_node&);

    // Use make_groupby_node to actually create one of these
    groupby_node(const ndarray_node_ptr& data_node, const ndarray_node_ptr& by_node, const dtype& groups);
public:
    virtual ~groupby_node() {
    }

    ndarray_node_category get_category() const
    {
        return groupby_node_category;
    }

    const dtype& get_dtype() const {
        return m_data_node->get_dtype();
    }

    int get_ndim() const
    {
        return 2;
    }

    const intptr_t *get_shape() const
    {
        return m_shape;
    }

    uint32_t get_access_flags() const
    {
        // Readable, and inherit the immutability of the 'data' and 'by' nodes
        return read_access_flag |
            (m_data_node->get_access_flags() & m_by_node->get_access_flags() & immutable_access_flag);
    }

    ndarray_node_ptr as_dtype(const dtype& dt,
                        dnd::assign_error_mode errmode, bool allow_in_place);

    ndarray_node_ptr apply_linear_index(
                    int ndim, const bool *remove_axis,
                    const intptr_t *start_index, const intptr_t *index_strides,
                    const intptr_t *shape,
                    bool allow_in_place);

    const char *node_name() const {
        return "groupby";
    }

    ndarray_node *get_data_node() const {
        return m_data_node.get_node();
    }

    ndarray_node *get_by_node() const {
        return m_by_node.get_node();
    }

    const dtype& get_groups() const {
        return m_groups;
    }

    int get_nop() const {
        return 2;
    }

    ndarray_node *get_opnode(int i) const {
        if (i == 0) {
            return m_data_node.get_node();
        } else if (i == 1) {
            return m_by_node.get_node();
        } else {
            return NULL;
        }
    }

    void debug_dump_extra(std::ostream& o, const std::string& indent) const;

    friend ndarray_node_ptr make_groupby_node(const ndarray_node_ptr& data_node,
                            const ndarray_node_ptr& by_node, const dtype& groups);
};

ndarray_node_ptr make_groupby_node(const ndarray_node_ptr& data_node,
                        const ndarray_node_ptr& by_node, const dtype& groups);


} // namespace dnd

#endif // _DND__GROUPBY_NODE_HPP_