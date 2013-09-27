//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt


#ifndef _DYND__GROUPBY_TYPE_HPP_
#define _DYND__GROUPBY_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/types/pointer_type.hpp>

namespace dynd {

struct groupby_type_metadata {
};

struct groupby_type_data {
    const char *data_values_pointer;
    const char *by_values_pointer;
};

/**
 * The groupby type represents a transformation of
 * operand values and by ndobjects into a 2D variable-sized
 * array whose rows are the groups as specified by a categorical
 * type.
 */
class groupby_type : public base_expression_type {
    ndt::type m_value_type, m_operand_type, m_groups_type;

public:
    groupby_type(const ndt::type& data_values_type, const ndt::type& by_values_type);

    virtual ~groupby_type();

    const ndt::type& get_value_type() const {
        return m_value_type;
    }
    const ndt::type& get_operand_type() const {
        return m_operand_type;
    }
    const ndt::type& get_groups_type() const {
        return m_groups_type;
    }
    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_type(std::ostream& o) const;

    ndt::type get_data_values_type() const;
    ndt::type get_by_values_type() const;

    /**
     * Given some metadata for the groupby type, return metadata
     * for a single element of the data_values array.
     */
    const char *get_data_value_metadata(const char *metadata) const {
        // First at_single gets us to the pointer<array<data_value>> type
        ndt::type d = m_operand_type.at_single(0, &metadata);
        // Second at_single gets us to the data_value type
        d.at_single(0, &metadata);
        return metadata;
    }

    /**
     * Given some metadata for the groupby type, returns the
     * metadata for the pointer type that points at the data
     * values.
     *
     * \param metadata  An instance of groupby type metadata.
     *
     * \returns  The pointer<data_values_type> metadata within the
     *           groupby metadata.
     */
    pointer_type_metadata *get_data_values_pointer_metadata(char *metadata) const {
        m_operand_type.at_single(0, const_cast<const char **>(&metadata));
        return reinterpret_cast<pointer_type_metadata *>(metadata);
    }

    /**
     * Given some metadata for the groupby type, returns the
     * metadata for the pointer type that points at the by
     * values.
     *
     * \param metadata  An instance of groupby type metadata.
     *
     * \returns  The pointer<by_values_type> metadata within the
     *           groupby metadata.
     */
    pointer_type_metadata *get_by_values_pointer_metadata(char *metadata) const {
        m_operand_type.at_single(1, const_cast<const char **>(&metadata));
        return reinterpret_cast<pointer_type_metadata *>(metadata);
    }

        void get_shape(size_t ndim, size_t i, intptr_t *out_shape, const char *metadata, const char *data) const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    ndt::type with_replaced_storage_type(const ndt::type& replacement_type) const;

    size_t make_operand_to_value_assignment_kernel(
                    ckernel_builder *out, size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;
    size_t make_value_to_operand_assignment_kernel(
                    ckernel_builder *out, size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;

    void get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;
};

namespace ndt {
    /**
     * Makes a groupby type.
     */
    inline ndt::type make_groupby(const ndt::type& data_values_type,
                    const ndt::type& by_values_type)
    {
        return ndt::type(new groupby_type(data_values_type,
                        by_values_type), false);
    }
} // namespace ndt

} // namespace dynd

#endif // _DYND__GROUPBY_TYPE_HPP_
