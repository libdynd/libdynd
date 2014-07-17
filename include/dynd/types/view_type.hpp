//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The view type reinterprets the bytes of
// one type as another.
//
#ifndef _DYND__VIEW_TYPE_HPP_
#define _DYND__VIEW_TYPE_HPP_

#include <dynd/type.hpp>

namespace dynd {

class view_type : public base_expr_type {
    ndt::type m_value_type, m_operand_type;

public:
    view_type(const ndt::type& value_type, const ndt::type& operand_type);

    virtual ~view_type();

    const ndt::type& get_value_type() const {
        return m_value_type;
    }
    const ndt::type& get_operand_type() const {
        return m_operand_type;
    }
    void print_data(std::ostream& o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream& o) const;
        void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    ndt::type with_replaced_storage_type(const ndt::type& replacement_type) const;

    size_t make_operand_to_value_assignment_kernel(
                    ckernel_builder *ckb, intptr_t ckb_offset,
                    const char *dst_arrmeta, const char *src_arrmeta,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;
    size_t make_value_to_operand_assignment_kernel(
                    ckernel_builder *ckb, intptr_t ckb_offset,
                    const char *dst_arrmeta, const char *src_arrmeta,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;

    // Propagate properties and functions from the value type
    void get_dynamic_array_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const
    {
        if (!m_value_type.is_builtin()) {
            m_value_type.extended()->get_dynamic_array_properties(out_properties, out_count);
        }
    }
    void get_dynamic_array_functions(
                    const std::pair<std::string, gfunc::callable> **out_functions,
                    size_t *out_count) const
    {
        if (!m_value_type.is_builtin()) {
            m_value_type.extended()->get_dynamic_array_functions(out_functions, out_count);
        }
    }
};

namespace ndt {
    /**
     * Makes an unaligned type to view the given type without alignment requirements.
     */
    inline ndt::type make_view(const ndt::type& value_type, const ndt::type& operand_type) {
        if (value_type.get_kind() != expr_kind) {
            return ndt::type(new view_type(value_type, operand_type), false);
        } else {
            // When the value type has an expr_kind, we need to chain things together
            // so that the view operation happens just at the primitive level.
            return value_type.tcast<base_expr_type>()->with_replaced_storage_type(
                ndt::type(new view_type(value_type.storage_type(), operand_type), false));
        }
    }

    template<typename Tvalue, typename Toperand>
    ndt::type make_view() {
        return ndt::type(new view_type(ndt::make_type<Tvalue>()), false);
    }
} // namespace ndt

} // namespace dynd

#endif // _DYND__VIEW_TYPE_HPP_
