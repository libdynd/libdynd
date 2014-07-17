//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

//
#ifndef _DYND__ADAPT_TYPE_HPP_
#define _DYND__ADAPT_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/view_type.hpp>

namespace dynd {

class adapt_type : public base_expr_type {
    ndt::type m_value_type, m_operand_type;
    nd::string m_op;
    nd::arrfunc m_forward, m_reverse;

public:
    adapt_type(const ndt::type &operand_type, const ndt::type &value_type,
               const nd::string &op);

    virtual ~adapt_type();

    const ndt::type& get_value_type() const {
        return m_value_type;
    }
    const ndt::type& get_operand_type() const {
        return m_operand_type;
    }
    void print_data(std::ostream& o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream& o) const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    ndt::type with_replaced_storage_type(const ndt::type& replacement_type) const;

    size_t make_operand_to_value_assignment_kernel(
        ckernel_builder *ckb, intptr_t ckb_offset, const char *dst_arrmeta,
        const char *src_arrmeta, kernel_request_t kernreq,
        const eval::eval_context *ectx) const;
    size_t make_value_to_operand_assignment_kernel(
        ckernel_builder *ckb, intptr_t ckb_offset, const char *dst_arrmeta,
        const char *src_arrmeta, kernel_request_t kernreq,
        const eval::eval_context *ectx) const;
};

namespace ndt {
    inline ndt::type make_adapt(const ndt::type &operand_type,
                                const ndt::type &value_type,
                                const nd::string &op)
    {
        return ndt::type(new adapt_type(operand_type, value_type, op), false);
    }
} // namespace ndt

} // namespace dynd

#endif // _DYND__ADAPT_TYPE_HPP_
