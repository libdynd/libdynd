//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The byteswap type represents one of the
// built-in types stored in non-native byte order.
//
// TODO: When needed, a mechanism for non built-in
//       types to expose a byteswap interface should
//       be added, which this type would use to
//       do the actual swapping.
//
#ifndef _DYND__BYTESWAP_TYPE_HPP_
#define _DYND__BYTESWAP_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/view_type.hpp>

namespace dynd {

class byteswap_type : public base_expr_type {
    ndt::type m_value_type, m_operand_type;

public:
    byteswap_type(const ndt::type& value_type);
    byteswap_type(const ndt::type& value_type, const ndt::type& operand_type);

    virtual ~byteswap_type();

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
                    ckernel_builder *ckb, intptr_t ckb_offset,
                    const char *dst_arrmeta, const char *src_arrmeta,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;
    size_t make_value_to_operand_assignment_kernel(
                    ckernel_builder *ckb, intptr_t ckb_offset,
                    const char *dst_arrmeta, const char *src_arrmeta,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;
};

namespace ndt {
    /**
     * Makes a byteswapped type to view the given type with a swapped byte order.
     */
    inline ndt::type make_byteswap(const ndt::type& native_tp) {
        return ndt::type(new byteswap_type(native_tp), false);
    }

    inline ndt::type make_byteswap(const ndt::type& native_tp, const ndt::type& operand_type) {
        return ndt::type(new byteswap_type(native_tp, operand_type), false);
    }

    template<typename Tnative>
    ndt::type make_byteswap() {
        return ndt::type(new byteswap_type(ndt::make_type<Tnative>()), false);
    }
} // namespace ndt

} // namespace dynd

#endif // _DYND__BYTESWAP_TYPE_HPP_
