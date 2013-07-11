//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The byteswap dtype represents one of the
// built-in dtypes stored in non-native byte order.
//
// TODO: When needed, a mechanism for non built-in
//       dtypes to expose a byteswap interface should
//       be added, which this dtype would use to
//       do the actual swapping.
//
#ifndef _DYND__BYTESWAP_TYPE_HPP_
#define _DYND__BYTESWAP_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/view_type.hpp>

namespace dynd {

class byteswap_type : public base_expression_type {
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
    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    bool is_lossless_assignment(const ndt::type& dst_dt, const ndt::type& src_dt) const;

    bool operator==(const base_type& rhs) const;

    ndt::type with_replaced_storage_type(const ndt::type& replacement_type) const;

    size_t make_operand_to_value_assignment_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;
    size_t make_value_to_operand_assignment_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;
};

namespace ndt {
    /**
     * Makes a byteswapped dtype to view the given dtype with a swapped byte order.
     */
    inline ndt::type make_byteswap(const ndt::type& native_dtype) {
        return ndt::type(new byteswap_type(native_dtype), false);
    }

    inline ndt::type make_byteswap(const ndt::type& native_dtype, const ndt::type& operand_type) {
        return ndt::type(new byteswap_type(native_dtype, operand_type), false);
    }

    template<typename Tnative>
    ndt::type make_byteswap() {
        return ndt::type(new byteswap_type(ndt::make_type<Tnative>()), false);
    }
} // namespace ndt

} // namespace dynd

#endif // _DYND__BYTESWAP_TYPE_HPP_
