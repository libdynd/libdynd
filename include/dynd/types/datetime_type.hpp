//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DATETIME_TYPE_HPP_
#define _DYND__DATETIME_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/view_type.hpp>
#include <dynd/string_encodings.hpp>
#include <dynd/types/datetime_util.hpp>
#include <dynd/types/static_type_instances.hpp>

namespace dynd {

enum datetime_unit_t {
    datetime_unit_hour,
    datetime_unit_minute,
    datetime_unit_second,
    datetime_unit_msecond,
    datetime_unit_usecond,
    datetime_unit_nsecond
};

std::ostream& operator<<(std::ostream& o, datetime_unit_t unit);

class datetime_type : public base_type {
    datetime_tz_t m_timezone;

public:
    datetime_type(datetime_tz_t timezone);

    virtual ~datetime_type();

    inline datetime_tz_t get_timezone() const {
        return m_timezone;
    }

    void set_cal(const char *arrmeta, char *data, assign_error_mode errmode,
                    int32_t year, int32_t month, int32_t day,
                    int32_t hour, int32_t min=0, int32_t sec=0, int32_t tick=0) const;
    void set_from_utf8_string(const char *arrmeta, char *data,
                              const char *utf8_begin, const char *utf8_end,
                              const eval::eval_context *ectx) const;

    void get_cal(const char *arrmeta, const char *data,
                    int32_t &out_year, int32_t &out_month, int32_t &out_day,
                    int32_t &out_hour, int32_t &out_min, int32_t &out_sec, int32_t &out_tick) const;

    void print_data(std::ostream& o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream& o) const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    void arrmeta_default_construct(char *DYND_UNUSED(arrmeta),
                                   intptr_t DYND_UNUSED(ndim),
                                   const intptr_t *DYND_UNUSED(shape),
                                   bool DYND_UNUSED(blockref_alloc)) const
    {
    }
    void arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta), memory_block_data *DYND_UNUSED(embedded_reference)) const {
    }
    void arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {
    }
    void arrmeta_debug_print(const char *DYND_UNUSED(arrmeta), std::ostream& DYND_UNUSED(o), const std::string& DYND_UNUSED(indent)) const {
    }

    size_t make_assignment_kernel(ckernel_builder *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp,
                                  const char *dst_arrmeta,
                                  const ndt::type &src_tp,
                                  const char *src_arrmeta,
                                  kernel_request_t kernreq,
                                  const eval::eval_context *ectx) const;

    void get_dynamic_type_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;
    void get_dynamic_type_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const;
    void get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;
    void get_dynamic_array_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const;

    size_t get_elwise_property_index(const std::string& property_name) const;
    ndt::type get_elwise_property_type(size_t elwise_property_index,
                    bool& out_readable, bool& out_writable) const;
    size_t make_elwise_property_getter_kernel(
                    ckernel_builder *ckb, intptr_t ckb_offset,
                    const char *dst_arrmeta,
                    const char *src_arrmeta, size_t src_elwise_property_index,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;
    size_t make_elwise_property_setter_kernel(
                    ckernel_builder *ckb, intptr_t ckb_offset,
                    const char *dst_arrmeta, size_t dst_elwise_property_index,
                    const char *src_arrmeta,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;

    nd::array get_option_nafunc() const;

    bool adapt_type(const ndt::type &operand_tp, const nd::string &op,
                    nd::arrfunc &out_forward, nd::arrfunc &out_reverse) const;
    bool reverse_adapt_type(const ndt::type &value_tp, const nd::string &op,
                            nd::arrfunc &out_forward,
                            nd::arrfunc &out_reverse) const;
};

namespace ndt {
  /** Returns type "datetime" (with abstract/naive time zone) */
  inline const ndt::type &make_datetime()
  {
    return *reinterpret_cast<const ndt::type *>(&types::datetime_tp);
  }
  /** Returns type "datetime[tz=<timezone>]" */
  inline ndt::type make_datetime(datetime_tz_t timezone)
  {
    return ndt::type(new datetime_type(timezone), false);
  }
} // namespace ndt

} // namespace dynd

#endif // _DYND__DATETIME_TYPE_HPP_
