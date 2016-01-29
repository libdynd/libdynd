//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>
#include <dynd/types/datetime_util.hpp>

namespace dynd {
namespace ndt {

  class DYND_API time_type : public base_type {
    datetime_tz_t m_timezone;

  public:
    time_type(datetime_tz_t);

    virtual ~time_type();

    inline datetime_tz_t get_timezone() const { return m_timezone; }

    void set_time(const char *arrmeta, char *data, assign_error_mode errmode, int32_t hour, int32_t minute,
                  int32_t second, int32_t tick) const;
    void set_from_utf8_string(const char *arrmeta, char *data, const char *utf8_begin, const char *utf8_end,
                              const eval::eval_context *ectx) const;

    time_hmst get_time(const char *arrmeta, const char *data) const;

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *DYND_UNUSED(arrmeta), bool DYND_UNUSED(blockref_alloc)) const {}
    void arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
                                const intrusive_ptr<memory_block_data> &DYND_UNUSED(embedded_reference)) const
    {
    }
    void arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {}
    void arrmeta_debug_print(const char *DYND_UNUSED(arrmeta), std::ostream &DYND_UNUSED(o),
                             const std::string &DYND_UNUSED(indent)) const
    {
    }

    std::map<std::string, nd::callable> get_dynamic_array_properties() const;
    std::map<std::string, nd::callable> get_dynamic_array_functions() const;

    /** Returns type "time" (with abstract/naive time zone) */
    static const type &make()
    {
      static const type time_tp(new time_type(tz_abstract), false);
      return time_tp;
    }

    /** Returns type "time[tz=<t>]" */
    static type make(datetime_tz_t t) { return type(new time_type(t), false); }
  };

} // namespace dynd::ndt
} // namespace dynd
