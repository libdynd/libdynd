//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  extern DYND_API struct DYND_API assign_na : declfunc<assign_na> {
    static callable make();
    static callable &get();
  } assign_na;

  DYND_API void old_assign_na(const ndt::type &option_tp, const char *arrmeta, char *data);

  extern DYND_API struct DYND_API is_na : declfunc<is_na> {
    static callable make();
    static callable &get();
  } is_na;

  DYND_API bool old_is_avail(const ndt::type &option_tp, const char *arrmeta, const char *data);

  DYND_API void set_option_from_utf8_string(const ndt::type &option_tp, const char *arrmeta, char *data, const char *utf8_begin,
                                   const char *utf8_end, const eval::eval_context *ectx);

  inline void set_option_from_utf8_string(const ndt::type &option_tp, const char *arrmeta, char *data,
                                          const std::string &utf8_str, const eval::eval_context *ectx)
  {
    set_option_from_utf8_string(option_tp, arrmeta, data, utf8_str.data(), utf8_str.data() + utf8_str.size(), ectx);
  }

} // namespace dynd::nd
} // namespace dynd
