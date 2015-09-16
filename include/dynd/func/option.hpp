//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/callable.hpp>

namespace dynd {
namespace nd {

  extern struct is_avail : declfunc<is_avail> {
    static callable children[DYND_TYPE_ID_MAX + 1];
    static callable dim_children[2];

    static callable &get_child(const ndt::type &value_tp)
    {
      get();
      return children[value_tp.get_type_id()];
    }

    static callable make();
  } is_avail;

  extern struct assign_na_decl : declfunc<assign_na_decl> {
    static callable children[DYND_TYPE_ID_MAX + 1];

    static callable &get_child(const ndt::type &value_tp)
    {
      get();
      return children[value_tp.get_type_id()];
    }

    static callable make();
  } assign_na_decl;

} // namespace dynd::nd
} // namespace dynd
