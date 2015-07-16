//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>

namespace dynd {
namespace nd {

  extern struct is_avail : declfunc<is_avail> {
    static arrfunc children[DYND_TYPE_ID_MAX + 1];
    static arrfunc default_child;

    static arrfunc &get_child(const ndt::type &value_tp)
    {
      get();
      return children[value_tp.get_type_id()];
    }

    static arrfunc make();
  } is_avail;

  extern struct assign_na_decl : declfunc<assign_na_decl> {
    static arrfunc children[DYND_TYPE_ID_MAX + 1];
    static arrfunc default_child;

    static arrfunc &get_child(const ndt::type &value_tp)
    {
      get();
      return children[value_tp.get_type_id()];
    }

    static arrfunc make();
  } assign_na_decl;

} // namespace dynd::nd
} // namespace dynd