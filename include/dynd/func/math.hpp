//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/math.hpp>

namespace dynd {
namespace nd {
  namespace decl {

    struct cos : arrfunc<cos> {
      static nd::arrfunc as_arrfunc();
    };

    struct sin : arrfunc<sin> {
      static nd::arrfunc as_arrfunc();
    };

    struct tan : arrfunc<tan> {
      static nd::arrfunc as_arrfunc();
    };

    struct exp : arrfunc<exp> {
      static nd::arrfunc as_arrfunc();
    };

  } // namespace dynd::nd::decl

  extern decl::cos cos;
  extern decl::sin sin;
  extern decl::tan tan;
  extern decl::exp exp;

} // namespace dynd::nd
} // namespace dynd