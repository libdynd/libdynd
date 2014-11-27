//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once
/*
#include <dynd/func/functor_arrfunc.hpp>
#include <dynd/func/lift_arrfunc.hpp>
#include <dynd/pp/list.hpp>
#include <dynd/pp/meta.hpp>

namespace dynd { namespace nd {

#define ELWISE(N)                                                              \
  template <typename func_type>                                                \
  nd::array elwise DYND_PP_PREPEND(                                            \
      func_type func, DYND_PP_META_NAME_RANGE(const nd::array &a, N))          \
  {                                                                            \
    nd::arrfunc af = make_functor_arrfunc(func, false);                        \
    nd::arrfunc laf = lift_arrfunc(af);                                        \
                                                                               \
    nd::array args[N] = {DYND_PP_JOIN((, ), DYND_PP_META_NAME_RANGE(a, N))};   \
    return laf.call(N, args, &eval::default_eval_context);                     \
  }

DYND_PP_JOIN_MAP(ELWISE, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_SRC_MAX)))

#undef ELWISE

#define ELWISE(N)                                                              \
  template <typename obj_type, typename mem_func_type>                         \
  nd::array elwise DYND_PP_PREPEND(                                            \
      obj_type obj,                                                            \
      DYND_PP_PREPEND(mem_func_type mem_func,                                  \
                      DYND_PP_META_NAME_RANGE(const nd::array &a, N)))         \
  {                                                                            \
    nd::arrfunc af = make_functor_arrfunc(obj, mem_func, false);               \
    nd::arrfunc laf = lift_arrfunc(af);                                        \
                                                                               \
    nd::array args[N] = {DYND_PP_JOIN((, ), DYND_PP_META_NAME_RANGE(a, N))};   \
    return laf.call(N, args, &eval::default_eval_context);                     \
  }

DYND_PP_JOIN_MAP(ELWISE, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_SRC_MAX)))

#undef ELWISE

}} // namespace dynd::nd
*/