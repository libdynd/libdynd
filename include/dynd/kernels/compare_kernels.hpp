//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t I0, type_id_t I1>
  struct less_kernel
      : base_kernel<less_kernel<I0, I1>, kernel_request_host, 2> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<dynd_bool *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) < *reinterpret_cast<A1 *>(src[1]);
    }

    static ndt::type make_type()
    {
      std::map<string, ndt::type> tp_vars;
      tp_vars["A0"] = ndt::make_type<A0>();
      tp_vars["A1"] = ndt::make_type<A1>();

      return ndt::substitute(ndt::type("(A0, A1) -> bool"), tp_vars, true);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct less_equal_kernel
      : base_kernel<less_equal_kernel<I0, I1>, kernel_request_host, 2> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<dynd_bool *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) <= *reinterpret_cast<A1 *>(src[1]);
    }

    static ndt::type make_type()
    {
      std::map<string, ndt::type> tp_vars;
      tp_vars["A0"] = ndt::make_type<A0>();
      tp_vars["A1"] = ndt::make_type<A1>();

      return ndt::substitute(ndt::type("(A0, A1) -> bool"), tp_vars, true);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct equal_kernel
      : base_kernel<equal_kernel<I0, I1>, kernel_request_host, 2> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<dynd_bool *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) == *reinterpret_cast<A1 *>(src[1]);
    }

    static ndt::type make_type()
    {
      std::map<string, ndt::type> tp_vars;
      tp_vars["A0"] = ndt::make_type<A0>();
      tp_vars["A1"] = ndt::make_type<A1>();

      return ndt::substitute(ndt::type("(A0, A1) -> bool"), tp_vars, true);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct not_equal_kernel
      : base_kernel<not_equal_kernel<I0, I1>, kernel_request_host, 2> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<dynd_bool *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) != *reinterpret_cast<A1 *>(src[1]);
    }

    static ndt::type make_type()
    {
      std::map<string, ndt::type> tp_vars;
      tp_vars["A0"] = ndt::make_type<A0>();
      tp_vars["A1"] = ndt::make_type<A1>();

      return ndt::substitute(ndt::type("(A0, A1) -> bool"), tp_vars, true);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct greater_equal_kernel
      : base_kernel<greater_equal_kernel<I0, I1>, kernel_request_host, 2> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<dynd_bool *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) >= *reinterpret_cast<A1 *>(src[1]);
    }

    static ndt::type make_type()
    {
      std::map<string, ndt::type> tp_vars;
      tp_vars["A0"] = ndt::make_type<A0>();
      tp_vars["A1"] = ndt::make_type<A1>();

      return ndt::substitute(ndt::type("(A0, A1) -> bool"), tp_vars, true);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct greater_kernel
      : base_kernel<greater_kernel<I0, I1>, kernel_request_host, 2> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<dynd_bool *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) > *reinterpret_cast<A1 *>(src[1]);
    }

    static ndt::type make_type()
    {
      std::map<string, ndt::type> tp_vars;
      tp_vars["A0"] = ndt::make_type<A0>();
      tp_vars["A1"] = ndt::make_type<A1>();

      return ndt::substitute(ndt::type("(A0, A1) -> bool"), tp_vars, true);
    }
  };

} // namespace dynd::nd
} // namespace dynd