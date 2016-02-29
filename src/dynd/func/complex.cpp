//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/complex.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/conj_kernel.hpp>
#include <dynd/kernels/imag_kernel.hpp>
#include <dynd/kernels/real_kernel.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::real::make()
{
  map<type_id_t, callable> children =
      callable::make_all<real_kernel, type_id_sequence<complex_float32_id, complex_float64_id>>();

  return functional::elwise(functional::dispatch(
      ndt::type("(Scalar) -> Scalar"),
      [children](const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                 const ndt::type *src_tp) mutable -> callable & { return children[src_tp[0].get_id()]; }));
}

DYND_DEFAULT_DECLFUNC_GET(nd::real)

DYND_API struct nd::real nd::real;

DYND_API nd::callable nd::imag::make()
{
  map<type_id_t, callable> children =
      callable::make_all<imag_kernel, type_id_sequence<complex_float32_id, complex_float64_id>>();

  return functional::elwise(functional::dispatch(
      ndt::type("(Scalar) -> Scalar"),
      [children](const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                 const ndt::type *src_tp) mutable -> callable & { return children[src_tp[0].get_id()]; }));
}

DYND_DEFAULT_DECLFUNC_GET(nd::imag)

DYND_API struct nd::imag nd::imag;

DYND_API nd::callable nd::conj::make()
{
  map<type_id_t, callable> children =
      callable::make_all<conj_kernel, type_id_sequence<complex_float32_id, complex_float64_id>>();

  return functional::elwise(functional::dispatch(
      ndt::type("(Scalar) -> Scalar"),
      [children](const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                 const ndt::type *src_tp) mutable -> callable & { return children[src_tp[0].get_id()]; }));
}

DYND_DEFAULT_DECLFUNC_GET(nd::conj)

DYND_API struct nd::conj nd::conj;
