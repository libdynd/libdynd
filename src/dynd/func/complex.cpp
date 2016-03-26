//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/complex.hpp>
#include <dynd/functional.hpp>
#include <dynd/callables/conj_dispatch_callable.hpp>
#include <dynd/callables/real_dispatch_callable.hpp>
#include <dynd/callables/imag_dispatch_callable.hpp>
#include <dynd/callables/imag_callable.hpp>
#include <dynd/callables/real_callable.hpp>
#include <dynd/callables/conj_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::real = nd::functional::elwise(nd::make_callable<nd::real_dispatch_callable>(
    ndt::type("(Scalar) -> Scalar"),
    nd::callable::new_make_all<nd::real_callable, type_id_sequence<complex_float32_id, complex_float64_id>>()));

DYND_API nd::callable nd::imag::make()
{
  dispatcher<callable> dispatcher =
      callable::new_make_all<imag_callable, type_id_sequence<complex_float32_id, complex_float64_id>>();

  return functional::elwise(make_callable<imag_dispatch_callable>(ndt::type("(Scalar) -> Scalar"), dispatcher));
}

DYND_DEFAULT_DECLFUNC_GET(nd::imag)

DYND_API struct nd::imag nd::imag;

DYND_API nd::callable nd::conj::make()
{
  dispatcher<callable> dispatcher =
      callable::new_make_all<conj_callable, type_id_sequence<complex_float32_id, complex_float64_id>>();

  return functional::elwise(make_callable<conj_dispatch_callable>(ndt::type("(Scalar) -> Scalar"), dispatcher));
}

DYND_DEFAULT_DECLFUNC_GET(nd::conj)

DYND_API struct nd::conj nd::conj;
