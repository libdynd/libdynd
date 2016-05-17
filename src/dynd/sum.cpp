//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/arithmetic.hpp>
#include <dynd/callable.hpp>
#include <dynd/callables/sum_callable.hpp>
#include <dynd/callables/sum_dispatch_callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/types/scalar_kind_type.hpp>

using namespace dynd;

DYND_API nd::callable nd::sum = nd::functional::reduction(nd::make_callable<nd::sum_dispatch_callable>(
    ndt::make_type<ndt::callable_type>(ndt::make_type<ndt::scalar_kind_type>(),
                                       {ndt::make_type<ndt::scalar_kind_type>()}),
    nd::callable::make_all<nd::sum_callable,
                           type_sequence<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t,
                                         float16, float, double, dynd::complex<float>, dynd::complex<double>>>()));
