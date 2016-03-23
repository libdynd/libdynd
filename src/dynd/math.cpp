//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/math.hpp>
#include <dynd/functional.hpp>

using namespace std;
using namespace dynd;

#ifdef DYND_CUDA

namespace dynd {

DYND_GET_CUDA_DEVICE_FUNC(get_cuda_device_cos, cos)
DYND_GET_CUDA_DEVICE_FUNC(get_cuda_device_sin, sin)
DYND_GET_CUDA_DEVICE_FUNC(get_cuda_device_tan, tan)
DYND_GET_CUDA_DEVICE_FUNC(get_cuda_device_exp, exp)

} // namespace dynd

#endif

namespace {
// CUDA and MSVC 2015 WORKAROUND: Using these functions directly in the apply
//                                template does not compile.
double mycos(double x) { return cos(x); }
double mysin(double x) { return sin(x); }
double mytan(double x) { return tan(x); }
double myexp(double x) { return exp(x); }
} // anonymous namespace

DYND_API nd::callable nd::cos::make() { return functional::elwise(functional::apply<double (*)(double), &mycos>()); }

DYND_API nd::callable nd::sin::make() { return functional::elwise(functional::apply<double (*)(double), &mysin>()); }

DYND_API nd::callable nd::tan::make() { return functional::elwise(functional::apply<double (*)(double), &mytan>()); }

DYND_API nd::callable nd::exp::make() { return functional::elwise(functional::apply<double (*)(double), &myexp>()); }

DYND_DEFAULT_DECLFUNC_GET(nd::cos)
DYND_DEFAULT_DECLFUNC_GET(nd::sin)
DYND_DEFAULT_DECLFUNC_GET(nd::tan)
DYND_DEFAULT_DECLFUNC_GET(nd::exp)

DYND_API struct nd::cos nd::cos;
DYND_API struct nd::sin nd::sin;
DYND_API struct nd::tan nd::tan;
DYND_API struct nd::exp nd::exp;
