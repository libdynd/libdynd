//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callable.hpp>
#include <dynd/func/math.hpp>
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
float mycos(float x) { return cos(x); }
double mycos(double x) { return cos(x); }
float mysin(float x) { return sin(x); }
double mysin(double x) { return sin(x); }
float mytan(float x) { return tan(x); }
double mytan(double x) { return tan(x); }
float myexp(float x) { return exp(x); }
double myexp(double x) { return exp(x); }
} // anonymous namespace

DYND_API nd::callable nd::cos::make()
{
  /*
  #ifdef DYND_CUDA
    ndt::type pattern_tp("(M[R]) -> M[R]");
  #else
    ndt::type pattern_tp("(R) -> R");
  #endif
  */
  ndt::type pattern_tp("(R) -> R");

  vector<nd::callable> children;
  children.push_back(functional::apply<float (*)(float), &mycos>());
  children.push_back(functional::apply<double (*)(double), &mycos>());
  /*
  #ifdef DYND_CUDA
    children.push_back(functional::apply<kernel_request_cuda_device>(
        get_cuda_device_cos<float (*)(float)>()));
    children.push_back(functional::apply<kernel_request_cuda_device>(
        get_cuda_device_cos<double (*)(double)>()));
  #endif
  */

  return functional::elwise(functional::multidispatch(pattern_tp, children.begin(), children.end()));
}

DYND_API nd::callable nd::sin::make()
{
  /*
  #ifdef DYND_CUDA
    ndt::type pattern_tp("(M[R]) -> M[R]");
  #else
    ndt::type pattern_tp("(R) -> R");
  #endif
  */
  ndt::type pattern_tp("(R) -> R");

  vector<nd::callable> children;
  children.push_back(functional::apply<float (*)(float), &mysin>());
  children.push_back(functional::apply<double (*)(double), &mysin>());
  /*
  #ifdef DYND_CUDA
    children.push_back(functional::apply<kernel_request_cuda_device>(
        get_cuda_device_sin<float (*)(float)>()));
    children.push_back(functional::apply<kernel_request_cuda_device>(
        get_cuda_device_sin<double (*)(double)>()));
  #endif
  */

  return functional::elwise(functional::multidispatch(pattern_tp, children.begin(), children.end()));
}

DYND_API nd::callable nd::tan::make()
{
  /*
  #ifdef DYND_CUDA
    ndt::type pattern_tp("(M[R]) -> M[R]");
  #else
    ndt::type pattern_tp("(R) -> R");
  #endif
  */
  ndt::type pattern_tp("(R) -> R");

  vector<nd::callable> children;
  children.push_back(functional::apply<float (*)(float), &mytan>());
  children.push_back(functional::apply<double (*)(double), &mytan>());
  /*
  #ifdef DYND_CUDA
    children.push_back(functional::apply<kernel_request_cuda_device>(
        get_cuda_device_tan<float (*)(float)>()));
    children.push_back(functional::apply<kernel_request_cuda_device>(
        get_cuda_device_tan<double (*)(double)>()));
  #endif
  */

  return functional::elwise(functional::multidispatch(pattern_tp, children.begin(), children.end()));
}

DYND_API nd::callable nd::exp::make()
{
  /*
  #ifdef DYND_CUDA
    ndt::type pattern_tp("(M[R]) -> M[R]");
  #else
    ndt::type pattern_tp("(R) -> R");
  #endif
  */
  ndt::type pattern_tp("(R) -> R");

  vector<nd::callable> children;
  children.push_back(functional::apply<float (*)(float), &myexp>());
  children.push_back(functional::apply<double (*)(double), &myexp>());
  /*
  #ifdef DYND_CUDA
    children.push_back(functional::apply<kernel_request_cuda_device>(
        get_cuda_device_exp<float (*)(float)>()));
    children.push_back(functional::apply<kernel_request_cuda_device>(
        get_cuda_device_exp<double (*)(double)>()));
  #endif
  */

  return functional::elwise(functional::multidispatch(pattern_tp, children.begin(), children.end()));
}

DYND_DEFAULT_DECLFUNC_GET(nd::cos)
DYND_DEFAULT_DECLFUNC_GET(nd::sin)
DYND_DEFAULT_DECLFUNC_GET(nd::tan)
DYND_DEFAULT_DECLFUNC_GET(nd::exp)

DYND_API struct nd::cos nd::cos;
DYND_API struct nd::sin nd::sin;
DYND_API struct nd::tan nd::tan;
DYND_API struct nd::exp nd::exp;
