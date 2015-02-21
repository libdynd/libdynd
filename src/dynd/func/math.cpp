//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/apply.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/func/math.hpp>

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

nd::arrfunc nd::decl::cos::as_arrfunc()
{
#ifdef DYND_CUDA
  ndt::type pattern_tp("(M[R]) -> M[R]");
#else
  ndt::type pattern_tp("(R) -> R");
#endif

  vector<nd::arrfunc> children;
  children.push_back(functional::apply<float (*)(float), &dynd::cos>());
  children.push_back(functional::apply<double (*)(double), &dynd::cos>());
#ifdef DYND_CUDA
  children.push_back(functional::apply<kernel_request_cuda_device>(
      get_cuda_device_cos<float (*)(float)>()));
  children.push_back(functional::apply<kernel_request_cuda_device>(
      get_cuda_device_cos<double (*)(double)>()));
#endif

  return functional::elwise(functional::multidispatch(pattern_tp, children));
}

nd::arrfunc nd::decl::sin::as_arrfunc()
{
#ifdef DYND_CUDA
  ndt::type pattern_tp("(M[R]) -> M[R]");
#else
  ndt::type pattern_tp("(R) -> R");
#endif

  vector<nd::arrfunc> children;
  children.push_back(functional::apply<float (*)(float), &dynd::sin>());
  children.push_back(functional::apply<double (*)(double), &dynd::sin>());
#ifdef DYND_CUDA
  children.push_back(functional::apply<kernel_request_cuda_device>(
      get_cuda_device_sin<float (*)(float)>()));
  children.push_back(functional::apply<kernel_request_cuda_device>(
      get_cuda_device_sin<double (*)(double)>()));
#endif

  return functional::elwise(functional::multidispatch(pattern_tp, children));
}

nd::arrfunc nd::decl::tan::as_arrfunc()
{
#ifdef DYND_CUDA
  ndt::type pattern_tp("(M[R]) -> M[R]");
#else
  ndt::type pattern_tp("(R) -> R");
#endif

  vector<nd::arrfunc> children;
  children.push_back(functional::apply<float (*)(float), &dynd::tan>());
  children.push_back(functional::apply<double (*)(double), &dynd::tan>());
#ifdef DYND_CUDA
  children.push_back(functional::apply<kernel_request_cuda_device>(
      get_cuda_device_tan<float (*)(float)>()));
  children.push_back(functional::apply<kernel_request_cuda_device>(
      get_cuda_device_tan<double (*)(double)>()));
#endif

  return functional::elwise(functional::multidispatch(pattern_tp, children));
}

nd::arrfunc nd::decl::exp::as_arrfunc()
{
#ifdef DYND_CUDA
  ndt::type pattern_tp("(M[R]) -> M[R]");
#else
  ndt::type pattern_tp("(R) -> R");
#endif

  vector<nd::arrfunc> children;
  children.push_back(functional::apply<float (*)(float), &dynd::exp>());
  children.push_back(functional::apply<double (*)(double), &dynd::exp>());
#ifdef DYND_CUDA
  children.push_back(functional::apply<kernel_request_cuda_device>(
      get_cuda_device_exp<float (*)(float)>()));
  children.push_back(functional::apply<kernel_request_cuda_device>(
      get_cuda_device_exp<double (*)(double)>()));
#endif

  return functional::elwise(functional::multidispatch(pattern_tp, children));
}


nd::decl::cos nd::cos;
nd::decl::sin nd::sin;
nd::decl::tan nd::tan;
nd::decl::exp nd::exp;