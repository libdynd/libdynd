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

// Trying to figure out what's going on here, getting the error message
//  math.cu
//
//c:\users\mwiebe\appdata\local\temp\tmpxft_00001264_00000000-5_math.cudafe1.stub.c(5): error C2970: 'dynd::nd::functional::apply_function_ck' : template parameter 'func' : 'cos' : an expression involving objects with internal linkage cannot be used as a non-type argument
//
//          C:\Dev\dynd-python\libraries\libdynd\include\dynd/kernels/apply.hpp(192) : see declaration of 'dynd::nd::functional::apply_function_ck'
//
//c:\users\mwiebe\appdata\local\temp\tmpxft_00001264_00000000-5_math.cudafe1.stub.c(6): error C2970: 'dynd::nd::functional::apply_function_ck' : template parameter 'func' : 'cos' : an expression involving objects with internal linkage cannot be used as a non-type argument
//
//          C:\Dev\dynd-python\libraries\libdynd\include\dynd/kernels/apply.hpp(192) : see declaration of 'dynd::nd::functional::apply_function_ck'
//
//c:\users\mwiebe\appdata\local\temp\tmpxft_00001264_00000000-5_math.cudafe1.stub.c(7): error C2970: 'dynd::nd::functional::apply_function_ck' : template parameter 'func' : 'cos' : an expression involving objects with internal linkage cannot be used as a non-type argument
//
//          C:\Dev\dynd-python\libraries\libdynd\include\dynd/kernels/apply.hpp(192) : see declaration of 'dynd::nd::functional::apply_function_ck'
//
//  CMake Error at libdynd_generated_math.cu.obj.cmake:264 (message):
//    Error generating file
//    C:/Dev/dynd-python/build_cuda/libraries/libdynd/CMakeFiles/libdynd.dir/src/dynd/func/RelWithDebInfo/libdynd_generated_math.cu.obj

template <typename func_type>
__global__ void get_cuda_device_cos(void *res)
{
  *reinterpret_cast<func_type *>(res) = static_cast<func_type>(&::dynd::cos);
}

template <typename func_type>
func_type get_cuda_device_cos()
{
  func_type func;
  func_type *cuda_device_func;
  cuda_throw_if_not_success(cudaMalloc(&cuda_device_func, sizeof(func_type)));
  get_cuda_device_cos<func_type> << <1, 1>>> (reinterpret_cast<void *>(cuda_device_func));
  cuda_throw_if_not_success(cudaMemcpy(
      &func, cuda_device_func, sizeof(func_type), cudaMemcpyDeviceToHost));
  cuda_throw_if_not_success(cudaFree(cuda_device_func));

  return func;
}

//DYND_GET_CUDA_DEVICE_FUNC(get_cuda_device_cos, cos)
DYND_GET_CUDA_DEVICE_FUNC(get_cuda_device_sin, sin)
DYND_GET_CUDA_DEVICE_FUNC(get_cuda_device_tan, tan)
DYND_GET_CUDA_DEVICE_FUNC(get_cuda_device_exp, exp)

} // namespace dynd

#endif

nd::arrfunc nd::cos::make()
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

nd::arrfunc nd::sin::make()
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

nd::arrfunc nd::tan::make()
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

nd::arrfunc nd::exp::make()
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


struct nd::cos nd::cos;
struct nd::sin nd::sin;
struct nd::tan nd::tan;
struct nd::exp nd::exp;