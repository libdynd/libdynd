//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/cuda_kernels.hpp>

using namespace std;
using namespace dynd;

#ifdef __CUDACC__

/*
__global__ void kernels::cuda_parallel_single(char *dst,
                                              array_wrapper<char *, 0> src,
                                              ckernel_prefix *self)
{
  expr_single_t func = self->get_function<expr_single_t>();
  func(dst, src, self);
}

__global__ void kernels::cuda_parallel_single(char *dst,
                                              array_wrapper<char *, 1> src,
                                              ckernel_prefix *self)
{
  expr_single_t func = self->get_function<expr_single_t>();
  func(dst, src, self);
}

__global__ void kernels::cuda_parallel_single(char *dst,
                                              array_wrapper<char *, 2> src,
                                              ckernel_prefix *self)
{
  expr_single_t func = self->get_function<expr_single_t>();
  func(dst, src, self);
}

__global__ void kernels::cuda_parallel_single(char *dst,
                                              array_wrapper<char *, 3> src,
                                              ckernel_prefix *self)
{
  expr_single_t func = self->get_function<expr_single_t>();
  func(dst, src, self);
}

__global__ void kernels::cuda_parallel_single(char *dst,
                                              array_wrapper<char *, 4> src,
                                              ckernel_prefix *self)
{
  expr_single_t func = self->get_function<expr_single_t>();
  func(dst, src, self);
}

__global__ void kernels::cuda_parallel_single(char *dst,
                                              array_wrapper<char *, 5> src,
                                              ckernel_prefix *self)
{
  expr_single_t func = self->get_function<expr_single_t>();
  func(dst, src, self);
}

__global__ void kernels::cuda_parallel_single(char *dst,
                                              array_wrapper<char *, 6> src,
                                              ckernel_prefix *self)
{
  expr_single_t func = self->get_function<expr_single_t>();
  func(dst, src, self);
}
*/

#endif