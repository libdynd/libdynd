//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/shortvector.hpp>
#include "single_assigner_builtin.hpp"

using namespace std;
using namespace dynd;

namespace dynd {
namespace kernels {
  template <class T>
  struct aligned_fixed_size_copy_assign_type
      : expr_ck<aligned_fixed_size_copy_assign_type<T>, kernel_request_host,
                1> {
    void single(char *dst, char **src)
    {
      *reinterpret_cast<T *>(dst) = **reinterpret_cast<T **>(src);
    }

    void strided(char *dst, intptr_t dst_stride, char **src,
                 const intptr_t *src_stride, size_t count)
    {
      char *src0 = *src;
      intptr_t src0_stride = *src_stride;
      for (size_t i = 0; i != count; ++i) {
        *reinterpret_cast<T *>(dst) = *reinterpret_cast<T *>(src0);
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  };

  template <int N>
  struct aligned_fixed_size_copy_assign;

  template <>
  struct aligned_fixed_size_copy_assign<1>
      : expr_ck<aligned_fixed_size_copy_assign<1>, kernel_request_host, 1> {
    void single(char *dst, char **src) { *dst = **src; }

    void strided(char *dst, intptr_t dst_stride, char **src,
                 const intptr_t *src_stride, size_t count)
    {
      char *src0 = *src;
      intptr_t src0_stride = *src_stride;
      for (size_t i = 0; i != count; ++i) {
        *dst = *src0;
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  };

  template <>
  struct aligned_fixed_size_copy_assign<2>
      : aligned_fixed_size_copy_assign_type<int16_t> {
  };

  template <>
  struct aligned_fixed_size_copy_assign<4>
      : aligned_fixed_size_copy_assign_type<int32_t> {
  };

  template <>
  struct aligned_fixed_size_copy_assign<8>
      : aligned_fixed_size_copy_assign_type<int64_t> {
  };

  template <int N>
  struct unaligned_fixed_size_copy_assign
      : expr_ck<unaligned_fixed_size_copy_assign<N>, kernel_request_host, 1> {
    static void single(char *dst, char **src) { memcpy(dst, *src, N); }

    static void strided(char *dst, intptr_t dst_stride, char **src,
                        const intptr_t *src_stride, size_t count)
    {
      char *src0 = *src;
      intptr_t src0_stride = *src_stride;
      for (size_t i = 0; i != count; ++i) {
        memcpy(dst, src0, N);
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  };

  struct unaligned_copy_ck
      : expr_ck<unaligned_copy_ck, kernel_request_host, 1> {
    size_t data_size;

    unaligned_copy_ck(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char **src) { memcpy(dst, *src, data_size); }

    void strided(char *dst, intptr_t dst_stride, char **src,
                 const intptr_t *src_stride, size_t count)
    {
      char *src0 = *src;
      intptr_t src0_stride = *src_stride;
      for (size_t i = 0; i != count; ++i) {
        memcpy(dst, src0, data_size);
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  };

#ifdef DYND_CUDA
  template <typename dst_type, typename src_type, assign_error_mode errmode>
  struct cuda_host_to_device_assign_ck
      : expr_ck<cuda_host_to_device_assign_ck<dst_type, src_type, errmode>,
                kernel_request_host, 1> {
    void single(char *dst, char **src)
    {
      dst_type tmp;
      single_assigner_builtin<dst_type, src_type, errmode>::assign(
          &tmp, reinterpret_cast<src_type *>(*src));
      throw_if_not_cuda_success(
          cudaMemcpy(dst, &tmp, sizeof(dst_type), cudaMemcpyHostToDevice));
    }
  };

  template <typename same_type, assign_error_mode errmode>
  struct cuda_host_to_device_assign_ck<same_type, same_type, errmode>
      : expr_ck<cuda_host_to_device_assign_ck<same_type, same_type, errmode>,
                kernel_request_host, 1> {
    void single(char *dst, char **src)
    {
      throw_if_not_cuda_success(
          cudaMemcpy(dst, *src, sizeof(same_type), cudaMemcpyHostToDevice));
    }
  };

  struct cuda_host_to_device_copy_ck
      : expr_ck<cuda_host_to_device_copy_ck, kernel_request_host, 1> {
    size_t data_size;

    cuda_host_to_device_copy_ck(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char **src)
    {
      throw_if_not_cuda_success(
          cudaMemcpy(dst, *src, data_size, cudaMemcpyHostToDevice));
    }
  };

  template <typename dst_type, typename src_type, assign_error_mode errmode>
  struct cuda_device_to_host_assign_ck
      : expr_ck<cuda_device_to_host_assign_ck<dst_type, src_type, errmode>,
                kernel_request_host, 1> {
    void single(char *dst, char **src)
    {
      src_type tmp;
      throw_if_not_cuda_success(
          cudaMemcpy(&tmp, *src, sizeof(src_type), cudaMemcpyDeviceToHost));
      single_assigner_builtin<dst_type, src_type, errmode>::assign(
          reinterpret_cast<dst_type *>(dst), &tmp);
    }
  };

  template <typename same_type, assign_error_mode errmode>
  struct cuda_device_to_host_assign_ck<same_type, same_type, errmode>
      : expr_ck<cuda_device_to_host_assign_ck<same_type, same_type, errmode>,
                kernel_request_host, 1> {
    void single(char *dst, char **src)
    {
      throw_if_not_cuda_success(
          cudaMemcpy(dst, *src, sizeof(same_type), cudaMemcpyDeviceToHost));
    }
  };

  struct cuda_device_to_host_copy_ck
      : expr_ck<cuda_device_to_host_copy_ck, kernel_request_host, 1> {
    size_t data_size;

    cuda_device_to_host_copy_ck(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char **src)
    {
      throw_if_not_cuda_success(
          cudaMemcpy(dst, *src, data_size, cudaMemcpyDeviceToHost));
    }
  };

  template <typename dst_type, typename src_type, assign_error_mode errmode>
  DYND_CUDA_GLOBAL void single_cuda_global_assign_builtin(dst_type *dst,
                                                          src_type *src)
  {
    single_assigner_builtin<dst_type, src_type, errmode>::assign(dst, src);
  }

  template <typename dst_type, typename src_type, assign_error_mode errmode>
  struct single_cuda_device_to_device_assigner_builtin {
    static void assign(dst_type *DYND_UNUSED(dst), src_type *DYND_UNUSED(src))
    {
      std::stringstream ss;
      ss << "assignment from " << ndt::make_type<src_type>()
         << " in CUDA global memory to ";
      ss << ndt::make_type<dst_type>() << " in CUDA global memory ";
      ss << "with error mode " << errmode << " is not implemented";
      throw std::runtime_error(ss.str());
    }
  };
  template <typename dst_type, typename src_type>
  struct single_cuda_device_to_device_assigner_builtin<dst_type, src_type,
                                                       assign_error_nocheck> {
    static void assign(dst_type *dst, src_type *src)
    {
      single_cuda_global_assign_builtin<dst_type, src_type, assign_error_nocheck> << <
          1, 1>>>
          (dst, src);
      throw_if_not_cuda_success(cudaDeviceSynchronize());
    }
  };

  template <class dst_type, class src_type, assign_error_mode errmode>
  struct cuda_device_to_device_assign_ck
      : expr_ck<cuda_device_to_device_assign_ck<dst_type, src_type, errmode>,
                kernel_request_host, 1> {
    void single(char *dst, char **src)
    {
      single_cuda_device_to_device_assigner_builtin<
          dst_type, src_type,
          errmode>::assign(reinterpret_cast<dst_type *>(dst),
                           reinterpret_cast<src_type *>(*src));
    }
  };

  struct cuda_device_to_device_copy_ck
      : expr_ck<cuda_device_to_device_copy_ck, kernel_request_host, 1> {
    size_t data_size;

    cuda_device_to_device_copy_ck(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char **src)
    {
      throw_if_not_cuda_success(
          cudaMemcpy(dst, *src, data_size, cudaMemcpyDeviceToDevice));
    }
  };
#endif

  template <class dst_type, class src_type, assign_error_mode errmode>
  struct assign_ck : expr_ck<assign_ck<dst_type, src_type, errmode>,
                             kernel_request_host, 1> {
    void single(char *dst, char **src)
    {
      single_assigner_builtin<dst_type, src_type, errmode>::assign(
          reinterpret_cast<dst_type *>(dst),
          reinterpret_cast<src_type *>(*src));
    }
  };

} // namespace kernels
} // namespace dynd

size_t dynd::make_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
  if (dst_tp.is_builtin()) {
    if (src_tp.is_builtin()) {
      if (dst_tp.extended() == src_tp.extended()) {
        return make_pod_typed_data_assignment_kernel(
            ckb, ckb_offset, dst_tp.get_data_size(),
            dst_tp.get_data_alignment(), kernreq);
      } else {
        return make_builtin_type_assignment_kernel(
            ckb, ckb_offset, dst_tp.get_type_id(), src_tp.get_type_id(),
            kernreq, ectx->errmode);
      }
    } else {
      return src_tp.extended()->make_assignment_kernel(
          ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta, kernreq,
          ectx);
    }
  } else {
    return dst_tp.extended()->make_assignment_kernel(
        ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta, kernreq,
        ectx);
  }
}

size_t dynd::make_pod_typed_data_assignment_kernel(void *ckb,
                                                   intptr_t ckb_offset,
                                                   size_t data_size,
                                                   size_t data_alignment,
                                                   kernel_request_t kernreq)
{
  if (data_size == data_alignment) {
    // Aligned specialization tables
    switch (data_size) {
    case 1:
      kernels::aligned_fixed_size_copy_assign<1>::create(ckb, kernreq,
                                                         ckb_offset);
      return ckb_offset;
    case 2:
      kernels::aligned_fixed_size_copy_assign<2>::create(ckb, kernreq,
                                                         ckb_offset);
      return ckb_offset;
    case 4:
      kernels::aligned_fixed_size_copy_assign<4>::create(ckb, kernreq,
                                                         ckb_offset);
      return ckb_offset;
    case 8:
      kernels::aligned_fixed_size_copy_assign<8>::create(ckb, kernreq,
                                                         ckb_offset);
      return ckb_offset;
    default:
      kernels::unaligned_copy_ck::create(ckb, kernreq, ckb_offset, data_size);
      return ckb_offset;
    }
  } else {
    // Unaligned specialization tables
    switch (data_size) {
    case 2:
      kernels::unaligned_fixed_size_copy_assign<2>::create(ckb, kernreq,
                                                           ckb_offset);
      return ckb_offset;
    case 4:
      kernels::unaligned_fixed_size_copy_assign<4>::create(ckb, kernreq,
                                                           ckb_offset);
      return ckb_offset;
    case 8:
      kernels::unaligned_fixed_size_copy_assign<8>::create(ckb, kernreq,
                                                           ckb_offset);
      return ckb_offset;
    default:
      kernels::unaligned_copy_ck::create(ckb, kernreq, ckb_offset, data_size);
      return ckb_offset;
    }
  }
}

typedef void *(*create_t)(void *, kernel_request_t, intptr_t &);

static kernels::create_t assign_create[builtin_type_id_count -
                                       2][builtin_type_id_count - 2][4] = {
#define SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, errmode)               \
  &kernels::assign_ck<dst_type, src_type, errmode>::create_opaque

#define ERROR_MODE_LEVEL(dst_type, src_type)                                   \
  {                                                                            \
    SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_nocheck),     \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type,                        \
                                    assign_error_overflow),                    \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type,                        \
                                    assign_error_fractional),                  \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_inexact)  \
  }

#define SRC_TYPE_LEVEL(dst_type)                                               \
  {                                                                            \
    ERROR_MODE_LEVEL(dst_type, dynd_bool), ERROR_MODE_LEVEL(dst_type, int8_t), \
        ERROR_MODE_LEVEL(dst_type, int16_t),                                   \
        ERROR_MODE_LEVEL(dst_type, int32_t),                                   \
        ERROR_MODE_LEVEL(dst_type, int64_t),                                   \
        ERROR_MODE_LEVEL(dst_type, dynd_int128),                               \
        ERROR_MODE_LEVEL(dst_type, uint8_t),                                   \
        ERROR_MODE_LEVEL(dst_type, uint16_t),                                  \
        ERROR_MODE_LEVEL(dst_type, uint32_t),                                  \
        ERROR_MODE_LEVEL(dst_type, uint64_t),                                  \
        ERROR_MODE_LEVEL(dst_type, dynd_uint128),                              \
        ERROR_MODE_LEVEL(dst_type, dynd_float16),                              \
        ERROR_MODE_LEVEL(dst_type, float), ERROR_MODE_LEVEL(dst_type, double), \
        ERROR_MODE_LEVEL(dst_type, dynd_float128),                             \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<float>),                       \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<double>)                       \
  }

    SRC_TYPE_LEVEL(dynd_bool),           SRC_TYPE_LEVEL(int8_t),
    SRC_TYPE_LEVEL(int16_t),             SRC_TYPE_LEVEL(int32_t),
    SRC_TYPE_LEVEL(int64_t),             SRC_TYPE_LEVEL(dynd_int128),
    SRC_TYPE_LEVEL(uint8_t),             SRC_TYPE_LEVEL(uint16_t),
    SRC_TYPE_LEVEL(uint32_t),            SRC_TYPE_LEVEL(uint64_t),
    SRC_TYPE_LEVEL(dynd_uint128),        SRC_TYPE_LEVEL(dynd_float16),
    SRC_TYPE_LEVEL(float),               SRC_TYPE_LEVEL(double),
    SRC_TYPE_LEVEL(dynd_float128),       SRC_TYPE_LEVEL(dynd_complex<float>),
    SRC_TYPE_LEVEL(dynd_complex<double>)
#undef SRC_TYPE_LEVEL
#undef ERROR_MODE_LEVEL
#undef SINGLE_OPERATION_PAIR_LEVEL
};

size_t dynd::make_builtin_type_assignment_kernel(void *ckb, intptr_t ckb_offset,
                                                 type_id_t dst_type_id,
                                                 type_id_t src_type_id,
                                                 kernel_request_t kernreq,
                                                 assign_error_mode errmode)
{
  // Do a table lookup for the built-in range of dynd types
  if (dst_type_id >= bool_type_id && dst_type_id <= complex_float64_type_id &&
      src_type_id >= bool_type_id && src_type_id <= complex_float64_type_id &&
      errmode != assign_error_default) {
    (*assign_create[dst_type_id - bool_type_id][src_type_id -
                                                bool_type_id][errmode])(
        ckb, kernreq, ckb_offset);
    return ckb_offset;
  } else {
    stringstream ss;
    ss << "Cannot assign from " << ndt::type(src_type_id) << " to "
       << ndt::type(dst_type_id);
    throw runtime_error(ss.str());
  }
}

namespace {
template <int N>
struct wrap_single_as_strided_fixedcount_ck {
  static void strided(char *dst, intptr_t dst_stride, char **src,
                      const intptr_t *src_stride, size_t count,
                      ckernel_prefix *self)
  {
    ckernel_prefix *echild = self->get_child_ckernel(sizeof(ckernel_prefix));
    expr_single_t opchild = echild->get_function<expr_single_t>();
    char *src_copy[N];
    for (int j = 0; j < N; ++j) {
      src_copy[j] = src[j];
    }
    for (size_t i = 0; i != count; ++i) {
      opchild(dst, src_copy, echild);
      dst += dst_stride;
      for (int j = 0; j < N; ++j) {
        src_copy[j] += src_stride[j];
      }
    }
  }
};

template <>
struct wrap_single_as_strided_fixedcount_ck<0> {
  static void strided(char *dst, intptr_t dst_stride, char **DYND_UNUSED(src),
                      const intptr_t *DYND_UNUSED(src_stride), size_t count,
                      ckernel_prefix *self)
  {
    ckernel_prefix *echild = self->get_child_ckernel(sizeof(ckernel_prefix));
    expr_single_t opchild = echild->get_function<expr_single_t>();
    for (size_t i = 0; i != count; ++i) {
      opchild(dst, NULL, echild);
      dst += dst_stride;
    }
  }
};

static const expr_strided_t wrap_single_as_strided_fixedcount[7] = {
    &wrap_single_as_strided_fixedcount_ck<0>::strided,
    &wrap_single_as_strided_fixedcount_ck<1>::strided,
    &wrap_single_as_strided_fixedcount_ck<2>::strided,
    &wrap_single_as_strided_fixedcount_ck<3>::strided,
    &wrap_single_as_strided_fixedcount_ck<4>::strided,
    &wrap_single_as_strided_fixedcount_ck<5>::strided,
    &wrap_single_as_strided_fixedcount_ck<6>::strided,
};

static void simple_wrapper_kernel_destruct(ckernel_prefix *self)
{
  self->destroy_child_ckernel(sizeof(ckernel_prefix));
}

struct wrap_single_as_strided_ck {
  typedef wrap_single_as_strided_ck self_type;
  ckernel_prefix base;
  intptr_t nsrc;

  static inline void strided(char *dst, intptr_t dst_stride, char **src,
                             const intptr_t *src_stride, size_t count,
                             ckernel_prefix *self)
  {
    intptr_t nsrc = reinterpret_cast<self_type *>(self)->nsrc;
    shortvector<char *> src_copy(nsrc, src);
    ckernel_prefix *child = self->get_child_ckernel(sizeof(self_type));
    expr_single_t child_fn = child->get_function<expr_single_t>();
    for (size_t i = 0; i != count; ++i) {
      child_fn(dst, src_copy.get(), child);
      dst += dst_stride;
      for (intptr_t j = 0; j < nsrc; ++j) {
        src_copy[j] += src_stride[j];
      }
    }
  }

  static void destruct(ckernel_prefix *self)
  {
    self->destroy_child_ckernel(sizeof(self_type));
  }
};

} // anonymous namespace

size_t dynd::make_kernreq_to_single_kernel_adapter(void *ckb,
                                                   intptr_t ckb_offset,
                                                   int nsrc,
                                                   kernel_request_t kernreq)
{
  switch (kernreq) {
  case kernel_request_single: {
    return ckb_offset;
  }
  case kernel_request_strided: {
    if (nsrc >= 0 &&
        nsrc < (int)(sizeof(wrap_single_as_strided_fixedcount) /
                     sizeof(wrap_single_as_strided_fixedcount[0]))) {
      ckernel_prefix *e =
          reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
              ->alloc_ck<ckernel_prefix>(ckb_offset);
      e->set_function<expr_strided_t>(wrap_single_as_strided_fixedcount[nsrc]);
      e->destructor = &simple_wrapper_kernel_destruct;
      return ckb_offset;
    } else {
      wrap_single_as_strided_ck *e =
          reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
              ->alloc_ck<wrap_single_as_strided_ck>(ckb_offset);
      e->base.set_function<expr_strided_t>(&wrap_single_as_strided_ck::strided);
      e->base.destructor = &wrap_single_as_strided_ck::destruct;
      e->nsrc = nsrc;
      return ckb_offset;
    }
  }
  default: {
    stringstream ss;
    ss << "make_kernreq_to_single_kernel_adapter: unrecognized request "
       << (int)kernreq;
    throw runtime_error(ss.str());
  }
  }
}
