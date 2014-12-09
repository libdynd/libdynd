//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/shortvector.hpp>

using namespace std;
using namespace dynd;

namespace dynd {
namespace kernels {
  template <class T>
  struct aligned_fixed_size_copy_assign_type
      : expr_ck<aligned_fixed_size_copy_assign_type<T>, kernel_request_host,
                1> {
    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<T *>(dst) = **reinterpret_cast<T *const *>(src);
    }

    void strided(char *dst, intptr_t dst_stride, char *const *src,
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
    void single(char *dst, char *const *src) { *dst = **src; }

    void strided(char *dst, intptr_t dst_stride, char *const *src,
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
    static void single(char *dst, char *const *src) { memcpy(dst, *src, N); }

    static void strided(char *dst, intptr_t dst_stride, char *const *src,
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

    void single(char *dst, char *const *src) { memcpy(dst, *src, data_size); }

    void strided(char *dst, intptr_t dst_stride, char *const *src,
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
} // namespace kernels
} // namespace dynd

intptr_t dynd::make_assignment_kernel(
    const arrfunc_type_data *self, const arrfunc_type *af_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type &src_tp, const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &kwds)
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
          self, af_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
          src_arrmeta, kernreq, ectx, kwds);
    }
  } else {
    return dst_tp.extended()->make_assignment_kernel(
        self, af_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta,
        kernreq, ectx, kwds);
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

static kernels::create_t assign_create[builtin_type_id_count -
                                       2][builtin_type_id_count - 2][4] = {
#define SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, errmode)               \
  &kernels::create<kernels::assign_ck<dst_type, src_type, errmode>>

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
  static void strided(char *dst, intptr_t dst_stride, char *const *src,
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
  static void strided(char *dst, intptr_t dst_stride,
                      char *const *DYND_UNUSED(src),
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

  static inline void strided(char *dst, intptr_t dst_stride, char *const *src,
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

#ifdef DYND_CUDA

#include "../types/dynd_complex.cu"
#include "../types/dynd_float16.cu"
#include "../types/dynd_float128.cu"
#include "../types/dynd_int128.cu"
#include "../types/dynd_uint128.cu"

size_t dynd::make_cuda_assignment_kernel(
    const arrfunc_type_data *self, const arrfunc_type *af_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type &src_tp, const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &kwds)
{
  assign_error_mode errmode;
  if (dst_tp.get_type_id() == cuda_device_type_id &&
      src_tp.get_type_id() == cuda_device_type_id) {
    errmode = ectx->cuda_device_errmode;
  } else {
    errmode = ectx->errmode;
  }

  if (dst_tp.without_memory_type().is_builtin()) {
    if (src_tp.without_memory_type().is_builtin()) {
      if (errmode != assign_error_nocheck &&
          is_lossless_assignment(dst_tp, src_tp)) {
        errmode = assign_error_nocheck;
      }

      if (dst_tp.without_memory_type().extended() ==
          src_tp.without_memory_type().extended()) {
        return make_cuda_pod_typed_data_assignment_kernel(
            ckb, ckb_offset, dst_tp.get_type_id() == cuda_device_type_id,
            src_tp.get_type_id() == cuda_device_type_id, dst_tp.get_data_size(),
            dst_tp.get_data_alignment(), kernreq);
      } else {
        return make_cuda_builtin_type_assignment_kernel(
            ckb, ckb_offset, dst_tp.get_type_id() == cuda_device_type_id,
            dst_tp.without_memory_type().get_type_id(), dst_tp.get_data_size(),
            src_tp.get_type_id() == cuda_device_type_id,
            src_tp.without_memory_type().get_type_id(), src_tp.get_data_size(),
            kernreq, errmode);
      }
    } else {
      return src_tp.extended()->make_assignment_kernel(
          self, af_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
          src_arrmeta, kernreq, ectx, kwds);
    }
  } else {
    return dst_tp.extended()->make_assignment_kernel(
        self, af_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta,
        kernreq, ectx, kwds);
  }
}

// This is meant to reflect make_builtin_type_assignment_kernel
size_t dynd::make_cuda_builtin_type_assignment_kernel(
    void *ckb, intptr_t ckb_offset, bool dst_device, type_id_t dst_type_id,
    size_t dst_size, bool src_device, type_id_t src_type_id, size_t src_size,
    kernel_request_t kernreq, assign_error_mode errmode)
{
  if (dst_type_id >= bool_type_id && dst_type_id <= complex_float64_type_id &&
      src_type_id >= bool_type_id && src_type_id <= complex_float64_type_id &&
      errmode != assign_error_default) {
    if (dst_device) {
      if (src_device) {
        kernels::cuda_parallel_ck<1> *self =
            kernels::cuda_parallel_ck<1>::create(ckb, kernreq, ckb_offset, 1,
                                                 1);
        ckb = &self->ckb;
        kernreq |= kernel_request_cuda_device;
        ckb_offset = 0;
      } else {
        kernels::cuda_host_to_device_assign_ck::create(ckb, kernreq, ckb_offset,
                                                       dst_size);
        kernreq = kernel_request_single;
      }
    } else {
      if (src_device) {
        kernels::cuda_device_to_host_assign_ck::create(ckb, kernreq, ckb_offset,
                                                       src_size);
        kernreq = kernel_request_single;
      }
    }
    return make_builtin_type_assignment_kernel(ckb, ckb_offset, dst_type_id,
                                               src_type_id, kernreq, errmode);
  } else {
    stringstream ss;
    ss << "Cannot assign from " << ndt::type(src_type_id);
    if (src_device) {
      ss << " in CUDA global memory";
    }
    ss << " to " << ndt::type(dst_type_id);
    if (dst_device) {
      ss << " in CUDA global memory";
    }
    throw runtime_error(ss.str());
  }
}

// This is meant to reflect make_pod_typed_data_assignment_kernel
size_t dynd::make_cuda_pod_typed_data_assignment_kernel(
    void *out, intptr_t offset_out, bool dst_device, bool src_device,
    size_t data_size, size_t data_alignment, kernel_request_t kernreq)
{
  if (dst_device) {
    if (src_device) {
      kernels::cuda_device_to_device_copy_ck::create(out, kernreq, offset_out,
                                                     data_size);
      return offset_out;
    } else {
      kernels::cuda_host_to_device_copy_ck::create(out, kernreq, offset_out,
                                                   data_size);
      return offset_out;
    }
  } else {
    if (src_device) {
      kernels::cuda_device_to_host_copy_ck::create(out, kernreq, offset_out,
                                                   data_size);
      return offset_out;
    } else {
      return make_pod_typed_data_assignment_kernel(out, offset_out, data_size,
                                                   data_alignment, kernreq);
    }
  }
}

#endif // DYND_CUDA