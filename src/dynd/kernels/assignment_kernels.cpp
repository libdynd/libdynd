//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/shortvector.hpp>
#include "single_assigner_builtin.hpp"

using namespace std;
using namespace dynd;

namespace {
    template<class T>
    struct aligned_fixed_size_copy_assign_type {
        static void single(char *dst, const char *const *src,
                           ckernel_prefix *DYND_UNUSED(self))
        {
            *reinterpret_cast<T *>(dst) =
                **reinterpret_cast<const T *const *>(src);
        }

        static void strided(char *dst, intptr_t dst_stride,
                            const char *const *src, const intptr_t *src_stride,
                            size_t count, ckernel_prefix *DYND_UNUSED(self))
        {
            const char *src0 = *src;
            intptr_t src0_stride = *src_stride;
            for (size_t i = 0; i != count; ++i) {
                *reinterpret_cast<T *>(dst) =
                    *reinterpret_cast<const T *>(src0);
                dst += dst_stride;
                src0 += src0_stride;
            }
        }
    };

    template<int N>
    struct aligned_fixed_size_copy_assign;
    template<>
    struct aligned_fixed_size_copy_assign<1> {
        static void single(char *dst, const char *const*src,
                           ckernel_prefix *DYND_UNUSED(self))
        {
            *dst = **src;
        }

        static void strided(char *dst, intptr_t dst_stride,
                            const char *const *src, const intptr_t *src_stride,
                            size_t count, ckernel_prefix *DYND_UNUSED(self))
        {
            const char *src0 = *src;
            intptr_t src0_stride = *src_stride;
            for (size_t i = 0; i != count; ++i) {
                *dst = *src0;
                dst += dst_stride;
                src0 += src0_stride;
            }
        }
    };
    template<>
    struct aligned_fixed_size_copy_assign<2> : public aligned_fixed_size_copy_assign_type<int16_t> {};
    template<>
    struct aligned_fixed_size_copy_assign<4> : public aligned_fixed_size_copy_assign_type<int32_t> {};
    template<>
    struct aligned_fixed_size_copy_assign<8> : public aligned_fixed_size_copy_assign_type<int64_t> {};

    template<int N>
    struct unaligned_fixed_size_copy_assign {
        static void single(char *dst, const char *const *src,
                           ckernel_prefix *DYND_UNUSED(self))
        {
            memcpy(dst, *src, N);
        }

        static void strided(char *dst, intptr_t dst_stride,
                            const char *const *src, const intptr_t *src_stride,
                            size_t count, ckernel_prefix *DYND_UNUSED(self))
        {
            const char *src0 = *src;
            intptr_t src0_stride = *src_stride;
            for (size_t i = 0; i != count; ++i) {
                memcpy(dst, src0, N);
                dst += dst_stride;
                src0 += src0_stride;
            }
        }
    };
}
struct unaligned_copy_ck {
    ckernel_prefix base;
    size_t data_size;
};
static void unaligned_copy_single(char *dst, const char *const *src,
                                  ckernel_prefix *self)
{
    size_t data_size =
        reinterpret_cast<unaligned_copy_ck *>(self)->data_size;
    memcpy(dst, *src, data_size);
}
static void unaligned_copy_strided(char *dst, intptr_t dst_stride,
                                   const char *const *src,
                                   const intptr_t *src_stride, size_t count,
                                   ckernel_prefix *self)
{
    size_t data_size = reinterpret_cast<unaligned_copy_ck *>(self)->data_size;
    const char *src0 = *src;
    intptr_t src0_stride = *src_stride;
    for (size_t i = 0; i != count; ++i) {
        memcpy(dst, src0, data_size);
        dst += dst_stride;
        src0 += src0_stride;
    }
}

size_t dynd::make_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
    if (dst_tp.is_builtin()) {
        if (src_tp.is_builtin()) {
            if (dst_tp.extended() == src_tp.extended()) {
                return make_pod_typed_data_assignment_kernel(ckb, ckb_offset,
                                dst_tp.get_data_size(), dst_tp.get_data_alignment(),
                                kernreq);
            } else {
                return make_builtin_type_assignment_kernel(ckb, ckb_offset,
                                dst_tp.get_type_id(), src_tp.get_type_id(),
                                kernreq, ectx->errmode);
            }
        } else {
            return src_tp.extended()->make_assignment_kernel(ckb, ckb_offset,
                            dst_tp, dst_arrmeta,
                            src_tp, src_arrmeta,
                            kernreq, ectx);
        }
    } else {
        return dst_tp.extended()->make_assignment_kernel(ckb, ckb_offset,
                        dst_tp, dst_arrmeta,
                        src_tp, src_arrmeta,
                        kernreq, ectx);
    }
}

size_t dynd::make_pod_typed_data_assignment_kernel(ckernel_builder *ckb,
                                                   intptr_t ckb_offset,
                                                   size_t data_size,
                                                   size_t data_alignment,
                                                   kernel_request_t kernreq)
{
  bool single = (kernreq == kernel_request_single);
  if (!single && kernreq != kernel_request_strided) {
    stringstream ss;
    ss << "make_pod_typed_data_assignment_kernel: unrecognized request "
       << (int)kernreq;
    throw runtime_error(ss.str());
  }
  ckernel_prefix *result = NULL;
  if (data_size == data_alignment) {
    // Aligned specialization tables
    // No need to reserve more space in the trivial cases, the space for a leaf
    // is already there
    switch (data_size) {
    case 1:
      result = ckb->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
      if (single) {
        result->set_function<expr_single_t>(
            &aligned_fixed_size_copy_assign<1>::single);
      } else {
        result->set_function<expr_strided_t>(
            &aligned_fixed_size_copy_assign<1>::strided);
      }
      return ckb_offset;
    case 2:
      result = ckb->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
      if (single) {
        result->set_function<expr_single_t>(
            &aligned_fixed_size_copy_assign<2>::single);
      } else {
        result->set_function<expr_strided_t>(
            &aligned_fixed_size_copy_assign<2>::strided);
      }
      return ckb_offset;
    case 4:
      result = ckb->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
      if (single) {
        result->set_function<expr_single_t>(
            &aligned_fixed_size_copy_assign<4>::single);
      } else {
        result->set_function<expr_strided_t>(
            &aligned_fixed_size_copy_assign<4>::strided);
      }
      return ckb_offset;
    case 8:
      result = ckb->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
      if (single) {
        result->set_function<expr_single_t>(
            &aligned_fixed_size_copy_assign<8>::single);
      } else {
        result->set_function<expr_strided_t>(
            &aligned_fixed_size_copy_assign<8>::strided);
      }
      return ckb_offset;
    default: {
      unaligned_copy_ck *self =
          ckb->alloc_ck_leaf<unaligned_copy_ck>(ckb_offset);
      if (single) {
        self->base.set_function<expr_single_t>(&unaligned_copy_single);
      } else {
        self->base.set_function<expr_strided_t>(&unaligned_copy_strided);
      }
      self->data_size = data_size;
      return ckb_offset;
    }
    }
  } else {
    // Unaligned specialization tables
    switch (data_size) {
    case 2:
      result = ckb->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
      if (single) {
        result->set_function<expr_single_t>(
            unaligned_fixed_size_copy_assign<2>::single);
      } else {
        result->set_function<expr_strided_t>(
            unaligned_fixed_size_copy_assign<2>::strided);
      }
      return ckb_offset;
    case 4:
      result = ckb->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
      if (single) {
        result->set_function<expr_single_t>(
            unaligned_fixed_size_copy_assign<4>::single);
      } else {
        result->set_function<expr_strided_t>(
            unaligned_fixed_size_copy_assign<4>::strided);
      }
      return ckb_offset;
    case 8:
      result = ckb->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
      if (single) {
        result->set_function<expr_single_t>(
            unaligned_fixed_size_copy_assign<8>::single);
      } else {
        result->set_function<expr_strided_t>(
            unaligned_fixed_size_copy_assign<8>::strided);
      }
      return ckb_offset;
    default: {
      unaligned_copy_ck *self =
          ckb->alloc_ck_leaf<unaligned_copy_ck>(ckb_offset);
      if (single) {
        self->base.set_function<expr_single_t>(&unaligned_copy_single);
      } else {
        self->base.set_function<expr_strided_t>(&unaligned_copy_strided);
      }
      self->data_size = data_size;
      return ckb_offset;
    }
    }
  }
}

namespace {
template<class dst_type, class src_type, assign_error_mode errmode>
struct single_assigner_as_expr_single {
  static void single(char *dst, const char *const *src,
                     ckernel_prefix *DYND_UNUSED(self))
  {
    single_assigner_builtin<dst_type, src_type, errmode>::assign(
        reinterpret_cast<dst_type *>(dst),
        reinterpret_cast<const src_type *>(*src));
  }
};
} // anonymous namespace

static expr_single_t assign_table_single_kernel[builtin_type_id_count-2][builtin_type_id_count-2][4] =
{
#define SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, errmode) \
            &single_assigner_as_expr_single<dst_type, src_type, errmode>::single
        
#define ERROR_MODE_LEVEL(dst_type, src_type) { \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_nocheck), \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_overflow), \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_fractional), \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_inexact) \
    }

#define SRC_TYPE_LEVEL(dst_type) { \
        ERROR_MODE_LEVEL(dst_type, dynd_bool), \
        ERROR_MODE_LEVEL(dst_type, int8_t), \
        ERROR_MODE_LEVEL(dst_type, int16_t), \
        ERROR_MODE_LEVEL(dst_type, int32_t), \
        ERROR_MODE_LEVEL(dst_type, int64_t), \
        ERROR_MODE_LEVEL(dst_type, dynd_int128), \
        ERROR_MODE_LEVEL(dst_type, uint8_t), \
        ERROR_MODE_LEVEL(dst_type, uint16_t), \
        ERROR_MODE_LEVEL(dst_type, uint32_t), \
        ERROR_MODE_LEVEL(dst_type, uint64_t), \
        ERROR_MODE_LEVEL(dst_type, dynd_uint128), \
        ERROR_MODE_LEVEL(dst_type, dynd_float16), \
        ERROR_MODE_LEVEL(dst_type, float), \
        ERROR_MODE_LEVEL(dst_type, double), \
        ERROR_MODE_LEVEL(dst_type, dynd_float128), \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<float>), \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<double>) \
    }

    SRC_TYPE_LEVEL(dynd_bool),
    SRC_TYPE_LEVEL(int8_t),
    SRC_TYPE_LEVEL(int16_t),
    SRC_TYPE_LEVEL(int32_t),
    SRC_TYPE_LEVEL(int64_t),
    SRC_TYPE_LEVEL(dynd_int128),
    SRC_TYPE_LEVEL(uint8_t),
    SRC_TYPE_LEVEL(uint16_t),
    SRC_TYPE_LEVEL(uint32_t),
    SRC_TYPE_LEVEL(uint64_t),
    SRC_TYPE_LEVEL(dynd_uint128),
    SRC_TYPE_LEVEL(dynd_float16),
    SRC_TYPE_LEVEL(float),
    SRC_TYPE_LEVEL(double),
    SRC_TYPE_LEVEL(dynd_float128),
    SRC_TYPE_LEVEL(dynd_complex<float>),
    SRC_TYPE_LEVEL(dynd_complex<double>)
#undef SRC_TYPE_LEVEL
#undef ERROR_MODE_LEVEL
#undef SINGLE_OPERATION_PAIR_LEVEL
};

namespace {
    template<typename dst_type, typename src_type, assign_error_mode errmode>
    struct multiple_assignment_builtin {
        static void strided_assign(
                        char *dst, intptr_t dst_stride,
                        const char *const *src, const intptr_t *src_stride,
                        size_t count, ckernel_prefix *DYND_UNUSED(self))
        {
            const char *src0 = src[0];
            intptr_t src0_stride = src_stride[0];
            for (size_t i = 0; i != count; ++i) {
                single_assigner_builtin<dst_type, src_type, errmode>::assign(
                    reinterpret_cast<dst_type *>(dst),
                    reinterpret_cast<const src_type *>(src0));
                dst += dst_stride;
                src0 += src0_stride;
            }
        }
    };
    template<typename dst_type, typename src_type>
    struct multiple_assignment_builtin<dst_type, src_type, assign_error_nocheck> {
         DYND_CUDA_HOST_DEVICE static void strided_assign(
                        char *dst, intptr_t dst_stride,
                        const char *const *src, const intptr_t *src_stride,
                        size_t count, ckernel_prefix *DYND_UNUSED(self))
        {
            const char *src0 = src[0];
            intptr_t src0_stride = src_stride[0];
            for (size_t i = 0; i != count; ++i) {
                single_assigner_builtin<dst_type, src_type, assign_error_nocheck>::
                    assign(reinterpret_cast<dst_type *>(dst),
                           reinterpret_cast<const src_type *>(src0));
                dst += dst_stride;
                src0 += src0_stride;
            }
        }
    };
} // anonymous namespace

static expr_strided_t assign_table_strided_kernel[builtin_type_id_count-2][builtin_type_id_count-2][4] =
{
#define STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, errmode) \
            &multiple_assignment_builtin<dst_type, src_type, errmode>::strided_assign
        
#define ERROR_MODE_LEVEL(dst_type, src_type) { \
        STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_nocheck), \
        STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_overflow), \
        STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_fractional), \
        STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_inexact) \
    }

#define SRC_TYPE_LEVEL(dst_type) { \
        ERROR_MODE_LEVEL(dst_type, dynd_bool), \
        ERROR_MODE_LEVEL(dst_type, int8_t), \
        ERROR_MODE_LEVEL(dst_type, int16_t), \
        ERROR_MODE_LEVEL(dst_type, int32_t), \
        ERROR_MODE_LEVEL(dst_type, int64_t), \
        ERROR_MODE_LEVEL(dst_type, dynd_int128), \
        ERROR_MODE_LEVEL(dst_type, uint8_t), \
        ERROR_MODE_LEVEL(dst_type, uint16_t), \
        ERROR_MODE_LEVEL(dst_type, uint32_t), \
        ERROR_MODE_LEVEL(dst_type, uint64_t), \
        ERROR_MODE_LEVEL(dst_type, dynd_int128), \
        ERROR_MODE_LEVEL(dst_type, dynd_float16), \
        ERROR_MODE_LEVEL(dst_type, float), \
        ERROR_MODE_LEVEL(dst_type, double), \
        ERROR_MODE_LEVEL(dst_type, dynd_float128), \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<float>), \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<double>) \
    }

    SRC_TYPE_LEVEL(dynd_bool),
    SRC_TYPE_LEVEL(int8_t),
    SRC_TYPE_LEVEL(int16_t),
    SRC_TYPE_LEVEL(int32_t),
    SRC_TYPE_LEVEL(int64_t),
    SRC_TYPE_LEVEL(dynd_int128),
    SRC_TYPE_LEVEL(uint8_t),
    SRC_TYPE_LEVEL(uint16_t),
    SRC_TYPE_LEVEL(uint32_t),
    SRC_TYPE_LEVEL(uint64_t),
    SRC_TYPE_LEVEL(dynd_uint128),
    SRC_TYPE_LEVEL(dynd_float16),
    SRC_TYPE_LEVEL(float),
    SRC_TYPE_LEVEL(double),
    SRC_TYPE_LEVEL(dynd_float128),
    SRC_TYPE_LEVEL(dynd_complex<float>),
    SRC_TYPE_LEVEL(dynd_complex<double>)
#undef SRC_TYPE_LEVEL
#undef ERROR_MODE_LEVEL
#undef STRIDED_OPERATION_PAIR_LEVEL
};

size_t dynd::make_builtin_type_assignment_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                type_id_t dst_type_id, type_id_t src_type_id,
                kernel_request_t kernreq, assign_error_mode errmode)
{
    // Do a table lookup for the built-in range of dynd types
    if (dst_type_id >= bool_type_id && dst_type_id <= complex_float64_type_id &&
                    src_type_id >= bool_type_id && src_type_id <= complex_float64_type_id &&
                    errmode != assign_error_default) {
        // No need to reserve more space, the space for a leaf is already there
        ckernel_prefix *result = ckb->get_at<ckernel_prefix>(ckb_offset);
        kernels::inc_ckb_offset<ckernel_prefix>(ckb_offset);
        switch (kernreq) {
            case kernel_request_single:
                result->set_function<expr_single_t>(
                                assign_table_single_kernel[dst_type_id-bool_type_id]
                                                [src_type_id-bool_type_id][errmode]);
                break;
            case kernel_request_strided:
                result->set_function<expr_strided_t>(
                                assign_table_strided_kernel[dst_type_id-bool_type_id]
                                                [src_type_id-bool_type_id][errmode]);
                break;
            default: {
                stringstream ss;
                ss << "make_builtin_type_assignment_function: unrecognized request " << (int)kernreq;
                throw runtime_error(ss.str());
            }   
        }
        return ckb_offset;
    } else {
        stringstream ss;
        ss << "Cannot assign from " << ndt::type(src_type_id) << " to " << ndt::type(dst_type_id);
        throw runtime_error(ss.str());
    }
}

namespace {
template<int N>
struct wrap_single_as_strided_fixedcount_ck {
    static void strided(char *dst, intptr_t dst_stride, const char *const *src,
                        const intptr_t *src_stride, size_t count,
                        ckernel_prefix *self)
    {
        ckernel_prefix *echild = self->get_child_ckernel(sizeof(ckernel_prefix));
        expr_single_t opchild = echild->get_function<expr_single_t>();
        const char *src_copy[N];
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

template<>
struct wrap_single_as_strided_fixedcount_ck<0> {
  static void strided(char *dst, intptr_t dst_stride,
                      const char *const *DYND_UNUSED(src),
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

  static inline void strided(char *dst, intptr_t dst_stride,
                             const char *const *src, const intptr_t *src_stride,
                             size_t count, ckernel_prefix *self)
  {
    intptr_t nsrc = reinterpret_cast<self_type *>(self)->nsrc;
    shortvector<const char *> src_copy(nsrc, src);
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

size_t dynd::make_kernreq_to_single_kernel_adapter(ckernel_builder *ckb,
                                                   intptr_t ckb_offset, int nsrc,
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
      ckernel_prefix *e = ckb->alloc_ck<ckernel_prefix>(ckb_offset);
      e->set_function<expr_strided_t>(wrap_single_as_strided_fixedcount[nsrc]);
      e->destructor = &simple_wrapper_kernel_destruct;
      return ckb_offset;
    } else {
      wrap_single_as_strided_ck *e =
          ckb->alloc_ck<wrap_single_as_strided_ck>(ckb_offset);
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
