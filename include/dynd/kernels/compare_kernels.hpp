//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/tuple_comparison_kernels.hpp>

namespace dynd {
namespace nd {

  template <typename K>
  struct base_comparison_kernel;

  template <template <type_id_t, type_id_t> class K, type_id_t I0, type_id_t I1>
  struct base_comparison_kernel<K<I0, I1>>
      : base_kernel<K<I0, I1>, kernel_request_host, 2> {
    static ndt::type make_type()
    {
      return ndt::arrfunc_type::make({ndt::type(I0), ndt::type(I1)},
                                     ndt::make_type<bool1>());
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct less_kernel : base_comparison_kernel<less_kernel<I0, I1>> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) <
          static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct less_kernel<I0, I0> : base_comparison_kernel<less_kernel<I0, I0>> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) < *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct less_equal_kernel : base_comparison_kernel<less_equal_kernel<I0, I1>> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) <=
          static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct less_equal_kernel<I0, I0>
      : base_comparison_kernel<less_equal_kernel<I0, I0>> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) <= *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct equal_kernel : base_comparison_kernel<equal_kernel<I0, I1>> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) ==
          static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct equal_kernel<I0, I0> : base_comparison_kernel<equal_kernel<I0, I0>> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) == *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <>
  struct equal_kernel<string_type_id, string_type_id>
      : base_comparison_kernel<equal_kernel<string_type_id, string_type_id>> {
    void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
    {
      std::cout << "equal_kernel<string_type_id, string_type_id>::single"
                << std::endl;
    }
  };

  struct tuple_compare_equality_kernel
      : base_kernel<tuple_compare_equality_kernel, kernel_request_host, 2> {
    typedef tuple_compare_equality_kernel extra_type;

    size_t field_count;
    const size_t *src0_data_offsets, *src1_data_offsets;
    // After this are field_count sorting_less kernel offsets, for
    // src0.field_i <op> src1.field_i
    // with each 0 <= i < field_count

    void single(char *dst, char *const *src)
    {
      const size_t *kernel_offsets = reinterpret_cast<const size_t *>(this + 1);
      char *child_src[2];
      for (size_t i = 0; i != field_count; ++i) {
        ckernel_prefix *echild = reinterpret_cast<ckernel_prefix *>(
            reinterpret_cast<char *>(this) + kernel_offsets[i]);
        expr_single_t opchild = echild->get_function<expr_single_t>();
        // if (src0.field_i < src1.field_i) return true
        child_src[0] = src[0] + src0_data_offsets[i];
        child_src[1] = src[1] + src1_data_offsets[i];
        int child_dst;
        opchild(reinterpret_cast<char *>(&child_dst), child_src, echild);
        if (!child_dst) {
          *reinterpret_cast<int *>(dst) = false;
          return;
        }
      }
      *reinterpret_cast<int *>(dst) = true;
    }

    static void destruct(ckernel_prefix *self)
    {
      extra_type *e = reinterpret_cast<extra_type *>(self);
      const size_t *kernel_offsets = reinterpret_cast<const size_t *>(e + 1);
      size_t field_count = e->field_count;
      for (size_t i = 0; i != field_count; ++i) {
        self->destroy_child_ckernel(kernel_offsets[i]);
      }
    }
  };

  struct tuple_compare_inequality_kernel
      : base_kernel<tuple_compare_equality_kernel, kernel_request_host, 2> {
    typedef tuple_compare_equality_kernel extra_type;

    size_t field_count;
    const size_t *src0_data_offsets, *src1_data_offsets;
    // After this are field_count sorting_less kernel offsets, for
    // src0.field_i <op> src1.field_i
    // with each 0 <= i < field_count

    void single(char *dst, char *const *src)
    {
      const size_t *kernel_offsets = reinterpret_cast<const size_t *>(this + 1);
      char *child_src[2];
      for (size_t i = 0; i != field_count; ++i) {
        ckernel_prefix *echild = reinterpret_cast<ckernel_prefix *>(
            reinterpret_cast<char *>(this) + kernel_offsets[i]);
        expr_single_t opchild = echild->get_function<expr_single_t>();
        // if (src0.field_i < src1.field_i) return true
        child_src[0] = src[0] + src0_data_offsets[i];
        child_src[1] = src[1] + src1_data_offsets[i];
        int child_dst;
        opchild(reinterpret_cast<char *>(&child_dst), child_src, echild);
        if (child_dst) {
          *reinterpret_cast<int *>(dst) = true;
          return;
        }
      }
      *reinterpret_cast<int *>(dst) = false;
    }

    static void destruct(ckernel_prefix *self)
    {
      extra_type *e = reinterpret_cast<extra_type *>(self);
      const size_t *kernel_offsets = reinterpret_cast<const size_t *>(e + 1);
      size_t field_count = e->field_count;
      for (size_t i = 0; i != field_count; ++i) {
        self->destroy_child_ckernel(kernel_offsets[i]);
      }
    }
  };

  template <>
  struct equal_kernel<tuple_type_id, tuple_type_id>
      : base_comparison_kernel<equal_kernel<tuple_type_id, tuple_type_id>> {
    void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
    {
      //      equal(dst, src, reinterpret_cast<ckernel_prefix *>(this));
    }

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const ndt::arrfunc_type *DYND_UNUSED(af_tp), char *DYND_UNUSED(data),
        void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context *ectx,
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      return make_tuple_comparison_kernel(ckb, ckb_offset, *src_tp,
                                          src_arrmeta[0], src_arrmeta[1],
                                          comparison_type_equal, ectx);
    }

    /*
       typedef equal_kernel extra_type;

        size_t field_count;
        const size_t *src0_data_offsets, *src1_data_offsets;

        static void destruct(ckernel_prefix *self)
        {
          extra_type *e = reinterpret_cast<extra_type *>(self);
          const size_t *kernel_offsets = reinterpret_cast<const size_t *>(e +
       1);
          size_t field_count = e->field_count;
          for (size_t i = 0; i != field_count; ++i) {
            self->destroy_child_ckernel(kernel_offsets[i]);
          }
        }

        static void equal(char *dst, char *const *src, ckernel_prefix *extra)
        {
          char *eraw = reinterpret_cast<char *>(extra);
          extra_type *e = reinterpret_cast<extra_type *>(extra);
          size_t field_count = e->field_count;
          const size_t *src0_data_offsets = e->src0_data_offsets;
          const size_t *src1_data_offsets = e->src1_data_offsets;
          const size_t *kernel_offsets = reinterpret_cast<const size_t *>(e +
       1);
          char *child_src[2];
          for (size_t i = 0; i != field_count; ++i) {
            ckernel_prefix *echild =
                reinterpret_cast<ckernel_prefix *>(eraw + kernel_offsets[i]);
            expr_single_t opchild = echild->get_function<expr_single_t>();
            // if (src0.field_i < src1.field_i) return true
            child_src[0] = src[0] + src0_data_offsets[i];
            child_src[1] = src[1] + src1_data_offsets[i];
            int child_dst;
            opchild(reinterpret_cast<char *>(&child_dst), child_src, echild);
            if (!child_dst) {
              *reinterpret_cast<int *>(dst) = false;
              return;
            }
          }
          *reinterpret_cast<int *>(dst) = true;
        }

        void single(char *dst, char *const *src)
        {
          equal(dst, src, reinterpret_cast<ckernel_prefix *>(this));
        }

        static intptr_t instantiate(
            const arrfunc_type_data *DYND_UNUSED(self),
            const ndt::arrfunc_type *DYND_UNUSED(af_tp), char
       *DYND_UNUSED(data),
            void *ckb, intptr_t ckb_offset, const ndt::type
       &DYND_UNUSED(dst_tp),
            const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
            const ndt::type *src_tp, const char *const *src_arrmeta,
            kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context
       *ectx,
            const nd::array &DYND_UNUSED(kwds),
            const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
        {
          intptr_t root_ckb_offset = ckb_offset;
          auto bsd = src_tp->extended<ndt::base_tuple_type>();
          size_t field_count = bsd->get_field_count();

          inc_ckb_offset(ckb_offset,
                         sizeof(extra_type) + field_count * sizeof(size_t));
          reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
              ->reserve(ckb_offset + sizeof(ckernel_prefix));
          extra_type *e = reinterpret_cast<ckernel_builder<kernel_request_host>
       *>(
                              ckb)->get_at<extra_type>(root_ckb_offset);
          e->set_function<expr_single_t>(&extra_type::equal);
          e->destructor = &extra_type::destruct;
          e->field_count = field_count;
          e->src0_data_offsets = bsd->get_data_offsets(src_arrmeta[0]);
          e->src1_data_offsets = bsd->get_data_offsets(src_arrmeta[1]);
          size_t *field_kernel_offsets;
          const uintptr_t *arrmeta_offsets = bsd->get_arrmeta_offsets_raw();
          for (size_t i = 0; i != field_count; ++i) {
            const ndt::type &ft = bsd->get_field_type(i);
            // Reserve space for the child, and save the offset to this
            // field comparison kernel. Have to re-get
            // the pointer because creating the field comparison kernel may
            // move the memory.
            reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                ->reserve(ckb_offset + sizeof(ckernel_prefix));
            e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                    ->get_at<extra_type>(root_ckb_offset);
            field_kernel_offsets = reinterpret_cast<size_t *>(e + 1);
            field_kernel_offsets[i] = ckb_offset - root_ckb_offset;
            const char *field_arrmeta = src_arrmeta[0] + arrmeta_offsets[i];
            ckb_offset =
                make_comparison_kernel(ckb, ckb_offset, ft, field_arrmeta, ft,
                                       field_arrmeta, comparison_type_equal,
       ectx);
          }
          return ckb_offset;
        }
    */
  };

  template <type_id_t I0, type_id_t I1>
  struct not_equal_kernel : base_comparison_kernel<not_equal_kernel<I0, I1>> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) !=
          static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct not_equal_kernel<I0, I0>
      : base_comparison_kernel<not_equal_kernel<I0, I0>> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) != *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct greater_equal_kernel
      : base_comparison_kernel<greater_equal_kernel<I0, I1>> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) >=
          static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct greater_equal_kernel<I0, I0>
      : base_comparison_kernel<greater_equal_kernel<I0, I0>> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) >= *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct greater_kernel : base_comparison_kernel<greater_kernel<I0, I1>> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) >
          static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct greater_kernel<I0, I0>
      : base_comparison_kernel<greater_kernel<I0, I0>> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) > *reinterpret_cast<A0 *>(src[1]);
    }
  };

} // namespace dynd::nd
} // namespace dynd