//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/type.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/tuple_comparison_kernels.hpp>
#include <dynd/kernels/compare_kernels.hpp>
#include <dynd/types/base_tuple_type.hpp>

using namespace std;
using namespace dynd;

namespace {
// Sorting less operation when the arrmeta is different
struct tuple_compare_sorting_less_matching_arrmeta_kernel {
  typedef tuple_compare_sorting_less_matching_arrmeta_kernel extra_type;

  ckernel_prefix base;
  size_t field_count;
  const size_t *src_data_offsets;
  // After this are field_count sorting_less kernel offsets, for
  // src#.field_i < src#.field_i with each 0 <= i < field_count

  static void sorting_less(ckernel_prefix *extra, char *dst, char *const *src)
  {
    char *eraw = reinterpret_cast<char *>(extra);
    extra_type *e = reinterpret_cast<extra_type *>(extra);
    size_t field_count = e->field_count;
    const size_t *src_data_offsets = e->src_data_offsets;
    const size_t *kernel_offsets = reinterpret_cast<const size_t *>(e + 1);
    char *child_src[2];
    for (size_t i = 0; i != field_count; ++i) {
      ckernel_prefix *sorting_less_kdp =
          reinterpret_cast<ckernel_prefix *>(eraw + kernel_offsets[i]);
      expr_single_t opchild = sorting_less_kdp->get_function<expr_single_t>();
      size_t data_offset = src_data_offsets[i];
      // if (src0.field_i < src1.field_i) return true
      child_src[0] = src[0] + data_offset;
      child_src[1] = src[1] + data_offset;
      int child_dst;
      opchild(sorting_less_kdp, reinterpret_cast<char *>(&child_dst),
              child_src);
      if (child_dst) {
        *reinterpret_cast<int *>(dst) = true;
        return;
      }
      // if (src1.field_i < src0.field_i) return false
      child_src[0] = src[1] + data_offset;
      child_src[1] = src[0] + data_offset;
      opchild(sorting_less_kdp, reinterpret_cast<char *>(&child_dst),
              child_src);
      if (child_dst) {
        *reinterpret_cast<int *>(dst) = false;
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
      self->get_child_ckernel(kernel_offsets[i])->destroy();
    }
  }
};

// Sorting less operation when the arrmeta is different
struct tuple_compare_sorting_less_diff_arrmeta_kernel {
  typedef tuple_compare_sorting_less_diff_arrmeta_kernel extra_type;

  ckernel_prefix base;
  size_t field_count;
  const size_t *src0_data_offsets, *src1_data_offsets;
  // After this are 2*field_count sorting_less kernel offsets, for
  // src0.field_i < src1.field_i and src1.field_i < src0.field_i
  // with each 0 <= i < field_count

  static void sorting_less(ckernel_prefix *extra, char *dst, char *const *src)
  {
    char *eraw = reinterpret_cast<char *>(extra);
    extra_type *e = reinterpret_cast<extra_type *>(extra);
    size_t field_count = e->field_count;
    const size_t *src0_data_offsets = e->src0_data_offsets;
    const size_t *src1_data_offsets = e->src1_data_offsets;
    const size_t *kernel_offsets = reinterpret_cast<const size_t *>(e + 1);
    char *child_src[2];
    for (size_t i = 0; i != field_count; ++i) {
      ckernel_prefix *src0_sorting_less_src1 =
          reinterpret_cast<ckernel_prefix *>(eraw + kernel_offsets[2 * i]);
      expr_single_t opchild =
          src0_sorting_less_src1->get_function<expr_single_t>();
      // if (src0.field_i < src1.field_i) return true
      child_src[0] = src[0] + src0_data_offsets[i];
      child_src[1] = src[1] + src1_data_offsets[i];
      int child_dst;
      opchild(src0_sorting_less_src1, reinterpret_cast<char *>(&child_dst),
              child_src);
      if (child_dst) {
        *reinterpret_cast<char *>(dst) = true;
        return;
      }
      ckernel_prefix *src1_sorting_less_src0 =
          reinterpret_cast<ckernel_prefix *>(eraw + kernel_offsets[2 * i + 1]);
      opchild = src1_sorting_less_src0->get_function<expr_single_t>();
      // if (src1.field_i < src0.field_i) return false
      child_src[0] = src[1] + src1_data_offsets[i];
      child_src[1] = src[0] + src0_data_offsets[i];
      opchild(src1_sorting_less_src0, reinterpret_cast<char *>(&child_dst),
              child_src);
      if (child_dst) {
        *reinterpret_cast<char *>(dst) = false;
        return;
      }
    }
    *reinterpret_cast<char *>(dst) = false;
  }

  static void destruct(ckernel_prefix *self)
  {
    extra_type *e = reinterpret_cast<extra_type *>(self);
    const size_t *kernel_offsets = reinterpret_cast<const size_t *>(e + 1);
    size_t field_count = e->field_count;
    for (size_t i = 0; i != 2 * field_count; ++i) {
      self->get_child_ckernel(kernel_offsets[i])->destroy();
    }
  }
};

} // anonymous namespace

size_t dynd::make_tuple_comparison_kernel(void *ckb, intptr_t ckb_offset,
                                          const ndt::type &src_tp,
                                          const char *src0_arrmeta,
                                          const char *src1_arrmeta,
                                          comparison_type_t comptype,
                                          const eval::eval_context *ectx)
{
  intptr_t root_ckb_offset = ckb_offset;
  auto bsd = src_tp.extended<ndt::base_tuple_type>();
  size_t field_count = bsd->get_field_count();
  if (comptype == comparison_type_sorting_less) {
    if (src0_arrmeta == src1_arrmeta || src_tp.get_arrmeta_size() == 0 ||
        memcmp(src0_arrmeta, src1_arrmeta, src_tp.get_arrmeta_size()) == 0) {
      // The arrmeta is identical, so can use a more specialized comparison
      // function
      inc_ckb_offset(
          ckb_offset,
          sizeof(tuple_compare_sorting_less_matching_arrmeta_kernel) +
              field_count * sizeof(size_t));
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
          ->reserve(ckb_offset + sizeof(ckernel_prefix));
      tuple_compare_sorting_less_matching_arrmeta_kernel *e =
          reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
              ->get_at<tuple_compare_sorting_less_matching_arrmeta_kernel>(
                    root_ckb_offset);
      e->base.set_function<expr_single_t>(
          &tuple_compare_sorting_less_matching_arrmeta_kernel::sorting_less);
      e->base.destructor =
          &tuple_compare_sorting_less_matching_arrmeta_kernel::destruct;
      e->field_count = field_count;
      e->src_data_offsets = bsd->get_data_offsets(src0_arrmeta);
      size_t *field_kernel_offsets;
      const uintptr_t *arrmeta_offsets = bsd->get_arrmeta_offsets_raw();
      for (size_t i = 0; i != field_count; ++i) {
        // Reserve space for the child, and save the offset to this
        // field comparison kernel. Have to re-get
        // the pointer because creating the field comparison kernel may
        // move the memory.
        reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
            ->reserve(ckb_offset + sizeof(ckernel_prefix));
        e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                ->get_at<tuple_compare_sorting_less_matching_arrmeta_kernel>(
                      root_ckb_offset);
        field_kernel_offsets = reinterpret_cast<size_t *>(e + 1);
        field_kernel_offsets[i] = ckb_offset - root_ckb_offset;
        const char *field_arrmeta = src0_arrmeta + arrmeta_offsets[i];
        const ndt::type &ft = bsd->get_field_type(i);
        ckb_offset = make_comparison_kernel(ckb, ckb_offset, ft, field_arrmeta,
                                            ft, field_arrmeta,
                                            comparison_type_sorting_less, ectx);
      }
      return ckb_offset;
    } else {
      // The arrmeta is different, so have to get the kernels both ways for the
      // fields
      inc_ckb_offset(ckb_offset,
                     sizeof(tuple_compare_sorting_less_diff_arrmeta_kernel) +
                         2 * field_count * sizeof(size_t));
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
          ->reserve(ckb_offset + sizeof(ckernel_prefix));
      tuple_compare_sorting_less_diff_arrmeta_kernel *e =
          reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
              ->get_at<tuple_compare_sorting_less_diff_arrmeta_kernel>(
                    root_ckb_offset);
      e->base.set_function<expr_single_t>(
          &tuple_compare_sorting_less_diff_arrmeta_kernel::sorting_less);
      e->base.destructor =
          &tuple_compare_sorting_less_diff_arrmeta_kernel::destruct;
      e->field_count = field_count;
      e->src0_data_offsets = bsd->get_data_offsets(src0_arrmeta);
      e->src1_data_offsets = bsd->get_data_offsets(src1_arrmeta);
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
                ->get_at<tuple_compare_sorting_less_diff_arrmeta_kernel>(
                      root_ckb_offset);
        field_kernel_offsets = reinterpret_cast<size_t *>(e + 1);
        field_kernel_offsets[2 * i] = ckb_offset - root_ckb_offset;
        ckb_offset = make_comparison_kernel(
            ckb, ckb_offset, ft, src0_arrmeta + arrmeta_offsets[i], ft,
            src1_arrmeta + arrmeta_offsets[i], comparison_type_sorting_less,
            ectx);
        // Repeat for comparing the other way
        reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
            ->reserve(ckb_offset + sizeof(ckernel_prefix));
        e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                ->get_at<tuple_compare_sorting_less_diff_arrmeta_kernel>(
                      root_ckb_offset);
        field_kernel_offsets = reinterpret_cast<size_t *>(e + 1);
        field_kernel_offsets[2 * i + 1] = ckb_offset - root_ckb_offset;
        ckb_offset = make_comparison_kernel(
            ckb, ckb_offset, ft, src1_arrmeta + arrmeta_offsets[i], ft,
            src0_arrmeta + arrmeta_offsets[i], comparison_type_sorting_less,
            ectx);
      }
      return ckb_offset;
    }
  } else if (comptype == comparison_type_equal ||
             comptype == comparison_type_not_equal) {
    //    inc_ckb_offset(ckb_offset, sizeof(nd::tuple_compare_equality_kernel) +
    //                                 field_count * sizeof(size_t));
    //    reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
    //      ->reserve(ckb_offset + sizeof(ckernel_prefix));
    if (comptype == comparison_type_equal) {
      std::map<std::string, ndt::type> tp_vars;
      const char *src_arrmeta[2] = {src0_arrmeta, src1_arrmeta};
      return nd::equal_kernel<tuple_type_id, tuple_type_id>::instantiate(
          NULL, 0, NULL, ckb, ckb_offset, ndt::type::make<int>(), NULL, 2,
          &src_tp, src_arrmeta, kernel_request_host | kernel_request_single,
          ectx, 0, NULL, tp_vars);
    } else {
      std::map<std::string, ndt::type> tp_vars;
      const char *src_arrmeta[2] = {src0_arrmeta, src1_arrmeta};
      return nd::not_equal_kernel<tuple_type_id, tuple_type_id>::instantiate(
          NULL, 0, NULL, ckb, ckb_offset, ndt::type::make<int>(), NULL, 2,
          &src_tp, src_arrmeta, kernel_request_host | kernel_request_single,
          ectx, 0, NULL, tp_vars);
    }
    return ckb_offset;
  } else {
    throw not_comparable_error(src_tp, src_tp, comptype);
  }
}
