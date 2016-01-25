//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/comparison.hpp>
#include <dynd/kernels/compare_kernels.hpp>

using namespace std;
using namespace dynd;

void nd::equal_kernel<tuple_type_id, tuple_type_id>::instantiate(
    char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
    const std::map<std::string, ndt::type> &tp_vars)
{
  intptr_t root_ckb_offset = ckb_offset;
  auto bsd = src_tp->extended<ndt::tuple_type>();
  size_t field_count = bsd->get_field_count();
  extra_type *e = extra_type::make(ckb, kernel_request_host | kernel_request_single, field_count,
                                   bsd->get_data_offsets(src_arrmeta[0]), bsd->get_data_offsets(src_arrmeta[1]));
  ckb_offset = ckb->m_size;
  e = extra_type::reserve(ckb, kernel_request_host | kernel_request_single, ckb_offset, field_count * sizeof(size_t));
  inc_ckb_offset(ckb_offset, field_count * sizeof(size_t));
  inc_ckb_offset(ckb->m_size, field_count * sizeof(size_t));
  //      e->field_count = field_count;
  //    e->src0_data_offsets = bsd->get_data_offsets(src0_arrmeta);
  //  e->src1_data_offsets = bsd->get_data_offsets(src1_arrmeta);
  size_t *field_kernel_offsets;
  const uintptr_t *arrmeta_offsets = bsd->get_arrmeta_offsets_raw();
  for (size_t i = 0; i != field_count; ++i) {
    const ndt::type &ft = bsd->get_field_type(i);
    // Reserve space for the child, and save the offset to this
    // field comparison kernel. Have to re-get
    // the pointer because creating the field comparison kernel may
    // move the memory.
    reinterpret_cast<kernel_builder *>(ckb)->reserve(ckb_offset + sizeof(ckernel_prefix));
    e = reinterpret_cast<kernel_builder *>(ckb)->get_at<extra_type>(root_ckb_offset);
    field_kernel_offsets = reinterpret_cast<size_t *>(e + 1);
    field_kernel_offsets[i] = ckb_offset - root_ckb_offset;
    const char *field_arrmeta = src_arrmeta[0] + arrmeta_offsets[i];
    ndt::type child_src_tp[2] = {ft, ft};
    const char *child_src_arrmeta[2] = {field_arrmeta, field_arrmeta};
    equal::get().get()->instantiate(equal::get().get()->static_data(), NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
                                    child_src_tp, child_src_arrmeta, kernreq, nkwd, kwds, tp_vars);
    ckb_offset = ckb->m_size;
  }
}

void nd::not_equal_kernel<tuple_type_id, tuple_type_id>::instantiate(
    char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
    const std::map<std::string, ndt::type> &tp_vars)
{
  intptr_t root_ckb_offset = ckb_offset;
  auto bsd = src_tp->extended<ndt::tuple_type>();
  size_t field_count = bsd->get_field_count();
  extra_type *e = extra_type::make(ckb, kernel_request_host | kernel_request_single, field_count,
                                   bsd->get_data_offsets(src_arrmeta[0]), bsd->get_data_offsets(src_arrmeta[1]));
  ckb_offset = ckb->m_size;
  e = extra_type::reserve(ckb, kernel_request_host | kernel_request_single, ckb_offset, field_count * sizeof(size_t));
  inc_ckb_offset(ckb_offset, field_count * sizeof(size_t));
  inc_ckb_offset(ckb->m_size, field_count * sizeof(size_t));
  //      e->field_count = field_count;
  //    e->src0_data_offsets = bsd->get_data_offsets(src0_arrmeta);
  //  e->src1_data_offsets = bsd->get_data_offsets(src1_arrmeta);
  size_t *field_kernel_offsets;
  const uintptr_t *arrmeta_offsets = bsd->get_arrmeta_offsets_raw();
  for (size_t i = 0; i != field_count; ++i) {
    const ndt::type &ft = bsd->get_field_type(i);
    // Reserve space for the child, and save the offset to this
    // field comparison kernel. Have to re-get
    // the pointer because creating the field comparison kernel may
    // move the memory.
    reinterpret_cast<kernel_builder *>(ckb)->reserve(ckb_offset + sizeof(ckernel_prefix));
    e = reinterpret_cast<kernel_builder *>(ckb)->get_at<extra_type>(root_ckb_offset);
    field_kernel_offsets = reinterpret_cast<size_t *>(e + 1);
    field_kernel_offsets[i] = ckb_offset - root_ckb_offset;
    const char *field_arrmeta = src_arrmeta[0] + arrmeta_offsets[i];
    ndt::type child_src_tp[2] = {ft, ft};
    const char *child_src_arrmeta[2] = {field_arrmeta, field_arrmeta};
    not_equal::get().get()->instantiate(not_equal::get().get()->static_data(), NULL, ckb, ckb_offset, dst_tp,
                                        dst_arrmeta, nsrc, child_src_tp, child_src_arrmeta, kernreq, nkwd, kwds,
                                        tp_vars);
    ckb_offset = ckb->m_size;
  }
}
