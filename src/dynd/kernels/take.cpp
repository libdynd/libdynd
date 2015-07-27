//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/shape_tools.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/take.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

void nd::masked_take_ck::single(char *dst, char *const *src)
{
  ckernel_prefix *child = get_child_ckernel();
  expr_strided_t child_fn = child->get_function<expr_strided_t>();
  char *src0 = src[0];
  char *mask = src[1];
  intptr_t dim_size = m_dim_size, src0_stride = m_src0_stride,
           mask_stride = m_mask_stride;
  // Start with the dst matching the dim size. (Maybe better to
  // do smaller? This means no resize required in the loop.)
  ndt::var_dim_element_initialize(m_dst_tp, m_dst_meta, dst, dim_size);
  var_dim_type_data *vdd = reinterpret_cast<var_dim_type_data *>(dst);
  char *dst_ptr = vdd->begin;
  intptr_t dst_stride =
      reinterpret_cast<const var_dim_type_arrmeta *>(m_dst_meta)->stride;
  intptr_t dst_count = 0;
  intptr_t i = 0;
  while (i < dim_size) {
    // Run of false
    for (; i < dim_size && *mask == 0;
         src0 += src0_stride, mask += mask_stride, ++i) {
    }
    // Run of true
    intptr_t i_saved = i;
    for (; i < dim_size && *mask != 0; mask += mask_stride, ++i) {
    }
    // Copy the run of true
    if (i > i_saved) {
      intptr_t run_count = i - i_saved;
      child_fn(dst_ptr, dst_stride, &src0, &src0_stride, run_count, child);
      dst_ptr += run_count * dst_stride;
      src0 += run_count * src0_stride;
      dst_count += run_count;
    }
  }
  // Shrink the var dim element to fit
  ndt::var_dim_element_resize(m_dst_tp, m_dst_meta, dst, dst_count);
}

void nd::masked_take_ck::destruct_children()
{
  // The child copy ckernel
  get_child_ckernel()->destroy();
}

intptr_t nd::masked_take_ck::instantiate(
    char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
    char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta,
    intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &DYND_UNUSED(kwds),
    const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  typedef nd::masked_take_ck self_type;

  self_type *self = self_type::make(ckb, kernreq, ckb_offset);

  if (dst_tp.get_type_id() != var_dim_type_id) {
    stringstream ss;
    ss << "masked take arrfunc: could not process type " << dst_tp;
    ss << " as a var dimension";
    throw type_error(ss.str());
  }
  self->m_dst_tp = dst_tp;
  self->m_dst_meta = dst_arrmeta;
  ndt::type dst_el_tp =
      self->m_dst_tp.extended<ndt::var_dim_type>()->get_element_type();
  const char *dst_el_meta = self->m_dst_meta + sizeof(var_dim_type_arrmeta);

  intptr_t src0_dim_size, mask_dim_size;
  ndt::type src0_el_tp, mask_el_tp;
  const char *src0_el_meta, *mask_el_meta;
  if (!src_tp[0].get_as_strided(src_arrmeta[0], &src0_dim_size,
                                &self->m_src0_stride, &src0_el_tp,
                                &src0_el_meta)) {
    stringstream ss;
    ss << "masked take arrfunc: could not process type " << src_tp[0];
    ss << " as a strided dimension";
    throw type_error(ss.str());
  }
  if (!src_tp[1].get_as_strided(src_arrmeta[1], &mask_dim_size,
                                &self->m_mask_stride, &mask_el_tp,
                                &mask_el_meta)) {
    stringstream ss;
    ss << "masked take arrfunc: could not process type " << src_tp[1];
    ss << " as a strided dimension";
    throw type_error(ss.str());
  }
  if (src0_dim_size != mask_dim_size) {
    stringstream ss;
    ss << "masked take arrfunc: source data and mask have different sizes, ";
    ss << src0_dim_size << " and " << mask_dim_size;
    throw invalid_argument(ss.str());
  }
  self->m_dim_size = src0_dim_size;
  if (mask_el_tp.get_type_id() != bool_type_id) {
    stringstream ss;
    ss << "masked take arrfunc: mask type should be bool, not ";
    ss << mask_el_tp;
    throw type_error(ss.str());
  }

  // Create the child element assignment ckernel
  return make_assignment_kernel(ckb, ckb_offset, dst_el_tp, dst_el_meta,
                                src0_el_tp, src0_el_meta,
                                kernel_request_strided, ectx);
}

void nd::indexed_take_ck::single(char *dst, char *const *src)
{
  ckernel_prefix *child = get_child_ckernel();
  expr_single_t child_fn = child->get_function<expr_single_t>();
  char *src0 = src[0];
  const char *index = src[1];
  intptr_t dst_dim_size = m_dst_dim_size, src0_dim_size = m_src0_dim_size,
           dst_stride = m_dst_stride, src0_stride = m_src0_stride,
           index_stride = m_index_stride;
  for (intptr_t i = 0; i < dst_dim_size; ++i) {
    intptr_t ix = *reinterpret_cast<const intptr_t *>(index);
    // Handle Python-style negative index, bounds checking
    ix = apply_single_index(ix, src0_dim_size, NULL);
    // Copy one element at a time
    char *child_src0 = src0 + ix * src0_stride;
    child_fn(dst, &child_src0, child);
    dst += dst_stride;
    index += index_stride;
  }
}

void nd::indexed_take_ck::destruct_children()
{
  // The child copy ckernel
  get_child_ckernel()->destroy();
}

intptr_t nd::indexed_take_ck::instantiate(
    char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
    char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta,
    intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &DYND_UNUSED(kwds),
    const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  typedef nd::indexed_take_ck self_type;

  self_type *self = self_type::make(ckb, kernreq, ckb_offset);

  ndt::type dst_el_tp;
  const char *dst_el_meta;
  if (!dst_tp.get_as_strided(dst_arrmeta, &self->m_dst_dim_size,
                             &self->m_dst_stride, &dst_el_tp, &dst_el_meta)) {
    stringstream ss;
    ss << "indexed take arrfunc: could not process type " << dst_tp;
    ss << " as a strided dimension";
    throw type_error(ss.str());
  }

  intptr_t index_dim_size;
  ndt::type src0_el_tp, index_el_tp;
  const char *src0_el_meta, *index_el_meta;
  if (!src_tp[0].get_as_strided(src_arrmeta[0], &self->m_src0_dim_size,
                                &self->m_src0_stride, &src0_el_tp,
                                &src0_el_meta)) {
    stringstream ss;
    ss << "indexed take arrfunc: could not process type " << src_tp[0];
    ss << " as a strided dimension";
    throw type_error(ss.str());
  }
  if (!src_tp[1].get_as_strided(src_arrmeta[1], &index_dim_size,
                                &self->m_index_stride, &index_el_tp,
                                &index_el_meta)) {
    stringstream ss;
    ss << "take arrfunc: could not process type " << src_tp[1];
    ss << " as a strided dimension";
    throw type_error(ss.str());
  }
  if (self->m_dst_dim_size != index_dim_size) {
    stringstream ss;
    ss << "indexed take arrfunc: index data and dest have different sizes, ";
    ss << index_dim_size << " and " << self->m_dst_dim_size;
    throw invalid_argument(ss.str());
  }
  if (index_el_tp.get_type_id() != (type_id_t)type_id_of<intptr_t>::value) {
    stringstream ss;
    ss << "indexed take arrfunc: index type should be intptr, not ";
    ss << index_el_tp;
    throw type_error(ss.str());
  }

  // Create the child element assignment ckernel
  return make_assignment_kernel(ckb, ckb_offset, dst_el_tp, dst_el_meta,
                                src0_el_tp, src0_el_meta, kernel_request_single,
                                ectx);
}

intptr_t nd::take_ck::instantiate(
    char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
    char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &kwds, const std::map<std::string, ndt::type> &tp_vars)
{
  ndt::type mask_el_tp = src_tp[1].get_type_at_dimension(NULL, 1);
  if (mask_el_tp.get_type_id() == bool_type_id) {
    return nd::masked_take_ck::instantiate(
        NULL, 0, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
        src_arrmeta, kernreq, ectx, kwds, tp_vars);
  } else if (mask_el_tp.get_type_id() ==
             (type_id_t)type_id_of<intptr_t>::value) {
    return nd::indexed_take_ck::instantiate(
        NULL, 0, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
        src_arrmeta, kernreq, ectx, kwds, tp_vars);
  } else {
    stringstream ss;
    ss << "take: unsupported type for the index " << mask_el_tp
       << ", need bool or intptr";
    throw invalid_argument(ss.str());
  }
}

void nd::take_ck::resolve_dst_type(
    char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
    char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
    const ndt::type *src_tp, const nd::array &DYND_UNUSED(kwds),
    const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  /*
    if (nsrc != 2) {
      stringstream ss;
      ss << "Wrong number of arguments to take arrfunc with prototype " << af_tp
         << ", got " << nsrc << " arguments";
      throw invalid_argument(ss.str());
    }
  */

  ndt::type mask_el_tp = src_tp[1].get_type_at_dimension(NULL, 1);
  if (mask_el_tp.get_type_id() == bool_type_id) {
    dst_tp = ndt::var_dim_type::make(
        src_tp[0].get_type_at_dimension(NULL, 1).get_canonical_type());
  } else if (mask_el_tp.get_type_id() ==
             (type_id_t)type_id_of<intptr_t>::value) {
    if (src_tp[1].get_type_id() == var_dim_type_id) {
      dst_tp = ndt::var_dim_type::make(
          src_tp[0].get_type_at_dimension(NULL, 1).get_canonical_type());
    } else {
      dst_tp = ndt::make_fixed_dim(
          src_tp[1].get_dim_size(NULL, NULL),
          src_tp[0].get_type_at_dimension(NULL, 1).get_canonical_type());
    }
  } else {
    stringstream ss;
    ss << "take: unsupported type for the index " << mask_el_tp
       << ", need bool or intptr";
    throw invalid_argument(ss.str());
  }
}