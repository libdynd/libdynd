//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/rolling_kernel.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

void nd::functional::strided_rolling_ck::single(char *dst, char *const *src)
{
  kernel_prefix *nachild = get_child();
  kernel_prefix *wopchild = get_child(m_window_op_offset);
  kernel_strided_t nachild_fn = nachild->get_function<kernel_strided_t>();
  kernel_strided_t wopchild_fn = wopchild->get_function<kernel_strided_t>();
  // Fill in NA/NaN at the beginning
  if (m_dim_size > 0) {
    nachild_fn(nachild, dst, m_dst_stride, NULL, NULL, std::min(m_window_size - 1, m_dim_size));
  }
  // Use stride trickery to do this as one strided call
  if (m_dim_size >= m_window_size) {
    wopchild_fn(wopchild, dst + m_dst_stride * (m_window_size - 1), m_dst_stride, src, &m_src_stride,
                m_dim_size - m_window_size + 1);
  }
}

void nd::functional::var_rolling_ck::single(char *dst, char *const *src)
{
  // Get the child ckernels
  kernel_prefix *nachild = get_child();
  kernel_prefix *wopchild = get_child(m_window_op_offset);
  kernel_strided_t nachild_fn = nachild->get_function<kernel_strided_t>();
  kernel_strided_t wopchild_fn = wopchild->get_function<kernel_strided_t>();
  // Get pointers to the src and dst data
  ndt::var_dim_type::data_type *dst_dat = reinterpret_cast<ndt::var_dim_type::data_type *>(dst);
  intptr_t dst_stride = reinterpret_cast<const ndt::var_dim_type::metadata_type *>(m_dst_meta)->stride;
  ndt::var_dim_type::data_type *src_dat = reinterpret_cast<ndt::var_dim_type::data_type *>(src[0]);
  char *src_arr_ptr = src_dat->begin + m_src_offset;
  intptr_t dim_size = src_dat->size;
  // Allocate the output data

  dst_dat->begin = reinterpret_cast<const ndt::var_dim_type::metadata_type *>(m_dst_meta)->blockref->alloc(dim_size);
  dst_dat->size = dim_size;

  char *dst_arr_ptr = dst_dat->begin;

  // Fill in NA/NaN at the beginning
  if (dim_size > 0) {
    nachild_fn(nachild, dst_arr_ptr, dst_stride, NULL, NULL, std::min(m_window_size - 1, dim_size));
  }
  // Use stride trickery to do this as one strided call
  if (dim_size >= m_window_size) {
    wopchild_fn(wopchild, dst_arr_ptr + dst_stride * (m_window_size - 1), dst_stride, &src_arr_ptr, &m_src_stride,
                dim_size - m_window_size + 1);
  }
}

// TODO This should handle both strided and var cases
void nd::functional::rolling_ck::instantiate(char *_static_data, char *data, kernel_builder *ckb,
                                             const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                             const ndt::type *src_tp, const char *const *src_arrmeta,
                                             kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                                             const std::map<std::string, ndt::type> &tp_vars)
{
  typedef dynd::nd::functional::strided_rolling_ck self_type;
  rolling_callable_data *static_data = *reinterpret_cast<rolling_callable_data **>(_static_data);

  intptr_t root_ckb_offset = ckb->size();
  ckb->emplace_back<self_type>(kernreq);
  self_type *self = ckb->get_at<self_type>(root_ckb_offset);
  const callable &window_af = static_data->window_op;
  ndt::type dst_el_tp, src_el_tp;
  const char *dst_el_arrmeta, *src_el_arrmeta;
  if (!dst_tp.get_as_strided(dst_arrmeta, &self->m_dim_size, &self->m_dst_stride, &dst_el_tp, &dst_el_arrmeta)) {
    stringstream ss;
    ss << "rolling window ckernel: could not process type " << dst_tp;
    ss << " as a strided dimension";
    throw type_error(ss.str());
  }
  intptr_t src_dim_size;
  if (!src_tp[0].get_as_strided(src_arrmeta[0], &src_dim_size, &self->m_src_stride, &src_el_tp, &src_el_arrmeta)) {
    stringstream ss;
    ss << "rolling window ckernel: could not process type " << src_tp[0];
    ss << " as a strided dimension";
    throw type_error(ss.str());
  }
  if (src_dim_size != self->m_dim_size) {
    stringstream ss;
    ss << "rolling window ckernel: source dimension size " << src_dim_size << " for type " << src_tp[0]
       << " does not match dest dimension size " << self->m_dim_size << " for type " << dst_tp;
    throw type_error(ss.str());
  }
  self->m_window_size = static_data->window_size;
  // Create the NA-filling child ckernel
  // TODO: Need to fix this
  //  ckb_offset = kernels::make_constant_value_assignment_ckernel(
  //    ckb, ckb_offset, dst_el_tp, dst_el_arrmeta,
  //  numeric_limits<double>::quiet_NaN(), kernel_request_strided, ectx);
  // Re-retrieve the self pointer, because it may be at a new memory location
  // now
  self = reinterpret_cast<kernel_builder *>(ckb)->get_at<self_type>(root_ckb_offset);
  // Create the window op child ckernel
  self->m_window_op_offset = ckb->size() - root_ckb_offset;
  // We construct array arrmeta for the window op ckernel to use,
  // without actually creating an nd::array to hold it.
  arrmeta_holder(ndt::make_fixed_dim(static_data->window_size, src_el_tp)).swap(self->m_src_winop_meta);
  self->m_src_winop_meta.get_at<fixed_dim_type_arrmeta>(0)->dim_size = self->m_window_size;
  self->m_src_winop_meta.get_at<fixed_dim_type_arrmeta>(0)->stride = self->m_src_stride;
  if (src_el_tp.get_arrmeta_size() > 0) {
    src_el_tp.extended()->arrmeta_copy_construct(self->m_src_winop_meta.get() + sizeof(fixed_dim_type_arrmeta),
                                                 src_el_arrmeta, intrusive_ptr<memory_block_data>());
  }

  const char *src_winop_meta = self->m_src_winop_meta.get();
  window_af->instantiate(const_cast<char *>(window_af->static_data()), data, ckb, dst_el_tp, dst_el_arrmeta, nsrc,
                         &self->m_src_winop_meta.get_type(), &src_winop_meta, kernel_request_strided, nkwd, kwds,
                         tp_vars);
}

void nd::functional::rolling_ck::resolve_dst_type(char *_static_data, char *data, ndt::type &dst_tp,
                                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, intptr_t nkwd,
                                                  const array *kwds, const std::map<std::string, ndt::type> &tp_vars)

{
  /*
    if (nsrc != 1) {
      stringstream ss;
      ss << "Wrong number of arguments to rolling callable with prototype ";
      ss << af_tp << ", got " << nsrc << " arguments";
      throw invalid_argument(ss.str());
    }
  */

  static_data_type *static_data = *reinterpret_cast<static_data_type **>(_static_data);
  const callable &child_af = static_data->window_op;
  // First get the type for the child callable
  ndt::type child_dst_tp;
  if (child_af->m_resolve_dst_type) {
    ndt::type child_src_tp = ndt::make_fixed_dim(static_data->window_size, src_tp[0].get_type_at_dimension(NULL, 1));
    child_af->resolve_dst_type(const_cast<char *>(child_af->static_data()), data, child_dst_tp, 1, &child_src_tp, nkwd,
                               kwds, tp_vars);
  }
  else {
    child_dst_tp = static_data->window_op.get_type()->get_return_type();
  }

  if (src_tp[0].get_id() == var_dim_id) {
    dst_tp = ndt::var_dim_type::make(child_dst_tp);
  }
  else {
    dst_tp = ndt::make_fixed_dim(src_tp[0].get_dim_size(NULL, NULL), child_dst_tp);
  }
}
