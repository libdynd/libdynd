//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/chain.hpp>

using namespace std;
using namespace dynd;

void nd::functional::unary_heap_chain_ck::single(char *dst, char *const *src)
{
  //        self_type *self = get_self(rawself);
  // Allocate a temporary buffer on the heap
  nd::array buf = nd::empty(m_buf_tp);
  char *buf_data = buf.get_readwrite_originptr();
  ckernel_prefix *first = get_child_ckernel();
  expr_single_t first_fn = first->get_function<expr_single_t>();
  ckernel_prefix *second = get_child_ckernel(m_second_offset);
  expr_single_t second_fn = second->get_function<expr_single_t>();
  first_fn(buf_data, src, first);
  second_fn(dst, &buf_data, second);
}

void nd::functional::unary_heap_chain_ck::strided(char *dst,
                                                  intptr_t dst_stride,
                                                  char *const *src,
                                                  const intptr_t *src_stride,
                                                  size_t count)
{
  // Allocate a temporary buffer on the heap
  nd::array buf = nd::empty(m_buf_shape[0], m_buf_tp);
  char *buf_data = buf.get_readwrite_originptr();
  intptr_t buf_stride = reinterpret_cast<const fixed_dim_type_arrmeta *>(
                            buf.get_arrmeta())->stride;
  ckernel_prefix *first = get_child_ckernel();
  expr_strided_t first_fn = first->get_function<expr_strided_t>();
  ckernel_prefix *second = get_child_ckernel(m_second_offset);
  expr_strided_t second_fn = second->get_function<expr_strided_t>();
  char *src0 = src[0];
  intptr_t src0_stride = src_stride[0];

  size_t chunk_size = std::min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
  first_fn(buf_data, buf_stride, &src0, src_stride, chunk_size, first);
  second_fn(dst, dst_stride, &buf_data, &buf_stride, chunk_size, second);
  count -= chunk_size;
  while (count) {
    src0 += chunk_size * src0_stride;
    dst += chunk_size * dst_stride;
    reset_strided_buffer_array(buf);
    chunk_size = std::min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
    first_fn(buf_data, buf_stride, &src0, src_stride, chunk_size, first);
    second_fn(dst, dst_stride, &buf_data, &buf_stride, chunk_size, second);
    count -= chunk_size;
  }
}

void nd::functional::unary_heap_chain_ck::destruct_children()
{
  // The first child ckernel
  get_child_ckernel()->destroy();
  // The second child ckernel
  base.destroy_child_ckernel(m_second_offset);
}

/**
 * Instantiate the chaining of arrfuncs ``first`` and ``second``, using
 * ``buf_tp`` as the intermediate type, without creating a temporary chained
 * arrfunc.
 */
intptr_t nd::functional::unary_heap_chain_ck::instantiate(
    const arrfunc_type_data *af_self, const arrfunc_type *DYND_UNUSED(af_tp),
    char *data, void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &DYND_UNUSED(kwds),
    const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  const instantiate_chain_data *icd =
      af_self->get_data_as<instantiate_chain_data>();

  const arrfunc_type_data *first = icd->first.get();
  const arrfunc_type *first_tp = icd->first.get_type();

  const arrfunc_type_data *second = icd->second.get();
  const arrfunc_type *second_tp = icd->second.get_type();

  const ndt::type &buf_tp = icd->buf_tp;

  if (first_tp->get_npos() == 1) {
    intptr_t root_ckb_offset = ckb_offset;
    nd::functional::unary_heap_chain_ck *self =
        nd::functional::unary_heap_chain_ck::make(ckb, kernreq, ckb_offset);
    self->m_buf_tp = buf_tp;
    arrmeta_holder(buf_tp).swap(self->m_buf_arrmeta);
    self->m_buf_arrmeta.arrmeta_default_construct(true);
    self->m_buf_shape.push_back(DYND_BUFFER_CHUNK_SIZE);
    ckb_offset = first->instantiate(
        first, first_tp, data, ckb, ckb_offset, buf_tp,
        self->m_buf_arrmeta.get(), first_tp->get_npos(), src_tp, src_arrmeta,
        kernreq, ectx, nd::array(), std::map<nd::string, ndt::type>());
    reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
        ->reserve(ckb_offset);
    self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
               ->get_at<nd::functional::unary_heap_chain_ck>(root_ckb_offset);
    self->m_second_offset = ckb_offset - root_ckb_offset;
    const char *buf_arrmeta = self->m_buf_arrmeta.get();
    ckb_offset = second->instantiate(
        second, second_tp, data + first->data_size, ckb, ckb_offset, dst_tp,
        dst_arrmeta, first_tp->get_npos(), &buf_tp, &buf_arrmeta, kernreq, ectx,
        nd::array(), std::map<nd::string, ndt::type>());
    return ckb_offset;
  } else {
    throw runtime_error("Multi-parameter arrfunc chaining is not implemented");
  }
}

void nd::functional::unary_heap_chain_ck::resolve_dst_type(
    const arrfunc_type_data *DYND_UNUSED(self), const arrfunc_type *self_tp,
    char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
    const ndt::type *DYND_UNUSED(src_tp),
    const dynd::nd::array &DYND_UNUSED(kwds),
    const std::map<dynd::nd::string, ndt::type> &tp_vars)
{
  dst_tp = ndt::substitute(self_tp->get_return_type(), tp_vars, true);
}