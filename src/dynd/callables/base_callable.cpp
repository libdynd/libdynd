//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/base_callable.hpp>
#include <dynd/callables/call_graph.hpp>

using namespace std;
using namespace dynd;

nd::base_callable::~base_callable() {}

nd::array nd::base_callable::call(ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                                  const char *const *src_arrmeta, char *const *src_data, intptr_t nkwd,
                                  const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
  call_graph g;
  ndt::type resolved_dst_tp = resolve(nullptr, nullptr, g, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Allocate, then initialize, the data
  char *data = data_init(dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Resolve the destination type
  if (dst_tp.is_symbolic()) {
    resolve_dst_type(data, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
  }

  if (resolved_dst_tp != dst_tp) {
    throw std::runtime_error("different types");
  }

  // Allocate the destination array
  array dst = alloc(&resolved_dst_tp);

  // Generate and evaluate the ckernel
  kernel_builder ckb;
  instantiate(nullptr, data, &ckb, dst_tp, dst.get()->metadata(), nsrc, src_tp, src_arrmeta, kernel_request_single,
              nkwd, kwds, tp_vars);
  kernel_single_t fn = ckb.get()->get_function<kernel_single_t>();
  fn(ckb.get(), dst.data(), src_data);

  return dst;
}

nd::array nd::base_callable::call(ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                                  const char *const *src_arrmeta, const array *src_data, intptr_t nkwd,
                                  const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {

  /*
    if (m_new_style) {
      call_graph g;
      if (!is_abstract()) {
        g.emplace_back(this);
      }
      new_resolve(nullptr, g, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

      kernel_builder ckb;
      array dst = empty(dst_tp);

      call_frame *frame = reinterpret_cast<call_frame *>(g.get());
      frame->callee->new_instantiate(frame, ckb, kernel_request_call, dst->metadata(), src_arrmeta, nkwd, kwds);

      kernel_call_t fn = ckb.get()->get_function<kernel_call_t>();
      fn(ckb.get(), &dst, src_data);

      return dst;
    }
  */
  call_graph cg;
  ndt::type resolved_dst_tp = resolve(nullptr, nullptr, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Allocate, then initialize, the data
  char *data = data_init(dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Resolve the destination type
  if (dst_tp.is_symbolic()) {
    resolve_dst_type(data, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
  }

  if (resolved_dst_tp != dst_tp) {
    throw std::runtime_error("different types");
  }

  // Allocate the destination array
  array dst = empty(resolved_dst_tp);

  // Generate and evaluate the ckernel
  kernel_builder ckb;
  cg.get()->callee->instantiate(cg.get(), data, &ckb, dst_tp, dst->metadata(), nsrc, src_tp, src_arrmeta,
                                kernel_request_call, nkwd, kwds, tp_vars);
  kernel_call_t fn = ckb.get()->get_function<kernel_call_t>();
  fn(ckb.get(), &dst, src_data);

  return dst;
}

void nd::base_callable::call(const ndt::type &dst_tp, const char *dst_arrmeta, char *dst_data, intptr_t nsrc,
                             const ndt::type *src_tp, const char *const *src_arrmeta, char *const *src_data,
                             intptr_t nkwd, const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
  call_graph g;
  resolve(nullptr, nullptr, g, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  char *data = data_init(dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Generate and evaluate the ckernel
  kernel_builder ckb;
  instantiate(nullptr, data, &ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernel_request_single, nkwd, kwds,
              tp_vars);
  kernel_single_t fn = ckb.get()->get_function<kernel_single_t>();
  fn(ckb.get(), dst_data, src_data);
}

void nd::base_callable::call(const ndt::type &dst_tp, const char *dst_arrmeta, array *dst, intptr_t nsrc,
                             const ndt::type *src_tp, const char *const *src_arrmeta, const array *src, intptr_t nkwd,
                             const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
  call_graph g;
  resolve(nullptr, nullptr, g, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  char *data = data_init(dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Generate and evaluate the ckernel
  kernel_builder ckb;
  instantiate(nullptr, data, &ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernel_request_call, nkwd, kwds,
              tp_vars);
  kernel_call_t fn = ckb.get()->get_function<kernel_call_t>();
  fn(ckb.get(), dst, src);
}

nd::call_graph::call_graph(base_callable *callee)
    : m_data(m_static_data), m_capacity(sizeof(m_static_data)), m_size(0) {

  size_t offset = m_size;
  m_size += aligned_size(sizeof(base_callable::call_node));
  reserve(m_size);
  new (this->get_at<base_callable::call_node>(offset)) base_callable::call_node(callee);

  m_back_offset = 0;
}

void nd::call_graph::emplace_back(base_callable *callee) {
  /* Alignment requirement of the type. */
  //      static_assert(alignof(KernelType) <= 8, "kernel types require alignment to be at most 8 bytes");

  m_back_offset = m_size;

  size_t offset = m_size;
  m_size += aligned_size(sizeof(base_callable::call_node));
  reserve(m_size);
  new (this->get_at<base_callable::call_node>(offset)) base_callable::call_node(callee);
}
