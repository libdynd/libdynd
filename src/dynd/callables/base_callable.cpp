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
  call_graph cg;
  dst_tp = resolve(nullptr, nullptr, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Allocate the destination array
  array dst = alloc(&dst_tp);

  // Generate and evaluate the ckernel
  kernel_builder ckb;
  call_node *node = cg.get();
  node->instantiate(node, &ckb, kernel_request_single, dst->metadata(), nsrc, src_arrmeta);
  kernel_single_t fn = ckb.get()->get_function<kernel_single_t>();
  fn(ckb.get(), dst.data(), src_data);

  return dst;
}

nd::array nd::base_callable::call(ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                                  const char *const *src_arrmeta, const array *src_data, intptr_t nkwd,
                                  const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
  call_graph cg;
  dst_tp = resolve(nullptr, nullptr, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  /*
    std::cout << "dst_tp = " << dst_tp << std::endl;
    for (int i = 0; i < nsrc; ++i) {
      std::cout << "src_tp[" << i << "] = " << src_tp[i] << std::endl;
    }
  */

  // Allocate the destination array
  array dst = empty(dst_tp);

  // Generate and evaluate the ckernel
  kernel_builder ckb;
  call_node *node = cg.get();
  node->instantiate(node, &ckb, kernel_request_call, dst->metadata(), nsrc, src_arrmeta);
  kernel_call_t fn = ckb.get()->get_function<kernel_call_t>();
  fn(ckb.get(), &dst, src_data);

  return dst;
}

void nd::base_callable::call(const ndt::type &dst_tp, const char *dst_arrmeta, char *dst_data, intptr_t nsrc,
                             const ndt::type *src_tp, const char *const *src_arrmeta, char *const *src_data,
                             intptr_t nkwd, const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
  call_graph cg;
  resolve(nullptr, nullptr, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Generate and evaluate the ckernel
  kernel_builder ckb;
  call_node *node = cg.get();
  node->instantiate(node, &ckb, kernel_request_single, dst_arrmeta, nsrc, src_arrmeta);
  kernel_single_t fn = ckb.get()->get_function<kernel_single_t>();
  fn(ckb.get(), dst_data, src_data);
}

void nd::base_callable::call(const ndt::type &dst_tp, const char *dst_arrmeta, array *dst, intptr_t nsrc,
                             const ndt::type *src_tp, const char *const *src_arrmeta, const array *src, intptr_t nkwd,
                             const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
  call_graph cg;
  resolve(nullptr, nullptr, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  /*
    std::cout << "dst_tp = " << dst_tp << std::endl;
    for (int i = 0; i < nsrc; ++i) {
      std::cout << "src_tp[" << i << "] = " << src_tp[i] << std::endl;
    }
  */

  // Generate and evaluate the ckernel
  kernel_builder ckb;
  call_node *node = cg.get();
  node->instantiate(node, &ckb, kernel_request_call, dst_arrmeta, nsrc, src_arrmeta);
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

void nd::call_graph::emplace_back(base_callable *DYND_UNUSED(callee)) {}
