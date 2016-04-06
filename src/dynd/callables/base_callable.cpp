//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/base_callable.hpp>
#include <dynd/callables/call_graph.hpp>

using namespace std;
using namespace dynd;

nd::base_callable::~base_callable() {}

nd::array nd::base_callable::call(ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp,
                                  const char *const *src_arrmeta, char *const *src_data, size_t nkwd, const array *kwds,
                                  const std::map<std::string, ndt::type> &tp_vars) {
  call_graph cg;
  dst_tp = resolve(nullptr, nullptr, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Allocate the destination array
  array dst = alloc(&dst_tp);

  // Generate and evaluate the ckernel
  kernel_builder kb(cg.get());
  kb(kernel_request_single, dst->metadata(), nsrc, src_arrmeta);

  kernel_single_t fn = kb.get()->get_function<kernel_single_t>();
  fn(kb.get(), dst.data(), src_data);

  return dst;
}

nd::array nd::base_callable::call(ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp,
                                  const char *const *src_arrmeta, const array *src_data, size_t nkwd, const array *kwds,
                                  const std::map<std::string, ndt::type> &tp_vars) {
  call_graph cg;
  dst_tp = resolve(nullptr, nullptr, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Allocate the destination array
  array dst = empty(dst_tp);

  // Generate and evaluate the kernel
  kernel_builder kb(cg.get());
  kb(kernel_request_call, dst->metadata(), nsrc, src_arrmeta);

  kernel_call_t fn = kb.get()->get_function<kernel_call_t>();
  fn(kb.get(), &dst, src_data);

  return dst;
}

void nd::base_callable::call(const ndt::type &dst_tp, const char *dst_arrmeta, char *dst_data, size_t nsrc,
                             const ndt::type *src_tp, const char *const *src_arrmeta, char *const *src_data,
                             size_t nkwd, const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
  call_graph cg;
  resolve(nullptr, nullptr, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Generate and evaluate the ckernel
  kernel_builder kb(cg.get());
  kb(kernel_request_single, dst_arrmeta, nsrc, src_arrmeta);

  kernel_single_t fn = kb.get()->get_function<kernel_single_t>();
  fn(kb.get(), dst_data, src_data);
}

void nd::base_callable::call(const ndt::type &dst_tp, const char *dst_arrmeta, array *dst, size_t nsrc,
                             const ndt::type *src_tp, const char *const *src_arrmeta, const array *src, size_t nkwd,
                             const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
  call_graph cg;
  resolve(nullptr, nullptr, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Generate and evaluate the ckernel
  kernel_builder kb(cg.get());
  kb(kernel_request_call, dst_arrmeta, nsrc, src_arrmeta);

  kernel_call_t fn = kb.get()->get_function<kernel_call_t>();
  fn(kb.get(), dst, src);
}
