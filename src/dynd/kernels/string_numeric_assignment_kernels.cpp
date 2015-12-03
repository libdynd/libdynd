//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>
#include <cctype>

#include <dynd/type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/string_numeric_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/parser_util.hpp>

using namespace std;
using namespace dynd;

namespace {

struct string_to_builtin_kernel {
  typedef string_to_builtin_kernel extra_type;

  ckernel_prefix base;
  ndt::type src_string_tp;
  assign_error_mode errmode;
  const char *src_arrmeta;

  static void destruct(ckernel_prefix *extra)
  {
    extra_type *e = reinterpret_cast<extra_type *>(extra);
    e->src_string_tp.~type();
  }
};
} // anonymous namespace

static expr_single_t static_string_to_builtin_kernels[builtin_type_id_count - 2] = {
    &nd::assignment_kernel<bool_type_id, string_type_id>::single_wrapper,
    &nd::assignment_kernel<int8_type_id, string_type_id>::single_wrapper,
    &nd::assignment_kernel<int16_type_id, string_type_id>::single_wrapper,
    &nd::assignment_kernel<int32_type_id, string_type_id>::single_wrapper,
    &nd::assignment_kernel<int64_type_id, string_type_id>::single_wrapper,
    &nd::assignment_kernel<int128_type_id, string_type_id>::single_wrapper,
    &nd::assignment_kernel<uint8_type_id, string_type_id>::single_wrapper,
    &nd::assignment_kernel<uint16_type_id, string_type_id>::single_wrapper,
    &nd::assignment_kernel<uint32_type_id, string_type_id>::single_wrapper,
    &nd::assignment_kernel<uint64_type_id, string_type_id>::single_wrapper,
    &nd::assignment_kernel<uint128_type_id, string_type_id>::single_wrapper,
    &nd::assignment_kernel<float16_type_id, string_type_id>::single_wrapper,
    &nd::assignment_kernel<float32_type_id, string_type_id>::single_wrapper,
    &nd::assignment_kernel<float64_type_id, string_type_id>::single_wrapper,
    &nd::assignment_kernel<float128_type_id, string_type_id>::single_wrapper,
    &nd::assignment_kernel<complex_float32_type_id, string_type_id>::single_wrapper,
    &nd::assignment_kernel<complex_float64_type_id, string_type_id>::single_wrapper};

size_t dynd::make_string_to_builtin_assignment_kernel(void *ckb, intptr_t ckb_offset, type_id_t dst_type_id,
                                                      const ndt::type &src_string_tp, const char *src_arrmeta,
                                                      kernel_request_t kernreq, const eval::eval_context *ectx)
{
  if (src_string_tp.get_kind() != string_kind) {
    stringstream ss;
    ss << "make_string_to_builtin_assignment_kernel: source type " << src_string_tp << " is not a string type";
    throw runtime_error(ss.str());
  }

  if (dst_type_id >= bool_type_id && dst_type_id <= complex_float64_type_id) {
    ckb_offset = make_kernreq_to_single_kernel_adapter(ckb, ckb_offset, 1, kernreq);
    string_to_builtin_kernel *e =
        reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->alloc_ck<string_to_builtin_kernel>(ckb_offset);
    e->base.function = reinterpret_cast<void *>(static_string_to_builtin_kernels[dst_type_id - bool_type_id]);
    e->base.destructor = &string_to_builtin_kernel::destruct;
    // The kernel data owns this reference
    e->src_string_tp = ndt::type(src_string_tp);
    e->errmode = ectx->errmode;
    e->src_arrmeta = src_arrmeta;
    return ckb_offset;
  }
  else {
    stringstream ss;
    ss << "make_string_to_builtin_assignment_kernel: destination type id " << dst_type_id << " is not builtin";
    throw runtime_error(ss.str());
  }
}

/////////////////////////////////////////
// string to builtin assignment

namespace {
struct builtin_to_string_kernel_extra {
  typedef builtin_to_string_kernel_extra extra_type;

  ckernel_prefix base;
  ndt::type dst_string_tp;
  type_id_t src_type_id;
  eval::eval_context ectx;
  const char *dst_arrmeta;

  static void single(ckernel_prefix *extra, char *dst, char *const *src)
  {
    extra_type *e = reinterpret_cast<extra_type *>(extra);

    // TODO: There are much faster ways to do this, but it's very generic!
    //       Also, for floating point values, a printing scheme like
    //       Python's, where it prints the shortest string that's
    //       guaranteed to parse to the same float number, would be
    //       better.
    stringstream ss;
    ndt::type(e->src_type_id).print_data(ss, NULL, src[0]);
    e->dst_string_tp->set_from_utf8_string(e->dst_arrmeta, dst, ss.str(), &e->ectx);
  }

  static void destruct(ckernel_prefix *extra)
  {
    extra_type *e = reinterpret_cast<extra_type *>(extra);
    e->dst_string_tp.~type();
  }
};
} // anonymous namespace

size_t dynd::make_builtin_to_string_assignment_kernel(void *ckb, intptr_t ckb_offset, const ndt::type &dst_string_tp,
                                                      const char *dst_arrmeta, type_id_t src_type_id,
                                                      kernel_request_t kernreq, const eval::eval_context *ectx)
{
  if (dst_string_tp.get_kind() != string_kind) {
    stringstream ss;
    ss << "make_builtin_to_string_assignment_kernel: destination type " << dst_string_tp << " is not a string type";
    throw runtime_error(ss.str());
  }

  if (src_type_id >= 0 && src_type_id < builtin_type_id_count) {
    ckb_offset = make_kernreq_to_single_kernel_adapter(ckb, ckb_offset, 1, kernreq);
    builtin_to_string_kernel_extra *e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                                            ->alloc_ck<builtin_to_string_kernel_extra>(ckb_offset);
    e->base.function = reinterpret_cast<void *>(builtin_to_string_kernel_extra::single);
    e->base.destructor = builtin_to_string_kernel_extra::destruct;
    // The kernel data owns this reference
    e->dst_string_tp = dst_string_tp;
    e->src_type_id = src_type_id;
    e->ectx = *ectx;
    e->dst_arrmeta = dst_arrmeta;
    return ckb_offset;
  }
  else {
    stringstream ss;
    ss << "make_builtin_to_string_assignment_kernel: source type id " << src_type_id << " is not builtin";
    throw runtime_error(ss.str());
  }
}

void dynd::assign_utf8_string_to_builtin(type_id_t dst_type_id, char *dst, const char *str_begin, const char *str_end,
                                         const eval::eval_context *ectx)
{
  ndt::type dt = ndt::string_type::make();
  dynd::string d(str_begin, str_end - str_begin);

  ckernel_builder<kernel_request_host> k;
  make_string_to_builtin_assignment_kernel(&k, 0, dst_type_id, dt, NULL, kernel_request_single, ectx);
  expr_single_t fn = k.get()->get_function<expr_single_t>();
  char *src = reinterpret_cast<char *>(&d);
  fn(k.get(), dst, &src);
}
