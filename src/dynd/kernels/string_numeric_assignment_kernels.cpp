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
