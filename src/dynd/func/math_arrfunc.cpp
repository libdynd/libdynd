//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/math_arrfunc.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

struct sin_ck : public kernels::unary_ck<sin_ck> {
  inline double operator()(double x) { return ::sin(x); }

  static intptr_t instantiate(const arrfunc_type_data *self_af,
                              dynd::ckernel_builder *ckb, intptr_t ckb_offset,
                              const ndt::type &dst_tp,
                              const char *DYND_UNUSED(dst_arrmeta),
                              const ndt::type *src_tp,
                              const char *const *DYND_UNUSED(src_arrmeta),
                              kernel_request_t kernreq,
                              const eval::eval_context *DYND_UNUSED(ectx))
  {
    if (dst_tp !=
            self_af->func_proto.tcast<funcproto_type>()->get_return_type() ||
        src_tp[0] !=
            self_af->func_proto.tcast<funcproto_type>()->get_param_type(0)) {
      stringstream ss;
      ss << "Cannot instantiate arrfunc with signature ";
      ss << self_af->func_proto << " with types (";
      ss << src_tp[0] << ") -> " << dst_tp;
      throw type_error(ss.str());
    }
    self_type::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  }
};

static nd::arrfunc make_sin_arrfunc()
{
  nd::array out_af = nd::empty(ndt::make_arrfunc());
  arrfunc_type_data *af =
      reinterpret_cast<arrfunc_type_data *>(out_af.get_readwrite_originptr());
  af->func_proto = ndt::type("(real) -> real");
  af->instantiate = &sin_ck::instantiate;
  out_af.flag_as_immutable();
  return out_af;
}

nd::arrfunc math::sin = make_sin_arrfunc();
