//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/chain_arrfunc.hpp>

using namespace std;
using namespace dynd;

static intptr_t
instantiate_chain(const arrfunc_type_data *af_self, dynd::ckernel_builder *ckb,
                 intptr_t ckb_offset, const ndt::type &dst_tp,
                 const char *dst_arrmeta, const ndt::type *src_tp,
                 const char *const *src_arrmeta, kernel_request_t kernreq,
                 const eval::eval_context *ectx)
{

}

static void free_chain_arrfunc(arrfunc_type_data *self_af)
{
  *self_af->get_data_as<ndt::type>() = ndt::type();
}

void dynd::make_chain_arrfunc(const nd::arrfunc &first,
                               const nd::arrfunc &second,
                               const ndt::type &buf_tp,
                               arrfunc_type_data *out_af)
{
  if (second.get()->func_proto.tcast<funcproto_type>()->get_param_count() !=
      1) {
    stringstream ss;
    ss << "Cannot chain functions " << first << " and " << second
       << ", because the second function is not unary";
    throw invalid_argument(ss.str());
  }
  out_af->free_func = &free_chain_arrfunc;
  out_af->func_proto = ndt::make_funcproto(
      first.get()->func_proto.tcast<funcproto_type>()->get_param_types(),
      second.get()->func_proto.tcast<funcproto_type>()->get_return_type());
  if (buf_tp.get_type_id() == uninitialized_type_id) {
    //out_af->resolve_dst_type = &resolve_chain_dst_type;
    //out_af->resolve_dst_shape = &resolve_chain_dst_shape;
    //out_af->instantiate = &instantiate_chain_resolve;
    throw runtime_error("Chaining functions without a provided intermediate "
                        "type is not implemented");
  } else {
    *out_af->get_data_as<ndt::type>() = buf_tp;
    out_af->instantiate = &instantiate_chain_buf_tp;
  }
}

nd::arrfunc dynd::make_chain_arrfunc(const nd::arrfunc &first,
                                     const nd::arrfunc &second,
                                     const ndt::type &buf_tp)
{
  nd::array af = nd::empty(ndt::make_arrfunc());
  make_chain_arrfunc(
      first, second, buf_tp,
      reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()));
  af.flag_as_immutable();
  return af;
}
