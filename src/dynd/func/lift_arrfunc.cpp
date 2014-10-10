//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/lift_arrfunc.hpp>
#include <dynd/kernels/make_lifted_ckernel.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/shape_tools.hpp>

using namespace std;
using namespace dynd;

static void delete_lifted_expr_arrfunc_data(arrfunc_type_data *self_af)
{
    memory_block_decref(*self_af->get_data_as<memory_block_data *>());
}

static intptr_t instantiate_lifted_expr_arrfunc_data(
    const arrfunc_type_data *self, dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const nd::array &DYND_UNUSED(aux), const eval::eval_context *ectx)
{
  const array_preamble *data = *self->get_data_as<const array_preamble *>();
  const arrfunc_type_data *child_af =
      reinterpret_cast<const arrfunc_type_data *>(data->m_data_pointer);
  intptr_t src_count = child_af->get_param_count();
  dimvector src_ndim(src_count);
  for (int i = 0; i < src_count; ++i) {
    src_ndim[i] = src_tp[i].get_ndim() - child_af->get_param_type(i).get_ndim();
  }
  return make_lifted_expr_ckernel(
      child_af, ckb, ckb_offset,
      dst_tp.get_ndim() - child_af->get_return_type().get_ndim(), dst_tp,
      dst_arrmeta, src_ndim.get(), src_tp, src_arrmeta,
      static_cast<dynd::kernel_request_t>(kernreq), ectx);
}

static int resolve_lifted_dst_type(const arrfunc_type_data *self, intptr_t nsrc,
                                   const ndt::type *src_tp,
                                   const nd::array &dyn_params,
                                   int throw_on_error, ndt::type &out_dst_tp)
{
    if (nsrc != self->get_param_count()) {
      if (throw_on_error) {
        stringstream ss;
        ss << "Wrong number of arguments to arrfunc with prototype ";
        ss << self->func_proto << ", got " << nsrc << " arguments";
        throw invalid_argument(ss.str());
      } else {
        return 0;
      }
    }
    const array_preamble *data = *self->get_data_as<const array_preamble *>();
    const arrfunc_type_data *child_af =
        reinterpret_cast<const arrfunc_type_data *>(data->m_data_pointer);
    intptr_t ndim = 0;
    // First get the type for the child arrfunc
    ndt::type child_dst_tp;
    if (child_af->resolve_dst_type) {
        std::vector<ndt::type> child_src_tp(nsrc);
        for (intptr_t i = 0; i < nsrc; ++i) {
            intptr_t child_ndim_i = child_af->get_param_type(i).get_ndim();
            if (child_ndim_i < src_tp[i].get_ndim()) {
              child_src_tp[i] = src_tp[i].get_dtype(child_ndim_i);
              ndim = std::max(ndim, src_tp[i].get_ndim() - child_ndim_i);
            } else {
              child_src_tp[i] = src_tp[i];
            }
        }
        if (!child_af->resolve_dst_type(child_af, nsrc, &child_src_tp[0],
                                        dyn_params, throw_on_error,
                                        child_dst_tp)) {
            return 0;
        }
    } else {
        // TODO: Should pattern match the source types here
        for (intptr_t i = 0; i < nsrc; ++i) {
            ndim = std::max(ndim, src_tp[i].get_ndim() -
                                      child_af->get_param_type(i).get_ndim());
        }
        child_dst_tp = child_af->get_return_type();
    }
    // Then build the type for the rest of the dimensions
    if (ndim > 0) {
        dimvector shape(ndim), tmp_shape(ndim);
        for (intptr_t i = 0; i < ndim; ++i) {
            shape[i] = -1;
        }
        for (intptr_t i = 0; i < nsrc; ++i) {
            intptr_t ndim_i =
                src_tp[i].get_ndim() - child_af->get_param_type(i).get_ndim();
            if (ndim_i > 0) {
                ndt::type tp = src_tp[i];
                intptr_t *shape_i = shape.get() + (ndim - ndim_i);
                intptr_t shape_at_j;
                for (intptr_t j = 0; j < ndim_i; ++j) {
                    switch (tp.get_type_id()) {
                        case fixed_dim_type_id:
                            shape_at_j = tp.tcast<fixed_dim_type>()->get_fixed_dim_size();
                            if (shape_i[j] < 0 || shape_i[j] == 1) {
                                if (shape_at_j != 1) {
                                    shape_i[j] = shape_at_j;
                                }
                            } else if (shape_i[j] != shape_at_j) {
                                if (throw_on_error) {
                                    throw broadcast_error(ndim, shape.get(),
                                                          ndim_i, shape_i);
                                } else {
                                    return 0;
                                }
                            }
                            break;
                        case cfixed_dim_type_id:
                            shape_at_j = tp.tcast<cfixed_dim_type>()->get_fixed_dim_size();
                            if (shape_i[j] < 0 || shape_i[j] == 1) {
                                if (shape_at_j != 1) {
                                    shape_i[j] = shape_at_j;
                                }
                            } else if (shape_i[j] != shape_at_j) {
                                if (throw_on_error) {
                                    throw broadcast_error(ndim, shape.get(),
                                                          ndim_i, shape_i);
                                } else {
                                    return 0;
                                }
                            }
                            break;
                        case var_dim_type_id:
                            break;
                        default:
                            if (throw_on_error) {
                                throw broadcast_error(ndim, shape.get(), ndim_i,
                                                      shape_i);
                            } else {
                                return 0;
                            }
                    }
                    tp = tp.get_dtype(tp.get_ndim() - 1);
                }
            }
        }
        for (intptr_t i = ndim - 1; i >= 0; --i) {
          if (shape[i] == -1) {
            child_dst_tp = ndt::make_var_dim(child_dst_tp);
          }
          else {
            child_dst_tp = ndt::make_fixed_dim(shape[i], child_dst_tp);
          }
        }
    }
    out_dst_tp = child_dst_tp;

    return 1;
}

/** Prepends "Dims..." to all the types in the proto */
static ndt::type lift_proto(const ndt::type& proto)
{
    const funcproto_type *p = proto.tcast<funcproto_type>();
    const ndt::type *param_types = p->get_param_types_raw();
    intptr_t param_count = p->get_param_count();
    nd::array out_param_types = nd::empty(param_count, ndt::make_type());
    nd::string dimsname("Dims");
    ndt::type *pt = reinterpret_cast<ndt::type *>(
        out_param_types.get_readwrite_originptr());
    for (intptr_t i = 0, i_end = p->get_param_count(); i != i_end; ++i) {
        pt[i] = ndt::make_ellipsis_dim(dimsname, param_types[i]);
    }
    return ndt::make_funcproto(
        out_param_types,
        ndt::make_ellipsis_dim(dimsname, p->get_return_type()));
}

void dynd::lift_arrfunc(arrfunc_type_data *out_af, const nd::arrfunc &af)
{
    const arrfunc_type_data *af_ptr = af.get();
    out_af->free_func = &delete_lifted_expr_arrfunc_data;
    *out_af->get_data_as<const array_preamble *>() = nd::array(af).release();
    out_af->instantiate = &instantiate_lifted_expr_arrfunc_data;
    out_af->resolve_dst_type = &resolve_lifted_dst_type;
    out_af->func_proto = lift_proto(af_ptr->func_proto);
}
