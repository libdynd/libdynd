//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/array.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/elwise.hpp>

#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>

namespace dynd {
namespace decl {
  namespace nd {
    class elwise : public arrfunc<elwise> {

    public:
      static ndt::type make_type()
      {
        return ndt::type("(..., func: (...) -> Any) -> Any");
      }

      static int resolve_dst_type_from_child(
          const arrfunc_type_data *child_af, const arrfunc_type *child_af_tp,
          intptr_t nsrc, const ndt::type *src_tp, int throw_on_error,
          ndt::type &out_dst_tp, const dynd::nd::array &kwds)
      {
        std::cout << "resolve_dst_type_from_child" << std::endl;
        intptr_t ndim = 0;
        // First get the type for the child arrfunc
        ndt::type child_dst_tp;
        if (child_af->resolve_dst_type) {
          std::vector<ndt::type> child_src_tp(nsrc);
          for (intptr_t i = 0; i < nsrc; ++i) {
            intptr_t child_ndim_i = child_af_tp->get_pos_type(i).get_ndim();
            if (child_ndim_i < src_tp[i].get_ndim()) {
              child_src_tp[i] = src_tp[i].get_dtype(child_ndim_i);
              ndim = std::max(ndim, src_tp[i].get_ndim() - child_ndim_i);
            } else {
              child_src_tp[i] = src_tp[i];
            }
          }
          if (!child_af->resolve_dst_type(child_af, child_af_tp, nsrc,
                                          &child_src_tp[0], throw_on_error,
                                          child_dst_tp, kwds)) {
            return 0;
          }
        } else {
          // TODO: Should pattern match the source types here
          for (intptr_t i = 0; i < nsrc; ++i) {
            ndim = std::max(ndim, src_tp[i].get_ndim() -
                                      child_af_tp->get_pos_type(i).get_ndim());
          }
          child_dst_tp = child_af_tp->get_return_type();
        }
        // Then build the type for the rest of the dimensions
        if (ndim > 0) {
          dimvector shape(ndim), tmp_shape(ndim);
          for (intptr_t i = 0; i < ndim; ++i) {
            shape[i] = -1;
          }
          for (intptr_t i = 0; i < nsrc; ++i) {
            intptr_t ndim_i =
                src_tp[i].get_ndim() - child_af_tp->get_pos_type(i).get_ndim();
            if (ndim_i > 0) {
              ndt::type tp = src_tp[i];
              intptr_t *shape_i = shape.get() + (ndim - ndim_i);
              intptr_t shape_at_j;
              for (intptr_t j = 0; j < ndim_i; ++j) {
                switch (tp.get_type_id()) {
                case fixed_dim_type_id:
                  shape_at_j =
                      tp.extended<fixed_dim_type>()->get_fixed_dim_size();
                  if (shape_i[j] < 0 || shape_i[j] == 1) {
                    if (shape_at_j != 1) {
                      shape_i[j] = shape_at_j;
                    }
                  } else if (shape_i[j] != shape_at_j) {
                    if (throw_on_error) {
                      throw broadcast_error(ndim, shape.get(), ndim_i, shape_i);
                    } else {
                      return 0;
                    }
                  }
                  break;
                case cfixed_dim_type_id:
                  shape_at_j =
                      tp.extended<cfixed_dim_type>()->get_fixed_dim_size();
                  if (shape_i[j] < 0 || shape_i[j] == 1) {
                    if (shape_at_j != 1) {
                      shape_i[j] = shape_at_j;
                    }
                  } else if (shape_i[j] != shape_at_j) {
                    if (throw_on_error) {
                      throw broadcast_error(ndim, shape.get(), ndim_i, shape_i);
                    } else {
                      return 0;
                    }
                  }
                  break;
                case var_dim_type_id:
                  break;
                default:
                  if (throw_on_error) {
                    throw broadcast_error(ndim, shape.get(), ndim_i, shape_i);
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
            } else {
              child_dst_tp = ndt::make_fixed_dim(shape[i], child_dst_tp);
            }
          }
        }
        out_dst_tp = child_dst_tp;

        return 1;
      }

      static int resolve_dst_type(const arrfunc_type_data *DYND_UNUSED(self),
                                  const arrfunc_type *DYND_UNUSED(self_tp),
                                  intptr_t nsrc, const ndt::type *src_tp,
                                  int throw_on_error, ndt::type &dst_tp,
                                  const dynd::nd::array &kwds)
      {
        const arrfunc_type_data *child =
            reinterpret_cast<const arrfunc_type_data *>(
                kwds.get_readonly_originptr());
        const arrfunc_type *child_tp = kwds.get_type()
                                           .extended<base_struct_type>()
                                           ->get_field_type(0)
                                           .extended<arrfunc_type>();

        return resolve_dst_type_from_child(child, child_tp, nsrc, src_tp,
                                           throw_on_error, dst_tp,
                                           dynd::nd::array());
      }

      static intptr_t
      instantiate(const arrfunc_type_data *DYND_UNUSED(self),
                  const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,
                  intptr_t ckb_offset, const ndt::type &dst_tp,
                  const char *dst_arrmeta, const ndt::type *src_tp,
                  const char *const *src_arrmeta, kernel_request_t kernreq,
                  const eval::eval_context *ectx, const dynd::nd::array &kwds)
      {
        const arrfunc_type_data *child =
            reinterpret_cast<const arrfunc_type_data *>(
                kwds.get_readonly_originptr());
        const arrfunc_type *child_tp = kwds.get_type()
                                           .extended<base_struct_type>()
                                           ->get_field_type(0)
                                           .extended<arrfunc_type>();

        return kernels::make_lifted_expr_ckernel(
            child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
            src_arrmeta, kernreq, ectx, dynd::nd::array());
      }
    };
  } // namespace nd
} // namespace decl

namespace nd {
  decl::nd::elwise elwise;
}

} // namespace dynd