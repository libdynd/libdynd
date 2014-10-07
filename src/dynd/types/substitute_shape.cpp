//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/substitute_shape.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/base_memory_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/types/typevar_dim_type.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/dim_fragment_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/ctuple_type.hpp>

using namespace std;
using namespace dynd;


namespace {
struct substitute_shape_data {
  intptr_t ndim, i;
  const intptr_t *shape;
  const ndt::type *fulltype;

  void throw_error()
  {
    stringstream ss;
    ss << "Cannot substitute shape ";
    print_shape(ss, ndim, shape);
    ss << " into type " << *fulltype;
    throw invalid_argument(ss.str());
  }
};

static void substitute_shape_visitor(const ndt::type &tp,
                                     intptr_t DYND_UNUSED(arrmeta_offset),
                                     void *extra, ndt::type &out_transformed_tp,
                                     bool &out_was_transformed)
{
  substitute_shape_data *ssd = reinterpret_cast<substitute_shape_data *>(extra);
  intptr_t ndim = ssd->ndim, i = ssd->i;
  if (tp.get_kind() == dim_kind) {
    intptr_t dim_size = ssd->shape[i];
    ndt::type subtp = tp.tcast<base_dim_type>()->get_element_type();
    if (i + 1 < ndim) {
      ssd->i = i + 1;
      substitute_shape_visitor(subtp, 0, extra, subtp, out_was_transformed);
    }
    switch (tp.get_type_id()) {
    case strided_dim_type_id:
      if (dim_size >= 0) {
        out_transformed_tp = ndt::make_fixed_dim(dim_size, subtp);
        out_was_transformed = true;
      }
      else {
        ssd->throw_error();
      }
      return;
    case fixed_dim_type_id:
      if (dim_size < 0 ||
          dim_size == tp.tcast<fixed_dim_type>()->get_fixed_dim_size()) {
        if (!out_was_transformed) {
          out_transformed_tp = tp;
        }
        else {
          out_transformed_tp = ndt::make_fixed_dim(
              tp.tcast<fixed_dim_type>()->get_fixed_dim_size(), subtp);
        }
      }
      else {
        ssd->throw_error();
      }
      break;
    case cfixed_dim_type_id:
      if (dim_size < 0 ||
          dim_size == tp.tcast<cfixed_dim_type>()->get_fixed_dim_size()) {
        // If nothing was transformed, it's all good!
        if (!out_was_transformed) {
          out_transformed_tp = tp;
        }
        // Can only substitute here if the size of the data type remained the
        // same
        else if (subtp.get_data_size() ==
                 tp.tcast<cfixed_dim_type>()
                     ->get_element_type()
                     .get_data_size()) {
          out_transformed_tp = ndt::make_cfixed_dim(
              tp.tcast<cfixed_dim_type>()->get_fixed_dim_size(), subtp,
              tp.tcast<cfixed_dim_type>()->get_fixed_stride());
        }
        else {
          ssd->throw_error();
        }
      }
      else {
        ssd->throw_error();
      }
      break;
    case var_dim_type_id:
      if (!out_was_transformed) {
        out_transformed_tp = tp;
      }
      else {
        out_transformed_tp = ndt::make_var_dim(subtp);
      }
      break;
    default:
      ssd->throw_error();
    }
  }
  // In non-dimension case, preserve everything
  else if (i < ndim) {
    tp.extended()->transform_child_types(&substitute_shape_visitor, 0, extra,
                                         out_transformed_tp,
                                         out_was_transformed);
  }
  else {
    out_transformed_tp = tp;
  }
}
} // anonymous namespace

ndt::type ndt::substitute_shape(const ndt::type &pattern, intptr_t ndim,
                                const intptr_t *shape)
{
  substitute_shape_data ssd;
  ssd.ndim = ndim;
  ssd.i = 0;
  ssd.shape = shape;
  ssd.fulltype = &pattern;
  ndt::type transformed_tp;
  bool was_transformed;
  if (ndim > pattern.get_ndim()) {
    ssd.throw_error();
  }
  substitute_shape_visitor(pattern, 0, &ssd, transformed_tp, was_transformed);
  return transformed_tp;
}
