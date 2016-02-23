//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/substitute_shape.hpp>
#include <dynd/callable.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/base_memory_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/types/typevar_dim_type.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/dim_fragment_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/tuple_type.hpp>

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
    throw type_error(ss.str());
  }
};

static void substitute_shape_visitor(const ndt::type &tp, intptr_t DYND_UNUSED(arrmeta_offset), void *extra,
                                     ndt::type &out_transformed_tp, bool &out_was_transformed)
{
  substitute_shape_data *ssd = reinterpret_cast<substitute_shape_data *>(extra);
  intptr_t ndim = ssd->ndim, i = ssd->i;
  if (!tp.is_scalar()) {
    intptr_t dim_size = ssd->shape[i];
    ndt::type subtp = tp.extended<ndt::base_dim_type>()->get_element_type();
    if (i + 1 < ndim) {
      ssd->i = i + 1;
      substitute_shape_visitor(subtp, 0, extra, subtp, out_was_transformed);
    }
    switch (tp.get_id()) {
    case fixed_dim_id:
      if (tp.is_symbolic()) {
        if (dim_size >= 0) {
          out_transformed_tp = ndt::make_fixed_dim(dim_size, subtp);
          out_was_transformed = true;
        }
        else {
          ssd->throw_error();
        }
      }
      else {
        if (dim_size < 0 || dim_size == tp.extended<ndt::fixed_dim_type>()->get_fixed_dim_size()) {
          if (!out_was_transformed) {
            out_transformed_tp = tp;
          }
          else {
            out_transformed_tp = ndt::make_fixed_dim(tp.extended<ndt::fixed_dim_type>()->get_fixed_dim_size(), subtp);
          }
        }
        else {
          ssd->throw_error();
        }
      }
      break;
    case var_dim_id:
      if (!out_was_transformed) {
        out_transformed_tp = tp;
      }
      else {
        out_transformed_tp = ndt::var_dim_type::make(subtp);
      }
      break;
    default:
      ssd->throw_error();
    }
  }
  // In non-dimension case, preserve everything
  else if (i < ndim) {
    tp.extended()->transform_child_types(&substitute_shape_visitor, 0, extra, out_transformed_tp, out_was_transformed);
  }
  else {
    out_transformed_tp = tp;
  }
}
} // anonymous namespace

ndt::type ndt::substitute_shape(const ndt::type &pattern, intptr_t ndim, const intptr_t *shape)
{
  substitute_shape_data ssd;
  ssd.ndim = ndim;
  ssd.i = 0;
  ssd.shape = shape;
  ssd.fulltype = &pattern;
  ndt::type transformed_tp;
  bool was_transformed = false;
  if (ndim > pattern.get_ndim()) {
    ssd.throw_error();
  }
  substitute_shape_visitor(pattern, 0, &ssd, transformed_tp, was_transformed);
  return transformed_tp;
}
