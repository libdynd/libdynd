//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/exceptions.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/types/any_kind_type.hpp>

using namespace std;
using namespace dynd;

bool ndt::any_kind_type::is_expression() const { return false; }

bool ndt::any_kind_type::is_unique_data_owner(const char *DYND_UNUSED(arrmeta)) const { return false; }

void ndt::any_kind_type::transform_child_types(type_transform_fn_t DYND_UNUSED(transform_fn),
                                               intptr_t DYND_UNUSED(arrmeta_offset), void *DYND_UNUSED(extra),
                                               type &out_transformed_tp, bool &DYND_UNUSED(out_was_transformed)) const {
  out_transformed_tp = type(this, true);
}

ndt::type ndt::any_kind_type::get_canonical_type() const { return type(this, true); }

ndt::type ndt::any_kind_type::at_single(intptr_t DYND_UNUSED(i0), const char **DYND_UNUSED(inout_arrmeta),
                                        const char **DYND_UNUSED(inout_data)) const {
  return type(this, true);
}

ndt::type ndt::any_kind_type::get_type_at_dimension(char **DYND_UNUSED(inout_arrmeta), intptr_t DYND_UNUSED(i),
                                                    intptr_t DYND_UNUSED(total_ndim)) const {
  return type(this, true);
}

intptr_t ndt::any_kind_type::get_dim_size(const char *DYND_UNUSED(arrmeta), const char *DYND_UNUSED(data)) const {
  return -1;
}

void ndt::any_kind_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *DYND_UNUSED(arrmeta),
                                   const char *DYND_UNUSED(data)) const {
  for (; i < ndim; ++i) {
    out_shape[i] = -1;
  }
}

bool ndt::any_kind_type::is_lossless_assignment(const type &DYND_UNUSED(dst_tp),
                                                const type &DYND_UNUSED(src_tp)) const {
  return false;
}
