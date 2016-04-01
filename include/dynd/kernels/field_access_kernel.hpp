//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/assignment.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/adapt_type.hpp>

namespace dynd {
namespace nd {

  struct field_access_kernel : base_strided_kernel<field_access_kernel, 1> {
    const uintptr_t data_offset;

    field_access_kernel(uintptr_t data_offset) : data_offset(data_offset) {}

    ~field_access_kernel() { get_child()->destroy(); }

    void single(char *res, char *const *src)
    {
      char *const field_src[1] = {src[0] + data_offset};
      get_child()->single(res, field_src);
    }
  };

  // Temporary solution (previously in struct_type.cpp).
  struct get_array_field_kernel : nd::base_kernel<get_array_field_kernel> {
    array self;
    intptr_t i;

    get_array_field_kernel(const array &self, intptr_t i) : self(self), i(i) {}

    void call(array *dst, const array *DYND_UNUSED(src))
    {
      array res = helper(self, i);
      *dst = res;
    }

    static array helper(const array &n, intptr_t i)
    {
      // Get the nd::array 'self' parameter
      intptr_t undim = n.get_ndim();
      ndt::type udt = n.get_dtype();
      if (udt.is_expression()) {
        std::string field_name = udt.value_type().extended<ndt::struct_type>()->get_field_name(i);
        return n.replace_dtype(ndt::make_type<ndt::adapt_type>(
            udt.value_type().extended<ndt::struct_type>()->get_field_type(i), udt, nd::callable(), nd::callable()));
      }
      else {
        if (undim == 0) {
          return n(i);
        }
        else {
          shortvector<irange> idx(undim + 1);
          idx[undim] = irange(i);
          return n.at_array(undim + 1, idx.get());
        }
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
