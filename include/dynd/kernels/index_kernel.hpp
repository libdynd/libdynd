//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_index_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID>
  struct index_kernel : base_index_kernel<index_kernel<Arg0ID>> {
    using base_index_kernel<index_kernel>::base_index_kernel;

    void single(char *DYND_UNUSED(metadata), char **DYND_UNUSED(data), const array *DYND_UNUSED(arg0)) {}
  };

  template <>
  struct index_kernel<fixed_dim_id> : base_index_kernel<index_kernel<fixed_dim_id>> {
    intptr_t index;
    intptr_t stride;

    index_kernel(int index, intptr_t stride) : index(index), stride(stride) {}

    ~index_kernel() { get_child()->destroy(); }

    void single(char *metadata, char **data, const array *DYND_UNUSED(arg0))
    {
      //      reinterpret_cast<ndt::fixed_dim_type::metadata_type *>(metadata)->stride = stride;
      *data += index * stride;

      get_child()->single(metadata, data, NULL);
    }
  };

} // namespace dynd::nd
} // namespace dynd
