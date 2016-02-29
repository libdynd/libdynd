//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  struct index_kernel_prefix : kernel_prefix {
    void single(char *dst, char **src, array *a)
    {
      (reinterpret_cast<void (*)(kernel_prefix *, char *, char **, array *)>(function))(this, dst, src, a);
    }
  };

  template <typename SelfType>
  struct base_index_kernel : base_kernel<index_kernel_prefix, SelfType> {
    struct data_type {
      intptr_t nindices;
      int *indices;

      data_type(intptr_t nindices, int *indices) : nindices(nindices), indices(indices) {}
      data_type(const array &index) : data_type(index.get_dim_size(), reinterpret_cast<int *>(index.data())) {}

      void next()
      {
        --nindices;
        ++indices;
      }
    };

    index_kernel_prefix *get_child()
    {
      return reinterpret_cast<index_kernel_prefix *>(kernel_prefix::get_child(sizeof(SelfType)));
    }

    void call(const array *res, const array *arg0)
    {
      res->get()->data = (*arg0)->data;
      reinterpret_cast<SelfType *>(this)->single(res->get()->metadata(), &res->get()->data, arg0);
    }

    void single(char *DYND_UNUSED(dst), char **DYND_UNUSED(src), array *)
    {
      std::stringstream ss;
      ss << "void single(char *dst, char **src) is not implemented in " << typeid(SelfType).name();
      throw std::runtime_error(ss.str());
    }

    static void single_wrapper(kernel_prefix *self, char *res_metadata, char **res_data, const array *arg0)
    {
      reinterpret_cast<SelfType *>(self)->single(res_metadata, res_data, arg0);
    }

    static char *data_init(char *DYND_UNUSED(static_data), const ndt::type &DYND_UNUSED(dst_tp),
                           intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp), intptr_t DYND_UNUSED(nkwd),
                           const array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      return reinterpret_cast<char *>(new data_type(kwds[0]));
    }
  };

} // namespace dynd::nd
} // namespace dynd
