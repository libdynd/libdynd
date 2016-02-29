//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename SelfType>
  struct base_index_kernel : kernel_prefix {
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

    kernel_prefix *get_child() { return kernel_prefix::get_child(sizeof(SelfType)); }

    static void call_wrapper(kernel_prefix *self, array *dst, const array *src)
    {
      reinterpret_cast<SelfType *>(self)->call(dst, src);
    }
    static const volatile char *DYND_USED(ir);

    static void strided_wrapper(kernel_prefix *, char *, intptr_t, char *const *, const intptr_t *, size_t)
    {
      throw std::runtime_error("invalid");
    }

    static void destruct(kernel_prefix *self) { reinterpret_cast<SelfType *>(self)->~SelfType(); }

    template <typename... ArgTypes>
    static void init(SelfType *self, kernel_request_t kernreq, ArgTypes &&... args)
    {
      new (self) SelfType(std::forward<ArgTypes>(args)...);

      self->destructor = SelfType::destruct;
      switch (kernreq) {
      case kernel_request_call:
        self->function = reinterpret_cast<void *>(SelfType::call_wrapper);
        break;
      case kernel_request_single:
        self->function = reinterpret_cast<void *>(SelfType::single_wrapper);
        break;
      default:
        DYND_HOST_THROW(std::invalid_argument,
                        "expr ckernel init: unrecognized ckernel request " + std::to_string(kernreq));
      }
    }

    void call(const array *res, const array *args)
    {
      res->get()->data = args[0]->data;
      reinterpret_cast<SelfType *>(this)->single(res->get()->metadata(), &res->get()->data);
    }

    void single(char *DYND_UNUSED(dst), char **DYND_UNUSED(src))
    {
      std::stringstream ss;
      ss << "void single(char *dst, char **src) is not implemented in " << typeid(SelfType).name();
      throw std::runtime_error(ss.str());
    }

    static void single_wrapper(kernel_prefix *self, char *res_metadata, char **res_data)
    {
      reinterpret_cast<SelfType *>(self)->single(res_metadata, res_data);
    }

    static char *data_init(char *DYND_UNUSED(static_data), const ndt::type &DYND_UNUSED(dst_tp),
                           intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp), intptr_t DYND_UNUSED(nkwd),
                           const array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      return reinterpret_cast<char *>(new data_type(kwds[0]));
    }
  };

  template <typename SelfType>
  const volatile char *DYND_USED(base_index_kernel<SelfType>::ir) = NULL;

} // namespace dynd::nd
} // namespace dynd
