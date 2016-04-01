//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/index_kernel.hpp>
#include <dynd/type.hpp>
#include <dynd/types/fixed_dim_type.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID>
  class index_callable : public base_callable {
  public:
    struct data_type {
      intptr_t nindices;
      int *indices;

      data_type(intptr_t nindices, int *indices) : nindices(nindices), indices(indices) {}
      data_type(const array &index) : data_type(index.get_dim_size(), reinterpret_cast<int *>(index.data())) {}

      void next() {
        --nindices;
        ++indices;
      }
    };

    index_callable() : base_callable(ndt::type("(Any, i: Any) -> Any")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), call_graph &cg, const ndt::type &DYND_UNUSED(dst_tp),
                      size_t DYND_UNUSED(nsrc), const ndt::type *src_tp, size_t DYND_UNUSED(nkwd),
                      const array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back(this);
      return src_tp[0];
    }

    char *data_init(const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                    const ndt::type *DYND_UNUSED(src_tp), intptr_t DYND_UNUSED(nkwd), const array *kwds,
                    const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      return reinterpret_cast<char *>(new data_type(kwds[0]));
    }

    void resolve_dst_type(char *data, ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                          intptr_t DYND_UNUSED(nkwd), const array *kwds,
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      dst_tp = src_tp[0];
      new (data) data_type(kwds[0]);
    }

    void instantiate(char *data, kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                     const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta),
                     kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      ckb->emplace_back<index_kernel<Arg0ID>>(kernreq);
      delete reinterpret_cast<data_type *>(data);
    }
  };

  template <>
  class index_callable<fixed_dim_id> : public base_callable {
  public:
    struct data_type {
      intptr_t nindices;
      int *indices;

      data_type(intptr_t nindices, int *indices) : nindices(nindices), indices(indices) {}
      data_type(const array &index) : data_type(index.get_dim_size(), reinterpret_cast<int *>(index.data())) {}

      void next() {
        --nindices;
        ++indices;
      }
    };

    index_callable() : base_callable(ndt::type("(Any, i: Any) -> Any")) {}

    char *data_init(const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                    const ndt::type *DYND_UNUSED(src_tp), intptr_t DYND_UNUSED(nkwd), const array *kwds,
                    const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      return reinterpret_cast<char *>(new data_type(kwds[0]));
    }

    ndt::type resolve(base_callable *DYND_UNUSED(caller), call_graph &cg, const ndt::type &dst_tp, size_t nsrc,
                      const ndt::type *src_tp, size_t nkwd, const array *kwds,
                      const std::map<std::string, ndt::type> &tp_vars) {
      cg.emplace_back(this);

      ndt::type child_src_tp = src_tp[0].extended<ndt::fixed_dim_type>()->get_element_type();
      return index->resolve(this, cg, dst_tp, nsrc, &child_src_tp, nkwd, kwds, tp_vars);
    }

    void resolve_dst_type(char *data, ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd,
                          const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      reinterpret_cast<data_type *>(data)->next();

      ndt::type child_src_tp = src_tp[0].extended<ndt::fixed_dim_type>()->get_element_type();
      index->resolve_dst_type(data, dst_tp, nsrc, &child_src_tp, nkwd, kwds, tp_vars);
    }

    void instantiate(char *data, kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                     const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      ckb->emplace_back<index_kernel<fixed_dim_id>>(
          kernreq, *reinterpret_cast<data_type *>(data)->indices,
          reinterpret_cast<const ndt::fixed_dim_type::metadata_type *>(src_arrmeta[0])->stride);

      reinterpret_cast<data_type *>(data)->next();

      ndt::type child_src_tp = src_tp[0].extended<ndt::fixed_dim_type>()->get_element_type();
      const char *child_src_arrmeta = src_arrmeta[0] + sizeof(ndt::fixed_dim_type::metadata_type);
      index->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, &child_src_tp, &child_src_arrmeta, kernel_request_single,
                         nkwd, kwds, tp_vars);
    }
  };

} // namespace dynd::nd
} // namespace dynd
