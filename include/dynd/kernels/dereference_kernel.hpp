//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  struct dereference_kernel : base_kernel<dereference_kernel, 1> {
    ndt::type src0_tp;

    dereference_kernel(const ndt::type &src0_tp) : src0_tp(src0_tp) {}

    void call(const array *dst, const array *src)
    {
      // Follow the pointers to eliminate them
      ndt::type dt = src0_tp;
      const char *arrmeta = src[0].get()->metadata();
      char *data = src[0].get()->data;
      memory_block_data *dataref = src[0].get()->owner.get();
      if (dataref == NULL) {
        dataref = src[0].get();
      }

      while (dt.get_id() == pointer_id) {
        const pointer_type_arrmeta *md = reinterpret_cast<const pointer_type_arrmeta *>(arrmeta);
        dt = dt.extended<ndt::pointer_type>()->get_target_type();
        arrmeta += sizeof(pointer_type_arrmeta);
        data = *reinterpret_cast<char **>(data) + md->offset;
        dataref = md->blockref.get();
      }

      if (!dt.is_builtin()) {
        dt.extended()->arrmeta_copy_construct((*dst)->metadata(), arrmeta,
                                              intrusive_ptr<memory_block_data>(src[0].get(), true));
      }
      (*dst)->data = data;
      (*dst)->owner = dataref;
    }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, intptr_t DYND_UNUSED(nkwd),
                                 const nd::array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = src_tp[0].extended<ndt::pointer_type>()->get_target_type();
    }

    static array alloc(const ndt::type *dst_tp) { return empty_shell(*dst_tp); }

    static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                            const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                            intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                            const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                            intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                            const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      ckb->emplace_back<dereference_kernel>(kernreq, src_tp[0]);
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <>
  struct traits<nd::dereference_kernel> {
    static type equivalent() { return type("(pointer[Any]) -> Any"); }
  };

} // namespace dynd::ndt

} // namespace dynd
