//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/func/assignment.hpp>

namespace dynd {
namespace nd {

struct dereference_kernel : nd::base_kernel<dereference_kernel> {
  nd::array self;

  dereference_kernel(const nd::array &self) : self(self) {}

  void call(nd::array *dst, const nd::array *DYND_UNUSED(src)) { *dst = helper(self); }

  static void resolve_dst_type(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), ndt::type &dst_tp,
                               intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                               intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                               const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    dst_tp = helper(kwds[0]).get_type();
  }

  static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), nd::kernel_builder *ckb,
                          const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                          intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                          const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                          intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    ckb->emplace_back<dereference_kernel>(kernreq, kwds[0]);
  }

  static nd::array helper(const nd::array &self)
  {
    // Follow the pointers to eliminate them
    ndt::type dt = self.get_type();
    const char *arrmeta = self.get()->metadata();
    char *data = self.get()->data;
    memory_block_data *dataref = self.get()->owner.get();
    if (dataref == NULL) {
      dataref = self.get();
    }
    uint64_t flags = self.get()->flags;

    while (dt.get_id() == pointer_id) {
      const pointer_type_arrmeta *md = reinterpret_cast<const pointer_type_arrmeta *>(arrmeta);
      dt = dt.extended<ndt::pointer_type>()->get_target_type();
      arrmeta += sizeof(pointer_type_arrmeta);
      data = *reinterpret_cast<char **>(data) + md->offset;
      dataref = md->blockref.get();
    }

    // Create an array without the pointers
    nd::array result = nd::empty(dt);
    if (!dt.is_builtin()) {
      dt.extended()->arrmeta_copy_construct(result.get()->metadata(), arrmeta,
                                            intrusive_ptr<memory_block_data>(self.get(), true));
    }
    result.get()->tp = dt;
    result.get()->data = data;
    result.get()->owner = dataref;
    result.get()->flags = flags;
    return result;
  }
};

} // namespace dynd::nd

namespace ndt {

  template <>
  struct traits<nd::dereference_kernel> {
    static type equivalent() { return type("(self: Any) -> Any"); }
  };

} // namespace dynd::ndt

} // namespace dynd
