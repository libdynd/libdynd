//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <map>

#include <dynd/types/type_id.hpp>
#include <dynd/kernels/ckernel_prefix.hpp>

namespace dynd {
namespace nd {

  class callable;

} // namespace dynd::nd

DYND_API void get_builtin_type_dynamic_array_properties(type_id_t builtin_type_id,
                                                        std::map<std::string, nd::callable> &properties);

DYND_API size_t get_builtin_type_elwise_property_index(type_id_t builtin_type_id, const std::string &property_name);

DYND_API ndt::type get_builtin_type_elwise_property_type(type_id_t builtin_type_id, size_t elwise_property_index,
                                                         bool &out_readable, bool &out_writable);

DYND_API size_t make_builtin_type_elwise_property_getter_kernel(
    void *ckb, intptr_t ckb_offset, type_id_t builtin_type_id, const char *dst_arrmeta, const char *src_arrmeta,
    size_t src_elwise_property_index, kernel_request_t kernreq, const eval::eval_context *ectx);

DYND_API size_t make_builtin_type_elwise_property_setter_kernel(void *ckb, intptr_t ckb_offset,
                                                                type_id_t builtin_type_id, const char *dst_arrmeta,
                                                                size_t dst_elwise_property_index,
                                                                const char *src_arrmeta, kernel_request_t kernreq,
                                                                const eval::eval_context *ectx);

} // namespace dynd
