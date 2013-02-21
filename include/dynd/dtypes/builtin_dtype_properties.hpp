//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BUILTIN_DTYPE_PROPERTIES_HPP_
#define _DYND__BUILTIN_DTYPE_PROPERTIES_HPP_

#include <dynd/dtypes/type_id.hpp>
#include <dynd/gfunc/callable.hpp>

namespace dynd {

void get_builtin_dtype_dynamic_ndobject_properties(
                type_id_t builtin_type_id,
                const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count);

size_t get_builtin_dtype_elwise_property_index(
                type_id_t builtin_type_id,
                const std::string& property_name,
                bool& out_readable, bool& out_writable);

dtype get_builtin_dtype_elwise_property_dtype(
                type_id_t builtin_type_id,
                size_t elwise_property_index);

size_t make_builtin_dtype_elwise_property_getter_kernel(
                assignment_kernel *out, size_t offset_out,
                type_id_t builtin_type_id,
                const char *dst_metadata,
                const char *src_metadata, size_t src_elwise_property_index,
                kernel_request_t kernreq, const eval::eval_context *ectx);

size_t make_builtin_dtype_elwise_property_setter_kernel(
                assignment_kernel *out, size_t offset_out,
                type_id_t builtin_type_id,
                const char *dst_metadata, size_t dst_elwise_property_index,
                const char *src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx);

} // namespace dynd

#endif // _DYND__BUILTIN_DTYPE_PROPERTIES_HPP_