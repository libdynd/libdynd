//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <climits>
#include <string>

#include <dynd/functional.hpp>
#include <dynd/kernels/parse_kernel.hpp>
#include <dynd/parse.hpp>
#include <dynd/string_encodings.hpp>
#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/option_type.hpp>

using namespace std;
using namespace dynd;

DYND_DEFAULT_DECLFUNC_GET(nd::json::parse)

DYND_API struct nd::json::parse nd::json::parse;

DYND_API nd::callable nd::json::parse::make()
{
  std::map<type_id_t, callable> children;
  children[bool_id] = callable::make<parse_kernel<bool_id>>();
  children[int8_id] = callable::make<parse_kernel<int8_id>>();
  children[int16_id] = callable::make<parse_kernel<int16_id>>();
  children[int32_id] = callable::make<parse_kernel<int32_id>>();
  children[int64_id] = callable::make<parse_kernel<int64_id>>();
  children[uint8_id] = callable::make<parse_kernel<uint8_id>>();
  children[uint16_id] = callable::make<parse_kernel<uint16_id>>();
  children[uint32_id] = callable::make<parse_kernel<uint32_id>>();
  children[uint64_id] = callable::make<parse_kernel<uint64_id>>();
  children[string_id] = callable::make<parse_kernel<string_id>>();
  children[struct_id] = callable::make<parse_kernel<struct_id>>();
  children[option_id] = callable::make<parse_kernel<option_id>>();
  children[fixed_dim_id] = callable::make<parse_kernel<fixed_dim_id>>();
  children[var_dim_id] = callable::make<parse_kernel<var_dim_id>>();

  return functional::dispatch(
      ndt::callable_type::make(ndt::make_type<ndt::any_kind_type>(), {ndt::make_type<string>()}),
      [children](const ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                 const ndt::type *DYND_UNUSED(src_tp)) mutable -> callable & { return children[dst_tp.get_id()]; });
}
