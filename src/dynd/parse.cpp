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
#include <dynd/callables/parse_dispatch_callable.hpp>

using namespace std;
using namespace dynd;

DYND_DEFAULT_DECLFUNC_GET(nd::json::parse)

DYND_API struct nd::json::parse nd::json::parse;

DYND_API nd::callable nd::json::parse::make()
{
  dispatcher<callable> dispatcher;
  dispatcher.insert({{bool_id}, callable::make<parse_kernel<bool_id>>()});
  dispatcher.insert({{int8_id}, callable::make<parse_kernel<int8_id>>()});
  dispatcher.insert({{int16_id}, callable::make<parse_kernel<int16_id>>()});
  dispatcher.insert({{int32_id}, callable::make<parse_kernel<int32_id>>()});
  dispatcher.insert({{int64_id}, callable::make<parse_kernel<int64_id>>()});
  dispatcher.insert({{uint8_id}, callable::make<parse_kernel<uint8_id>>()});
  dispatcher.insert({{uint16_id}, callable::make<parse_kernel<uint16_id>>()});
  dispatcher.insert({{uint32_id}, callable::make<parse_kernel<uint32_id>>()});
  dispatcher.insert({{uint64_id}, callable::make<parse_kernel<uint64_id>>()});
  dispatcher.insert({{string_id}, callable::make<parse_kernel<string_id>>()});
  dispatcher.insert({{struct_id}, callable::make<parse_kernel<struct_id>>()});
  dispatcher.insert({{option_id}, callable::make<parse_kernel<option_id>>()});
  dispatcher.insert({{fixed_dim_id}, callable::make<parse_kernel<fixed_dim_id>>()});
  dispatcher.insert({{var_dim_id}, callable::make<parse_kernel<var_dim_id>>()});

  return make_callable<parse_dispatch_callable>(
      ndt::callable_type::make(ndt::make_type<ndt::any_kind_type>(), {ndt::make_type<string>()}), dispatcher);
}
