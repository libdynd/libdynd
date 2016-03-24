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
#include <dynd/callables/parse_callable.hpp>

using namespace std;
using namespace dynd;

DYND_DEFAULT_DECLFUNC_GET(nd::json::parse)

DYND_API struct nd::json::parse nd::json::parse;

DYND_API nd::callable nd::json::parse::make()
{
  dispatcher<callable> dispatcher;
  dispatcher.insert({{bool_id}, make_callable<parse_callable<bool_id>>()});
  dispatcher.insert({{int8_id}, make_callable<parse_callable<int8_id>>()});
  dispatcher.insert({{int16_id}, make_callable<parse_callable<int16_id>>()});
  dispatcher.insert({{int32_id}, make_callable<parse_callable<int32_id>>()});
  dispatcher.insert({{int64_id}, make_callable<parse_callable<int64_id>>()});
  dispatcher.insert({{uint8_id}, make_callable<parse_callable<uint8_id>>()});
  dispatcher.insert({{uint16_id}, make_callable<parse_callable<uint16_id>>()});
  dispatcher.insert({{uint32_id}, make_callable<parse_callable<uint32_id>>()});
  dispatcher.insert({{uint64_id}, make_callable<parse_callable<uint64_id>>()});
  dispatcher.insert({{string_id}, make_callable<parse_callable<string_id>>()});
  dispatcher.insert({{struct_id}, make_callable<parse_callable<struct_id>>()});
  dispatcher.insert({{option_id}, make_callable<parse_callable<option_id>>()});
  dispatcher.insert({{fixed_dim_id}, make_callable<parse_callable<fixed_dim_id>>()});
  dispatcher.insert({{var_dim_id}, make_callable<parse_callable<var_dim_id>>()});

  return make_callable<parse_dispatch_callable>(
      ndt::callable_type::make(ndt::make_type<ndt::any_kind_type>(), {ndt::make_type<string>()}), dispatcher);
}
