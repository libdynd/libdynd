//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <climits>
#include <string>

#include <dynd/callables/parse_callable.hpp>
#include <dynd/callables/parse_dispatch_callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/parse_kernel.hpp>
#include <dynd/parse.hpp>
#include <dynd/string_encodings.hpp>
#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/option_type.hpp>

using namespace std;
using namespace dynd;

namespace {

nd::callable make_dynamic_parse() {
  dispatcher<1, nd::callable> dispatcher;
  dispatcher.insert({{bool_id}, nd::make_callable<nd::json::parse_callable<bool_id>>()});
  dispatcher.insert({{int8_id}, nd::make_callable<nd::json::parse_callable<int8_id>>()});
  dispatcher.insert({{int16_id}, nd::make_callable<nd::json::parse_callable<int16_id>>()});
  dispatcher.insert({{int32_id}, nd::make_callable<nd::json::parse_callable<int32_id>>()});
  dispatcher.insert({{int64_id}, nd::make_callable<nd::json::parse_callable<int64_id>>()});
  dispatcher.insert({{uint8_id}, nd::make_callable<nd::json::parse_callable<uint8_id>>()});
  dispatcher.insert({{uint16_id}, nd::make_callable<nd::json::parse_callable<uint16_id>>()});
  dispatcher.insert({{uint32_id}, nd::make_callable<nd::json::parse_callable<uint32_id>>()});
  dispatcher.insert({{uint64_id}, nd::make_callable<nd::json::parse_callable<uint64_id>>()});
  dispatcher.insert({{string_id}, nd::make_callable<nd::json::parse_callable<string_id>>()});
  dispatcher.insert({{struct_id}, nd::make_callable<nd::json::parse_callable<struct_id>>()});
  dispatcher.insert({{option_id}, nd::make_callable<nd::json::parse_callable<option_id>>()});
  dispatcher.insert({{fixed_dim_id}, nd::make_callable<nd::json::parse_callable<fixed_dim_id>>()});
  dispatcher.insert({{var_dim_id}, nd::make_callable<nd::json::parse_callable<var_dim_id>>()});

  return nd::make_callable<nd::parse_dispatch_callable>(
      ndt::make_type<ndt::callable_type>(ndt::make_type<ndt::any_kind_type>(), {ndt::make_type<dynd::string>()}),
      dispatcher);
}

} // unnamed namespace

DYND_API nd::callable nd::json::dynamic_parse = make_dynamic_parse();
