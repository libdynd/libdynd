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

static std::vector<ndt::type> func_ptr(const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                                       const ndt::type *DYND_UNUSED(src_tp)) {
  return {dst_tp};
}

nd::callable make_dynamic_parse() {
  dispatcher<1, nd::callable> dispatcher(func_ptr);
  dispatcher.insert(nd::make_callable<nd::json::parse_callable<bool>>());
  dispatcher.insert(nd::make_callable<nd::json::parse_callable<int8_t>>());
  dispatcher.insert(nd::make_callable<nd::json::parse_callable<int16_t>>());
  dispatcher.insert(nd::make_callable<nd::json::parse_callable<int32_t>>());
  dispatcher.insert(nd::make_callable<nd::json::parse_callable<int64_t>>());
  dispatcher.insert(nd::make_callable<nd::json::parse_callable<uint8_t>>());
  dispatcher.insert(nd::make_callable<nd::json::parse_callable<uint16_t>>());
  dispatcher.insert(nd::make_callable<nd::json::parse_callable<uint32_t>>());
  dispatcher.insert(nd::make_callable<nd::json::parse_callable<uint64_t>>());
  dispatcher.insert(nd::make_callable<nd::json::parse_callable<dynd::string>>());
  dispatcher.insert(nd::make_callable<nd::json::parse_callable<ndt::struct_type>>());
  dispatcher.insert(nd::make_callable<nd::json::parse_callable<ndt::option_type>>());
  dispatcher.insert(nd::make_callable<nd::json::parse_callable<ndt::fixed_dim_type>>());
  dispatcher.insert(nd::make_callable<nd::json::parse_callable<ndt::var_dim_type>>());

  return nd::make_callable<nd::parse_dispatch_callable>(
      ndt::make_type<ndt::callable_type>(ndt::make_type<ndt::any_kind_type>(), {ndt::make_type<dynd::string>()}),
      dispatcher);
}

} // unnamed namespace

DYND_API nd::callable nd::json::dynamic_parse = make_dynamic_parse();
