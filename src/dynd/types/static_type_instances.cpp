//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/static_type_instances.hpp>
#include <dynd/type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/json_type.hpp>
#include <dynd/types/ndarrayarg_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/categorical_type.hpp>
#include <dynd/types/builtin_type_properties.hpp>

namespace dynd { namespace types {
// Static instances of selected types
base_type *bytes_tp;
base_type *char_tp;
base_type *date_tp;
base_type *datetime_tp;
base_type *json_tp;
base_type *ndarrayarg_tp;
base_type *string_tp;
base_type *time_tp;
base_type *type_tp;
}} // namespace dynd::types

void dynd::init::static_types_init()
{
  static bytes_type bt(1);
  types::bytes_tp = &bt;
  static char_type ct(string_encoding_utf_32);
  types::char_tp = &ct;
  static date_type dt;
  types::date_tp = &dt;
  static datetime_type dtt(tz_abstract);
  types::datetime_tp = &dtt;
  static json_type jt;
  types::json_tp = &jt;
  static ndarrayarg_type naat;
  types::ndarrayarg_tp = &naat;
  static string_type st(string_encoding_utf_8);
  types::string_tp = &st;
  static time_type tt(tz_abstract);
  types::time_tp = &tt;
  static type_type tpt;
  types::type_tp = &tpt;
  // Call initialization of individual types
  init::builtins_type_init();
  init::categorical_type_init();
  init::base_string_type_init();
}

void dynd::init::static_types_cleanup()
{
  init::base_string_type_cleanup();
  init::categorical_type_cleanup();
  init::builtins_type_cleanup();
}
