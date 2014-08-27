//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/static_type_instances.hpp>
#include <dynd/type.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/json_type.hpp>
#include <dynd/types/ndarrayarg_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/type_type.hpp>

namespace dynd { namespace types {
// Static instances of selected types
base_type *arrfunc_tp;
base_type *bytes_tp;
base_type *char_tp;
base_type *date_tp;
base_type *datetime_tp;
base_type *json_tp;
base_type *ndarrayarg_tp;
base_type *string_tp;
base_type *strided_of_string_tp;
base_type *time_tp;
base_type *type_tp;
base_type *strided_of_type_tp;
}} // namespace dynd::types

void dynd::init::static_types_init()
{
  static arrfunc_type aft;
  types::arrfunc_tp = &aft;
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
  static strided_dim_type sst(
      *reinterpret_cast<const ndt::type *>(&types::string_tp));
  types::strided_of_string_tp = &sst;
  static time_type tt(tz_abstract);
  types::time_tp = &tt;
  static type_type tpt;
  types::type_tp = &tpt;
  static strided_dim_type stpt(
      *reinterpret_cast<const ndt::type *>(&types::type_tp));
  types::strided_of_type_tp = &stpt;
}

void dynd::init::static_types_cleanup()
{
}
