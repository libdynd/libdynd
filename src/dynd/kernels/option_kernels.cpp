//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/kernels/option_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/kernels/is_avail_kernel.hpp>
#include <dynd/kernels/assign_na_kernel.hpp>

using namespace std;
using namespace dynd;

// c{is_avail : (T) -> bool, assign_na : () -> T}
// naf.p("assign_na").vals() =
// nd::as_arrfunc<nd::assign_na_ck<T>>(naf.p("assign_na").get_type(), 0);

const nd::arrfunc &dynd::get_option_builtin_is_avail(type_id_t tid)
{
  static nd::arrfunc bna =
      nd::arrfunc::make<nd::is_avail_kernel<bool_type_id>>(0);
  static nd::arrfunc i8na =
      nd::arrfunc::make<nd::is_avail_kernel<int8_type_id>>(0);
  static nd::arrfunc i16na =
      nd::arrfunc::make<nd::is_avail_kernel<int16_type_id>>(0);
  static nd::arrfunc i32na =
      nd::arrfunc::make<nd::is_avail_kernel<int32_type_id>>(0);
  static nd::arrfunc i64na =
      nd::arrfunc::make<nd::is_avail_kernel<int64_type_id>>(0);
  static nd::arrfunc i128na =
      nd::arrfunc::make<nd::is_avail_kernel<int128_type_id>>(0);
  static nd::arrfunc f32na =
      nd::arrfunc::make<nd::is_avail_kernel<float32_type_id>>(0);
  static nd::arrfunc f64na =
      nd::arrfunc::make<nd::is_avail_kernel<float64_type_id>>(0);
  static nd::arrfunc cf32na =
      nd::arrfunc::make<nd::is_avail_kernel<complex_float32_type_id>>(0);
  static nd::arrfunc cf64na =
      nd::arrfunc::make<nd::is_avail_kernel<complex_float64_type_id>>(0);
  static nd::arrfunc vna =
      nd::arrfunc::make<nd::is_avail_kernel<void_type_id>>(0);
  static nd::arrfunc nullarr;

  switch (tid) {
  case bool_type_id:
    return bna;
  case int8_type_id:
    return i8na;
  case int16_type_id:
    return i16na;
  case int32_type_id:
    return i32na;
  case int64_type_id:
    return i64na;
  case int128_type_id:
    return i128na;
  case float32_type_id:
    return f32na;
  case float64_type_id:
    return f64na;
  case complex_float32_type_id:
    return cf32na;
  case complex_float64_type_id:
    return cf64na;
  case void_type_id:
    return vna;
  default:
    return nullarr;
  }
}

const nd::arrfunc &dynd::get_option_builtin_assign_na(type_id_t tid)
{
  static nd::arrfunc bna =
      nd::arrfunc::make<nd::assign_na_kernel<bool_type_id>>(0);
  static nd::arrfunc i8na =
      nd::arrfunc::make<nd::assign_na_kernel<int8_type_id>>(0);
  static nd::arrfunc i16na =
      nd::arrfunc::make<nd::assign_na_kernel<int16_type_id>>(0);
  static nd::arrfunc i32na =
      nd::arrfunc::make<nd::assign_na_kernel<int32_type_id>>(0);
  static nd::arrfunc i64na =
      nd::arrfunc::make<nd::assign_na_kernel<int64_type_id>>(0);
  static nd::arrfunc i128na =
      nd::arrfunc::make<nd::assign_na_kernel<int128_type_id>>(0);
  static nd::arrfunc f32na =
      nd::arrfunc::make<nd::assign_na_kernel<float32_type_id>>(0);
  static nd::arrfunc f64na =
      nd::arrfunc::make<nd::assign_na_kernel<float64_type_id>>(0);
  static nd::arrfunc cf32na =
      nd::arrfunc::make<nd::assign_na_kernel<complex_float32_type_id>>(0);
  static nd::arrfunc cf64na =
      nd::arrfunc::make<nd::assign_na_kernel<complex_float64_type_id>>(0);
  static nd::arrfunc vna =
      nd::arrfunc::make<nd::assign_na_kernel<void_type_id>>(0);
  static nd::arrfunc nullarr;

  switch (tid) {
  case bool_type_id:
    return bna;
  case int8_type_id:
    return i8na;
  case int16_type_id:
    return i16na;
  case int32_type_id:
    return i32na;
  case int64_type_id:
    return i64na;
  case int128_type_id:
    return i128na;
  case float32_type_id:
    return f32na;
  case float64_type_id:
    return f64na;
  case complex_float32_type_id:
    return cf32na;
  case complex_float64_type_id:
    return cf64na;
  case void_type_id:
    return vna;
  default:
    return nullarr;
  }
}