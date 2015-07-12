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

template <type_id_t TypeID>
struct nafunc {
  static nd::array get()
  {
    nd::array naf = nd::empty(ndt::option_type::make_nafunc_type());
    arrfunc_type_data *is_avail =
        reinterpret_cast<arrfunc_type_data *>(naf.get_ndo()->m_data_pointer);
    arrfunc_type_data *assign_na = is_avail + 1;

    new (is_avail) arrfunc_type_data(0, NULL, NULL,
                                     nd::is_avail_kernel<TypeID>::instantiate);
    new (assign_na) arrfunc_type_data(
        0, NULL, NULL, nd::assign_na_kernel<TypeID>::instantiate);
    return naf;
  }
};

const nd::array &dynd::get_option_builtin_nafunc(type_id_t tid)
{
  static nd::array bna = nafunc<bool_type_id>::get();
  static nd::array i8na = nafunc<int8_type_id>::get();
  static nd::array i16na = nafunc<int16_type_id>::get();
  static nd::array i32na = nafunc<int32_type_id>::get();
  static nd::array i64na = nafunc<int64_type_id>::get();
  static nd::array i128na = nafunc<int128_type_id>::get();
  static nd::array f32na = nafunc<float32_type_id>::get();
  static nd::array f64na = nafunc<float64_type_id>::get();
  static nd::array cf32na = nafunc<complex_float32_type_id>::get();
  static nd::array cf64na = nafunc<complex_float64_type_id>::get();
  static nd::array vna = nafunc<void_type_id>::get();
  static nd::array nullarr;
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