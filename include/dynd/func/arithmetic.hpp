//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/arrfunc.hpp>

namespace dynd {
namespace nd {

  typedef integer_sequence<
      type_id_t, int8_type_id, int16_type_id, int32_type_id, int64_type_id,
      uint8_type_id, uint16_type_id, uint32_type_id, uint64_type_id,
      float32_type_id, float64_type_id, complex_float32_type_id,
      complex_float64_type_id> arithmetic_type_ids;

  extern struct plus : declfunc<plus> {
    static arrfunc make();
  } plus;

  extern struct minus : declfunc<minus> {
    static arrfunc make();
  } minus;

  extern struct add : declfunc<add> {
    static arrfunc make();
  } add;

  extern struct subtract : declfunc<subtract> {
    static arrfunc make();
  } subtract;

  extern struct multiply : declfunc<multiply> {
    static arrfunc make();
  } multiply;

  extern struct divide : declfunc<divide> {
    static arrfunc make();
  } divide;

} // namespace nd
} // namespace dynd