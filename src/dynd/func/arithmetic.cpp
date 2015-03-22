//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/arithmetic.hpp>
#include <dynd/kernels/arithmetic.hpp>
#include <dynd/array.hpp>

using namespace dynd;

nd::arrfunc nd::plus::make()
{
  return functional::elwise(as_arrfunc<multidispatch_plus_ck>());
}

struct nd::plus nd::plus;

nd::array nd::operator+(const nd::array &a0) { return nd::plus(a0); }

nd::arrfunc nd::minus::make()
{
  return functional::elwise(as_arrfunc<multidispatch_minus_ck>());
}

struct nd::minus nd::minus;

nd::array nd::operator-(const nd::array &a0) { return nd::minus(a0); }

nd::arrfunc nd::add::make()
{
  return functional::elwise(as_arrfunc<virtual_add_ck>());
}

struct nd::add nd::add;

nd::array nd::operator+(const nd::array &a0, const nd::array &a1)
{
  return nd::add(a0, a1);
}

nd::arrfunc nd::subtract::make()
{
  return functional::elwise(as_arrfunc<virtual_subtract_ck>());
}

struct nd::subtract nd::subtract;

nd::array nd::operator-(const nd::array &a0, const nd::array &a1)
{
  return nd::subtract(a0, a1);
}

#define BUILTIN_ROW(NAME, A0)                                                  \
  {                                                                            \
    NULL, &create<nd::NAME<A0, int8_t>>, &create<nd::NAME<A0, int16_t>>,       \
        &create<nd::NAME<A0, int32_t>>, &create<nd::NAME<A0, int64_t>>, NULL,  \
        &create<nd::NAME<A0, uint8_t>>, &create<nd::NAME<A0, uint16_t>>,       \
        &create<nd::NAME<A0, uint32_t>>, &create<nd::NAME<A0, uint64_t>>,      \
        NULL, NULL, &create<nd::NAME<A0, float>>,                              \
        &create<nd::NAME<A0, double>>, NULL,                                   \
        &create<nd::NAME<A0, dynd::complex<float>>>,                           \
        &create<nd::NAME<A0, dynd::complex<double>>>,                          \
  }

#define BUILTIN_TABLE(NAME)                                                    \
  const create_t nd::NAME::builtin_table[builtin_type_id_count -               \
                                         2][builtin_type_id_count - 2] = {     \
      {NULL},                                                                  \
      BUILTIN_ROW(NAME##_ck, int8_t),                                          \
      BUILTIN_ROW(NAME##_ck, int16_t),                                         \
      BUILTIN_ROW(NAME##_ck, int32_t),                                         \
      BUILTIN_ROW(NAME##_ck, int64_t),                                         \
      {NULL},                                                                  \
      BUILTIN_ROW(NAME##_ck, uint8_t),                                         \
      BUILTIN_ROW(NAME##_ck, uint16_t),                                        \
      BUILTIN_ROW(NAME##_ck, uint32_t),                                        \
      BUILTIN_ROW(NAME##_ck, uint64_t),                                        \
      {NULL},                                                                  \
      {NULL},                                                                  \
      BUILTIN_ROW(NAME##_ck, float),                                           \
      BUILTIN_ROW(NAME##_ck, double),                                          \
      {NULL},                                                                  \
      BUILTIN_ROW(NAME##_ck, dynd::complex<float>),                            \
      BUILTIN_ROW(NAME##_ck, dynd::complex<double>)};

BUILTIN_TABLE(mul);
BUILTIN_TABLE(div);

#undef BUILTIN_TABLE
#undef BUILTIN_ROW

struct nd::mul nd::mul;
struct nd::div nd::div;

nd::array nd::operator*(const nd::array &a0, const nd::array &a1)
{
  return nd::mul(a0, a1);
}

nd::array nd::operator/(const nd::array &a0, const nd::array &a1)
{
  return nd::div(a0, a1);
}
