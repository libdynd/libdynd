//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/arithmetic.hpp>

using namespace std;
using namespace dynd;

#define BUILTIN_ROW(NAME, A0)                                                  \
  {                                                                            \
    NULL, &kernels::create<kernels::NAME<A0, int8_t>>,                         \
        &kernels::create<kernels::NAME<A0, int16_t>>,                          \
        &kernels::create<kernels::NAME<A0, int32_t>>,                          \
        &kernels::create<kernels::NAME<A0, int64_t>>,                          \
        NULL,                                                                  \
        &kernels::create<kernels::NAME<A0, uint8_t>>,                          \
        &kernels::create<kernels::NAME<A0, uint16_t>>,                         \
        &kernels::create<kernels::NAME<A0, uint32_t>>,                         \
        &kernels::create<kernels::NAME<A0, uint64_t>>,                         \
        NULL,                                                                  \
        NULL,                                                                  \
        &kernels::create<kernels::NAME<A0, float>>,                            \
        &kernels::create<kernels::NAME<A0, double>>,                           \
        NULL,                                                                  \
        &kernels::create<kernels::NAME<A0, dynd_complex<float>>>,              \
        &kernels::create<kernels::NAME<A0, dynd_complex<double>>>,             \
  }

#define BUILTIN_TABLE(NAME)                                                    \
  const kernels::create_t                                                      \
      nd::decl::NAME::builtin_table[builtin_type_id_count -                    \
                                    2][builtin_type_id_count - 2] = {          \
          {NULL},                                                              \
          BUILTIN_ROW(NAME##_ck, int8_t),                                      \
          BUILTIN_ROW(NAME##_ck, int16_t),                                     \
          BUILTIN_ROW(NAME##_ck, int32_t),                                     \
          BUILTIN_ROW(NAME##_ck, int64_t),                                     \
          {NULL},                                                              \
          BUILTIN_ROW(NAME##_ck, uint8_t),                                     \
          BUILTIN_ROW(NAME##_ck, uint16_t),                                    \
          BUILTIN_ROW(NAME##_ck, uint32_t),                                    \
          BUILTIN_ROW(NAME##_ck, uint64_t),                                    \
          {NULL},                                                              \
          {NULL},                                                              \
          BUILTIN_ROW(NAME##_ck, float),                                       \
          BUILTIN_ROW(NAME##_ck, double),                                      \
          {NULL},                                                              \
          BUILTIN_ROW(NAME##_ck, dynd_complex<float>),                         \
          BUILTIN_ROW(NAME##_ck, dynd_complex<double>)};

BUILTIN_TABLE(add);
BUILTIN_TABLE(sub);
BUILTIN_TABLE(mul);
BUILTIN_TABLE(div);

#undef BUILTIN_TABLE
#undef BUILTIN_ROW

nd::decl::add nd::add;
nd::decl::sub nd::sub;
nd::decl::mul nd::mul;
nd::decl::div nd::div;

