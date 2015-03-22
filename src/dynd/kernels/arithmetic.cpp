//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/arithmetic.hpp>

using namespace std;
using namespace dynd;

#define DYND_BUILTIN_TYPE_IDS                                                  \
  (int8_type_id, int16_type_id, int32_type_id, int64_type_id, uint8_type_id,   \
   uint16_type_id, uint32_type_id, uint64_type_id, float32_type_id,            \
   float64_type_id, complex_float32_type_id, complex_float64_type_id)

#define MAKE_PAIRS(I) DYND_PP_JOIN_MAP(MAKE_PAIR, (, ), I)

#define MAKE_PAIR(I) std::make_pair(I, as_arrfunc<plus_ck<I>>())

detail::array_by_type_id<nd::arrfunc, 1> nd::multidispatch_plus_ck::children(
    {MAKE_PAIRS((int8_type_id, int16_type_id, int32_type_id, int64_type_id,
                 uint8_type_id, uint16_type_id, uint32_type_id, uint64_type_id,
                 float32_type_id, float64_type_id, complex_float32_type_id))});

#undef MAKE_PAIR

ndt::type nd::multidispatch_plus_ck::make_type()
{
  return ndt::type("(A0) -> R");
}

detail::array_by_type_id<nd::arrfunc, 1> nd::multidispatch_minus_ck::children(
    {std::make_pair(int32_type_id, as_arrfunc<minus_ck<int32_t>>()),
     std::make_pair(int64_type_id, as_arrfunc<minus_ck<int64_t>>()),
     std::make_pair(float32_type_id, as_arrfunc<minus_ck<float>>()),
     std::make_pair(float64_type_id, as_arrfunc<minus_ck<double>>())});

ndt::type nd::multidispatch_minus_ck::make_type()
{
  return ndt::type("(A0) -> R");
}

#undef MAKE_PAIRS

#define MAKE_PAIRS(I, J)                                                       \
  DYND_PP_JOIN_ELWISE(_MAKE_PAIRS, (, ), I, DYND_PP_REPEAT(J, DYND_PP_LEN(I)))
#define _MAKE_PAIRS(I, J)                                                      \
  DYND_PP_JOIN_ELWISE_1(MAKE_PAIR, (, ), DYND_PP_REPEAT(I, DYND_PP_LEN(J)), J)

#define MAKE_PAIR(I, J)                                                        \
  std::make_pair(std::make_pair(I, J), as_arrfunc<add_ck<I, J>>())

detail::array_by_type_id<nd::arrfunc, 2> nd::virtual_add_ck::children(
    {MAKE_PAIRS(DYND_BUILTIN_TYPE_IDS, DYND_BUILTIN_TYPE_IDS)});

#undef MAKE_PAIR

ndt::type nd::virtual_add_ck::make_type() { return ndt::type("(A0, A1) -> R"); }

#define MAKE_PAIR(I, J)                                                        \
  std::make_pair(std::make_pair(I, J), as_arrfunc<subtract_ck<I, J>>())

detail::array_by_type_id<nd::arrfunc, 2> nd::virtual_subtract_ck::children(
    {MAKE_PAIRS(DYND_BUILTIN_TYPE_IDS, DYND_BUILTIN_TYPE_IDS)});

#undef MAKE_PAIR

ndt::type nd::virtual_subtract_ck::make_type()
{
  return ndt::type("(A0, A1) -> R");
}

#define MAKE_PAIR(I, J)                                                        \
  std::make_pair(std::make_pair(I, J), as_arrfunc<multiply_ck<I, J>>())

detail::array_by_type_id<nd::arrfunc, 2> nd::virtual_multiply_ck::children(
    {MAKE_PAIRS(DYND_BUILTIN_TYPE_IDS, DYND_BUILTIN_TYPE_IDS)});

#undef MAKE_PAIR

ndt::type nd::virtual_multiply_ck::make_type()
{
  return ndt::type("(A0, A1) -> R");
}

#define MAKE_PAIR(I, J)                                                        \
  std::make_pair(std::make_pair(I, J), as_arrfunc<divide_ck<I, J>>())

detail::array_by_type_id<nd::arrfunc, 2> nd::virtual_divide_ck::children(
    {MAKE_PAIRS(DYND_BUILTIN_TYPE_IDS, DYND_BUILTIN_TYPE_IDS)});

#undef MAKE_PAIR

ndt::type nd::virtual_divide_ck::make_type()
{
  return ndt::type("(A0, A1) -> R");
}

#undef _MAKE_PAIRS
#undef MAKE_PAIRS