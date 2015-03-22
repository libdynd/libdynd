//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/arithmetic.hpp>

using namespace std;
using namespace dynd;

detail::array_by_type_id<nd::arrfunc, 1> nd::multidispatch_plus_ck::children(
    {std::make_pair(int32_type_id, as_arrfunc<plus_ck<int32_t>>()),
     std::make_pair(int64_type_id, as_arrfunc<plus_ck<int64_t>>()),
     std::make_pair(float32_type_id, as_arrfunc<plus_ck<float>>()),
     std::make_pair(float64_type_id, as_arrfunc<plus_ck<double>>())});

ndt::type nd::multidispatch_plus_ck::make_type()
{
  return ndt::type("(R) -> R");
}

detail::array_by_type_id<nd::arrfunc, 1> nd::multidispatch_minus_ck::children(
    {std::make_pair(int32_type_id, as_arrfunc<minus_ck<int32_t>>()),
     std::make_pair(int64_type_id, as_arrfunc<minus_ck<int64_t>>()),
     std::make_pair(float32_type_id, as_arrfunc<minus_ck<float>>()),
     std::make_pair(float64_type_id, as_arrfunc<minus_ck<double>>())});

ndt::type nd::multidispatch_minus_ck::make_type()
{
  return ndt::type("(R) -> R");
}