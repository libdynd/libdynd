//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/arithmetic.hpp>

using namespace std;
using namespace dynd;

const nd::arrfunc nd::multidispatch_plus_ck::children[builtin_type_id_count -
                                                      2] = {
    arrfunc(),                      arrfunc(),                      arrfunc(),
    as_arrfunc<plus_ck<int32_t>>(), as_arrfunc<plus_ck<int64_t>>(), arrfunc(),
    arrfunc(),                      arrfunc(),                      arrfunc(),
    arrfunc(),                      arrfunc(),                      arrfunc(),
    as_arrfunc<plus_ck<float>>(),   as_arrfunc<plus_ck<double>>(),  arrfunc(),
    arrfunc(),                      arrfunc(),
};

ndt::type nd::multidispatch_plus_ck::make_type()
{
  return ndt::type("(R) -> R");
}

const nd::arrfunc nd::multidispatch_minus_ck::children[builtin_type_id_count -
                                                       2] = {
    arrfunc(),                       arrfunc(),                       arrfunc(),
    as_arrfunc<minus_ck<int32_t>>(), as_arrfunc<minus_ck<int64_t>>(), arrfunc(),
    arrfunc(),                       arrfunc(),                       arrfunc(),
    arrfunc(),                       arrfunc(),                       arrfunc(),
    as_arrfunc<minus_ck<float>>(),   as_arrfunc<minus_ck<double>>(),  arrfunc(),
    arrfunc(),                       arrfunc(),
};

ndt::type nd::multidispatch_minus_ck::make_type()
{
  return ndt::type("(R) -> R");
}