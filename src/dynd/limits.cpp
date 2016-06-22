//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/limits/min_callable.hpp>
#include <dynd/callables/multidispatch_callable.hpp>
#include <dynd/limits.hpp>

using namespace std;
using namespace dynd;

static std::vector<ndt::type> func_ptr(const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                                       const ndt::type *DYND_UNUSED(src_tp)) {
  return {dst_tp};
}

DYND_API nd::callable nd::limits::min = nd::make_callable<nd::multidispatch_callable<1>>(
    ndt::type("() -> Any"),
    nd::callable::make_all<nd::limits::min_callable, type_sequence<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t,
                                                                   uint32_t, uint64_t, float, double>>(func_ptr));
