//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <chrono>

#include <dynd/callables/multidispatch_callable.hpp>
#include <dynd/callables/uniform_callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/random.hpp>

using namespace std;
using namespace dynd;

namespace {

static std::vector<ndt::type> func_ptr(const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                                       const ndt::type *DYND_UNUSED(src_tp)) {
  return {dst_tp};
}

template <typename GeneratorType>
struct uniform_callable_alias {
  template <typename ReturnType>
  using type = nd::random::uniform_callable<ReturnType, GeneratorType>;
};

} // unnamed namespace

DYND_API nd::callable nd::random::uniform = nd::functional::elwise(nd::make_callable<nd::multidispatch_callable<1>>(
    ndt::type("(a: ?R, b: ?R) -> R"),
    nd::callable::make_all<uniform_callable_alias<std::default_random_engine>::type,
                           type_sequence<int32_t, int64_t, uint32_t, uint64_t, float, double, dynd::complex<float>,
                                         dynd::complex<double>>>(func_ptr)));
