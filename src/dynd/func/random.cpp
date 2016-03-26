//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <chrono>

#include <dynd/func/random.hpp>
#include <dynd/functional.hpp>
#include <dynd/callables/uniform_callable.hpp>
#include <dynd/callables/uniform_dispatch_callable.hpp>

using namespace std;
using namespace dynd;

namespace {

template <typename GeneratorType>
struct uniform_callable_alias {
  template <type_id_t DstTypeID>
  using type = nd::random::uniform_callable<DstTypeID, GeneratorType>;
};

} // unnamed namespace

DYND_API nd::callable nd::random::uniform = nd::functional::elwise(nd::make_callable<nd::uniform_dispatch_callable>(
    ndt::type("(a: ?R, b: ?R) -> R"),
    nd::callable::new_make_all<uniform_callable_alias<std::default_random_engine>::type,
                               type_id_sequence<int32_id, int64_id, uint32_id, uint64_id, float32_id, float64_id,
                                                complex_float32_id, complex_float64_id>>()));
