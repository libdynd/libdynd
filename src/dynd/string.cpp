//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/elwise.hpp>
#include <dynd/kernels/string_concat_kernel.hpp>
#include <dynd/kernels/string_count_kernel.hpp>
#include <dynd/kernels/string_find_kernel.hpp>
#include <dynd/kernels/string_rfind_kernel.hpp>
#include <dynd/kernels/string_replace_kernel.hpp>
#include <dynd/kernels/string_split_kernel.hpp>
#include <dynd/string.hpp>

using namespace std;
using namespace dynd;

////////////////////////////////////////////////////////////
// String kernels

namespace dynd {
namespace nd {

  DYND_API callable string_concatenation::make()
  {
    return functional::elwise(callable::make<string_concatenation_kernel>());
  }

  DYND_DEFAULT_DECLFUNC_GET(string_concatenation)

  DYND_API struct string_concatenation string_concatenation;

  DYND_API callable string_count::make() { return functional::elwise(callable::make<string_count_kernel>()); }

  DYND_DEFAULT_DECLFUNC_GET(string_count)

  DYND_API struct string_count string_count;

  DYND_API callable string_find::make() { return functional::elwise(callable::make<string_find_kernel>()); }

  DYND_DEFAULT_DECLFUNC_GET(string_find)

  DYND_API struct string_find string_find;

  DYND_API callable string_rfind::make() { return functional::elwise(callable::make<string_rfind_kernel>()); }

  DYND_API struct string_rfind string_rfind;

  DYND_API callable string_replace::make() { return functional::elwise(callable::make<string_replace_kernel>()); }

  DYND_DEFAULT_DECLFUNC_GET(string_replace)

  DYND_API struct string_replace string_replace;

  DYND_API callable string_split::make() { return functional::elwise(callable::make<string_split_kernel>()); }

  DYND_DEFAULT_DECLFUNC_GET(string_split)

  DYND_API struct string_split string_split;

} // namespace nd
} // namespace dynd
