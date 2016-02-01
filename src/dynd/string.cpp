//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/string.hpp>
#include <dynd/kernels/string_concat_kernel.hpp>
#include <dynd/kernels/string_count_kernel.hpp>
#include <dynd/kernels/string_find_kernel.hpp>
#include <dynd/kernels/string_replace_kernel.hpp>
#include <dynd/func/elwise.hpp>

using namespace std;
using namespace dynd;


////////////////////////////////////////////////////////////
// String kernels


DYND_API nd::callable nd::string_concatenation::make()
{
  return nd::functional::elwise(nd::callable::make<nd::string_concatenation_kernel>());
}

DYND_API struct nd::string_concatenation nd::string_concatenation;

DYND_API nd::callable nd::string_count::make()
{
  return nd::functional::elwise(nd::callable::make<nd::string_count_kernel>());
}

DYND_API struct nd::string_count nd::string_count;

DYND_API nd::callable nd::string_find::make()
{
  return nd::functional::elwise(nd::callable::make<nd::string_find_kernel>());
}

DYND_API struct nd::string_find nd::string_find;

DYND_API nd::callable nd::string_replace::make()
{
  return nd::functional::elwise(nd::callable::make<nd::string_replace_kernel>());
}

DYND_API struct nd::string_replace nd::string_replace;
