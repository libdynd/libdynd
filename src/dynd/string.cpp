//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/string.hpp>
#include <dynd/func/elwise.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::string_concatenation::make()
{
  return nd::functional::elwise(nd::callable::make<nd::string_concatenation_kernel>());
}

DYND_API struct nd::string_concatenation nd::string_concatenation;
