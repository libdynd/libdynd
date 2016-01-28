//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// Implement a string concatenation kernel

#include <dynd/kernels/string_concat_kernels.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::string_concatenation::make()
{
  nd::functional::elwise(nd::callable::make<nd::string_concatenation_kernel>())
}

DYND_API struct nd::string_concatenation nd::string_concatenation;
