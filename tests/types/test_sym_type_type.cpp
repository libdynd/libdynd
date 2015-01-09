//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/types/sym_type_type.hpp>

using namespace std;
using namespace dynd;

TEST(SymTypeType, Simple)
{

// Fixed, Type[T], type_type, sym_type_type
//  std::cout << ndt::type("(tp: <Dims... * T>, ) -> void") << std::endl;

// "(Type[Dims... * T], 

  std::cout << ndt::make_sym_type_type(ndt::type("T")) << std::endl;

  std::exit(-1);
}
