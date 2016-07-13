//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/arithmetic.hpp>
#include <dynd/assignment.hpp>
#include <dynd/comparison.hpp>
#include <dynd/index.hpp>
#include <dynd/io.hpp>
#include <dynd/math.hpp>
#include <dynd/option.hpp>
#include <dynd/pointer.hpp>
#include <dynd/random.hpp>
#include <dynd/range.hpp>
#include <dynd/registry.hpp>
#include <dynd/statistics.hpp>

using namespace std;
using namespace dynd;

registry_entry &dynd::registered() {
  static registry_entry entry{{"dynd", {{"nd", {{"add", nd::add},
                                                {"assign", nd::assign},
                                                {"assign_na", nd::assign_na},
                                                {"bitwise_and", nd::bitwise_and},
                                                {"bitwise_not", nd::bitwise_not},
                                                {"bitwise_or", nd::bitwise_or},
                                                {"bitwise_xor", nd::bitwise_xor},
                                                {"cbrt", nd::cbrt},
                                                {"compound_add", nd::compound_add},
                                                {"compound_div", nd::compound_div},
                                                {"conj", nd::conj},
                                                {"cos", nd::cos},
                                                {"dereference", nd::dereference},
                                                {"divide", nd::divide},
                                                {"equal", nd::equal},
                                                {"exp", nd::exp},
                                                {"greater", nd::greater},
                                                {"greater_equal", nd::greater_equal},
                                                {"imag", nd::imag},
                                                {"is_na", nd::is_na},
                                                {"left_shift", nd::left_shift},
                                                {"less", nd::less},
                                                {"less_equal", nd::less_equal},
                                                {"logical_and", nd::logical_and},
                                                {"logical_not", nd::logical_not},
                                                {"logical_or", nd::logical_or},
                                                {"logical_xor", nd::logical_xor},
                                                {"max", nd::max},
                                                {"min", nd::min},
                                                {"minus", nd::minus},
                                                {"mod", nd::mod},
                                                {"multiply", nd::multiply},
                                                {"not_equal", nd::not_equal},
                                                {"plus", nd::plus},
                                                {"pow", nd::pow},
                                                {"range", nd::range},
                                                {"real", nd::real},
                                                {"right_shift", nd::right_shift},
                                                {"serialize", nd::serialize},
                                                {"sin", nd::sin},
                                                {"sqrt", nd::sqrt},
                                                {"subtract", nd::subtract},
                                                {"sum", nd::sum},
                                                {"take", nd::take},
                                                {"tan", nd::tan},
                                                {"total_order", nd::total_order},
                                                {"random", {{"uniform", nd::random::uniform}}}}}}}};

  return entry;
}
