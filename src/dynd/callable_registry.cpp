//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <map>
#include <cmath>

#include <dynd/callable_registry.hpp>
#include <dynd/func/arithmetic.hpp>
#include <dynd/func/assignment.hpp>
#include <dynd/io.hpp>
#include <dynd/math.hpp>
#include <dynd/option.hpp>
#include <dynd/func/random.hpp>
#include <dynd/func/sum.hpp>
#include <dynd/func/take.hpp>
#include <dynd/func/min.hpp>
#include <dynd/func/max.hpp>
#include <dynd/func/complex.hpp>
#include <dynd/func/pointer.hpp>

using namespace std;
using namespace dynd;

std::map<std::string, nd::callable> &nd::callable_registry::get_regfunctions()
{
  static map<std::string, nd::callable> registry;
  if (registry.empty()) {
    registry["take"] = take::get();
    registry["sum"] = sum::get();
    registry["min"] = min::get();
    registry["max"] = max::get();

    // dynd/arithmetic.hpp
    registry["add"] = add::get();
    registry["subtract"] = subtract::get();
    registry["multiply"] = multiply::get();
    registry["divide"] = divide::get();

    // dynd/assign.hpp
    registry["assign"] = assign::get();

    // dynd/complex.hpp
    registry["real"] = real::get();
    registry["imag"] = imag::get();
    registry["conj"] = conj::get();

    // dynd/io.hpp
    registry["serialize"] = serialize::get();

    // dynd/math.hpp
    registry["sin"] = sin::get();
    registry["cos"] = cos::get();
    registry["tan"] = tan::get();
    registry["exp"] = exp::get();

    // dynd/option.hpp
    registry["assign_na"] = assign_na::get();
    registry["is_na"] = is_na::get();

    // dynd/pointer.hpp
    registry["dereference"] = dereference::get();

    // dynd/random.hpp
    registry["uniform"] = random::uniform::get();
  }

  return registry;
}

nd::callable &nd::callable_registry::operator[](const std::string &name)
{
  std::map<std::string, nd::callable> &registry = get_regfunctions();

  auto it = registry.find(name);
  if (it != registry.end()) {
    return it->second;
  }
  else {
    stringstream ss;
    ss << "No dynd function ";
    print_escaped_utf8_string(ss, name);
    ss << " has been registered";
    throw invalid_argument(ss.str());
  }
}

DYND_API class nd::callable_registry nd::callable_registry;
