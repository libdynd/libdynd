//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <map>
#include <cmath>

#include <dynd/arithmetic.hpp>
#include <dynd/callable_registry.hpp>
#include <dynd/io.hpp>
#include <dynd/math.hpp>
#include <dynd/option.hpp>
#include <dynd/pointer.hpp>
#include <dynd/random.hpp>
#include <dynd/statistics.hpp>


#include <dynd/func/assignment.hpp>
#include <dynd/func/take.hpp>



using namespace std;
using namespace dynd;

std::map<std::string, nd::callable> &nd::callable_registry::get_regfunctions()
{
  static map<std::string, nd::callable> registry;
  if (registry.empty()) {
    registry["take"] = take;
    registry["min"] = min;
    registry["max"] = max;

    // dynd/arithmetic.hpp
    registry["add"] = add;
    registry["subtract"] = subtract;
    registry["multiply"] = multiply;
    registry["divide"] = divide;
    registry["sum"] = sum;

    // dynd/assign.hpp
    registry["assign"] = assign;

    // dynd/complex.hpp
    registry["real"] = real;
    registry["imag"] = imag;
    registry["conj"] = conj;

    // dynd/io.hpp
    registry["serialize"] = serialize;

    // dynd/math.hpp
    registry["sin"] = sin;
    registry["cos"] = cos;
    registry["tan"] = tan;
    registry["exp"] = exp;

    // dynd/option.hpp
    registry["assign_na"] = assign_na;
    registry["is_na"] = is_na;

    // dynd/pointer.hpp
    registry["dereference"] = dereference;

    // dynd/random.hpp
    registry["uniform"] = random::uniform;
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
