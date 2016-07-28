//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <dynd/registry.hpp>
#include <dynd_assertions.hpp>

using namespace std;
using namespace dynd;

/*
TEST(CallableRegistry, Dispatch)
{
  nd::callable af;
  af = nd::callable_registry["sin");
  // These are exact overloads of ``sin``
  EXPECT_DOUBLE_EQ(sin(1.0), af(1.0).as<double>());
  EXPECT_FLOAT_EQ(sin(2.0f), af(2.0f).as<float>());
  // Implicit int -> double conversion
  EXPECT_DOUBLE_EQ(sin(3.0), af(3).as<double>());
  EXPECT_DOUBLE_EQ(sin(4.0), af(4u).as<double>());
  // Bool doesn't implicitly convert to float
  EXPECT_THROW(af(true), type_error);
}
*/

TEST(CallableRegistry, Arithmetic) {
  registry_entry &entry = registered("dynd.nd");

  // Simple sanity checks
  nd::callable af;
  af = entry["add"].value();
  EXPECT_EQ(ndt::type("int32"), af((int8_t)3, (int8_t)4).get_type());
  EXPECT_EQ(8, af(3, 5).as<int>());
  EXPECT_EQ(ndt::type("float32"), af(3.5f, 5.25f).get_type());
  EXPECT_EQ(8.75, af(3.5f, 5.25f).as<float>());
  af = entry["subtract"].value();
  EXPECT_EQ(ndt::type("float64"), af(3.5, 4).get_type());
  EXPECT_EQ(-0.5, af(3.5, 4).as<double>());
  af = entry["multiply"].value();
  EXPECT_EQ(ndt::type("float32"), af(3.5f, (int8_t)4).get_type());
  EXPECT_EQ(14, af(3.5f, (int8_t)4).as<float>());
  af = entry["divide"].value();
  EXPECT_EQ(ndt::type("float64"), af(12.0, (int8_t)4).get_type());
  EXPECT_EQ(3, af(12.0, (int8_t)4).as<double>());
}

TEST(CallableRegistry, Trig) {
  registry_entry &entry = registered("dynd.nd");

  // Simple sanity checks
  nd::callable af;
  af = entry["sin"].value();
  //  EXPECT_FLOAT_EQ(sinf(2.0f), af(2.0f).as<float>());
  EXPECT_DOUBLE_EQ(sin(1.0), af(1.0).as<double>());
  af = entry["cos"].value();
  // EXPECT_FLOAT_EQ(cosf(1.f), af(1.f).as<float>());
  EXPECT_DOUBLE_EQ(cos(1.0), af(1.0).as<double>());
  af = entry["tan"].value();
  //  EXPECT_FLOAT_EQ(tanf(1.f), af(1.f).as<float>());
  EXPECT_DOUBLE_EQ(tan(1.0), af(1.0).as<double>());
  af = entry["exp"].value();
  // EXPECT_FLOAT_EQ(expf(1.f), af(1.f).as<float>());
  EXPECT_DOUBLE_EQ(exp(1.0), af(1.0).as<double>());
  /*
    af = nd::callable_registry["arcsin"];
    EXPECT_FLOAT_EQ(asinf(0.4f), af(0.4f).as<float>());
    EXPECT_DOUBLE_EQ(asin(1.0), af(1.0).as<double>());
    af = nd::callable_registry["arccos"];
    EXPECT_FLOAT_EQ(acosf(1.f), af(1.f).as<float>());
    EXPECT_DOUBLE_EQ(acos(1.0), af(1.0).as<double>());
    af = nd::callable_registry["arctan"];
    EXPECT_FLOAT_EQ(atanf(1.f), af(1.f).as<float>());
    EXPECT_DOUBLE_EQ(atan(1.0), af(1.0).as<double>());
    af = nd::callable_registry["arctan2"];
    EXPECT_FLOAT_EQ(atan2f(1.f, 2.f), af(1.f, 2.f).as<float>());
    EXPECT_DOUBLE_EQ(atan2(1.0, 2.0), af(1.0, 2.0).as<double>());
    af = nd::callable_registry["hypot"];
    EXPECT_FLOAT_EQ(5, af(3.f, 4.f).as<float>());
    EXPECT_DOUBLE_EQ(hypot(3.0, 4.5), af(3.0, 4.5).as<double>());
    af = nd::callable_registry["sinh"];
    EXPECT_FLOAT_EQ(sinhf(2.0f), af(2.0f).as<float>());
    EXPECT_DOUBLE_EQ(sinh(1.0), af(1.0).as<double>());
    af = nd::callable_registry["cosh"];
    EXPECT_FLOAT_EQ(coshf(1.f), af(1.f).as<float>());
    EXPECT_DOUBLE_EQ(cosh(1.0), af(1.0).as<double>());
    af = nd::callable_registry["tanh"];
    EXPECT_FLOAT_EQ(tanhf(1.f), af(1.f).as<float>());
    EXPECT_DOUBLE_EQ(tanh(1.0), af(1.0).as<double>());
    af = nd::callable_registry["power"];
    EXPECT_FLOAT_EQ(powf(1.5f, 2.25f), af(1.5f, 2.25f).as<float>());
    EXPECT_DOUBLE_EQ(pow(1.5, 2.25), af(1.5, 2.25).as<double>());
  */
}

// TEST(Registry, Insert) {
// registry_entry &entry = registered();
// entry.insert({"x", {{"y", nd::callable([] { return 0; })}}});

//  std::cout << entry.path() << std::endl;

//  registry_entry &entry2 = registered("dynd.nd.random");
// std::cout << entry2.path() << std::endl;

//  std::exit(-1);
/*


  entry.observe([](registry_entry *, const char *name) { std::cout << (std::string(name) + " changed") << std::endl;
});
  std::cout << entry.name() << std::endl;

  registry_entry &entry2 = registered("dynd");

  entry2.observe([](registry_entry *, const char *name) { std::cout << (std::string(name) + " changed") << std::endl;
});
//  std::cout << entry.name() << std::endl;

//  registry_entry &entry = registered("dynd.nd");
//  std::cout << (entry.parent() == NULL) << std::endl;
  //std::cout << entry.name() << std::endl;

//  entry.observe([](registry_entry *, const char *name) {
  //  std::cout << (std::string(name) + " changed") << std::endl;

//    EXPECT_EQ("f", std::string(name));

  //  flag = true;
//  });

  entry.insert({"f", nd::callable([] { return 0; })});
//  EXPECT_TRUE(flag);
*/
//}
