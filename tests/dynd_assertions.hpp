//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <gtest/gtest.h>

#include <memory>

#include <dynd/json_parser.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/type_promotion.hpp>

#include <dynd/math.hpp>
#include <dynd/functional.hpp>
#include <dynd/option.hpp>

inline std::string ShapeFormatter(const std::vector<intptr_t> &shape) {
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0, i_end = shape.size(); i != i_end; ++i) {
    ss << shape[i];
    if (i != i_end - 1) {
      ss << ", ";
    }
  }
  ss << ")";
  return ss.str();
}

inline ::testing::AssertionResult CompareDyNDArrays(const char *expr1, const char *expr2, const dynd::nd::array &val1,
                                                    const dynd::nd::array &val2) {
  using namespace dynd;

  if (val1.get_type().get_id() == cuda_device_id && val2.get_type().get_id() == cuda_device_id) {
    return CompareDyNDArrays(expr1, expr2, val1.to_host(), val2.to_host());
  }

  if (val1.equals_exact(val2)) {
    return ::testing::AssertionSuccess();
  } else {
    if (val1.get_type() != val2.get_type()) {
      return ::testing::AssertionFailure() << "The types of " << expr1 << " and " << expr2 << " do not match\n" << expr1
                                           << " has type " << val1.get_type() << ",\n" << expr2 << " has type "
                                           << val2.get_type() << ".";
    } else if (val1.get_shape() != val2.get_shape()) {
      return ::testing::AssertionFailure() << "The shapes of " << expr1 << " and " << expr2 << " do not match\n"
                                           << expr1 << " has shape " << ShapeFormatter(val1.get_shape()) << ",\n"
                                           << expr2 << " has shape " << ShapeFormatter(val2.get_shape()) << ".";
    } else if (val1.get_type().get_id() == struct_id) {
      const ndt::struct_type *bsd = val1.get_type().extended<ndt::struct_type>();
      intptr_t field_count = bsd->get_field_count();
      for (intptr_t i = 0; i < field_count; ++i) {
        nd::array field1 = val1(i), field2 = val2(i);
        if (!field1.equals_exact(field2)) {
          return ::testing::AssertionFailure()
                 << "The values of " << expr1 << " and " << expr2 << " do not match at field index " << i << ", name \""
                 << bsd->get_field_name(i) << "\"\n" << expr1 << " has field value " << field1 << ",\n" << expr2
                 << " has field value " << field2 << ".";
        }
      }
      return ::testing::AssertionFailure() << "DYND ASSERTION INTERNAL ERROR: One of the struct fields "
                                              "should have compared unequal";
    } else if (val1.get_type().get_id() == tuple_id) {
      const ndt::tuple_type *bsd = val1.get_type().extended<ndt::tuple_type>();
      intptr_t field_count = bsd->get_field_count();
      for (intptr_t i = 0; i < field_count; ++i) {
        nd::array field1 = val1(i), field2 = val2(i);
        if (!field1.equals_exact(field2)) {
          return ::testing::AssertionFailure()
                 << "The values of " << expr1 << " and " << expr2 << " do not match at field index " << i << "\"\n"
                 << expr1 << " has field value " << field1 << ",\n" << expr2 << " has field value " << field2 << ".";
        }
      }
      return ::testing::AssertionFailure() << "DYND ASSERTION INTERNAL ERROR: One of the tuple fields "
                                              "should have compared unequal";
    } else if (val1.get_ndim() > 0) {
      intptr_t dim_size = val1.get_dim_size();
      for (intptr_t i = 0; i < dim_size; ++i) {
        nd::array sub1 = val1(i), sub2 = val2(i);
        if (!sub1.equals_exact(sub2)) {
          return ::testing::AssertionFailure()
                 << "The values of " << expr1 << " and " << expr2 << " do not match at index " << i << "\"\n" << expr1
                 << " has subarray value " << sub1 << ",\n" << expr2 << " has subarray value " << sub2 << ".";
        }
      }
      return ::testing::AssertionFailure() << "DYND ASSERTION INTERNAL ERROR: One of the subarrays "
                                              "should have compared unequal\n"
                                           << expr1 << " has value " << val1 << ",\n" << expr2 << " has value " << val2
                                           << ".";
    } else {
      return ::testing::AssertionFailure() << "The values of " << expr1 << " and " << expr2 << " do not match\n"
                                           << expr1 << " has value " << val1 << ",\n" << expr2 << " has value " << val2
                                           << ".";
    }
  }
}

inline ::testing::AssertionResult CompareDyNDArrayValues(const char *expr1, const char *expr2,
                                                         const dynd::nd::array &val1, const dynd::nd::array &val2) {
  using namespace dynd;
  ndt::type common_tp;
  try {
    common_tp = promote_types_arithmetic(val1.get_type(), val2.get_type());
  } catch (const type_error &) {
    return ::testing::AssertionFailure() << "The types of " << expr1 << " and " << expr2
                                         << " do not have mutually promotable types\n" << expr1 << " has type "
                                         << val1.get_type() << ",\n" << expr2 << " has type " << val2.get_type() << ".";
  }
  nd::array v1 = nd::empty(common_tp), v2 = nd::empty(common_tp);
  v1.vals() = val1;
  v2.vals() = val2;
  return CompareDyNDArrays(expr1, expr2, v1, v2);
}

inline ::testing::AssertionResult CompareDyNDArrayToJSON(const char *expr1, const char *expr2, const char *json,
                                                         const dynd::nd::array &b) {
  using namespace dynd;
  nd::array a(nd::empty(b.get_type()));
  parse_json(a, json);
  return CompareDyNDArrays(expr1, expr2, a, b);
}

inline ::testing::AssertionResult MatchNdtTypes(const char *expr1, const char *expr2, const dynd::ndt::type &pattern,
                                                const dynd::ndt::type &candidate) {
  if (pattern.match(candidate)) {
    return ::testing::AssertionSuccess();
  } else {
    return ::testing::AssertionFailure() << "The type of candidate " << expr2 << " does not match pattern " << expr1
                                         << "\n" << expr1 << " has value " << pattern << ",\n" << expr2 << " has value "
                                         << candidate << ".";
  }
}

inline ::testing::AssertionResult MatchNdtTypes(const char *expr1, const char *expr2, const char *pattern,
                                                const dynd::ndt::type &candidate) {
  return MatchNdtTypes(expr1, expr2, dynd::ndt::type(pattern), candidate);
}

inline ::testing::AssertionResult MatchNdtTypes(const char *expr1, const char *expr2, const dynd::ndt::type &pattern,
                                                const char *candidate) {
  return MatchNdtTypes(expr1, expr2, pattern, dynd::ndt::type(candidate));
}

inline ::testing::AssertionResult MatchNdtTypes(const char *expr1, const char *expr2, const char *pattern,
                                                const char *candidate) {
  return MatchNdtTypes(expr1, expr2, dynd::ndt::type(pattern), dynd::ndt::type(candidate));
}

inline ::testing::AssertionResult CompareNdtTypeToString(const char *DYND_UNUSED(expr1), const char *DYND_UNUSED(expr2),
                                                         const char *repr, const dynd::ndt::type &t) {
  std::stringstream ss;

  ss << t;
  if (repr == ss.str()) {
    return ::testing::AssertionSuccess();
  } else {
    return ::testing::AssertionFailure() << "expected repr: "
                                         << "\"" << repr << "\" "
                                         << "actual repr: "
                                         << "\"" << t << "\"";
  }
}

inline ::testing::AssertionResult ExpectAllTrue(const char *DYND_UNUSED(expr1), const dynd::nd::array actual) {
  if (actual.as<bool>()) {
    return ::testing::AssertionSuccess();
  }

  return ::testing::AssertionFailure();
}

inline ::testing::AssertionResult ExpectAllFalse(const char *DYND_UNUSED(expr1), const dynd::nd::array actual) {
  if (!actual.as<bool>()) {
    return ::testing::AssertionSuccess();
  }

  return ::testing::AssertionFailure();
}

/**
 * Macro to compare two arrays which should
 * be exactly equal
 *
 * nd::array a = {1, 2, 3};-
 * int bvals[3] = {1, 2, 3}
 * nd::array b = bvals;
 * EXPECT_ARRAY_EQ(b, a);
 */
#define EXPECT_ARRAY_EQ(expected, actual) EXPECT_PRED_FORMAT2(CompareDyNDArrays, expected, actual)

#define EXPECT_ARRAY_VALS_EQ(expected, actual) EXPECT_PRED_FORMAT2(CompareDyNDArrayValues, expected, actual)

#define EXPECT_ALL_TRUE(ACTUAL) EXPECT_PRED_FORMAT1(ExpectAllTrue, ACTUAL)
#define EXPECT_ALL_FALSE(ACTUAL) EXPECT_PRED_FORMAT1(ExpectAllFalse, ACTUAL)

/**
 * Macro to compare an array's values to those
 * provided in a JSON string, parsed using a
 * type matching that of the array.
 *
 * nd::array a = {1, 2, 3};
 * EXPECT_JSON_EQ_ARR("[1, 2, 3]", a);
 */
#define EXPECT_JSON_EQ_ARR(expected, actual) EXPECT_PRED_FORMAT2(CompareDyNDArrayToJSON, expected, actual)

/**
 * Macro to validate that a candidate type matches against
 * the provided pattern.
 *
 * EXPECT_TYPE_MATCH("Fixed * T", "10 * int32");
 */
#define EXPECT_TYPE_MATCH(pattern, candidate) EXPECT_PRED_FORMAT2(MatchNdtTypes, pattern, candidate)

/**
 * Macro to compare a type's string representation to the expected string.
 *
 * EXPECT_TYPE_REPR_EQ("int32", ndt::type("int32"));
 */
#define EXPECT_TYPE_REPR_EQ(expected, actual) EXPECT_PRED_FORMAT2(CompareNdtTypeToString, expected, actual)

inline float rel_error(float expected, float actual) {
  if ((expected == 0.0f) && (actual == 0.0f)) {
    return 0.0f;
  }

  return fabs(1.0f - actual / expected);
}

inline float rel_error(dynd::complex<float> expected, dynd::complex<float> actual) {
  if (expected == 0.0f) {
    if (actual == 0.0f) {
      return 0.0f;
    } else {
      return fabs(abs(expected - actual));
    }
  }

  return fabs(abs(expected - actual) / abs(expected));
}

inline double rel_error(double expected, double actual) {
  if ((expected == 0.0) && (actual == 0.0)) {
    return 0.0;
  }

  return fabs(1.0 - actual / expected);
}

inline double rel_error(dynd::complex<double> expected, dynd::complex<double> actual) {
  if (expected == 0.0) {
    if (actual == 0.0) {
      return 0.0;
    } else {
      return fabs(abs(expected - actual));
    }
  }

  return fabs(abs(expected - actual) / abs(expected));
}

template <typename T>
::testing::AssertionResult
AssertRelErrorLE(const char *DYND_UNUSED(expected_expr), const char *DYND_UNUSED(actual_expr),
                 const char *DYND_UNUSED(rel_error_max_expr), T expected, T actual, float rel_error_max) {
  float rel_error_val = rel_error(expected, actual);

  if (rel_error_val <= rel_error_max) {
    return ::testing::AssertionSuccess();
  }

  return ::testing::AssertionFailure() << "Expected: rel_error(" << expected << ", " << actual
                                       << ") <= " << rel_error_max << "\n"
                                       << "  Actual: " << rel_error_val << " vs " << rel_error_max;
}

template <typename T>
::testing::AssertionResult
AssertRelErrorLE(const char *DYND_UNUSED(expected_expr), const char *DYND_UNUSED(actual_expr),
                 const char *DYND_UNUSED(rel_error_max_expr), T expected, T actual, double rel_error_max) {
  double rel_error_val = rel_error(expected, actual);

  if (rel_error_val <= rel_error_max) {
    return ::testing::AssertionSuccess();
  }

  return ::testing::AssertionFailure() << "Expected: rel_error(" << expected << ", " << actual
                                       << ") <= " << rel_error_max << "\n"
                                       << "  Actual: " << rel_error_val << " vs " << rel_error_max;
}

inline ::testing::AssertionResult AssertRelErrorLE(const char *expected_expr, const char *actual_expr,
                                                   const char *rel_error_max_expr, float expected,
                                                   dynd::complex<float> actual, float rel_error_max) {
  return AssertRelErrorLE(expected_expr, actual_expr, rel_error_max_expr, dynd::complex<float>(expected), actual,
                          rel_error_max);
}

inline ::testing::AssertionResult AssertRelErrorLE(const char *expected_expr, const char *actual_expr,
                                                   const char *rel_error_max_expr, double expected,
                                                   dynd::complex<double> actual, double rel_error_max) {
  return AssertRelErrorLE(expected_expr, actual_expr, rel_error_max_expr, dynd::complex<double>(expected), actual,
                          rel_error_max);
}

#define EXPECT_EQ_RELERR(expected, actual, rel_error_max)                                                              \
  ASSERT_PRED_FORMAT3(AssertRelErrorLE, expected, actual, rel_error_max)

template <typename T>
struct test_class {
  double rel_error_max;
  bool &flag;

  test_class(double rel_error_max, bool &flag) : rel_error_max(rel_error_max), flag(flag) { this->flag = true; }

  void operator()(T a, T b) {
    if (rel_error(a, b) > rel_error_max) {
      flag = false;
    }
  }
};

inline ::testing::AssertionResult AssertArrayNear(const char *DYND_UNUSED(expected_expr),
                                                  const char *DYND_UNUSED(actual_expr),
                                                  const char *DYND_UNUSED(rel_error_max_expr),
                                                  const dynd::nd::array &expected, const dynd::nd::array &actual,
                                                  double rel_error_max) {
  //  const dynd::ndt::type &expected_tp = expected.get_type();
  const dynd::ndt::type &actual_tp = actual.get_dtype();

  bool flag;
  dynd::nd::callable f;
  switch (actual_tp.get_id()) {
  case dynd::float32_id:
    f = dynd::nd::functional::elwise(test_class<float>(rel_error_max, flag));
    break;
  case dynd::float64_id:
    f = dynd::nd::functional::elwise(test_class<double>(rel_error_max, flag));
    break;
  case dynd::complex_float64_id:
    f = dynd::nd::functional::elwise(test_class<dynd::complex<double>>(rel_error_max, flag));
    break;
  default:
    throw std::runtime_error("unsupported type for near comparision");
  }

  f(expected, actual);

  if (flag) {
    return ::testing::AssertionSuccess();
  }

  return ::testing::AssertionFailure() << "the values do not match";
}

#define EXPECT_ARRAY_NEAR(EXPECTED, ACTUAL) ASSERT_PRED_FORMAT3(AssertArrayNear, EXPECTED, ACTUAL, 0.01)
