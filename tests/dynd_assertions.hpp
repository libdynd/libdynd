//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND_ASSERTIONS_HPP
#define DYND_ASSERTIONS_HPP

#include "inc_gtest.hpp"

#include <dynd/json_parser.hpp>
#include <dynd/types/base_struct_type.hpp>

inline std::string ShapeFormatter(const std::vector<intptr_t>& shape)
{
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

inline ::testing::AssertionResult CompareDyNDArrays(const char *expr1,
                                                    const char *expr2,
                                                    const dynd::nd::array &val1,
                                                    const dynd::nd::array &val2)
{
    using namespace dynd;
    if (val1.equals_exact(val2)) {
        return ::testing::AssertionSuccess();
    } else {
        if (val1.get_type() != val2.get_type()) {
            return ::testing::AssertionFailure()
                   << "The types of " << expr1 << " and " << expr2
                   << " do not match\n" << expr1 << " has type "
                   << val1.get_type() << ",\n" << expr2 << " has type "
                   << val2.get_type() << ".";
        } else if (val1.get_shape() != val2.get_shape()) {
            return ::testing::AssertionFailure()
                   << "The shapes of " << expr1 << " and " << expr2
                   << " do not match\n" << expr1 << " has shape "
                   << ShapeFormatter(val1.get_shape()) << ",\n" << expr2
                   << " has shape " << ShapeFormatter(val2.get_shape()) << ".";
        } else if (val1.get_type().get_kind() == struct_kind) {
            const base_struct_type *bsd = val1.get_type().tcast<base_struct_type>();
            intptr_t field_count = bsd->get_field_count();
            for (intptr_t i = 0; i < field_count; ++i) {
                nd::array field1 = val1(i), field2 = val2(i);
                if (!field1.equals_exact(field2)) {
                    return ::testing::AssertionFailure()
                           << "The values of " << expr1 << " and " << expr2
                           << " do not match at field index " << i
                           << ", name \"" << bsd->get_field_name(i) << "\"\n"
                           << expr1 << " has field value " << field1 << ",\n"
                           << expr2 << " has field value " << field2 << ".";
                }
            }
            return ::testing::AssertionFailure()
                   << "DYND ASSERTION INTERNAL ERROR: One of the struct fields "
                      "should have compared unequal";
        } else if (val1.get_type().get_kind() == tuple_kind) {
            const base_tuple_type *bsd = val1.get_type().tcast<base_tuple_type>();
            intptr_t field_count = bsd->get_field_count();
            for (intptr_t i = 0; i < field_count; ++i) {
                nd::array field1 = val1(i), field2 = val2(i);
                if (!field1.equals_exact(field2)) {
                    return ::testing::AssertionFailure()
                           << "The values of " << expr1 << " and " << expr2
                           << " do not match at field index " << i << "\"\n"
                           << expr1 << " has field value " << field1 << ",\n"
                           << expr2 << " has field value " << field2 << ".";
                }
            }
            return ::testing::AssertionFailure()
                   << "DYND ASSERTION INTERNAL ERROR: One of the tuple fields "
                      "should have compared unequal";
        } else if (val1.get_ndim() > 0) {
            intptr_t dim_size = val1.get_dim_size();
            for (intptr_t i = 0; i < dim_size; ++i) {
                nd::array sub1 = val1(i), sub2 = val2(i);
                if (!sub1.equals_exact(sub2)) {
                    return ::testing::AssertionFailure()
                           << "The values of " << expr1 << " and " << expr2
                           << " do not match at index " << i << "\"\n"
                           << expr1 << " has subarray value " << sub1 << ",\n"
                           << expr2 << " has subarray value " << sub2 << ".";
                }
            }
            return ::testing::AssertionFailure()
                   << "DYND ASSERTION INTERNAL ERROR: One of the subarrays "
                      "should have compared unequal";
        } else {
            return ::testing::AssertionFailure()
                   << "The values of " << expr1 << " and " << expr2
                   << " do not match\n" << expr1 << " has value " << val1
                   << ",\n" << expr2 << " has value " << val2 << ".";
        }
    }
}

inline ::testing::AssertionResult
CompareDyNDArrayToJSON(const char *expr1, const char *expr2,
                       const char *json, const dynd::nd::array &b)
{
    using namespace dynd;
    dimvector shape(b.get_ndim());
    b.get_shape(shape.get());
    nd::array a(
        make_array_memory_block(b.get_type(), b.get_ndim(), shape.get()));
    parse_json(a, json);
    return CompareDyNDArrays(expr1, expr2, a, b);
}

/**
 * Macro to compare two arrays which should
 * be exactly equal
 *
 * nd::array a = {1, 2, 3};-
 * int bvals[3] = {1, 2, 3}
 * nd::array b = bvals;
 * EXPECT_ARR_EQ(b, a);
 */
#define EXPECT_ARR_EQ(expected, actual) \
  ASSERT_PRED_FORMAT2(CompareDyNDArrays, \
                      expected, actual)

/**
 * Macro to compare an array's values to those
 * provided in a JSON string, parsed using a
 * type matching that of the array.
 *
 * nd::array a = {1, 2, 3};
 * EXPECT_JSON_EQ_ARR("[1, 2, 3]", a);
 */
#define EXPECT_JSON_EQ_ARR(expected, actual) \
  ASSERT_PRED_FORMAT2(CompareDyNDArrayToJSON, \
                      expected, actual)

#endif // DYND_ASSERTIONS_HPP
