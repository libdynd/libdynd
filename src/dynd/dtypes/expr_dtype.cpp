//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/expr_dtype.hpp>

using namespace std;
using namespace dynd;

expr_dtype::expr_dtype(const dtype& value_dtype, const dtype& operand_dtype)
    : base_expression_dtype(expr_type_id, expression_kind,
                        operand_dtype.get_data_size(), operand_dtype.get_alignment(),
                        dtype_flag_none,
                        0, value_dtype.get_undim()),
                    m_value_dtype(value_dtype), m_operand_dtype(operand_dtype)
{
}