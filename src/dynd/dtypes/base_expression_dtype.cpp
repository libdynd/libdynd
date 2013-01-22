//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtype.hpp>

using namespace std;
using namespace dynd;

base_expression_dtype::~base_expression_dtype()
{
}

bool base_expression_dtype::is_expression() const
{
    return true;
}

dtype base_expression_dtype::get_canonical_dtype() const
{
    return get_value_dtype();
}

size_t base_expression_dtype::get_metadata_size() const
{
    const dtype& dt = get_operand_dtype();
    if (!dt.is_builtin()) {
        return dt.extended()->get_metadata_size();
    } else {
        return 0;
    }
}

void base_expression_dtype::metadata_default_construct(char *metadata, int ndim, const intptr_t* shape) const
{
    const dtype& dt = get_operand_dtype();
    if (!dt.is_builtin()) {
        dt.extended()->metadata_default_construct(metadata, ndim, shape);
    }
}

void base_expression_dtype::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    const dtype& dt = get_operand_dtype();
    if (!dt.is_builtin()) {
        dt.extended()->metadata_copy_construct(dst_metadata, src_metadata, embedded_reference);
    }
}

void base_expression_dtype::metadata_destruct(char *metadata) const
{
    const dtype& dt = get_operand_dtype();
    if (!dt.is_builtin()) {
        dt.extended()->metadata_destruct(metadata);
    }
}

void base_expression_dtype::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    const dtype& dt = get_operand_dtype();
    if (!dt.is_builtin()) {
        dt.extended()->metadata_debug_print(metadata, o, indent);
    }
}

size_t base_expression_dtype::get_iterdata_size(int DYND_UNUSED(ndim)) const
{
    return 0;
}

