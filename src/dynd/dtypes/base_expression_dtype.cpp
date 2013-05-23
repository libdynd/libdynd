//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dynd/dtype.hpp>
#include <dynd/kernels/expression_assignment_kernels.hpp>
#include <dynd/kernels/expression_comparison_kernels.hpp>

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

void base_expression_dtype::metadata_default_construct(char *metadata, size_t ndim, const intptr_t* shape) const
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

size_t base_expression_dtype::get_iterdata_size(size_t DYND_UNUSED(ndim)) const
{
    return 0;
}

size_t base_expression_dtype::make_operand_to_value_assignment_kernel(
                hierarchical_kernel *DYND_UNUSED(out), size_t DYND_UNUSED(offset_out),
                const char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata),
                kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context *DYND_UNUSED(ectx)) const
{
    stringstream ss;
    ss << "dynd dtype " << dtype(this, true) << " does not support reading of its values";
    throw runtime_error(ss.str());
}

size_t base_expression_dtype::make_value_to_operand_assignment_kernel(
                hierarchical_kernel *DYND_UNUSED(out), size_t DYND_UNUSED(offset_out),
                const char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata),
                kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context *DYND_UNUSED(ectx)) const
{
    stringstream ss;
    ss << "dynd dtype " << dtype(this, true) << " does not support writing to its values";
    throw runtime_error(ss.str());
}

size_t base_expression_dtype::make_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    return make_expression_assignment_kernel(out, offset_out,
                    dst_dt, dst_metadata, src_dt, src_metadata,
                    kernreq, errmode, ectx);
}

size_t base_expression_dtype::make_comparison_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const dtype& src0_dt, const char *src0_metadata,
                const dtype& src1_dt, const char *src1_metadata,
                comparison_type_t comptype,
                const eval::eval_context *ectx) const
{
    return make_expression_comparison_kernel(out, offset_out,
                    src0_dt, src0_metadata,
                    src1_dt, src1_metadata,
                    comptype, ectx);
}
