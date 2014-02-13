//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dynd/type.hpp>
#include <dynd/kernels/expression_assignment_kernels.hpp>
#include <dynd/kernels/expression_comparison_kernels.hpp>

using namespace std;
using namespace dynd;

base_expression_type::~base_expression_type()
{
}

bool base_expression_type::is_expression() const
{
    return true;
}

ndt::type base_expression_type::get_canonical_type() const
{
    return get_value_type();
}

void base_expression_type::metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const
{
    const ndt::type& dt = get_operand_type();
    if (!dt.is_builtin()) {
        dt.extended()->metadata_default_construct(metadata, ndim, shape);
    }
}

void base_expression_type::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    const ndt::type& dt = get_operand_type();
    if (!dt.is_builtin()) {
        dt.extended()->metadata_copy_construct(dst_metadata, src_metadata, embedded_reference);
    }
}

void base_expression_type::metadata_destruct(char *metadata) const
{
    const ndt::type& dt = get_operand_type();
    if (!dt.is_builtin()) {
        dt.extended()->metadata_destruct(metadata);
    }
}

void base_expression_type::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    const ndt::type& dt = get_operand_type();
    if (!dt.is_builtin()) {
        dt.extended()->metadata_debug_print(metadata, o, indent);
    }
}

size_t base_expression_type::get_iterdata_size(intptr_t DYND_UNUSED(ndim)) const
{
    return 0;
}

size_t base_expression_type::make_operand_to_value_assignment_kernel(
                ckernel_builder *DYND_UNUSED(out), size_t DYND_UNUSED(offset_out),
                const char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata),
                kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context *DYND_UNUSED(ectx)) const
{
    stringstream ss;
    ss << "dynd type " << ndt::type(this, true) << " does not support reading of its values";
    throw dynd::type_error(ss.str());
}

size_t base_expression_type::make_value_to_operand_assignment_kernel(
                ckernel_builder *DYND_UNUSED(out), size_t DYND_UNUSED(offset_out),
                const char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata),
                kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context *DYND_UNUSED(ectx)) const
{
    stringstream ss;
    ss << "dynd type " << ndt::type(this, true) << " does not support writing to its values";
    throw dynd::type_error(ss.str());
}

size_t base_expression_type::make_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& dst_tp, const char *dst_metadata,
                const ndt::type& src_tp, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    return make_expression_assignment_kernel(out, offset_out,
                    dst_tp, dst_metadata, src_tp, src_metadata,
                    kernreq, errmode, ectx);
}

size_t base_expression_type::make_comparison_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& src0_dt, const char *src0_metadata,
                const ndt::type& src1_dt, const char *src1_metadata,
                comparison_type_t comptype,
                const eval::eval_context *ectx) const
{
    return make_expression_comparison_kernel(out, offset_out,
                    src0_dt, src0_metadata,
                    src1_dt, src1_metadata,
                    comptype, ectx);
}
