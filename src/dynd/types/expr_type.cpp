//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/expr_type.hpp>
#include <dynd/shortvector.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/builtin_type_properties.hpp>
#include <dynd/shape_tools.hpp>

using namespace std;
using namespace dynd;

expr_type::expr_type(const ndt::type& value_type, const ndt::type& operand_type,
                const expr_kernel_generator *kgen)
    : base_expression_type(expr_type_id, expression_kind,
                        operand_type.get_data_size(), operand_type.get_data_alignment(),
                        inherited_flags(value_type.get_flags(), operand_type.get_flags()),
                        operand_type.get_metadata_size(), value_type.get_undim()),
                    m_value_type(value_type), m_operand_type(operand_type),
                    m_kgen(kgen)
{
    if (operand_type.get_type_id() != cstruct_type_id) {
        stringstream ss;
        ss << "expr_type can only be constructed with a cstruct as its operand, given ";
        ss << operand_type;
        throw runtime_error(ss.str());
    }
    const cstruct_type *fsd = static_cast<const cstruct_type *>(operand_type.extended());
    size_t field_count = fsd->get_field_count();
    if (field_count == 1) {
        throw runtime_error("expr_type is for 2 or more operands, use unary_expr_type for 1 operand");
    }
    const ndt::type *field_types = fsd->get_field_types();
    for (size_t i = 0; i != field_count; ++i) {
        if (field_types[i].get_type_id() != pointer_type_id) {
            stringstream ss;
            ss << "each field of the expr_type's operand must be a pointer, field " << i;
            ss << " is " << field_types[i];
            throw runtime_error(ss.str());
        }
    }
}

expr_type::~expr_type()
{
    expr_kernel_generator_decref(m_kgen);
}

void expr_type::print_data(std::ostream& DYND_UNUSED(o),
                const char *DYND_UNUSED(metadata), const char *DYND_UNUSED(data)) const
{
    throw runtime_error("internal error: expr_type::print_data isn't supposed to be called");
}

void expr_type::print_type(std::ostream& o) const
{
    const cstruct_type *fsd = static_cast<const cstruct_type *>(m_operand_type.extended());
    size_t field_count = fsd->get_field_count();
    const ndt::type *field_types = fsd->get_field_types();
    o << "expr<";
    o << m_value_type;
    for (size_t i = 0; i != field_count; ++i) {
        const pointer_type *pd = static_cast<const pointer_type *>(field_types[i].extended());
        o << ", op" << i << "=" << pd->get_target_dtype();
    }
    o << ", expr=";
    m_kgen->print_type(o);
    o << ">";
}

ndt::type expr_type::apply_linear_index(size_t nindices, const irange *indices,
            size_t current_i, const ndt::type& root_dt, bool DYND_UNUSED(leading_dimension)) const
{
    if (m_kgen->is_elwise()) {
        size_t undim = get_undim();
        const cstruct_type *fsd = static_cast<const cstruct_type *>(m_operand_type.extended());
        size_t field_count = fsd->get_field_count();
        const ndt::type *field_types = fsd->get_field_types();

        ndt::type result_value_dt = m_value_type.apply_linear_index(nindices, indices,
                        current_i, root_dt, true);
        vector<ndt::type> result_src_dt(field_count);
        // Apply the portion of the indexing to each of the src operand types
        for (size_t i = 0; i != field_count; ++i) {
            const ndt::type& dt = field_types[i];
            size_t field_undim = dt.get_undim();
            if (nindices + field_undim <= undim) {
                result_src_dt[i] = dt;
            } else {
                size_t index_offset = undim - field_undim;
                result_src_dt[i] = dt.apply_linear_index(
                                                nindices - index_offset, indices + index_offset,
                                                current_i, root_dt, false);
            }
        }
        ndt::type result_operand_type = ndt::make_cstruct(field_count, &result_src_dt[0],
                        fsd->get_field_names());
        expr_kernel_generator_incref(m_kgen);
        return ndt::make_expr(result_value_dt, result_operand_type, m_kgen);
    } else {
        throw runtime_error("expr_type::apply_linear_index is only implemented for elwise kernel generators");
    }
}

intptr_t expr_type::apply_linear_index(size_t nindices, const irange *indices, const char *metadata,
                const ndt::type& result_dtype, char *out_metadata,
                memory_block_data *embedded_reference,
                size_t current_i, const ndt::type& root_dt,
                bool DYND_UNUSED(leading_dimension), char **DYND_UNUSED(inout_data),
                memory_block_data **DYND_UNUSED(inout_dataref)) const
{
    if (m_kgen->is_elwise()) {
        size_t undim = get_undim();
        const expr_type *out_ed = static_cast<const expr_type *>(result_dtype.extended());
        const cstruct_type *fsd = static_cast<const cstruct_type *>(m_operand_type.extended());
        const cstruct_type *out_fsd = static_cast<const cstruct_type *>(out_ed->m_operand_type.extended());
        const size_t *metadata_offsets = fsd->get_metadata_offsets();
        const size_t *out_metadata_offsets = out_fsd->get_metadata_offsets();
        size_t field_count = fsd->get_field_count();
        const ndt::type *field_types = fsd->get_field_types();
        const ndt::type *out_field_types = out_fsd->get_field_types();
        // Apply the portion of the indexing to each of the src operand types
        for (size_t i = 0; i != field_count; ++i) {
            const pointer_type *pd = static_cast<const pointer_type *>(field_types[i].extended());
            size_t field_undim = pd->get_undim();
            if (nindices + field_undim <= undim) {
                pd->metadata_copy_construct(out_metadata + out_metadata_offsets[i],
                                metadata + metadata_offsets[i],
                                embedded_reference);
            } else {
                size_t index_offset = undim - field_undim;
                intptr_t offset = pd->apply_linear_index(nindices - index_offset, indices + index_offset,
                                metadata + metadata_offsets[i],
                                out_field_types[i], out_metadata + out_metadata_offsets[i],
                                embedded_reference, current_i, root_dt, false, NULL, NULL);
                if (offset != 0) {
                    throw runtime_error("internal error: expr_type::apply_linear_index"
                                    " expected 0 offset from pointer_type::apply_linear_index");
                }
            }
        }
        return 0;
    } else {
        throw runtime_error("expr_type::apply_linear_index is only implemented for elwise kernel generators");
    }
}

void expr_type::get_shape(size_t ndim, size_t i, intptr_t *out_shape, const char *metadata) const
{
    size_t undim = get_undim();
    // Initialize the shape to all ones
    dimvector bcast_shape(undim);
    for (size_t j = 0; j != undim; ++j) {
        bcast_shape[j] = 1;
    }

    // Get each operand shape, and broadcast them together
    dimvector shape(undim);
    const cstruct_type *fsd = static_cast<const cstruct_type *>(m_operand_type.extended());
    const size_t *metadata_offsets = fsd->get_metadata_offsets();
    size_t field_count = fsd->get_field_count();
    for (size_t fi = 0; fi != field_count; ++fi) {
        const ndt::type& dt = fsd->get_field_types()[fi];
        size_t field_undim = dt.get_undim();
        if (field_undim > 0) {
            dt.extended()->get_shape(field_undim, 0, shape.get(),
                            metadata ? (metadata + metadata_offsets[fi]) : NULL);
            incremental_broadcast(undim, bcast_shape.get(), field_undim, shape.get());
        }
    }

    // Copy this shape to the output
    memcpy(out_shape + i, bcast_shape.get(), min(undim, ndim - i) * sizeof(intptr_t));

    // If more shape is requested, get it from the value dtype
    if (ndim - i > undim) {
        const ndt::type& dt = m_value_type.get_udtype();
        if (!dt.is_builtin()) {
            dt.extended()->get_shape(ndim, i + undim, out_shape, NULL);
        } else {
            stringstream ss;
            ss << "requested too many dimensions from type " << ndt::type(this, true);
            throw runtime_error(ss.str());
        }
    }
}

bool expr_type::is_lossless_assignment(
                const ndt::type& DYND_UNUSED(dst_dt),
                const ndt::type& DYND_UNUSED(src_dt)) const
{
    return false;
}

bool expr_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != expr_type_id) {
        return false;
    } else {
        const expr_type *dt = static_cast<const expr_type*>(&rhs);
        return m_value_type == dt->m_value_type &&
                        m_operand_type == dt->m_operand_type &&
                        m_kgen == dt->m_kgen;
    }
}

namespace {
    template<int N>
    struct expr_type_offset_applier_extra {
        typedef expr_type_offset_applier_extra<N> extra_type;

        kernel_data_prefix base;
        size_t offsets[N];

        // Only the single kernel is needed for this one
        static void single(char *dst, const char * const *src,
                        kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const size_t *offsets = e->offsets;
            const char *src_modified[N];
            for (int i = 0; i < N; ++i) {
                src_modified[i] = src[i] + offsets[i];
            }
            kernel_data_prefix *echild = &(e + 1)->base;
            expr_single_operation_t opchild = echild->get_function<expr_single_operation_t>();
            opchild(dst, src_modified, echild);
        }

        static void destruct(kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            kernel_data_prefix *echild = &(e + 1)->base;
            if (echild->destructor) {
                echild->destructor(echild);
            }
        }
    };

    struct expr_type_offset_applier_general_extra {
        typedef expr_type_offset_applier_general_extra extra_type;

        kernel_data_prefix base;
        size_t src_count;
        // After this are src_count size_t offsets

       // Only the single kernel is needed for this one
        static void single(char *dst, const char * const *src,
                        kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            size_t src_count = e->src_count;
            const size_t *offsets = reinterpret_cast<const size_t *>(e + 1);
            shortvector<const char *> src_modified(src_count);
            for (size_t i = 0; i != src_count; ++i) {
                src_modified[i] = src[i] + offsets[i];
            }
            kernel_data_prefix *echild = reinterpret_cast<kernel_data_prefix *>(
                            reinterpret_cast<char *>(extra) + sizeof(extra_type) +
                            src_count * sizeof(size_t));
            expr_single_operation_t opchild = echild->get_function<expr_single_operation_t>();
            opchild(dst, src_modified.get(), echild);
        }

        static void destruct(kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            size_t src_count = e->src_count;
            kernel_data_prefix *echild = reinterpret_cast<kernel_data_prefix *>(
                            reinterpret_cast<char *>(extra) + sizeof(extra_type) +
                            src_count * sizeof(size_t));
            if (echild->destructor) {
                echild->destructor(echild);
            }
        }
    };
} // anonymous namespace

static size_t make_expr_type_offset_applier(
                hierarchical_kernel *out, size_t offset_out,
                size_t src_count, const intptr_t *src_data_offsets)
{
    // A few specializations with fixed size, and a general case version
    // NOTE: src_count == 1 must never happen here, it is handled by the unary_expr dtype
    switch (src_count) {
        case 2: {
            out->ensure_capacity(offset_out + sizeof(expr_type_offset_applier_extra<2>));
            expr_type_offset_applier_extra<2> *e = out->get_at<expr_type_offset_applier_extra<2> >(offset_out);
            memcpy(e->offsets, src_data_offsets, sizeof(e->offsets));
            e->base.set_function<expr_single_operation_t>(&expr_type_offset_applier_extra<2>::single);
            e->base.destructor = &expr_type_offset_applier_extra<2>::destruct;
            return offset_out + sizeof(expr_type_offset_applier_extra<2>);
        }
        case 3: {
            out->ensure_capacity(offset_out + sizeof(expr_type_offset_applier_extra<3>));
            expr_type_offset_applier_extra<3> *e = out->get_at<expr_type_offset_applier_extra<3> >(offset_out);
            memcpy(e->offsets, src_data_offsets, sizeof(e->offsets));
            e->base.set_function<expr_single_operation_t>(&expr_type_offset_applier_extra<3>::single);
            e->base.destructor = &expr_type_offset_applier_extra<3>::destruct;
            return offset_out + sizeof(expr_type_offset_applier_extra<3>);
        }
        case 4: {
            out->ensure_capacity(offset_out + sizeof(expr_type_offset_applier_extra<4>));
            expr_type_offset_applier_extra<4> *e = out->get_at<expr_type_offset_applier_extra<4> >(offset_out);
            memcpy(e->offsets, src_data_offsets, sizeof(e->offsets));
            e->base.set_function<expr_single_operation_t>(&expr_type_offset_applier_extra<4>::single);
            e->base.destructor = &expr_type_offset_applier_extra<4>::destruct;
            return offset_out + sizeof(expr_type_offset_applier_extra<4>);
        }
        default: {
            size_t extra_size = sizeof(expr_type_offset_applier_general_extra) +
                            src_count * sizeof(size_t);
            out->ensure_capacity(offset_out + extra_size);
            expr_type_offset_applier_general_extra *e = out->get_at<expr_type_offset_applier_general_extra>(offset_out);
            e->src_count = src_count;
            size_t *out_offsets = reinterpret_cast<size_t *>(e + 1);
            memcpy(out_offsets, src_data_offsets, src_count * sizeof(size_t));
            e->base.set_function<expr_single_operation_t>(&expr_type_offset_applier_general_extra::single);
            e->base.destructor = &expr_type_offset_applier_general_extra::destruct;
            return offset_out + extra_size;
        }
    }
    
}

size_t expr_type::make_operand_to_value_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const char *dst_metadata, const char *src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    const cstruct_type *fsd = static_cast<const cstruct_type *>(m_operand_type.extended());

    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
    size_t input_count = fsd->get_field_count();
    const size_t *metadata_offsets = fsd->get_metadata_offsets();
    shortvector<const char *> src_metadata_array(input_count);
    dimvector src_data_offsets(input_count);
    bool nonzero_offsets = false;

    const ndt::type *src_ptr_dt = fsd->get_field_types();
    vector<ndt::type> src_dt(input_count);
    for (size_t i = 0; i != input_count; ++i) {
        const pointer_type *pd = static_cast<const pointer_type *>(src_ptr_dt[i].extended());
        src_dt[i] = pd->get_target_dtype();
    }
    for (size_t i = 0; i != input_count; ++i) {
        const char *ptr_metadata = src_metadata + metadata_offsets[i];
        intptr_t offset = reinterpret_cast<const pointer_type_metadata *>(ptr_metadata)->offset;
        if (offset != 0) {
            nonzero_offsets = true;
        }
        src_data_offsets[i] = offset;
        src_metadata_array[i] = ptr_metadata + sizeof(pointer_type_metadata);
    }
    // If there were any non-zero pointer offsets, we need to add a kernel
    // adapter which applies those offsets.
    if (nonzero_offsets) {
        offset_out = make_expr_type_offset_applier(out, offset_out,
                        input_count, src_data_offsets.get());
    }
    return m_kgen->make_expr_kernel(out, offset_out,
                    m_value_type, dst_metadata,
                    input_count, &src_dt[0],
                    src_metadata_array.get(),
                    kernel_request_single, ectx);
}

size_t expr_type::make_value_to_operand_assignment_kernel(
                hierarchical_kernel *DYND_UNUSED(out), size_t DYND_UNUSED(offset_out),
                const char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata),
                kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context *DYND_UNUSED(ectx)) const
{
    throw runtime_error("Cannot assign to a dynd expr object value");
}

ndt::type expr_type::with_replaced_storage_type(const ndt::type& DYND_UNUSED(replacement_type)) const
{
    throw runtime_error("TODO: implement expr_type::with_replaced_storage_type");
}

void expr_type::get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    const ndt::type& udt = m_value_type.get_udtype();
    if (!udt.is_builtin()) {
        udt.extended()->get_dynamic_array_properties(out_properties, out_count);
    } else {
        get_builtin_type_dynamic_array_properties(udt.get_type_id(), out_properties, out_count);
    }
}

void expr_type::get_dynamic_array_functions(const std::pair<std::string, gfunc::callable> **out_functions,
                size_t *out_count) const
{
    const ndt::type& udt = m_value_type.get_udtype();
    if (!udt.is_builtin()) {
        udt.extended()->get_dynamic_array_functions(out_functions, out_count);
    } else {
        //get_builtin_type_dynamic_ndobject_functions(udt.get_type_id(), out_functions, out_count);
    }
}
