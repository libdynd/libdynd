//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <vector>

#include <dynd/dtypes/groupby_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <dynd/dtypes/pointer_dtype.hpp>
#include <dynd/dtypes/fixed_dim_dtype.hpp>
#include <dynd/dtypes/var_dim_dtype.hpp>
#include <dynd/dtypes/categorical_dtype.hpp>
#include <dynd/ndobject_iter.hpp>
#include <dynd/gfunc/make_callable.hpp>

using namespace std;
using namespace dynd;

groupby_dtype::groupby_dtype(const dtype& data_values_dtype,
                const dtype& by_values_dtype)
    : base_expression_dtype(groupby_type_id, expression_kind,
                    sizeof(groupby_dtype_data), sizeof(void *), dtype_flag_none,
                    0, 1 + data_values_dtype.get_undim())
{
    m_groups_dtype = by_values_dtype.at_single(0).value_dtype();
    if (m_groups_dtype.get_type_id() != categorical_type_id) {
        stringstream ss;
        ss << "to construct a groupby dtype, the by dtype, " << by_values_dtype.at_single(0);
        ss << ", must have a categorical value type";
        throw runtime_error(ss.str());
    }
    if (data_values_dtype.get_undim() < 1) {
        throw runtime_error("to construct a groupby dtype, its values dtype must have at least one uniform dimension");
    }
    if (by_values_dtype.get_undim() < 1) {
        throw runtime_error("to construct a groupby dtype, its values dtype must have at least one uniform dimension");
    }
    m_operand_dtype = make_fixedstruct_dtype(make_pointer_dtype(data_values_dtype), "data",
                    make_pointer_dtype(by_values_dtype), "by");
    m_members.metadata_size = m_operand_dtype.get_metadata_size();
    const categorical_dtype *cd = static_cast<const categorical_dtype *>(m_groups_dtype.extended());
    m_value_dtype = make_fixed_dim_dtype(cd->get_category_count(),
                    make_var_dim_dtype(data_values_dtype.at_single(0)));
}

groupby_dtype::~groupby_dtype()
{
}

void groupby_dtype::print_data(std::ostream& DYND_UNUSED(o),
                const char *DYND_UNUSED(metadata), const char *DYND_UNUSED(data)) const
{
    throw runtime_error("internal error: groupby_dtype::print_data isn't supposed to be called");
}

dtype groupby_dtype::get_data_values_dtype() const
{
    const pointer_dtype *pd = static_cast<const pointer_dtype *>(m_operand_dtype.at_single(0).extended());
    return pd->get_target_dtype();
}

dtype groupby_dtype::get_by_values_dtype() const
{
    const pointer_dtype *pd = static_cast<const pointer_dtype *>(m_operand_dtype.at_single(1).extended());
    return pd->get_target_dtype();
}

void groupby_dtype::print_dtype(std::ostream& o) const
{
    o << "groupby<values=" << get_data_values_dtype();
    o << ", by=" << get_by_values_dtype() << ">";
}

void groupby_dtype::get_shape(size_t i, intptr_t *out_shape) const
{
    if (!m_value_dtype.is_builtin()) {
        m_value_dtype.extended()->get_shape(i + 2, out_shape);
    }
}

void groupby_dtype::get_shape(size_t i, intptr_t *out_shape, const char *metadata) const
{
    // The first dimension is the groups, the second variable-sized
    out_shape[i] = reinterpret_cast<const categorical_dtype *>(m_groups_dtype.extended())->get_category_count();
    out_shape[i+1] = -1;
    // Get the rest of the shape if necessary
    if (get_undim() > 2) {
        // Get the dtype for a single data_value element, and its corresponding metadata
        dtype data_values_dtype = m_operand_dtype.at_single(0, &metadata);
        data_values_dtype.at_single(0, &metadata);
        // Use this to get the rest of the shape
        data_values_dtype.extended()->get_shape(i + 2, out_shape, metadata);
    }
}

bool groupby_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    // Treat this dtype as the value dtype for whether assignment is always lossless
    if (src_dt.extended() == this) {
        return ::dynd::is_lossless_assignment(dst_dt, m_value_dtype);
    } else {
        return ::dynd::is_lossless_assignment(m_value_dtype, src_dt);
    }
}

bool groupby_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != groupby_type_id) {
        return false;
    } else {
        const groupby_dtype *dt = static_cast<const groupby_dtype*>(&rhs);
        return m_value_dtype == dt->m_value_dtype && m_operand_dtype == dt->m_operand_dtype;
    }
}

namespace {
    // Assign from a categorical dtype to some other dtype
    struct groupby_to_value_assign_extra {
        typedef groupby_to_value_assign_extra extra_type;

        kernel_data_prefix base;
        // The groupby dtype
        const groupby_dtype *src_groupby_dt;
        const char *src_metadata, *dst_metadata;

        template<typename UIntType>
        inline static void single(char *dst, const char *src, kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const groupby_dtype *gd = e->src_groupby_dt;

            // Get the data_values raw ndobject
            dtype data_values_dt = gd->get_operand_dtype();
            const char *data_values_metadata = e->src_metadata, *data_values_data = src;
            data_values_dt = data_values_dt.extended()->at_single(0, &data_values_metadata, &data_values_data);
            data_values_dt = static_cast<const pointer_dtype *>(data_values_dt.extended())->get_target_dtype();
            data_values_metadata += sizeof(pointer_dtype_metadata);
            data_values_data = *reinterpret_cast<const char * const *>(data_values_data);

            // Get the by_values raw ndobject
            dtype by_values_dt = gd->get_operand_dtype();
            const char *by_values_metadata = e->src_metadata, *by_values_data = src;
            by_values_dt = by_values_dt.extended()->at_single(1, &by_values_metadata, &by_values_data);
            by_values_dt = static_cast<const pointer_dtype *>(by_values_dt.extended())->get_target_dtype();
            by_values_metadata += sizeof(pointer_dtype_metadata);
            by_values_data = *reinterpret_cast<const char * const *>(by_values_data);

            // If by_values is an expression, evaluate it since we're doing two passes through them
            ndobject by_values_tmp;
            if (by_values_dt.is_expression() || !by_values_dt.extended()->is_strided()) {
                by_values_tmp = eval_raw_copy(by_values_dt, by_values_metadata, by_values_data);
                by_values_dt = by_values_tmp.get_dtype();
                by_values_metadata = by_values_tmp.get_ndo_meta();
                by_values_data = by_values_tmp.get_readonly_originptr();
            }

            // Get a strided representation of by_values for processing
            const char *by_values_origin = NULL;
            intptr_t by_values_stride, by_values_size;
            by_values_dt.extended()->process_strided(by_values_metadata, by_values_data,
                            by_values_dt, by_values_origin, by_values_stride, by_values_size);

            const dtype& result_dt = gd->get_value_dtype();
            const fixed_dim_dtype *fad = static_cast<const fixed_dim_dtype *>(result_dt.extended());
            intptr_t fad_stride = fad->get_fixed_stride();
            const var_dim_dtype *vad = static_cast<const var_dim_dtype *>(fad->get_element_dtype().extended());
            const var_dim_dtype_metadata *vad_md = reinterpret_cast<const var_dim_dtype_metadata *>(e->dst_metadata);
            if (vad_md->offset != 0) {
                throw runtime_error("dynd groupby: destination var_dim offset must be zero to allocate output");
            }
            intptr_t vad_stride = vad_md->stride;

            // Do a pass through by_values to get the size of each variable-sized dimension
            vector<size_t> cat_sizes(fad->get_fixed_dim_size());
            const char *by_values_ptr = by_values_origin;
            for (intptr_t i = 0; i < by_values_size; ++i, by_values_ptr += by_values_stride) {
                UIntType value = *reinterpret_cast<const UIntType *>(by_values_ptr);
                if (value >= cat_sizes.size()) {
                    stringstream ss;
                    ss << "dynd groupby: 'by' array contains an out of bounds value " << (uint32_t)value;
                    ss << ", range is [0, " << cat_sizes.size() << ")";
                    throw runtime_error(ss.str());
                }
                ++cat_sizes[value];
            }

            // Allocate the output, and create a vector of pointers to the start
            // of each group's output
            memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(vad_md->blockref);
            char *out_begin = NULL, *out_end = NULL;
            allocator->allocate(vad_md->blockref, by_values_size * vad_stride,
                            vad->get_element_dtype().get_alignment(), &out_begin, &out_end);
            vector<char *> cat_pointers(cat_sizes.size());
            for (size_t i = 0, i_end = cat_pointers.size(); i != i_end; ++i) {
                cat_pointers[i] = out_begin;
                reinterpret_cast<var_dim_dtype_data *>(dst + i * fad_stride)->begin = out_begin;
                size_t csize = cat_sizes[i];
                reinterpret_cast<var_dim_dtype_data *>(dst + i * fad_stride)->size = csize;
                out_begin += csize * vad_stride;
            }

            // Loop through both by_values and data_values,
            // copying the data to the right place in the output
            kernel_data_prefix *echild = &(e + 1)->base;
            unary_single_operation_t opchild = echild->get_function<unary_single_operation_t>();
            ndobject_iter<0, 1> iter(data_values_dt, data_values_metadata, data_values_data);
            if (!iter.empty()) {
                by_values_ptr = by_values_origin;
                do {
                    UIntType value = *reinterpret_cast<const UIntType *>(by_values_ptr);
                    char *&cp = cat_pointers[value];
                    opchild(cp, iter.data(), echild);
                    // Advance the pointer inside the cat_pointers array
                    cp += vad_stride;
                    by_values_ptr += by_values_stride;
                } while (iter.next());
            }
        }

        // Some compilers are finicky about getting single<T> as a function pointer, so this...
        static void single_uint8(char *dst, const char *src, kernel_data_prefix *extra) {
            single<uint8_t>(dst, src, extra);
        }
        static void single_uint16(char *dst, const char *src, kernel_data_prefix *extra) {
            single<uint16_t>(dst, src, extra);
        }
        static void single_uint32(char *dst, const char *src, kernel_data_prefix *extra) {
            single<uint32_t>(dst, src, extra);
        }

        static void destruct(kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            if (e->src_groupby_dt != NULL) {
                base_dtype_decref(e->src_groupby_dt);
            }
            kernel_data_prefix *echild = &(e + 1)->base;
            if (echild->destructor) {
                echild->destructor(echild);
            }
        }
    };
} // anonymous namespace

size_t groupby_dtype::make_operand_to_value_assignment_kernel(
                assignment_kernel *out, size_t offset_out,
                const char *dst_metadata, const char *src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
    out->ensure_capacity(offset_out + sizeof(groupby_to_value_assign_extra));
    groupby_to_value_assign_extra *e = out->get_at<groupby_to_value_assign_extra>(offset_out);
    const categorical_dtype *cd = static_cast<const categorical_dtype *>(m_groups_dtype.extended());
    switch (cd->get_category_int_dtype().get_type_id()) {
        case uint8_type_id:
            e->base.set_function<unary_single_operation_t>(&groupby_to_value_assign_extra::single_uint8);
            break;
        case uint16_type_id:
            e->base.set_function<unary_single_operation_t>(&groupby_to_value_assign_extra::single_uint16);
            break;
        case uint32_type_id:
            e->base.set_function<unary_single_operation_t>(&groupby_to_value_assign_extra::single_uint32);
            break;
        default:
            throw runtime_error("internal error in groupby_dtype::get_operand_to_value_kernel");
    }
    e->base.destructor = &groupby_to_value_assign_extra::destruct;
    // The kernel dtype owns a reference to this dtype
    e->src_groupby_dt = this;
    base_dtype_incref(e->src_groupby_dt);
    e->src_metadata = src_metadata;
    e->dst_metadata = dst_metadata;

    // The following is the setup for copying a single 'data' value to the output
    // The destination element type and metadata
    const dtype& dst_element_dtype = static_cast<const var_dim_dtype *>(
                    static_cast<const fixed_dim_dtype *>(m_value_dtype.extended())->get_element_dtype().extended()
                    )->get_element_dtype();
    const char *dst_element_metadata = dst_metadata + 0 + sizeof(var_dim_dtype_metadata);
    // Get source element type and metadata
    dtype src_element_dtype = m_operand_dtype;
    const char *src_element_metadata = e->src_metadata;
    src_element_dtype = src_element_dtype.extended()->at_single(0, &src_element_metadata, NULL);
    src_element_dtype = static_cast<const pointer_dtype *>(src_element_dtype.extended())->get_target_dtype();
    src_element_metadata += sizeof(pointer_dtype_metadata);
    src_element_dtype = src_element_dtype.extended()->at_single(0, &src_element_metadata, NULL);

    return ::make_assignment_kernel(out, offset_out + sizeof(groupby_to_value_assign_extra),
                    dst_element_dtype, dst_element_metadata,
                    src_element_dtype, src_element_metadata,
                    kernel_request_single, assign_error_none, ectx);
}

size_t groupby_dtype::make_value_to_operand_assignment_kernel(
                assignment_kernel *DYND_UNUSED(out),
                size_t DYND_UNUSED(offset_out),
                const char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata),
                kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context *DYND_UNUSED(ectx)) const
{
    throw runtime_error("Cannot assign to a dynd groupby object value");
}

dtype groupby_dtype::with_replaced_storage_dtype(const dtype& DYND_UNUSED(replacement_dtype)) const
{
    throw runtime_error("TODO: implement groupby_dtype::with_replaced_storage_dtype");
}

///////// properties on the ndobject

static ndobject property_ndo_get_groups(const ndobject& n) {
    dtype d = n.get_dtype();
    while (d.get_type_id() != groupby_type_id) {
        d = d.at_single(0);
    }
    const groupby_dtype *gd = static_cast<const groupby_dtype *>(d.extended());
    return gd->get_groups_dtype().p("categories");
}

static pair<string, gfunc::callable> groupby_ndobject_properties[] = {
    pair<string, gfunc::callable>("groups", gfunc::make_callable(&property_ndo_get_groups, "self")),
};

void groupby_dtype::get_dynamic_ndobject_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = groupby_ndobject_properties;
    *out_count = sizeof(groupby_ndobject_properties) / sizeof(groupby_ndobject_properties[0]);
}
