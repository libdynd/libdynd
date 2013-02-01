//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <vector>

#include <dynd/dtypes/groupby_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <dynd/dtypes/pointer_dtype.hpp>
#include <dynd/dtypes/fixedarray_dtype.hpp>
#include <dynd/dtypes/var_array_dtype.hpp>
#include <dynd/dtypes/categorical_dtype.hpp>
#include <dynd/ndobject_iter.hpp>

using namespace std;
using namespace dynd;

groupby_dtype::groupby_dtype(const dtype& data_values_dtype,
                const dtype& by_values_dtype, const dtype& groups_dtype)
    : base_expression_dtype(groupby_type_id, expression_kind,
                    sizeof(groupby_dtype_data), sizeof(void *), 2 + data_values_dtype.get_undim())
{
    if (groups_dtype.get_type_id() != categorical_type_id) {
        throw runtime_error("to construct a groupby dtype, its groups dtype must be categorical");
    }
    if (data_values_dtype.get_undim() < 1) {
        throw runtime_error("to construct a groupby dtype, its values dtype must have at least one uniform dimension");
    }
    if (by_values_dtype.get_undim() < 1) {
        throw runtime_error("to construct a groupby dtype, its values dtype must have at least one uniform dimension");
    }
    if (by_values_dtype.at_single(0).value_dtype() !=
                    reinterpret_cast<const categorical_dtype *>(groups_dtype.extended())->get_category_dtype()) {
        stringstream ss;
        ss << "to construct a groupby dtype, the by dtype, " << by_values_dtype.at_single(0);
        ss << ", should match the category dtype, ";
        ss << reinterpret_cast<const categorical_dtype *>(groups_dtype.extended())->get_category_dtype();
    }
    m_operand_dtype = make_fixedstruct_dtype(make_pointer_dtype(data_values_dtype), "data",
                    make_pointer_dtype(by_values_dtype), "by");
    const categorical_dtype *cd = static_cast<const categorical_dtype *>(groups_dtype.extended());
    m_value_dtype = make_fixedarray_dtype(make_var_array_dtype(
                    data_values_dtype.at_single(0)), cd->get_category_count());
    m_groups_dtype = groups_dtype;
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
    o << ", by=" << get_by_values_dtype();
    o << ", groups=" << m_groups_dtype << ">";
}

bool groupby_dtype::is_scalar() const
{
    return false;
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
    } else if (rhs.get_type_id() != view_type_id) {
        return false;
    } else {
        const groupby_dtype *dt = static_cast<const groupby_dtype*>(&rhs);
        return m_value_dtype == dt->m_value_dtype;
    }
}

namespace {
   struct groupby_to_value_assign {
        // Assign from a categorical dtype to some other dtype
        struct auxdata_storage {
            // The groupby dtype
            dtype gb;
            // Kernel for copying a data_value
            kernel_instance<unary_operation_pair_t> value_copy;
        };

        template<typename UIntType>
        static void single_kernel(char *dst, const char *src, unary_kernel_static_data *extra)
        {
            auxdata_storage& ad = get_auxiliary_data<auxdata_storage>(extra->auxdata);
            const groupby_dtype *gd = static_cast<const groupby_dtype *>(ad.gb.extended());
            ad.value_copy.extra.dst_metadata = extra->dst_metadata + sizeof(var_array_dtype_metadata);

            // Get the by_values raw ndobject
            dtype by_values_dt = gd->get_operand_dtype();
            const char *by_values_metadata = extra->src_metadata, *by_values_data = src;
            by_values_dt = by_values_dt.extended()->at_single(0, &by_values_metadata, &by_values_data);
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

            // Get the data_values raw ndobject
            dtype data_values_dt = gd->get_operand_dtype();
            const char *data_values_metadata = extra->src_metadata, *data_values_data = src;
            data_values_dt = data_values_dt.extended()->at_single(1, &data_values_metadata, &data_values_data);
            data_values_dt = static_cast<const pointer_dtype *>(data_values_dt.extended())->get_target_dtype();
            data_values_metadata += sizeof(pointer_dtype_metadata);
            data_values_data = *reinterpret_cast<const char * const *>(data_values_data);

            const dtype& result_dt = gd->get_value_dtype();
            const fixedarray_dtype *fad = static_cast<const fixedarray_dtype *>(result_dt.extended());
            intptr_t fad_stride = fad->get_fixed_stride();
            const var_array_dtype *vad = static_cast<const var_array_dtype *>(fad->get_element_dtype().extended());
            const var_array_dtype_metadata *vad_md = reinterpret_cast<const var_array_dtype_metadata *>(extra->dst_metadata);
            if (vad_md->offset != 0) {
                throw runtime_error("dynd groupby: destination var_array offset must be zero to allocate output");
            }
            intptr_t vad_stride = vad_md->stride;

            // Do a pass through by_values to get the size of each variable-sized dimension
            vector<size_t> cat_sizes(fad->get_fixed_dim_size());
            const char *by_values_ptr = by_values_origin;
            for (intptr_t i = 0; i < by_values_size; ++i, by_values_ptr += by_values_stride) {
                UIntType value = *reinterpret_cast<const UIntType *>(by_values_ptr);
                if (value >= cat_sizes.size()) {
                    throw runtime_error("dynd groupby: 'by' array contains an out of bounds value");
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
                reinterpret_cast<var_array_dtype_data *>(dst + i * fad_stride)->begin = out_begin;
                size_t csize = cat_sizes[i];
                reinterpret_cast<var_array_dtype_data *>(dst + i * fad_stride)->size = csize;
                out_begin += csize * vad_stride;
            }

            // Loop through both by_values and data_values,
            // copying the data to the right place in the output
            ndobject_iter<0, 1> iter(data_values_dt, data_values_metadata, data_values_data);
            if (!iter.empty()) {
                ad.value_copy.extra.src_metadata = iter.metadata();
                by_values_ptr = by_values_origin;
                do {
                    UIntType value = *reinterpret_cast<const UIntType *>(by_values_ptr);
                    char *&cp = cat_pointers[value];
                    ad.value_copy.kernel.single(cp, iter.data(), &ad.value_copy.extra);
                    ++cp;
                    by_values_ptr += by_values_stride;
                } while (iter.next());
            }
        }
    };
} // anonymous namespace

void groupby_dtype::get_operand_to_value_kernel(const eval::eval_context *ectx,
                        kernel_instance<unary_operation_pair_t>& out_kernel) const
{
    const categorical_dtype *cd = static_cast<const categorical_dtype *>(m_groups_dtype.extended());
    switch (cd->get_category_int_dtype().get_type_id()) {
        case uint8_type_id:
            out_kernel.kernel = unary_operation_pair_t(
                            groupby_to_value_assign::single_kernel<uint8_t>, NULL);
            break;
        case uint16_type_id:
            out_kernel.kernel = unary_operation_pair_t(
                            groupby_to_value_assign::single_kernel<uint16_t>, NULL);
            break;
        case uint32_type_id:
            out_kernel.kernel = unary_operation_pair_t(
                            groupby_to_value_assign::single_kernel<uint32_t>, NULL);
            break;
        default:
            throw runtime_error("internal error in groupby_dtype::get_operand_to_value_kernel");
    }
    make_auxiliary_data<groupby_to_value_assign::auxdata_storage>(out_kernel.extra.auxdata);
    groupby_to_value_assign::auxdata_storage& ad =
                out_kernel.extra.auxdata.get<groupby_to_value_assign::auxdata_storage>();
    // A reference to this dtype
    ad.gb = dtype(this, true);
    dtype dvdt = get_data_values_dtype().at_single(0);
    // A kernel assigning from the data_values array to a value in the result
    ::get_dtype_assignment_kernel(dvdt.value_dtype(), dvdt, assign_error_default, ectx, ad.value_copy);
}

void groupby_dtype::get_value_to_operand_kernel(const eval::eval_context *DYND_UNUSED(ectx),
                        kernel_instance<unary_operation_pair_t>& DYND_UNUSED(out_borrowed_kernel)) const
{
    throw runtime_error("TODO: implement groupby_dtype::get_value_to_operand_kernel");
}

dtype groupby_dtype::with_replaced_storage_dtype(const dtype& DYND_UNUSED(replacement_dtype)) const
{
    throw runtime_error("TODO: implement groupby_dtype::with_replaced_storage_dtype");
}
