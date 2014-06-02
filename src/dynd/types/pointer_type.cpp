//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/make_callable.hpp>
#include <dynd/pp/list.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

pointer_type::pointer_type(const ndt::type& target_tp)
    : base_expression_type(pointer_type_id, expression_kind, sizeof(void *),
                    sizeof(void *),
                    inherited_flags(target_tp.get_flags(), type_flag_zeroinit|type_flag_blockref),
                    sizeof(pointer_type_metadata) + target_tp.get_metadata_size(),
                    target_tp.get_ndim()),
                    m_target_tp(target_tp)
{
    // I'm not 100% sure how blockref pointer types should interact with
    // the computational subsystem, the details will have to shake out
    // when we want to actually do something with them.
    if (target_tp.get_kind() == expression_kind && target_tp.get_type_id() != pointer_type_id) {
        stringstream ss;
        ss << "A dynd pointer type's target cannot be the expression type ";
        ss << target_tp;
        throw dynd::type_error(ss.str());
    }
}

pointer_type::~pointer_type()
{
}

void pointer_type::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    const pointer_type_metadata *md = reinterpret_cast<const pointer_type_metadata *>(metadata);
    const char *target_data = *reinterpret_cast<const char * const *>(data) + md->offset;
    m_target_tp.print_data(o, metadata + sizeof(pointer_type_metadata), target_data);
}

void pointer_type::print_type(std::ostream& o) const
{
    o << "pointer[" << m_target_tp << "]";
}

bool pointer_type::is_expression() const
{
    // Even though the pointer is an instance of an base_expression_type,
    // we'll only call it an expression if the target is.
    return m_target_tp.is_expression();
}

bool pointer_type::is_unique_data_owner(const char *metadata) const
{
    const pointer_type_metadata *md = reinterpret_cast<const pointer_type_metadata *>(*metadata);
    if (md->blockref != NULL &&
            (md->blockref->m_use_count != 1 ||
             (md->blockref->m_type != pod_memory_block_type &&
              md->blockref->m_type != fixed_size_pod_memory_block_type))) {
        return false;
    }
    return true;
}

void pointer_type::transform_child_types(type_transform_fn_t transform_fn, void *extra,
                ndt::type& out_transformed_tp, bool& out_was_transformed) const
{
    ndt::type tmp_tp;
    bool was_transformed = false;
    transform_fn(m_target_tp, extra, tmp_tp, was_transformed);
    if (was_transformed) {
        out_transformed_tp = ndt::make_pointer(tmp_tp);
        out_was_transformed = true;
    } else {
        out_transformed_tp = ndt::type(this, true);
    }
}


ndt::type pointer_type::get_canonical_type() const
{
    // The canonical version doesn't include the pointer
    return m_target_tp;
}

const ndt::type& pointer_type::get_operand_type() const
{
    static ndt::type vpt = ndt::make_pointer<void>();

    if (m_target_tp.get_type_id() == pointer_type_id) {
        return m_target_tp;
    } else {
        return vpt;
    }
}


ndt::type pointer_type::apply_linear_index(intptr_t nindices, const irange *indices,
                size_t current_i, const ndt::type& root_tp, bool leading_dimension) const
{
    if (nindices == 0) {
        if (leading_dimension) {
            // Even with 0 indices, throw away the pointer when it's a leading dimension
            return m_target_tp.apply_linear_index(0, NULL, current_i, root_tp, true);
        } else {
            return ndt::type(this, true);
        }
    } else {
        ndt::type dt = m_target_tp.apply_linear_index(nindices, indices, current_i, root_tp, leading_dimension);
        if (leading_dimension) {
            // If it's a leading dimension, throw away the pointer
            return dt;
        } else if (dt == m_target_tp) {
            return ndt::type(this, true);
        } else {
            return ndt::make_pointer(dt);
        }
    }
}

intptr_t pointer_type::apply_linear_index(intptr_t nindices, const irange *indices, const char *metadata,
                const ndt::type& result_tp, char *out_metadata,
                memory_block_data *embedded_reference,
                size_t current_i, const ndt::type& root_tp,
                bool leading_dimension, char **inout_data,
                memory_block_data **inout_dataref) const
{
    if (leading_dimension) {
        // If it's a leading dimension, we always throw away the pointer
        const pointer_type_metadata *md = reinterpret_cast<const pointer_type_metadata *>(metadata);
        *inout_data = *reinterpret_cast<char **>(*inout_data) + md->offset;
        if (*inout_dataref) {
            memory_block_decref(*inout_dataref);
        }
        *inout_dataref = md->blockref ? md->blockref : embedded_reference;
        memory_block_incref(*inout_dataref);
        if (m_target_tp.is_builtin()) {
            return 0;
        } else {
            return m_target_tp.extended()->apply_linear_index(nindices, indices,
                            metadata + sizeof(pointer_type_metadata),
                            result_tp, out_metadata,
                            embedded_reference, current_i, root_tp,
                            true, inout_data, inout_dataref);
        }
    } else {
        const pointer_type_metadata *md = reinterpret_cast<const pointer_type_metadata *>(metadata);
        pointer_type_metadata *out_md = reinterpret_cast<pointer_type_metadata *>(out_metadata);
        // If there are no more indices, copy the rest verbatim
        out_md->blockref = md->blockref;
        memory_block_incref(out_md->blockref);
        out_md->offset = md->offset;
        if (!m_target_tp.is_builtin()) {
            const pointer_type *pdt = result_tp.tcast<pointer_type>();
            // The indexing may cause a change to the metadata offset
            out_md->offset += m_target_tp.extended()->apply_linear_index(nindices, indices,
                            metadata + sizeof(pointer_type_metadata),
                            pdt->m_target_tp, out_metadata + sizeof(pointer_type_metadata),
                            embedded_reference, current_i, root_tp,
                            false, NULL, NULL);
        }
        return 0;
    }
}

ndt::type pointer_type::at_single(intptr_t i0, const char **inout_metadata, const char **inout_data) const
{
    // If metadata/data is provided, follow the pointer and call the target type's at_single
    if (inout_metadata) {
        const pointer_type_metadata *md = reinterpret_cast<const pointer_type_metadata *>(*inout_metadata);
        // Modify the metadata
        *inout_metadata += sizeof(pointer_type_metadata);
        // If requested, modify the data pointer
        if (inout_data) {
            *inout_data = *reinterpret_cast<const char * const *>(inout_data) + md->offset;
        }
    }
    return m_target_tp.at_single(i0, inout_metadata, inout_data);
}

ndt::type pointer_type::get_type_at_dimension(char **inout_metadata, intptr_t i, intptr_t total_ndim) const
{
    if (i == 0) {
        return ndt::type(this, true);
    } else {
        *inout_metadata += sizeof(pointer_type_metadata);
        return m_target_tp.get_type_at_dimension(inout_metadata, i, total_ndim);
    }
}

void pointer_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                const char *metadata, const char *data) const
{
    if (!m_target_tp.is_builtin()) {
        const char *target_data = NULL;
        if (metadata != NULL && data != NULL) {
            const pointer_type_metadata *md = reinterpret_cast<const pointer_type_metadata *>(metadata);
            target_data = *reinterpret_cast<const char * const *>(data) + md->offset;
        }
        m_target_tp.extended()->get_shape(ndim, i, out_shape,
                        metadata ? (metadata + sizeof(pointer_type_metadata)) : NULL,
                        target_data);
    } else {
        stringstream ss;
        ss << "requested too many dimensions from type " << m_target_tp;
        throw runtime_error(ss.str());
    }
}

axis_order_classification_t pointer_type::classify_axis_order(const char *metadata) const
{
    // Return the classification of the target type
    if (m_target_tp.get_ndim() > 1) {
        return m_target_tp.extended()->classify_axis_order(
                        metadata + sizeof(pointer_type_metadata));
    } else {
        return axis_order_none;
    }
}

bool pointer_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
{
    if (dst_tp.extended() == this) {
        return ::is_lossless_assignment(m_target_tp, src_tp);
    } else {
        return ::is_lossless_assignment(dst_tp, m_target_tp);
    }
}

bool pointer_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != pointer_type_id) {
        return false;
    } else {
        const pointer_type *dt = static_cast<const pointer_type*>(&rhs);
        return m_target_tp == dt->m_target_tp;
    }
}

ndt::type pointer_type::with_replaced_storage_type(const ndt::type& /*replacement_tp*/) const
{
    throw runtime_error("TODO: implement pointer_type::with_replaced_storage_type");
}

void pointer_type::metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const
{
    // Simply allocate a POD memory block
    // TODO: Will need a different kind of memory block if the data isn't POD.
    pointer_type_metadata *md = reinterpret_cast<pointer_type_metadata *>(metadata);
    md->blockref = make_pod_memory_block().release();
    if (!m_target_tp.is_builtin()) {
        m_target_tp.extended()->metadata_default_construct(metadata + sizeof(pointer_type_metadata), ndim, shape);
    }
}

void pointer_type::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    // Copy the blockref, switching it to the embedded_reference if necessary
    const pointer_type_metadata *src_md = reinterpret_cast<const pointer_type_metadata *>(src_metadata);
    pointer_type_metadata *dst_md = reinterpret_cast<pointer_type_metadata *>(dst_metadata);
    dst_md->blockref = src_md->blockref ? src_md->blockref : embedded_reference;
    if (dst_md->blockref) {
        memory_block_incref(dst_md->blockref);
    }
    dst_md->offset = src_md->offset;
    // Copy the target metadata
    if (!m_target_tp.is_builtin()) {
        m_target_tp.extended()->metadata_copy_construct(dst_metadata + sizeof(pointer_type_metadata),
                        src_metadata + sizeof(pointer_type_metadata), embedded_reference);
    }
}

void pointer_type::metadata_reset_buffers(char *DYND_UNUSED(metadata)) const
{
    throw runtime_error("TODO implement pointer_type::metadata_reset_buffers");
}

void pointer_type::metadata_finalize_buffers(char *metadata) const
{
    pointer_type_metadata *md = reinterpret_cast<pointer_type_metadata *>(metadata);
    if (md->blockref != NULL) {
        // Finalize the memory block
        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        if (allocator != NULL) {
            allocator->finalize(md->blockref);
        }
    }
}

void pointer_type::metadata_destruct(char *metadata) const
{
    pointer_type_metadata *md =
                    reinterpret_cast<pointer_type_metadata *>(metadata);
    if (md->blockref) {
        memory_block_decref(md->blockref);
    }
    if (!m_target_tp.is_builtin()) {
        m_target_tp.extended()->metadata_destruct(
                        metadata + sizeof(pointer_type_metadata));
    }
}

void pointer_type::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    const pointer_type_metadata *md = reinterpret_cast<const pointer_type_metadata *>(metadata);
    o << indent << "pointer metadata\n";
    o << indent << " offset: " << md->offset << "\n";
    memory_block_debug_print(md->blockref, o, indent + " ");
    if (!m_target_tp.is_builtin()) {
        m_target_tp.extended()->metadata_debug_print(metadata + sizeof(pointer_type_metadata), o, indent + " ");
    }
}

static ndt::type property_get_target_type(const ndt::type& tp) {
    const pointer_type *pd = tp.tcast<pointer_type>();
    return pd->get_target_type();
}

void pointer_type::get_dynamic_type_properties(
                const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    static pair<string, gfunc::callable> type_properties[] = {
        pair<string, gfunc::callable>(
            "target_type",
            gfunc::make_callable(&property_get_target_type, "self"))};

    *out_properties = type_properties;
    *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}

namespace {
    // TODO: use the PP meta stuff, but DYND_PP_LEN_MAX is set to 8 right now, would need to be 19
    struct static_pointer {
        pointer_type bt1;
        pointer_type bt2;
        pointer_type bt3;
        pointer_type bt4;
        pointer_type bt5;
        pointer_type bt6;
        pointer_type bt7;
        pointer_type bt8;
        pointer_type bt9;
        pointer_type bt10;
        pointer_type bt11;
        pointer_type bt12;
        pointer_type bt13;
        pointer_type bt14;
        pointer_type bt15;
        pointer_type bt16;
        pointer_type bt17;
        void_pointer_type bt18;

        ndt::type static_builtins_instance[builtin_type_id_count];

        static_pointer()
            : bt1(ndt::type((type_id_t)1)),
              bt2(ndt::type((type_id_t)2)),
              bt3(ndt::type((type_id_t)3)),
              bt4(ndt::type((type_id_t)4)),
              bt5(ndt::type((type_id_t)5)),
              bt6(ndt::type((type_id_t)6)),
              bt7(ndt::type((type_id_t)7)),
              bt8(ndt::type((type_id_t)8)),
              bt9(ndt::type((type_id_t)9)),
              bt10(ndt::type((type_id_t)10)),
              bt11(ndt::type((type_id_t)11)),
              bt12(ndt::type((type_id_t)12)),
              bt13(ndt::type((type_id_t)13)),
              bt14(ndt::type((type_id_t)14)),
              bt15(ndt::type((type_id_t)15)),
              bt16(ndt::type((type_id_t)16)),
              bt17(ndt::type((type_id_t)17)),
              bt18()
        {
            static_builtins_instance[1] = ndt::type(&bt1, true);
            static_builtins_instance[2] = ndt::type(&bt2, true);
            static_builtins_instance[3] = ndt::type(&bt3, true);
            static_builtins_instance[4] = ndt::type(&bt4, true);
            static_builtins_instance[5] = ndt::type(&bt5, true);
            static_builtins_instance[6] = ndt::type(&bt6, true);
            static_builtins_instance[7] = ndt::type(&bt7, true);
            static_builtins_instance[8] = ndt::type(&bt8, true);
            static_builtins_instance[9] = ndt::type(&bt9, true);
            static_builtins_instance[10] = ndt::type(&bt10, true);
            static_builtins_instance[11] = ndt::type(&bt11, true);
            static_builtins_instance[12] = ndt::type(&bt12, true);
            static_builtins_instance[13] = ndt::type(&bt13, true);
            static_builtins_instance[14] = ndt::type(&bt14, true);
            static_builtins_instance[15] = ndt::type(&bt15, true);
            static_builtins_instance[16] = ndt::type(&bt16, true);
            static_builtins_instance[17] = ndt::type(&bt17, true);
            static_builtins_instance[18] = ndt::type(&bt18, true);
        }
    };
} // anonymous namespace

ndt::type ndt::make_pointer(const ndt::type& target_tp)
{
    // Static instance of strided_dim_type, which have a reference
    // count > 0 for the lifetime of the program. This static
    // construction is inside a function to ensure correct creation
    // order during startup.
    static static_pointer sp;

    if (target_tp.is_builtin()) {
        return sp.static_builtins_instance[target_tp.get_type_id()];
    } else {
        return ndt::type(new pointer_type(target_tp), false);
    }
}
