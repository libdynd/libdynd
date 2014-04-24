//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/view.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/shape_tools.hpp>

using namespace std;
using namespace dynd;

/**
 * Scans through the types, and tries to view data
 * for 'tp'/'metadata' as 'view_tp'. For this to be
 * possible, one must be able to construct
 * metadata for 'tp' corresponding to the same data.
 *
 * \param tp  The type of the data.
 * \param metadata  The array metadata of the data.
 * \param view_tp  The type the data should be viewed as.
 * \param view_metadata The array metadata of the view, which should be populated.
 * \param embedded_reference  The containing memory block in case the data was embedded.
 *
 * \returns If it worked, returns true, otherwise false.
 */
static bool try_view(const ndt::type &tp, const char *metadata,
                     const ndt::type &view_tp, char *view_metadata,
                     dynd::memory_block_data *embedded_reference)
{
    switch (tp.get_type_id()) {
    case strided_dim_type_id: {
        const strided_dim_type *sdt = tp.tcast<strided_dim_type>();
        const strided_dim_type_metadata *md =
            reinterpret_cast<const strided_dim_type_metadata *>(metadata);
        switch (view_tp.get_type_id()) {
        case strided_dim_type_id: { // strided as strided
            const strided_dim_type *view_sdt =
                view_tp.tcast<strided_dim_type>();
            strided_dim_type_metadata *view_md =
                reinterpret_cast<strided_dim_type_metadata *>(view_metadata);
            if (try_view(sdt->get_element_type(),
                         metadata + sizeof(strided_dim_type_metadata),
                         view_sdt->get_element_type(),
                         view_metadata + sizeof(strided_dim_type_metadata),
                         embedded_reference)) {
                view_md->size = md->size;
                view_md->stride = md->stride;
                return true;
            } else {
                return false;
            }
        }
        case fixed_dim_type_id: { // strided as fixed
            const fixed_dim_type *view_fdt = view_tp.tcast<fixed_dim_type>();
            // The size must match exactly in this case
            if (md->size != (intptr_t)view_fdt->get_fixed_dim_size()) {
                return false;
            }
            fixed_dim_type_metadata *view_md =
                reinterpret_cast<fixed_dim_type_metadata *>(view_metadata);
            if (try_view(sdt->get_element_type(),
                         metadata + sizeof(strided_dim_type_metadata),
                         view_fdt->get_element_type(),
                         view_metadata + sizeof(fixed_dim_type_metadata),
                         embedded_reference)) {
                view_md->stride = md->stride;
                return true;
            } else {
                return false;
            }
        }
        case cfixed_dim_type_id: { // strided as cfixed
            const cfixed_dim_type *view_fdt =
                view_tp.tcast<cfixed_dim_type>();
            // The size and stride must match exactly in this case
            if (md->size != (intptr_t)view_fdt->get_fixed_dim_size() ||
                    md->stride != view_fdt->get_fixed_stride()) {
                return false;
            }
            return try_view(sdt->get_element_type(),
                            metadata + sizeof(strided_dim_type_metadata),
                            view_fdt->get_element_type(), view_metadata,
                            embedded_reference);
        }
        default: // other cases cannot be handled
            return false;
        }
    }
    case fixed_dim_type_id: {
        const fixed_dim_type *fdt = tp.tcast<fixed_dim_type>();
        const fixed_dim_type_metadata *md =
            reinterpret_cast<const fixed_dim_type_metadata *>(metadata);
        switch (view_tp.get_type_id()) {
        case strided_dim_type_id: { // fixed as strided
            const strided_dim_type *view_sdt =
                view_tp.tcast<strided_dim_type>();
            strided_dim_type_metadata *view_md =
                reinterpret_cast<strided_dim_type_metadata *>(view_metadata);
            if (try_view(fdt->get_element_type(),
                         metadata + sizeof(fixed_dim_type_metadata),
                         view_sdt->get_element_type(),
                         view_metadata + sizeof(strided_dim_type_metadata),
                         embedded_reference)) {
                view_md->size = fdt->get_fixed_dim_size();
                view_md->stride = md->stride;
                return true;
            } else {
                return false;
            }
        }
        case fixed_dim_type_id: { // fixed as fixed
            const fixed_dim_type *view_fdt = view_tp.tcast<fixed_dim_type>();
            // The size must match exactly in this case
            if (fdt->get_fixed_dim_size() != view_fdt->get_fixed_dim_size()) {
                return false;
            }
            fixed_dim_type_metadata *view_md =
                reinterpret_cast<fixed_dim_type_metadata *>(view_metadata);
            if (try_view(fdt->get_element_type(),
                         metadata + sizeof(fixed_dim_type_metadata),
                         view_fdt->get_element_type(),
                         view_metadata + sizeof(fixed_dim_type_metadata),
                         embedded_reference)) {
                view_md->stride = md->stride;
                return true;
            } else {
                return false;
            }
        }
        case cfixed_dim_type_id: { // strided as cfixed
            const cfixed_dim_type *view_fdt =
                view_tp.tcast<cfixed_dim_type>();
            // The size and stride must match exactly in this case
            if (fdt->get_fixed_dim_size() != view_fdt->get_fixed_dim_size() ||
                    md->stride != view_fdt->get_fixed_stride()) {
                return false;
            }
            return try_view(fdt->get_element_type(),
                            metadata + sizeof(fixed_dim_type_metadata),
                            view_fdt->get_element_type(), view_metadata,
                            embedded_reference);
        }
        default: // other cases cannot be handled
            return false;
        }
    }
    case cfixed_dim_type_id: {
        const cfixed_dim_type *fdt =
            tp.tcast<cfixed_dim_type>();
        switch (view_tp.get_type_id()) {
        case cfixed_dim_type_id: { // cfixed as cfixed
            const cfixed_dim_type *view_fdt =
                view_tp.tcast<cfixed_dim_type>();
            // The size and stride must match exactly in this case
            if (fdt->get_fixed_dim_size() != view_fdt->get_fixed_dim_size() ||
                    fdt->get_fixed_stride() != view_fdt->get_fixed_stride()) {
                return false;
            }
            return try_view(fdt->get_element_type(), metadata,
                            view_fdt->get_element_type(), view_metadata,
                            embedded_reference);
        }
        case fixed_dim_type_id: { // cfixed as fixed
            const fixed_dim_type *view_fdt =
                view_tp.tcast<fixed_dim_type>();
            // The size must match exactly in this case
            if (fdt->get_fixed_dim_size() != view_fdt->get_fixed_dim_size()) {
                return false;
            }
            fixed_dim_type_metadata *view_md =
                reinterpret_cast<fixed_dim_type_metadata *>(view_metadata);
            if (try_view(fdt->get_element_type(), metadata,
                         view_fdt->get_element_type(),
                         view_metadata + sizeof(fixed_dim_type_metadata),
                         embedded_reference)) {
                view_md->stride = fdt->get_fixed_stride();
                return true;
            } else {
                return false;
            }
        }
        case strided_dim_type_id: { // cfixed as strided
            const strided_dim_type *view_sdt =
                view_tp.tcast<strided_dim_type>();
            strided_dim_type_metadata *view_md =
                reinterpret_cast<strided_dim_type_metadata *>(view_metadata);
            if (try_view(fdt->get_element_type(), metadata,
                         view_sdt->get_element_type(),
                         view_metadata + sizeof(strided_dim_type_metadata),
                         embedded_reference)) {
                view_md->size = fdt->get_fixed_dim_size();
                view_md->stride = fdt->get_fixed_stride();
                return true;
            } else {
                return false;
            }
        }
        default: // other cases cannot be handled
            return false;
        }
        }
    default: // require equal types otherwise
        if (tp == view_tp) {
            if (tp.get_metadata_size() > 0) {
                tp.extended()->metadata_copy_construct(view_metadata, metadata,
                                                       embedded_reference);
            }
            return true;
        } else {
            return false;
        }
    }
}

static void refine_bytes_view(memory_block_ptr &data_ref, char *&data_ptr,
                              ndt::type &data_tp, const char *&data_meta,
                              intptr_t &data_dim_size, intptr_t &data_stride)
{
    // Handle sequence of strided dims
    intptr_t dim_size, stride;
    ndt::type el_tp;
    const char *el_meta;
    if (data_tp.get_as_strided_dim(data_meta, dim_size, stride, el_tp,
                                   el_meta)) {
        dimvector shape(data_tp.get_ndim());
        dimvector strides(data_tp.get_ndim());
        intptr_t ndim = 1;
        shape[0] = dim_size;
        strides[0] = stride;
        bool csorted = true;
        // Get all the strided dimensions we can in a row
        while (el_tp.get_as_strided_dim(el_meta, dim_size, stride, el_tp,
                                        el_meta)) {
            shape[ndim] = dim_size;
            strides[ndim] = stride;
            if (stride > strides[ndim-1]) {
                csorted = false;
            }
            ++ndim;
        }
        if (!csorted) {
            // If the strides weren't sorted in C order, sort them
            shortvector<int> axis_perm(ndim);
            strides_to_axis_perm(ndim, strides.get(), axis_perm.get());
            dimvector shape_sorted(ndim);
            dimvector strides_sorted(ndim);
            for (intptr_t i = 0; i < ndim; ++i) {
                int i_perm = axis_perm[i];
                shape_sorted[ndim-i-1] = shape[i_perm];
                strides_sorted[ndim-i-1] = strides[i_perm];
            }
            shape.swap(shape_sorted);
            strides.swap(strides_sorted);
        }
        // Try to collapse the shape/strides into a single strided array
        intptr_t i = 0;
        while (data_dim_size == -1 && i < ndim) {
            // If there's not already a dim_size/stride, start one
            if (shape[i] != 1) {
                data_dim_size = shape[i];
                data_stride = strides[i];
                ++i;
            }
        }
        for (; i < ndim; ++i) {
            if (shape[i] != 1) {
                if (shape[i] * strides[i] != data_stride) {
                    // Indicate we couldn't view this as bytes
                    data_tp = ndt::type();
                    data_dim_size = -1;
                    return;
                }
                data_dim_size *= shape[i];
                data_stride = strides[i];
            }
        }
        data_tp = el_tp;
        data_meta = el_meta;
        return;
    }

    switch (data_tp.get_type_id()) {
    case var_dim_type_id: {
        // We can only allow leading var_dim
        if (data_dim_size != -1) {
            break;
        }
        const var_dim_type_metadata *meta =
            reinterpret_cast<const var_dim_type_metadata *>(data_meta);
        if (meta->blockref != NULL) {
            data_ref = meta->blockref;
        }
        var_dim_type_data *d = reinterpret_cast<var_dim_type_data *>(data_ptr);
        data_ptr = d->begin + meta->offset;
        if (d->size != 1) {
            data_dim_size = d->size;
            data_stride = meta->stride;
        }
        data_tp = data_tp.tcast<var_dim_type>()->get_element_type();
        data_meta += sizeof(var_dim_type_metadata);
        return;
    }
    case pointer_type_id: {
        // We can only strip away leading pointers
        if (data_dim_size != -1) {
            break;
        }
        const pointer_type_metadata *meta =
            reinterpret_cast<const pointer_type_metadata *>(data_meta);
        if (meta->blockref != NULL) {
            data_ref = meta->blockref;
        }
        data_ptr = *reinterpret_cast<char **>(data_ptr) + meta->offset;
        data_tp = data_tp.tcast<pointer_type>()->get_target_type();
        data_meta += sizeof(pointer_type_metadata);
        return;
    }
    default:
        break;
    }

    // If the data type has a fixed size, check if it fits the strides
    size_t data_tp_size = data_tp.get_data_size();
    if (data_tp_size > 0) {
        if (data_dim_size == -1) {
            // Indicate success (one item)
            data_tp = ndt::type();
            data_dim_size = data_tp_size;
            data_stride = 1;
            return;
        } else if (data_tp_size == data_stride) {
            data_tp = ndt::type();
            data_dim_size *= data_tp_size;
            data_stride = 1;
            return;
        }
    }

    // Indicate we couldn't view this as bytes
    data_tp = ndt::type();
    data_dim_size = -1;
}

static nd::array view_as_bytes(const nd::array& arr, const ndt::type& tp)
{
    if (arr.get_type().get_flags()&type_flag_destructor) {
        // Can't view arrays of object type
        return nd::array();
    }

    // Get the essential components of the array to analyze
    memory_block_ptr data_ref = arr.get_data_memblock();
    char *data_ptr = arr.get_ndo()->m_data_pointer;
    ndt::type data_tp = arr.get_type();
    const char *data_meta = arr.get_ndo_meta();
    intptr_t data_dim_size = -1, data_stride = 0;
    // Repeatedly refine the data
    while (data_tp.get_type_id() != uninitialized_type_id) {
        refine_bytes_view(data_ref, data_ptr, data_tp, data_meta, data_dim_size,
                          data_stride);
    }
    // Check that it worked, and that the resulting data pointer is aligned
    if (data_dim_size < 0 ||
            !offset_is_aligned(reinterpret_cast<size_t>(data_ptr),
                           tp.tcast<bytes_type>()->get_target_alignment())) {
        // This signals we could not view the data as a
        // contiguous chunk of bytes
        return nd::array();
    }

    char *result_data_ptr = NULL;
    nd::array result(make_array_memory_block(
        tp.extended()->get_metadata_size(), tp.get_data_size(),
        tp.get_data_alignment(), &result_data_ptr));
    // Set the bytes extents
    ((char **)result_data_ptr)[0] = data_ptr;
    ((char **)result_data_ptr)[1] = data_ptr + data_dim_size;
    // Set the array metadata
    array_preamble *ndo = result.get_ndo();
    ndo->m_type = ndt::type(tp).release();
    ndo->m_data_pointer = result_data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = arr.get_flags();
    // Set the bytes metadata
    bytes_type_metadata *ndo_meta =
        reinterpret_cast<bytes_type_metadata *>(result.get_ndo_meta());
    ndo_meta->blockref = data_ref.release();
    return result;
}

nd::array nd::view(const nd::array& arr, const ndt::type& tp)
{
    if (arr.get_type() == tp) {
        // If the types match exactly, simply return 'arr'
        return arr;
    } else if (tp.get_type_id() == bytes_type_id) {
        // If it's a request to view the data as raw bytes
        nd::array result = view_as_bytes(arr, tp);
        if (!result.is_empty()) {
            return result;
        }
        // Otherwise fall through to error message
    } else if (arr.get_ndim() == tp.get_ndim()) {
        // Allocate a result array to attempt the view in it
        nd::array result(make_array_memory_block(tp.get_metadata_size()));
        // Copy the fields
        result.get_ndo()->m_data_pointer = arr.get_ndo()->m_data_pointer;
        if (arr.get_ndo()->m_data_reference == NULL) {
            // Embedded data, need reference to the array
            result.get_ndo()->m_data_reference = arr.get_memblock().release();
        } else {
            // Use the same data reference, avoid producing a chain
            result.get_ndo()->m_data_reference = arr.get_data_memblock().release();
        }
        result.get_ndo()->m_type = ndt::type(tp).release();
        result.get_ndo()->m_flags = arr.get_ndo()->m_flags;
        // Now try to copy the metadata as a view
        if (try_view(arr.get_type(), arr.get_ndo_meta(), tp,
                     result.get_ndo_meta(), arr.get_memblock().get())) {
            // If it succeeded, return it
            return result;
        }
        // Otherwise fall through, let it get destructed, and raise an error
    }

    stringstream ss;
    ss << "Unable to view nd::array of type " << arr.get_type();
    ss << " as type " << tp;
    throw type_error(ss.str());
}
