//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/view.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>

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
        const strided_dim_type *sdt =
            static_cast<const strided_dim_type *>(tp.extended());
        const strided_dim_type_metadata *md =
            reinterpret_cast<const strided_dim_type_metadata *>(metadata);
        switch (view_tp.get_type_id()) {
        case strided_dim_type_id: { // strided as strided
            const strided_dim_type *view_sdt =
                static_cast<const strided_dim_type *>(view_tp.extended());
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
            const fixed_dim_type *view_fdt =
                static_cast<const fixed_dim_type *>(view_tp.extended());
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
        const fixed_dim_type *fdt =
            static_cast<const fixed_dim_type *>(tp.extended());
        switch (view_tp.get_type_id()) {
        case fixed_dim_type_id: { // fixed as fixed
            const fixed_dim_type *view_fdt =
                static_cast<const fixed_dim_type *>(view_tp.extended());
            // The size and stride must match exactly in this case
            if (fdt->get_fixed_dim_size() != view_fdt->get_fixed_dim_size() ||
                    fdt->get_fixed_stride() != view_fdt->get_fixed_stride()) {
                return false;
            }
            return try_view(fdt->get_element_type(), metadata,
                            view_fdt->get_element_type(), view_metadata,
                            embedded_reference);
        }
        case strided_dim_type_id: { // fixed as strided
            const strided_dim_type *view_sdt =
                static_cast<const strided_dim_type *>(view_tp.extended());
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

nd::array nd::view(const nd::array& arr, const ndt::type& tp)
{
    // If the types match exactly, simply return 'arr'
    if (arr.get_type() == tp) {
        return arr;
    } else if (arr.get_ndim() == tp.get_ndim()) {
        // Allocate a result array to attempt the view in it
        array result(make_array_memory_block(tp.get_metadata_size()));
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
    ss << "as type " << tp;
    throw type_error(ss.str());
}
