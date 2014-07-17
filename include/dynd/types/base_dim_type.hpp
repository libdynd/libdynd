//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND__TYPES__BASE_DIM_TYPE_HPP
#define DYND__TYPES__BASE_DIM_TYPE_HPP

#include <dynd/types/base_type.hpp>
#include <dynd/type.hpp>

namespace dynd {


/**
 * Base class for all array dimension types. If a type
 * has kind dim_kind, it must be a subclass of
 * base_dim_type.
 */
class base_dim_type : public base_type {
protected:
    ndt::type m_element_tp;
    size_t m_element_arrmeta_offset;
public:
  inline base_dim_type(type_id_t type_id, const ndt::type &element_tp,
                               size_t data_size, size_t alignment,
                               size_t element_arrmeta_offset, flags_type flags,
                               bool strided)
      : base_type(type_id, dim_kind, data_size, alignment, flags,
                  element_arrmeta_offset + element_tp.get_arrmeta_size(),
                  1 + element_tp.get_ndim(),
                  strided ? (1 + element_tp.get_strided_ndim()) : 0),
        m_element_tp(element_tp),
        m_element_arrmeta_offset(element_arrmeta_offset)
    {
    }

    virtual ~base_dim_type();

    /** The element type. */
    inline const ndt::type& get_element_type() const {
        return m_element_tp;
    }

    inline bool is_type_subarray(const ndt::type& subarray_tp) const {
        // Uniform dimensions can share one implementation
        intptr_t this_ndim = get_ndim(), stp_ndim = subarray_tp.get_ndim();
        if (this_ndim > stp_ndim) {
            return get_element_type().is_type_subarray(subarray_tp);
        } else if (this_ndim == stp_ndim) {
            return (*this) == (*subarray_tp.extended());
        } else {
            return false;
        }
    }

    /**
     * The offset to add to the arrmeta to get to the
     * element type's arrmeta.
     */
    inline size_t get_element_arrmeta_offset() const {
        return m_element_arrmeta_offset;
    }

    /**
     * The dimension size, or -1 if it can't be determined
     * from the information given.
     *
     * \param arrmeta  A arrmeta instance for the type, or NULL.
     * \param data  A data instance for the type/arrmeta, or NULL.
     *
     * \returns  The size of the dimension, or -1.
     */
    virtual intptr_t get_dim_size(const char *arrmeta = NULL, const char *data = NULL) const = 0;

    /**
     * Constructs the nd::array arrmeta for one dimension of this type, leaving
     * the arrmeta for deeper dimensions uninitialized. Returns the size of
     * the arrmeta that was copied.
     *
     * \param dst_arrmeta  The new arrmeta memory which is constructed.
     * \param src_arrmeta   Existing arrmeta memory from which to copy.
     * \param embedded_reference  For references which are NULL, add this reference in the output.
     *                            A NULL means the data was embedded in the original nd::array, so
     *                            when putting it in a new nd::array, need to hold a reference to
     *                            that memory.
     */
    virtual size_t arrmeta_copy_construct_onedim(char *dst_arrmeta, const char *src_arrmeta,
                    memory_block_data *embedded_reference) const = 0;
};


} // namespace dynd

#endif // DYND__TYPES__BASE_DIM_TYPE_HPP
