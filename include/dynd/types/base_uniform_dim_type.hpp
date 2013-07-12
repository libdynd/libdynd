//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BASE_UNIFORM_DIM_TYPE_HPP_
#define _DYND__BASE_UNIFORM_DIM_TYPE_HPP_

#include <dynd/types/base_type.hpp>
#include <dynd/type.hpp>

namespace dynd {


/**
 * Base class for all array dimension types. If a type
 * has kind uniform_dim_kind, it must be a subclass of
 * base_uniform_dim_type.
 */
class base_uniform_dim_type : public base_type {
protected:
    ndt::type m_element_tp;
    size_t m_element_metadata_offset;
public:
    inline base_uniform_dim_type(type_id_t type_id, const ndt::type& element_tp, size_t data_size,
                    size_t alignment, size_t element_metadata_offset,
                    flags_type flags)
        : base_type(type_id, uniform_dim_kind, data_size,
                        alignment, flags, element_metadata_offset + element_tp.get_metadata_size(),
                        1 + element_tp.get_undim()),
            m_element_tp(element_tp), m_element_metadata_offset(element_metadata_offset)
    {
    }

    virtual ~base_uniform_dim_type();

    /** The element type. */
    inline const ndt::type& get_element_type() const {
        return m_element_tp;
    }

    /**
     * The offset to add to the metadata to get to the
     * element type's metadata.
     */
    inline size_t get_element_metadata_offset() const {
        return m_element_metadata_offset;
    }

    /**
     * The dimension size, or -1 if it can't be determined
     * from the information given.
     *
     * \param metadata  A metadata instance for the type, or NULL.
     * \param data  A data instance for the type/metadata, or NULL.
     *
     * \returns  The size of the dimension, or -1.
     */
    virtual intptr_t get_dim_size(const char *metadata = NULL, const char *data = NULL) const = 0;

    /**
     * Constructs the ndobject metadata for one dimension of this type, leaving
     * the metadata for deeper dimensions uninitialized. Returns the size of
     * the metadata that was copied.
     *
     * \param dst_metadata  The new metadata memory which is constructed.
     * \param src_metadata   Existing metadata memory from which to copy.
     * \param embedded_reference  For references which are NULL, add this reference in the output.
     *                            A NULL means the data was embedded in the original ndobject, so
     *                            when putting it in a new ndobject, need to hold a reference to
     *                            that memory.
     */
    virtual size_t metadata_copy_construct_onedim(char *dst_metadata, const char *src_metadata,
                    memory_block_data *embedded_reference) const = 0;
};


} // namespace dynd

#endif // _DYND__BASE_UNIFORM_DIM_TYPE_HPP_
