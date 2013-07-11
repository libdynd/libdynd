//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BASE_UNIFORM_DIM_TYPE_HPP_
#define _DYND__BASE_UNIFORM_DIM_TYPE_HPP_

#include <dynd/dtypes/base_type.hpp>
#include <dynd/type.hpp>

namespace dynd {


/**
 * Base class for all array dimension dtypes. If a dtype
 * has kind uniform_dim_kind, it must be a subclass of
 * base_uniform_dim_dtype.
 */
class base_uniform_dim_dtype : public base_type {
protected:
    ndt::type m_element_dtype;
    size_t m_element_metadata_offset;
public:
    inline base_uniform_dim_dtype(type_id_t type_id, const ndt::type& element_dtype, size_t data_size,
                    size_t alignment, size_t element_metadata_offset,
                    flags_type flags)
        : base_type(type_id, uniform_dim_kind, data_size,
                        alignment, flags, element_metadata_offset + element_dtype.get_metadata_size(),
                        1 + element_dtype.get_undim()),
            m_element_dtype(element_dtype), m_element_metadata_offset(element_metadata_offset)
    {
    }

    virtual ~base_uniform_dim_dtype();

    /** The element dtype. */
    inline const ndt::type& get_element_type() const {
        return m_element_dtype;
    }

    /**
     * The offset to add to the metadata to get to the
     * element_dtype's metadata.
     */
    inline size_t get_element_metadata_offset() const {
        return m_element_metadata_offset;
    }

    /**
     * The dimension size, or -1 if it can't be determined
     * from the information given.
     *
     * \param metadata  A metadata instance for the dtype, or NULL.
     * \param data  A data instance for the type/metadata, or NULL.
     *
     * \returns  The size of the dimension, or -1.
     */
    virtual intptr_t get_dim_size(const char *metadata = NULL, const char *data = NULL) const = 0;

    /**
     * Constructs the ndobject metadata for one dimension of this dtype, leaving
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
