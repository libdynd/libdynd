//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>
#include <dynd/types/base_type.hpp>

namespace dynd {
namespace ndt {

  /**
   * Base class for all array dimension types. If a type
   * has kind dim_kind, it must be a subclass of
   * base_dim_type.
   */
  class DYNDT_API base_dim_type : public base_type {
  protected:
    type m_element_tp;
    size_t m_element_arrmeta_offset;

  public:
    base_dim_type(type_id_t tp_id, const type &element_tp, size_t data_size, size_t data_alignment,
                  size_t element_arrmeta_offset, uint32_t flags, bool strided);

    /** The element type. */
    const type &get_element_type() const { return m_element_tp; }

    void get_element_types(std::size_t ndim, const type **element_tp) const;

    std::vector<const type *> get_element_types(std::size_t ndim) const
    {
      std::vector<const type *> element_tp(ndim);
      get_element_types(ndim, element_tp.data());

      return element_tp;
    }

    bool is_type_subarray(const type &subarray_tp) const
    {
      // Uniform dimensions can share one implementation
      intptr_t this_ndim = get_ndim(), stp_ndim = subarray_tp.get_ndim();
      if (this_ndim > stp_ndim) {
        return get_element_type().is_type_subarray(subarray_tp);
      }
      else if (this_ndim == stp_ndim) {
        return (*this) == (*subarray_tp.extended());
      }
      else {
        return false;
      }
    }

    /**
     * The offset to add to the arrmeta to get to the
     * element type's arrmeta.
     */
    size_t get_element_arrmeta_offset() const { return m_element_arrmeta_offset; }

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

    intptr_t get_size(const char *arrmeta) const
    {
      std::intptr_t dim_size = get_dim_size(arrmeta, NULL);
      if (dim_size == -1) {
        return -1;
      }

      return dim_size * m_element_tp.get_size(NULL);
    }

    virtual void get_vars(std::unordered_set<std::string> &vars) const { m_element_tp.get_vars(vars); }

    /**
     * Constructs the nd::array arrmeta for one dimension of this type, leaving
     * the arrmeta for deeper dimensions uninitialized. Returns the size of
     * the arrmeta that was copied.
     *
     * \param dst_arrmeta  The new arrmeta memory which is constructed.
     * \param src_arrmeta   Existing arrmeta memory from which to copy.
     * \param embedded_reference  For references which are NULL, add this
     *                            reference in the output.
     *                            A NULL means the data was embedded in the
     *                            original nd::array, so
     *                            when putting it in a new nd::array, need to
     *                            hold a reference to that memory.
     */
    virtual size_t arrmeta_copy_construct_onedim(char *dst_arrmeta, const char *src_arrmeta,
                                                 const intrusive_ptr<memory_block_data> &embedded_reference) const = 0;

    virtual bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    virtual type with_element_type(const type &element_tp) const = 0;
  };

} // namespace dynd::ndt
} // namespace dynd
