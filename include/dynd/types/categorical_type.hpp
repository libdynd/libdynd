//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The categorical type always represents categorical data
//

#pragma once

#include <dynd/type.hpp>
#include <dynd/array.hpp>
#include <dynd/types/fixed_dim_type.hpp>

namespace {

struct assign_to_same_category_type;

} // anonymous namespace

namespace dynd {
namespace ndt {

  class DYND_API categorical_type : public base_type {
    // The data type of the category
    type m_category_tp;
    // The integer type used for storage
    type m_storage_type;
    // list of categories, in sorted order
    nd::array m_categories;
    // mapping from category indices to values
    nd::array m_category_index_to_value;
    // mapping from values to category indices
    nd::array m_value_to_category_index;

  public:
    categorical_type(const nd::array &categories, bool presorted = false);

    virtual ~categorical_type()
    {
    }

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;

    size_t get_category_count() const
    {
      return (size_t) reinterpret_cast<const fixed_dim_type_arrmeta *>(m_categories.get()->metadata())->dim_size;
    }

    /**
     * Returns the type of the category values.
     */
    const type &get_category_type() const
    {
      return m_category_tp;
    }

    /**
     * Return the type of the underlying integer used
     * to index the category list.
     */
    const type &get_storage_type() const
    {
      return m_storage_type;
    }

    uint32_t get_value_from_category(const char *category_arrmeta, const char *category_data) const;
    uint32_t get_value_from_category(const nd::array &category) const;

    const char *get_category_data_from_value(uint32_t value) const
    {
      if (value >= get_category_count()) {
        throw std::runtime_error("category value is out of bounds");
      }
      return m_categories.cdata() +
             unchecked_fixed_dim_get<intptr_t>(m_value_to_category_index, value) *
                 reinterpret_cast<const fixed_dim_type_arrmeta *>(m_categories.get()->metadata())->stride;
    }
    /** Returns the arrmeta corresponding to data from
     * get_category_data_from_value */
    const char *get_category_arrmeta() const;

    /** Returns the categories as an immutable nd::array */
    nd::array get_categories() const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const intrusive_ptr<memory_block_data> &embedded_reference) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const;

    intptr_t make_assignment_kernel(void *ckb, intptr_t ckb_offset, const type &dst_tp, const char *dst_arrmeta,
                                    const type &src_tp, const char *src_arrmeta, kernel_request_t kernreq,
                                    const eval::eval_context *ectx) const;

    void get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties,
                                      size_t *out_count) const;
    void get_dynamic_type_properties(const std::pair<std::string, nd::callable> **out_properties,
                                     size_t *out_count) const;

    friend struct assign_to_same_category_type;
    friend struct assign_from_same_category_type;
    friend struct assign_from_commensurate_category_type;

    static type make(const nd::array &values)
    {
      return type(new categorical_type(values), false);
    }
  };

  DYND_API type factor_categorical(const nd::array &values);

} // namespace dynd::ndt
} // namespace dynd
