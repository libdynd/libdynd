//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <atomic>
#include <map>
#include <unordered_set>
#include <vector>

#include <dynd/irange.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/memblock/memory_block.hpp>
#include <dynd/types/type_id.hpp>

namespace dynd {
namespace ndt {

  class type;

} // namespace dynd::ndt

struct iterdata_common;

/** This is the callback function type used by the base_type::foreach function
 */
typedef void (*foreach_fn_t)(const ndt::type &dt, const char *arrmeta, char *data, void *callback_data);

/**
 * This is the iteration increment function used by iterdata. It increments the
 * iterator at the specified level, resetting all the more inner levels to 0.
 */
typedef char *(*iterdata_increment_fn_t)(iterdata_common *iterdata, intptr_t level);

/**
 * This is the iteration advance function used by iterdata. It advances the
 * iterator at the specified level by the specified amount, resetting all the
 * more inner levels to 0.
 */

typedef char *(*iterdata_advance_fn_t)(iterdata_common *iterdata, intptr_t level, intptr_t i);
/**
 * This is the reset function which is called when an outer dimension
 * increment resets all the lower dimensions to index 0. It returns
 * the data pointer for the next inner level of iteration.
 */

typedef char *(*iterdata_reset_fn_t)(iterdata_common *iterdata, char *data, intptr_t ndim);

/**
 * This is a generic function which applies a transformation to a type.
 * Usage of the function pointer is typically paired with the
 * base_type::transform_child_types virtual function on the type
 *
 * An implementation of this function should either copy 'dt' into
 * 'out_transformed_tp', and leave 'out_was_transformed' alone, or it
 * should place a different type in 'out_transformed_type', then set
 * 'out_was_transformed' to true.
 */
typedef void (*type_transform_fn_t)(const ndt::type &dt, intptr_t arrmeta_offset, void *extra,
                                    ndt::type &out_transformed_type, bool &out_was_transformed);

// Common preamble of all iterdata instances
struct DYNDT_API iterdata_common {
  // This increments the iterator at the requested level
  iterdata_increment_fn_t incr;
  // This advances the iterator at the requested level by the requested amount
  iterdata_advance_fn_t adv;
  // This resets the data pointers of the iterator
  iterdata_reset_fn_t reset;
};

namespace ndt {

  /**
   * This is the virtual base class for defining new types which are not so
   * basic that we want them in the small list of builtin types. This is a reference
   * counted class, and is immutable, so once a base_type instance is constructed,
   * it should never be modified.
   *
   * Typically, the base_type is used by manipulating a type instance, which acts
   * as a smart pointer to base_type, which special handling for the builtin types.
   */
  class DYNDT_API base_type {
    /** Embedded reference counting */
    mutable std::atomic_long m_use_count;

  protected:
    type_id_t m_id;          // The type id
    size_t m_metadata_size;  // The size of a arrmeta instance for the type.
    size_t m_data_size;      // The size of one instance of the type, or 0 if there is not one fixed size.
    size_t m_data_alignment; // The data alignment
    uint32_t flags;          // The flags.
    intptr_t m_ndim;         // The number of array dimensions this type has
    intptr_t m_fixed_ndim;   // The number of fixed dimensions in a row with no var dims, pointers, etc in between

  public:
    /** Starts off the extended type instance with a use count of 1. */
    base_type(type_id_t id, size_t data_size, size_t data_alignment, uint32_t flags, size_t arrmeta_size, size_t ndim,
              size_t strided_ndim)
        : m_use_count(1), m_id(id), m_metadata_size(arrmeta_size), m_data_size(data_size),
          m_data_alignment(data_alignment), flags(flags), m_ndim(ndim), m_fixed_ndim(strided_ndim)
    {
    }

    virtual ~base_type();

    /** For debugging purposes, the type's use count */
    int32_t get_use_count() const { return m_use_count; }

    /**
      * The type's id.
      */
    type_id_t get_id() const { return m_id; }

    /** The size of the nd::array arrmeta for this type */
    size_t get_arrmeta_size() const { return m_metadata_size; }

    /** The size of one instance of the type, or 0 if there is not one fixed
     * size. */
    size_t get_data_size() const { return m_data_size; }

    /** The type's data alignment. Every data pointer for this type _must_ be
     * aligned. */
    size_t get_data_alignment() const { return m_data_alignment; }

    /** The number of array dimensions this type has */
    intptr_t get_ndim() const { return m_ndim; }

    /** The number of outer strided dimensions this type has in a row */
    intptr_t get_strided_ndim() const { return m_fixed_ndim; }

    uint32_t get_flags() const { return flags; }

    virtual size_t get_default_data_size() const;

    /**
     * Print a representation of the type itself
     *
     * \param o  The std::ostream to print to.
     */
    virtual void print_type(std::ostream &o) const = 0;

    /**
     * Print the raw data interpreted as a single instance of this type.
     *
     * \param o  The std::ostream to print to.
     * \param arrmeta  Pointer to the type arrmeta of the data element to print.
     * \param data  Pointer to the data element to print.
     */
    virtual void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    inline bool is_indexable() const { return (flags & type_flag_indexable) != 0; }

    /**
     * Returns true if the type is a scalar.
     *
     * This precludes a dynamic type from switching between scalar and array
     *behavior,
     * but the simplicity seems to probably be worth it.
     */
    bool is_scalar() const { return m_ndim == 0 && (flags & type_flag_variadic) == 0; }

    /**
     * Returns true if the given type is a subarray of this type.
     * For example, "int" is a subarray of "strided, int". This
     * relationship may exist for unequal types with the same number
     * of dimensions, for example "int" is a subarray of "pointer(int)".
     *
     * \param subarray_tp  Testing if it is a subarray of 'this'.
     */
    virtual bool is_type_subarray(const ndt::type &subarray_tp) const;

    /**
     * Returns true if the type contains an expression type anywhere within it.
     */
    virtual bool is_expression() const;

    /**
     * Should return true if there is no additional blockref which might point
     * to data not owned by the arrmeta. For example, a blockref which points
     * to an 'external' memory block does not own its data uniquely.
     */
    virtual bool is_unique_data_owner(const char *arrmeta) const;

    /**
     * Applies the transform function to all the child types, creating
     * a new type of the same type but with the transformed children.
     *
     * \param transform_fn  The function for transforming types.
     * \param arrmeta_offset  An offset for arrmeta corresponding to the
     *                        type. This is adjusted and passed to the
     *                        transform_fn for each child type's arrmeta_offset.
     * \param extra  Extra data to pass to the transform function
     * \param out_transformed_type  The transformed type is placed here.
     * \param out_was_transformed  Is set to true if a transformation was done,
     *                             is left alone otherwise.
     */
    virtual void transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                                       ndt::type &out_transformed_type, bool &out_was_transformed) const;

    /**
     * Returns a modified type with all expression types replaced with
     * their value types, and types replaced with "standard versions"
     * whereever appropriate. For example, an offset-based uniform array
     * would be replaced by a strided uniform array.
     */
    virtual ndt::type get_canonical_type() const;

    /** Sets the value from a UTF8 string */
    virtual void set_from_utf8_string(const char *arrmeta, char *data, const char *utf8_begin, const char *utf8_end,
                                      const eval::eval_context *ectx) const;
    /** Copies a C++ std::string with a UTF8 encoding to a string element */
    inline void set_from_utf8_string(const char *arrmeta, char *data, const std::string &utf8_str,
                                     const eval::eval_context *ectx) const
    {
      this->set_from_utf8_string(arrmeta, data, utf8_str.data(), utf8_str.data() + utf8_str.size(), ectx);
    }

    /**
     * Indexes into the type. This function returns the type which results
     * from applying the same index to an ndarray of this type.
     *
     * \param nindices     The number of elements in the 'indices' array. This
     *                     is shrunk by one for each recursive call.
     * \param indices      The indices to apply. This is incremented by one for
     *                     each recursive call.
     * \param current_i    The current index position. Used for error messages.
     * \param root_tp      The data type in the first call, before any
     *                     recursion. Used for error messages.
     * \param leading_dimension  If this is true, the current dimension is one
     *                           for which there is only a single
     *                           data instance, and the type can do operations
     *                           relying on the data. An example
     *                           of this is a pointer data throwing away the
     *                           pointer part, so the result
     *                           doesn't contain that indirection.
     */
    virtual ndt::type apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i,
                                         const ndt::type &root_tp, bool leading_dimension) const;

    /**
     * Indexes into an nd::array using the provided linear index, and a type and
     *freshly allocated output
     * set to point to the same base data reference.
     *
     * \param nindices     The number of elements in the 'indices' array. This
     *is shrunk by one for each recursive call.
     * \param indices      The indices to apply. This is incremented by one for
     *each recursive call.
     * \param arrmeta     The arrmeta of the input array.
     * \param result_type The result of an apply_linear_index call.
     * \param out_arrmeta The arrmeta of the output array. The output data
     *should all be references to the data
     *                     of the input array, so there is no out_data
     *parameter.
     * \param embedded_reference  For references which are NULL, add this
     *reference in the output.
     *                            A NULL means the data was embedded in the
     *original nd::array, so
     *                            when putting it in a new nd::array, need to
     *hold a reference to
     *                            that memory.
     * \param current_i    The current index position. Used for error messages.
     * \param root_tp      The data type in the first call, before any
     *recursion.
     *                     Used for error messages.
     * \param leading_dimension  If this is true, the current dimension is one
     *for
     *                           which there is only a single data instance, and
     *                           the type can do operations relying on the data.
     *                           An example of this is a pointer data throwing
     *away
     *                           the pointer part, so the result doesn't contain
     *                           that indirection.
     * \param inout_data  This may *only* be used/modified if leading_dimension
     *                    is true. In the case of eliminating a pointer, this is
     *                    a pointer to the pointer data. The pointer type would
     *                    dereference the pointer data, and modify both the data
     *                    pointer and the data reference to reflect that change.
     * \param inout_dataref  This may only be used/modified if leading_dimension
     *                       is true. If the target of inout_data is modified,
     *then
     *                       in many cases the data will be pointing into a
     *different
     *                       memory block than before. This must be modified to
     *                       be a reference to the updated memory block.
     *
     * @return  An offset to apply to the data pointer(s).
     */
    virtual intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta,
                                        const ndt::type &result_type, char *out_arrmeta,
                                        const intrusive_ptr<memory_block_data> &embedded_reference, size_t current_i,
                                        const ndt::type &root_tp, bool leading_dimension, char **inout_data,
                                        intrusive_ptr<memory_block_data> &inout_dataref) const;

    /**
     * The 'at' function is used for indexing. Indexing one dimension with
     * an integer index is special-cased, both for higher performance and
     * to provide a way to get a arrmeta pointer for the result type.
     *
     * \param i0  The index to apply.
     * \param inout_arrmeta  If non-NULL, points to a arrmeta pointer for
     *                        this type that is modified to point to the
     *                        result's arrmeta.
     * \param inout_data  If non-NULL, points to a data pointer that is modified
     *                    to point to the result's data. If `inout_data` is
     *non-NULL,
     *                    `inout_arrmeta` must also be non-NULL.
     *
     * \returns  The type that results from the indexing operation.
     */
    virtual ndt::type at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const;

    /**
     * Retrieves the type starting at the requested dimension. This is
     * generally equivalent to apply_linear_index with a count of 'dim'
     * scalar indices.
     *
     * \param inout_arrmeta  NULL to ignore, or point it at some arrmeta for the
     *type,
     *                        and it will be updated to point to the arrmeta for
     *the returned
     *                        type.
     * \param i         The dimension number to retrieve.
     * \param total_ndim  A count of how many dimensions have been traversed
     *from the
     *                    type start, for producing error messages.
     */
    virtual ndt::type get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim = 0) const;

    /**
     * Retrieves the shape of the type nd::array instance,
     * populating up to 'ndim' elements of out_shape. For dimensions with
     * variable or unknown shape, -1 is returned.
     *
     * The 'arrmeta' may be NULL, in which case -1 should be used when
     * the shape cannot be determined.
     * The 'data' may be NULL, and only gets fed deeper when an element
     * is unique (i.e. the dimension size is 1, it's a pointer type, etc).
     *
     * The output must be pre-initialized to have 'ndim' elements.
     */
    virtual void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;

    /**
     * Retrieves the strides of the type nd::array instance,
     * expanding the vector as needed. For dimensions where
     * there is not a simple stride (e.g. a tuple/struct type),
     * 0 is returned and the caller should handle this.
     *
     * The output must be pre-initialized to have get_ndim() elements.
     */
    virtual void get_strides(size_t i, intptr_t *out_strides, const char *arrmeta) const;

    virtual bool is_c_contiguous(const char *arrmeta) const;

    bool is_symbolic() const { return (flags & type_flag_symbolic) != 0; }

    /**
     * Classifies the order the axes occur in the memory
     * layout of the array.
     */
    virtual axis_order_classification_t classify_axis_order(const char *arrmeta) const;

    /**
     * Called by ::dynd::is_lossless_assignment, with (this ==
     * dst_tp->extended()).
     */
    virtual bool is_lossless_assignment(const ndt::type &dst_tp, const ndt::type &src_tp) const;

    virtual bool operator==(const base_type &rhs) const = 0;

    /**
     * Constructs the nd::array arrmeta for this type using default settings.
     * The element size of the result must match that from
     * get_default_data_size().
     *
     * \param arrmeta  The arrmeta to default construct.
     * \param blockref_alloc  If ``true``, blockref types should allocate
     *                        writable memory blocks, and if ``false``, they
     *                        should set their blockrefs to NULL. The latter
     *                        indicates the blockref memory is owned by
     *                        the parent nd::array, and is useful for viewing
     *                        external memory with compatible layout.
     */
    virtual void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    /**
     * Constructs the nd::array arrmeta for this type, copying everything
     *exactly from
     * input arrmeta for the same type.
     *
     * \param dst_arrmeta  The new arrmeta memory which is constructed.
     * \param src_arrmeta   Existing arrmeta memory from which to copy.
     * \param embedded_reference  For references which are NULL, add this reference in the output.
     *                            A NULL means the data was embedded in the original nd::array, so
     *                            when putting it in a new nd::array, need to hold a reference to
     *                            that memory.
     */
    virtual void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                        const intrusive_ptr<memory_block_data> &embedded_reference) const;

    /** Destructs any references or other state contained in the nd::arrays'
     * arrmeta */
    virtual void arrmeta_destruct(char *arrmeta) const;
    /**
     * When arrmeta is used for temporary buffers of a type,
     * and that usage is finished one execution cycle, this function
     * is called to clear usage of that memory so it can be reused in
     * the next cycle.
     */
    virtual void arrmeta_reset_buffers(char *arrmeta) const;
    /**
     * For blockref types, once all the elements have been written
     * we want to turn off further memory allocation, and possibly
     * trim excess memory that was allocated. This function
     * does this.
     */
    virtual void arrmeta_finalize_buffers(char *arrmeta) const;
    /** Debug print of the metdata */
    virtual void arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const;

    /**
     * For types that have the flag type_flag_constructor set, this function
     * or the strided version is called to construct data.
     */
    virtual void data_construct(const char *arrmeta, char *data) const;

    /**
     * For types that have the flag type_flag_destructor set, this function
     * or the strided version is called to destruct data.
     */
    virtual void data_destruct(const char *arrmeta, char *data) const;
    /**
     * For types that have the flag type_flag_destructor set, this function
     * or the non-strided version is called to destruct data.
     */
    virtual void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const;

    /** The size of the data required for uniform iteration */
    virtual size_t get_iterdata_size(intptr_t ndim) const;
    /**
     * Constructs the iterdata for processing iteration at this level of the
     * datashape
     */
    virtual size_t iterdata_construct(iterdata_common *iterdata, const char **inout_arrmeta, intptr_t ndim,
                                      const intptr_t *shape, ndt::type &out_uniform_tp) const;
    /** Destructs any references or other state contained in the iterdata */
    virtual size_t iterdata_destruct(iterdata_common *iterdata, intptr_t ndim) const;

    virtual bool match(const ndt::type &candidate_tp, std::map<std::string, ndt::type> &tp_vars) const;

    /**
     * Call the callback on each element of the array with given data/arrmeta
     *along the leading
     * dimension. For array dimensions, the type provided is the same each call,
     *but for
     * heterogeneous dimensions it changes.
     *
     * \param arrmeta  The arrmeta.
     * \param data  The nd::array data.
     * \param callback  Callback function called for each subelement.
     * \param callback_data  Data provided to the callback function.
     */
    virtual void foreach_leading(const char *arrmeta, char *data, foreach_fn_t callback, void *callback_data) const;

    virtual void get_vars(std::unordered_set<std::string> &DYND_UNUSED(vars)) const {}

    /**
     * Additional dynamic properties exposed by the type.
     */
    virtual std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;

    friend void intrusive_ptr_retain(const base_type *ptr);
    friend void intrusive_ptr_release(const base_type *ptr);
    friend long intrusive_ptr_use_count(const base_type *ptr);

    friend type make_dynamic_type(type_id_t tp_id);
  };

  /**
   * Checks if the type is builtin or not, and if not,
   * increments the reference count of the type.
   */
  inline void intrusive_ptr_retain(const base_type *ptr)
  {
    if (!is_builtin_type(ptr)) {
      ++ptr->m_use_count;
    }
  }

  /**
   * Checks if the type is builtin or not, and if not,
   * decrements the reference count of the type,
   * freeing it if the count reaches zero.
   */
  inline void intrusive_ptr_release(const base_type *ptr)
  {
    if (!is_builtin_type(ptr)) {
      if (--ptr->m_use_count == 0) {
        delete ptr;
      }
    }
  }

  inline long intrusive_ptr_use_count(const base_type *ptr) { return ptr->m_use_count; }

} // namespace dynd::ndt

/**
 * A pair of values describing the parameters of a single
 * strided dimension. When a type ``tp`` describes a multi-dimensional
 * strided array, its arrmeta always begins with an array
 * of ``size_stride_t`` with length ``tp.get_strided_ndim()``.
 */
struct DYNDT_API size_stride_t {
  intptr_t dim_size;
  intptr_t stride;
};

} // namespace dynd
