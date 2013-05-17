//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BASE_DTYPE_HPP_
#define _DYND__BASE_DTYPE_HPP_

#include <vector>

#include <dynd/config.hpp>
#include <dynd/atomic_refcount.hpp>
#include <dynd/irange.hpp>
#include <dynd/memblock/memory_block.hpp>
#include <dynd/kernels/kernel_instance.hpp>
#include <dynd/kernels/comparison_kernels.hpp>
#include <dynd/dtypes/type_id.hpp>
#include <dynd/dtype_assign.hpp>

namespace dynd {

// Forward definition from dynd/gfunc/callable.hpp
namespace gfunc {
    class callable;
};

// Forward definition from dynd/dtype.hpp
class dtype;
class base_dtype;
class assignment_kernel;

struct iterdata_common;

/** This is the callback function type used by the base_dtype::foreach function */
typedef void (*foreach_fn_t)(const dtype &dt, char *data, const char *metadata, void *callback_data);

/**
 * This is the iteration increment function used by iterdata. It increments the
 * iterator at the specified level, resetting all the more inner levels to 0.
 */
typedef char * (*iterdata_increment_fn_t)(iterdata_common *iterdata, size_t level);
/**
 * This is the reset function which is called when an outer dimension
 * increment resets all the lower dimensions to index 0. It returns
 * the data pointer for the next inner level of iteration.
 */
typedef char * (*iterdata_reset_fn_t)(iterdata_common *iterdata, char *data, size_t ndim);

/**
 * This is a generic function which applies a transformation to a dtype.
 * Usage of the function pointer is typically paired with the
 * base_dtype::transform_child_dtypes virtual function on the dtype
 *
 * An implementation of this function should either copy 'dt' into
 * 'out_transformed_dtype', and leave 'out_was_transformed' alone, or it
 * should place a different dtype in 'out_transformed_dtype', then set
 * 'out_was_transformed' to true.
 */
typedef void (*dtype_transform_fn_t)(const dtype& dt, void *extra,
                dtype& out_transformed_dtype, bool& out_was_transformed);

// Common preamble of all iterdata instances
struct iterdata_common {
    // This increments the iterator at the requested level
    iterdata_increment_fn_t incr;
    // This resets the data pointers of the iterator
    iterdata_reset_fn_t reset;
};

struct base_dtype_members {
    typedef uint32_t flags_type;

    /** The dtype's type id (type_id_t is the enum) */
    uint16_t type_id;
    /** The dtype's kind (dtype_kind_t is the enum) */
    uint8_t kind;
    /** The dtype's data alignment */
    uint8_t alignment;
    /** The dtype's flags */
    flags_type flags;
    /** The size of one instance of the dtype, or 0 if there is not one fixed size. */
    size_t data_size;
    /** The size of a metadata instance for the dtype. */
    size_t metadata_size;
    /** The number of uniform dimensions this dtype has */
    uint8_t undim;

    base_dtype_members(uint16_t type_id_, uint8_t kind_, uint8_t alignment_,
                    flags_type flags_, size_t data_size_, size_t metadata_size_, uint8_t undim_)
        : type_id(type_id_), kind(kind_), alignment(alignment_), flags(flags_),
                data_size(data_size_), metadata_size(metadata_size_), undim(undim_)
    {}
};

/**
 * This is the virtual base class for defining new dtypes which are not so basic
 * that we want them in the small list of builtin dtypes. This is a reference
 * counted class, and is immutable, so once an base_dtype instance is constructed,
 * it should never be modified.
 *
 * Typically, the base_dtype is used by manipulating a dtype instance, which acts
 * as a smart pointer to base_dtype, which special handling for the builtin types.
 */
class base_dtype {
    /** Embedded reference counting */
    mutable atomic_refcount m_use_count;
protected:
    /// Standard dtype data
    base_dtype_members m_members;
    

protected:
    // Helper function for uniform dimension dtypes
    void get_nonuniform_ndobject_properties_and_functions(
                    std::vector<std::pair<std::string, gfunc::callable> >& out_properties,
                    std::vector<std::pair<std::string, gfunc::callable> >& out_functions) const;
public:
    typedef base_dtype_members::flags_type flags_type;

    /** Starts off the extended dtype instance with a use count of 1. */
    inline base_dtype(type_id_t type_id, dtype_kind_t kind, size_t data_size,
                    size_t alignment, flags_type flags, size_t metadata_size, size_t undim)
        : m_use_count(1), m_members(static_cast<uint16_t>(type_id), static_cast<uint8_t>(kind),
                static_cast<uint8_t>(alignment), flags, data_size, metadata_size, static_cast<uint8_t>(undim))
    {}

    virtual ~base_dtype();

    /** For debugging purposes, the dtype's use count */
    inline int32_t get_use_count() const {
        return m_use_count;
    }

    /** Returns the struct of data common to all dtypes. */
    inline const base_dtype_members& get_base_dtype_members() const {
        return m_members;
    }

    /** The dtype's type id */
    inline type_id_t get_type_id() const {
        return static_cast<type_id_t>(m_members.type_id);
    }
    /** The dtype's kind */
    inline dtype_kind_t get_kind() const {
        return static_cast<dtype_kind_t>(m_members.kind);
    }
    /** The size of one instance of the dtype, or 0 if there is not one fixed size. */
    inline size_t get_data_size() const {
        return m_members.data_size;
    }
    /** The dtype's data alignment. Every data pointer for this dtype _must_ be aligned. */
    inline size_t get_alignment() const {
        return m_members.alignment;
    }
    /** The number of uniform dimensions this dtype has */
    inline size_t get_undim() const {
        return m_members.undim;
    }
    inline base_dtype_members::flags_type get_flags() const {
        return m_members.flags;
    }
    virtual size_t get_default_data_size(size_t ndim, const intptr_t *shape) const;

    /**
     * Print the raw data interpreted as a single instance of this dtype.
     *
     * \param o  The std::ostream to print to.
     * \param metadata  Pointer to the dtype metadata of the data element to print.
     * \param data  Pointer to the data element to print.
     */
    virtual void print_data(std::ostream& o, const char *metadata, const char *data) const = 0;

    /**
     * Print a representation of the dtype itself
     *
     * \param o  The std::ostream to print to.
     */
    virtual void print_dtype(std::ostream& o) const = 0;

    /**
     * Returns true if the dtype is a scalar.
     *
     * This precludes a dynamic dtype from switching between scalar and array behavior,
     * but the simplicity seems to probably be worth it.
     */
    inline bool is_scalar() const {
        return (m_members.flags & dtype_flag_scalar) != 0;
    }

    /**
     * Returns true if the dtype contains an expression dtype anywhere within it.
     */
    virtual bool is_expression() const;

    /**
     * Should return true if there is no additional blockref which might point
     * to data not owned by the metadata. For example, a blockref which points
     * to an 'external' memory block does not own its data uniquely.
     */
    virtual bool is_unique_data_owner(const char *metadata) const;

    /**
     * Applies the transform function to all the child dtypes, creating
     * a new dtype of the same type but with the transformed children.
     *
     * \param transform_fn  The function for transforming dtypes.
     * \param extra  Extra data to pass to the transform function
     * \param out_transformed_dtype  The transformed dtype is placed here.
     * \param out_was_transformed  Is set to true if a transformation was done,
     *                             is left alone otherwise.
     */
    virtual void transform_child_dtypes(dtype_transform_fn_t transform_fn, void *extra,
                    dtype& out_transformed_dtype, bool& out_was_transformed) const;

    /**
     * Returns a modified dtype with all expression dtypes replaced with
     * their value dtypes, and dtypes replaced with "standard versions"
     * whereever appropriate. For example, an offset-based uniform array
     * would be replaced by a strided uniform array.
     */
    virtual dtype get_canonical_dtype() const;

    /**
     * Returns true if this level of the dtype can be processed as an
     * origin pointer, a stride, and a size.
     */
    virtual bool is_strided() const;

    /**
     * When is_strided() returns true, this function can be used to
     * get the striding parameters for a given metadata/data instance
     * of the dtype.
     */
    virtual void process_strided(const char *metadata, const char *data,
                    dtype& out_dt, const char *&out_origin,
                    intptr_t& out_stride, intptr_t& out_dim_size) const;

    /**
     * Indexes into the dtype. This function returns the dtype which results
     * from applying the same index to an ndarray of this dtype.
     *
     * \param nindices     The number of elements in the 'indices' array. This is shrunk by one for each recursive call.
     * \param indices      The indices to apply. This is incremented by one for each recursive call.
     * \param current_i    The current index position. Used for error messages.
     * \param root_dt      The data type in the first call, before any recursion. Used for error messages.
     * \param leading_dimension  If this is true, the current dimension is one for which there is only a single
     *                           data instance, and the dtype can do operations relying on the data. An example
     *                           of this is a pointer data throwing away the pointer part, so the result
     *                           doesn't contain that indirection.
     */
    virtual dtype apply_linear_index(size_t nindices, const irange *indices,
                size_t current_i, const dtype& root_dt, bool leading_dimension) const;

    /**
     * Indexes into an ndobject using the provided linear index, and a dtype and freshly allocated output
     * set to point to the same base data reference.
     *
     * \param nindices     The number of elements in the 'indices' array. This is shrunk by one for each recursive call.
     * \param indices      The indices to apply. This is incremented by one for each recursive call.
     * \param metadata     The metadata of the input array.
     * \param result_dtype The result of an apply_linear_index call.
     * \param out_metadata The metadata of the output array. The output data should all be references to the data
     *                     of the input array, so there is no out_data parameter.
     * \param embedded_reference  For references which are NULL, add this reference in the output.
     *                            A NULL means the data was embedded in the original ndobject, so
     *                            when putting it in a new ndobject, need to hold a reference to
     *                            that memory.
     * \param current_i    The current index position. Used for error messages.
     * \param root_dt      The data type in the first call, before any recursion.
     *                     Used for error messages.
     * \param leading_dimension  If this is true, the current dimension is one for
     *                           which there is only a single data instance, and
     *                           the dtype can do operations relying on the data.
     *                           An example of this is a pointer data throwing away
     *                           the pointer part, so the result doesn't contain
     *                           that indirection.
     * \param inout_data  This may *only* be used/modified if leading_dimension
     *                    is true. In the case of eliminating a pointer, this is
     *                    a pointer to the pointer data. The pointer dtype would
     *                    dereference the pointer data, and modify both the data
     *                    pointer and the data reference to reflect that change.
     * \param inout_dataref  This may only be used/modified if leading_dimension
     *                       is true. If the target of inout_data is modified, then
     *                       in many cases the data will be pointing into a different
     *                       memory block than before. This must be modified to
     *                       be a reference to the updated memory block.
     *
     * @return  An offset to apply to the data pointer(s).
     */
    virtual intptr_t apply_linear_index(size_t nindices, const irange *indices, const char *metadata,
                    const dtype& result_dtype, char *out_metadata,
                    memory_block_data *embedded_reference,
                    size_t current_i, const dtype& root_dt,
                    bool leading_dimension, char **inout_data,
                    memory_block_data **inout_dataref) const;

    /**
     * The 'at' function is used for indexing. Indexing one dimension with
     * an integer index is special-cased, both for higher performance and
     * to provide a way to get a metadata pointer for the result dtype.
     *
     * \param i0  The index to apply.
     * \param inout_metadata  If non-NULL, points to a metadata pointer for
     *                        this dtype that is modified to point to the
     *                        result's metadata.
     * \param inout_data  If non-NULL, points to a data pointer that is modified
     *                    to point to the result's data. If `inout_data` is non-NULL,
     *                    `inout_metadata` must also be non-NULL.
     *
     * \returns  The dtype that results from the indexing operation.
     */
    virtual dtype at_single(intptr_t i0, const char **inout_metadata, const char **inout_data) const;

    /**
     * Retrieves the dtype starting at the requested dimension. This is
     * generally equivalent to apply_linear_index with a count of 'dim'
     * scalar indices.
     *
     * \param inout_metadata  NULL to ignore, or point it at some metadata for the dtype,
     *                        and it will be updated to point to the metadata for the returned
     *                        dtype.
     * \param i         The dimension number to retrieve.
     * \param total_ndim  A count of how many dimensions have been traversed from the
     *                    dtype start, for producing error messages.
     */
    virtual dtype get_dtype_at_dimension(char **inout_metadata, size_t i, size_t total_ndim = 0) const;

    /**
     * Retrieves the shape of the dtype, expanding the vector as needed. For dimensions with
     * unknown or variable shape, -1 is returned.
     *
     * The output must be pre-initialized to have get_undim() elements.
     */
    virtual void get_shape(size_t i, intptr_t *out_shape) const;

    /**
     * Retrieves the shape of the dtype ndobject instance,
     expanding the vector as needed. For dimensions with
     * variable shape, -1 is returned.
     *
     * The output must be pre-initialized to have get_undim() elements.
     */
    virtual void get_shape(size_t i, intptr_t *out_shape, const char *metadata) const;

    /**
     * Retrieves the strides of the dtype ndobject instance,
     * expanding the vector as needed. For dimensions where
     * there is not a simple stride (e.g. a tuple/struct dtype),
     * 0 is returned and the caller should handle this.
     *
     * The output must be pre-initialized to have get_undim() elements.
     */
    virtual void get_strides(size_t i, intptr_t *out_strides, const char *metadata) const;

    /**
     * Classifies the order the axes occur in the memory
     * layout of the array.
     */
    virtual axis_order_classification_t classify_axis_order(const char *metadata) const;

    /**
     * Called by ::dynd::is_lossless_assignment, with (this == dst_dt->extended()).
     */
    virtual bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    virtual bool operator==(const base_dtype& rhs) const = 0;

    /** The size of the ndobject metadata for this dtype */
    inline size_t get_metadata_size() const {
        return m_members.metadata_size;
    }
    /**
     * Constructs the ndobject metadata for this dtype, prepared for writing.
     * The element size of the result must match that from get_default_data_size().
     */
    virtual void metadata_default_construct(char *metadata, size_t ndim, const intptr_t* shape) const;
    /**
     * Constructs the ndobject metadata for this dtype, copying everything exactly from
     * input metadata for the same dtype.
     *
     * \param dst_metadata  The new metadata memory which is constructed.
     * \param src_metadata   Existing metadata memory from which to copy.
     * \param embedded_reference  For references which are NULL, add this reference in the output.
     *                            A NULL means the data was embedded in the original ndobject, so
     *                            when putting it in a new ndobject, need to hold a reference to
     *                            that memory.
     */
    virtual void metadata_copy_construct(char *dst_metadata, const char *src_metadata,
                    memory_block_data *embedded_reference) const;
    /** Destructs any references or other state contained in the ndobjects' metdata */
    virtual void metadata_destruct(char *metadata) const;
    /**
     * When metadata is used for temporary buffers of a dtype,
     * and that usage is finished one execution cycle, this function
     * is called to clear usage of that memory so it can be reused in
     * the next cycle.
     */
    virtual void metadata_reset_buffers(char *metadata) const;
    /**
     * For blockref dtypes, once all the elements have been written
     * we want to turn off further memory allocation, and possibly
     * trim excess memory that was allocated. This function
     * does this.
     */
    virtual void metadata_finalize_buffers(char *metadata) const;
    /** Debug print of the metdata */
    virtual void metadata_debug_print(const char *metadata, std::ostream& o,
                    const std::string& indent) const;

    /**
     * For dtypes that have the flag dtype_flag_destructor set, this function
     * or the strided version is called to destruct data.
     */
    virtual void data_destruct(const char *metadata, char *data) const;
    /**
     * For dtypes that have the flag dtype_flag_destructor set, this function
     * or the non-strided version is called to destruct data.
     */
    virtual void data_destruct_strided(const char *metadata, char *data,
                    intptr_t stride, size_t count) const;

    /** The size of the data required for uniform iteration */
    virtual size_t get_iterdata_size(size_t ndim) const;
    /**
     * Constructs the iterdata for processing iteration at this level of the datashape
     */
    virtual size_t iterdata_construct(iterdata_common *iterdata,
                    const char **inout_metadata, size_t ndim,
                    const intptr_t* shape, dtype& out_uniform_dtype) const;
    /** Destructs any references or other state contained in the iterdata */
    virtual size_t iterdata_destruct(iterdata_common *iterdata, size_t ndim) const;

    /**
     * Creates an assignment kernel for one data value from the
     * src dtype/metadata to the dst dtype/metadata. This adds the
     * kernel at the 'out_offset' position in 'out's data, as part
     * of a hierarchy matching the dtype's hierarchy.
     *
     * This function should always be called with this == dst_dt first,
     * and dtypes which don't support the particular assignment should
     * then call the corresponding function with this == src_dt.
     *
     * \returns  The offset at the end of 'out' after adding this
     *           kernel.
     */
    virtual size_t make_assignment_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const dtype& dst_dt, const char *dst_metadata,
                    const dtype& src_dt, const char *src_metadata,
                    kernel_request_t kernreq, assign_error_mode errmode,
                    const eval::eval_context *ectx) const;

    /**
     * Creates a comparison kernel for one data value from one
     * dtype/metadata to another dtype/metadata. This adds the
     * kernel at the 'out_offset' position in 'out's data, as part
     * of a hierarchy matching the dtype's hierarchy.
     *
     * This function should always be called with this == src0_dt first,
     * and dtypes which don't support the particular comparison should
     * then call the corresponding function with this == src1_dt.
     *
     * \returns  The offset at the end of 'out' after adding this
     *           kernel.
     */
    virtual size_t make_comparison_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const dtype& src0_dt, const char *src0_metadata,
                    const dtype& src1_dt, const char *src1_metadata,
                    comparison_type_t comptype,
                    const eval::eval_context *ectx) const;

    /**
     * Call the callback on each element of the array with given data/metadata along the leading
     * dimension. For uniform dimensions, the dtype provided is the same each call, but for
     * heterogeneous dimensions it changes.
     *
     * \param data  The ndobject data.
     * \param metadata  The ndobject metadata.
     * \param callback  Callback function called for each subelement.
     * \param callback_data  Data provided to the callback function.
     */
    virtual void foreach_leading(char *data, const char *metadata,
                    foreach_fn_t callback, void *callback_data) const;

    /**
     * Additional dynamic properties exposed by the dtype as gfunc::callable.
     */
    virtual void get_dynamic_dtype_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;

    /**
     * Additional dynamic functions exposed by the dtype as gfunc::callable.
     */
    virtual void get_dynamic_dtype_functions(
                    const std::pair<std::string, gfunc::callable> **out_functions,
                    size_t *out_count) const;

    /**
     * Additional dynamic properties exposed by any ndobject of this dtype as gfunc::callable.
     *
     * \note Uniform dtypes copy these properties from the first non-uniform dtype, so such properties must
     *       be able to handle the case where they are the first non-uniform dtype in an array type, not
     *       just strictly of the non-uniform dtype.
     */
    virtual void get_dynamic_ndobject_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;

    /**
     * Additional dynamic functions exposed by any ndobject of this dtype as gfunc::callable.
     *
     * \note Uniform dtypes copy these functions from the first non-uniform dtype, so such properties must
     *       be able to handle the case where they are the first non-uniform dtype in an array type, not
     *       just strictly of the non-uniform dtype.
     */
    virtual void get_dynamic_ndobject_functions(
                    const std::pair<std::string, gfunc::callable> **out_functions,
                    size_t *out_count) const;

    /**
     * Returns the index for the element-wise property of the given name.
     *
     * \param property_name  The name of the element-wise property.
     *
     * \returns  The index of the property, to be provided to the other elwise_property
     *           functions.
     */
    virtual size_t get_elwise_property_index(const std::string& property_name) const;

    /**
     * Returns the dtype for the element-wise property of the given index.
     *
     * \param elwise_property_index  The index of the property, typically from
     *                               a call to get_elwise_property_index.
     * \param out_readable  The dtype should set this to true/false depending on
     *                      whether the property is readable.
     * \param out_writable  The dtype should set this to true/false depending on
     *                      whether the property is writable.
     *
     * \returns  The dtype of the property.
     */
    virtual dtype get_elwise_property_dtype(size_t elwise_property_index,
            bool& out_readable, bool& out_writable) const;

    /**
     * Returns a kernel to transform instances of this dtype into values of the
     * element-wise property.
     *
     * \param out  The hierarchical assignment kernel being constructed.
     * \param offset_out  The offset within 'out'.
     * \param dst_metadata  Metadata for the destination property being written to.
     * \param src_metadata  Metadata for the operand dtype being read from.
     * \param src_elwise_property_index  The index of the property, from get_elwise_property_index().
     * \param ectx  DyND evaluation context.
     */
    virtual size_t make_elwise_property_getter_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const char *dst_metadata,
                    const char *src_metadata, size_t src_elwise_property_index,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;

    /**
     * Returns a kernel to transform instances of the element-wise property
     * into values of this dtype.
     *
     * \note This can only be used when the source value sets all of the
     *       destination, not when it only sets part of it. For example,
     *       setting just the "year" property of a date is not possible
     *       with this mechanism.
     *
     * \param out  The hierarchical assignment kernel being constructed.
     * \param offset_out  The offset within 'out'.
     * \param dst_metadata  Metadata for the operand dtype being written to.
     * \param dst_elwise_property_index  The index of the property, from get_elwise_property_index().
     * \param src_metadata  Metadata for the source property being read from.
     * \param ectx  DyND evaluation contrext.
     */
    virtual size_t make_elwise_property_setter_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const char *dst_metadata, size_t dst_elwise_property_index,
                    const char *src_metadata,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;

    friend void base_dtype_incref(const base_dtype *ed);
    friend void base_dtype_decref(const base_dtype *ed);
};

/**
 * Increments the reference count of the dtype
 */
inline void base_dtype_incref(const base_dtype *bd)
{
    //std::cout << "dtype " << (void *)ed << " inc: " << ed->m_use_count + 1 << "\t"; ed->print_dtype(std::cout); std::cout << std::endl;
    ++bd->m_use_count;
}

/**
 * Checks if the dtype is builtin or not, and if not,
 * increments the reference count of the dtype.
 */
inline void base_dtype_xincref(const base_dtype *bd)
{
    if (!is_builtin_dtype(bd)) {
        base_dtype_incref(bd);
    }
}

/**
 * Decrements the reference count of the dtype,
 * freeing it if the count reaches zero.
 */
inline void base_dtype_decref(const base_dtype *bd)
{
    //std::cout << "dtype " << (void *)ed << " dec: " << ed->m_use_count - 1 << "\t"; ed->print_dtype(std::cout); std::cout << std::endl;
    if (--bd->m_use_count == 0) {
        delete bd;
    }
}

/**
 * Checks if the dtype is builtin or not, and if not,
 * decrements the reference count of the dtype,
 * freeing it if the count reaches zero.
 */
inline void base_dtype_xdecref(const base_dtype *bd)
{
    if (!is_builtin_dtype(bd)) {
        base_dtype_decref(bd);
    }
}

} // namespace dynd

#endif // _DYND__BASE_DTYPE_HPP_
