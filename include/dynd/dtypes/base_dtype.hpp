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
#include <dynd/dtype_assign.hpp>


namespace dynd {

// Forward definition from dynd/gfunc/callable.hpp
namespace gfunc {
    class callable;
};

// Forward definition from dynd/dtype.hpp
class dtype;

enum dtype_kind_t {
    bool_kind,
    int_kind,
    uint_kind,
    real_kind,
    complex_kind,
    // string_kind means subclass of base_string_dtype
    string_kind,
    bytes_kind,
    void_kind,
    datetime_kind,
    // For any array dtypes which have elements of all the same type
    uniform_array_kind,
    // For struct_type_id and fixedstruct_type_id
    struct_kind,
    // For dtypes whose value_dtype != the dtype, signals
    // that calculations should look at the value_dtype for
    // type promotion, etc.
    expression_kind,
    // For pattern-matching dtypes
    pattern_kind,
    // For use when it becomes possible to register custom dtypes
    custom_kind
};

enum type_id_t {
    // The value zero is reserved for an uninitialized dtype.
    uninitialized_type_id,
    // A 1-byte boolean type
    bool_type_id,
    // Signed integer types
    int8_type_id,
    int16_type_id,
    int32_type_id,
    int64_type_id,
    // Unsigned integer types
    uint8_type_id,
    uint16_type_id,
    uint32_type_id,
    uint64_type_id,
    // Floating point types
    float32_type_id,
    float64_type_id,
    // Complex floating-point types
    complex_float32_type_id,
    complex_float64_type_id,
    // Means no type, just like in C. (Different from NumPy)
    void_type_id,
    // Like C/C++ (void*), the storage of pointer_dtype
    void_pointer_type_id,

    // A pointer type
    pointer_type_id,

    // blockref primitive dtypes
    bytes_type_id,
    // A bytes buffer of a fixed size
    fixedbytes_type_id,

    // A variable-sized string type
    string_type_id,
    // A NULL-terminated string buffer of a fixed size
    fixedstring_type_id,

    // A categorical (enum-like) type
    categorical_type_id,
    // A 32-bit date type
    date_type_id,
    // A 32-bit date type limited to business days
    busdate_type_id,
    // A UTF-8 encoded string type for holding JSON
    json_type_id,

    // A strided array dimension type (like NumPy)
    strided_array_type_id,
    // A fixed-sized array dimension type
    fixedarray_type_id,
    // A variable-sized array dimension type
    var_array_type_id,

    // A struct type with variable layout
    struct_type_id,
    // A struct type with fixed layout
    fixedstruct_type_id,
    tuple_type_id,
    ndobject_type_id,

    // Adapter dtypes
    convert_type_id,
    byteswap_type_id,
    align_type_id,
    view_type_id,

    // A type for holding property access in the 'date' type (will generalize to property_type_id)
    date_property_type_id,

    // Advanced expression dtypes
    groupby_type_id,

    // The number of built-in, atomic types (including uninitialized and void)
    builtin_type_id_count = 15
};

enum {
    /** A mask within which alll the built-in type ids are guaranteed to fit */
    builtin_type_id_mask = 0x1f
};

enum dtype_memory_management_t {
    /** The dtype's memory is POD (plain old data) */
    pod_memory_management,
    /** The dtype contains pointers into another memory_block */
    blockref_memory_management,
    /** The dtype requires full object lifetime management (construct/copy/move/destroy) */
    object_memory_management
};

struct iterdata_common;

/** This is the callback function type used by the base_dtype::foreach function */
typedef void (*foreach_fn_t)(const dtype &dt, char *data, const char *metadata, void *callback_data);

/**
 * This is the iteration increment function used by iterdata. It increments the
 * iterator at the specified level, resetting all the more inner levels to 0.
 */
typedef char * (*iterdata_increment_fn_t)(iterdata_common *iterdata, int level);
/**
 * This is the reset function which is called when an outer dimension
 * increment resets all the lower dimensions to index 0. It returns
 * the data pointer for the next inner level of iteration.
 */
typedef char * (*iterdata_reset_fn_t)(iterdata_common *iterdata, char *data, int ndim);

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
typedef void (*dtype_transform_fn_t)(const dtype& dt, const void *extra,
                dtype& out_transformed_dtype, bool& out_was_transformed);

// Common preamble of all iterdata instances
struct iterdata_common {
    // This increments the iterator at the requested level
    iterdata_increment_fn_t incr;
    // This resets the data pointers of the iterator
    iterdata_reset_fn_t reset;
};

struct base_dtype_members {
    /** The dtype's type id (type_id_t is the enum) */
    uint16_t type_id;
    /** The dtype's kind (dtype_kind_t is the enum) */
    uint8_t kind;
    /** The dtype's data alignment */
    uint8_t alignment;
    /** The size of one instance of the dtype, or 0 if there is not one fixed size. */
    size_t data_size;
    /** The number of uniform dimensions this dtype has */
    uint8_t undim;

    base_dtype_members(uint16_t type_id_, uint8_t kind_, uint8_t alignment_,
                    size_t data_size_, uint8_t undim_)
        : type_id(type_id_), kind(kind_), alignment(alignment_),
                data_size(data_size_), undim(undim_)
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
    /** Starts off the extended dtype instance with a use count of 1. */
    inline base_dtype(type_id_t type_id, dtype_kind_t kind, size_t data_size, size_t alignment, size_t undim=0)
        : m_use_count(1), m_members(static_cast<uint16_t>(type_id), static_cast<uint8_t>(kind),
                static_cast<uint8_t>(alignment), data_size, static_cast<uint8_t>(undim))
    {}

    virtual ~base_dtype();

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
    virtual size_t get_default_data_size(int ndim, const intptr_t *shape) const;

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

    /** Returns what kind of memory management the dtype uses, e.g. construct/copy/move/destruct semantics */
    virtual dtype_memory_management_t get_memory_management() const = 0;

    /**
     * Returns true if the dtype is a scalar.
     *
     * This precludes a dynamic dtype from switching between scalar and array behavior,
     * but the simplicity seems to probably be worth it.
     */
    virtual bool is_scalar() const;

    /**
     * Returns true if the first dimension of the dtype is a uniform dimension.
     */
    virtual bool is_uniform_dim() const;

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
    virtual void transform_child_dtypes(dtype_transform_fn_t transform_fn, const void *extra,
                    dtype& out_transformed_dtype, bool& out_was_transformed) const;

    /**
     * Returns a modified dtype with all expression dtypes replaced with
     * their value dtypes, and dtypes replaced with "standard versions"
     * whereever appropriate. For example, an offset-based uniform array
     * would be replaced by a strided uniform array.
     */
    virtual dtype get_canonical_dtype() const;

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
    virtual dtype apply_linear_index(int nindices, const irange *indices,
                int current_i, const dtype& root_dt, bool leading_dimension) const;

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
    virtual intptr_t apply_linear_index(int nindices, const irange *indices, const char *metadata,
                    const dtype& result_dtype, char *out_metadata,
                    memory_block_data *embedded_reference,
                    int current_i, const dtype& root_dt,
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
    virtual dtype at(intptr_t i0, const char **inout_metadata, const char **inout_data) const;

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
     * Retrieves the leading dimension size of the shape.
     *
     * \param data      Data corresponding to the dtype.
     * \param metadata  Metadata corresponding to the data.
     */
    virtual intptr_t get_dim_size(const char *data, const char *metadata) const;

    /**
     * Retrieves the shape of the dtype, expanding the vector as needed. For dimensions with
     * unknown or variable shape, -1 is returned.
     *
     * The output must be pre-initialized to have get_undim() elements.
     */
    virtual void get_shape(size_t i, intptr_t *out_shape) const;

    /**
     * Retrieves the shape of the dtype ndobject instance, expanding the vector as needed. For dimensions with
     * variable shape, -1 is returned.
     *
     * The output must be pre-initialized to have get_undim() elements.
     */
    virtual void get_shape(size_t i, intptr_t *out_shape, const char *metadata) const;

    /**
     * Retrieves the strides of the dtype ndobject instance, expanding the vector as needed. For dimensions
     * where there is not a simple stride (e.g. a tuple/struct dtype), 0 is returned and
     * the caller should handle this.
     *
     * The output must be pre-initialized to have get_undim() elements.
     */
    virtual void get_strides(size_t i, intptr_t *out_strides, const char *metadata) const;

    /**
     * \brief Returns a value representative of a stride for the dimension, used for axis sorting.
     *
     * For dimensions which are strided, returns the stride. For dimensions which
     * are not, for example a dimension with an array of offsets, returns a
     * non-zero value which represents roughly what a stride would be. In this
     * example, the first non-zero offset would work.
     *
     * \param metadata  Metadata corresponding to the dtype.
     *
     * \returns  The representative stride.
     */
    virtual intptr_t get_representative_stride(const char *metadata) const;

    /**
     * Called by ::dynd::is_lossless_assignment, with (this == dst_dt->extended()).
     */
    virtual bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const = 0;

    /*
     * Return a comparison kernel that can perform the requested single comparison on
     * data of this dtype
     *
     * \param compare_id the identifier of the comparison
     */
    virtual void get_single_compare_kernel(kernel_instance<compare_operations_t>& out_kernel) const;

    /**
     * Called by ::dynd::get_dtype_assignment_kernel with (this == dst_dt.extended()) or
     * by another implementation of this function with (this == src_dt.extended()).
     *
     * If (this == dst_dt.extended()), and the function can't produce an assignment kernel,
     * should call dst_dt.extended()->get_dtype_assignment_kernel(...) to let the other
     * dtype provide the function if it can be done.
     */
    virtual void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    kernel_instance<unary_operation_pair_t>& out_kernel) const;

    virtual bool operator==(const base_dtype& rhs) const = 0;

    /** The size of the ndobject metadata for this dtype */
    virtual size_t get_metadata_size() const;
    /**
     * Constructs the ndobject metadata for this dtype, prepared for writing.
     * The element size of the result must match that from get_default_data_size().
     */
    virtual void metadata_default_construct(char *metadata, int ndim, const intptr_t* shape) const;
    /**
     * Constructs the ndobject metadata for this dtype, copying everything exactly from
     * input metadata for the same dtype.
     *
     * \param out_metadata  The new metadata memory which is constructed.
     * \param in_metadata   Existing metadata memory from which to copy.
     * \param embedded_reference  For references which are NULL, add this reference in the output.
     *                            A NULL means the data was embedded in the original ndobject, so
     *                            when putting it in a new ndobject, need to hold a reference to
     *                            that memory.
     */
    virtual void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    /** Destructs any references or other state contained in the ndobjects' metdata */
    virtual void metadata_destruct(char *metadata) const;
    /**
     * When metadata is used for temporary buffers of a dtype, and that usage is finished one
     * execution cycle, this function is called to clear usage of that memory so it can be reused in
     * the next cycle.
     */
    virtual void metadata_reset_buffers(char *metadata) const;
    /**
     * For blockref dtypes, once all the elements have been written we want to turn off further
     * memory allocation, and possibly trim excess memory that was allocated. This function
     * does this.
     */
    virtual void metadata_finalize_buffers(char *metadata) const;
    /** Debug print of the metdata */
    virtual void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;

    /** The size of the data required for uniform iteration */
    virtual size_t get_iterdata_size(int ndim) const;
    /**
     * Constructs the iterdata for processing iteration at this level of the datashape
     */
    virtual size_t iterdata_construct(iterdata_common *iterdata, const char **inout_metadata, int ndim, const intptr_t* shape, dtype& out_uniform_dtype) const;
    /** Destructs any references or other state contained in the iterdata */
    virtual size_t iterdata_destruct(iterdata_common *iterdata, int ndim) const;

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
    virtual void foreach_leading(char *data, const char *metadata, foreach_fn_t callback, void *callback_data) const;

    /**
     * Modifies metadata allocated using the metadata_default_construct function, to be used
     * immediately after ndobject construction. Given an input dtype/metadata, edits the output
     * metadata in place to match.
     *
     * \param dst_metadata  The metadata created by metadata_default_construct, which is modified in place
     * \param src_dtype  The dtype of the input ndobject whose stride ordering is to be matched.
     * \param src_metadata  The metadata of the input ndobject whose stride ordering is to be matched.
     */
    virtual void reorder_default_constructed_strides(char *dst_metadata, const dtype& src_dtype, const char *src_metadata) const;

    /**
     * Additional dynamic properties exposed by the dtype as gfunc::callable.
     */
    virtual void get_dynamic_dtype_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;

    /**
     * Additional dynamic functions exposed by the dtype as gfunc::callable.
     */
    virtual void get_dynamic_dtype_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const;

    /**
     * Additional dynamic properties exposed by any ndobject of this dtype as gfunc::callable.
     *
     * \note Uniform dtypes copy these properties from the first non-uniform dtype, so such properties must
     *       be able to handle the case where they are the first non-uniform dtype in an array type, not
     *       just strictly of the non-uniform dtype.
     */
    virtual void get_dynamic_ndobject_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;

    /**
     * Additional dynamic functions exposed by any ndobject of this dtype as gfunc::callable.
     *
     * \note Uniform dtypes copy these functions from the first non-uniform dtype, so such properties must
     *       be able to handle the case where they are the first non-uniform dtype in an array type, not
     *       just strictly of the non-uniform dtype.
     */
    virtual void get_dynamic_ndobject_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const;

    friend void base_dtype_incref(const base_dtype *ed);
    friend void base_dtype_decref(const base_dtype *ed);
};

/**
 * Increments the reference count of a memory block object.
 */
inline void base_dtype_incref(const base_dtype *ed)
{
    //std::cout << "dtype " << (void *)ed << " inc: " << ed->m_use_count + 1 << "\t"; ed->print_dtype(std::cout); std::cout << std::endl;
    ++ed->m_use_count;
}

/**
 * Decrements the reference count of a memory block object,
 * freeing it if the count reaches zero.
 */
inline void base_dtype_decref(const base_dtype *ed)
{
    //std::cout << "dtype " << (void *)ed << " dec: " << ed->m_use_count - 1 << "\t"; ed->print_dtype(std::cout); std::cout << std::endl;
    if (--ed->m_use_count == 0) {
        delete ed;
    }
}

} // namespace dynd

#endif // _DYND__BASE_DTYPE_HPP_
