//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DTYPE_HPP_
#define _DYND__DTYPE_HPP_

#include <iostream>
#include <complex>
#include <stdexcept>
#include <vector>

#include <dynd/config.hpp>
#include <dynd/atomic_refcount.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/dtype_comparisons.hpp>
#include <dynd/kernels/single_compare_kernel_instance.hpp>
#include <dynd/string_encodings.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/irange.hpp>
#include <dynd/exceptions.hpp>

namespace dynd {

// Forward definition from dynd/gfunc/callable.hpp
namespace gfunc {
    class callable;
};

// A boolean class for dynamicndarray which is one-byte big
class dynd_bool {
    char m_value;
public:
    dynd_bool() : m_value(0) {}

    dynd_bool(bool value) : m_value(value) {}

    // Special case complex conversion to avoid ambiguous overload
    template<class T>
    dynd_bool(std::complex<T> value) : m_value(value != std::complex<T>(0)) {}

    operator bool() const {
        return m_value != 0;
    }
};

enum dtype_kind_t {
    bool_kind,
    int_kind,
    uint_kind,
    real_kind,
    complex_kind,
    // string_kind means subclass of extended_string_dtype
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
    void_pointer_type_id,

    // Other primitives (not builtin)
    fixedbytes_type_id,
    fixedstring_type_id,
    categorical_type_id,
    date_type_id,
    busdate_type_id,
    pointer_type_id,

    date_property_type_id,

    // blockref primitive dtypes
    bytes_type_id,
    string_type_id,

    // blockref composite dtypes
    array_type_id,

    // Composite dtypes
    strided_array_type_id,
    fixedarray_type_id,
    struct_type_id,
    fixedstruct_type_id,
    tuple_type_id,
    ndobject_type_id,

    // Adapter dtypes
    convert_type_id,
    byteswap_type_id,
    align_type_id,
    view_type_id,

    // pattern matches against other types - cannot instantiate
    pattern_type_id,

    // The number of built-in, atomic types
    builtin_type_id_count = 13
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


namespace detail {
    // Simple metaprogram taking log base 2 of 1, 2, 4, and 8
    template <int I> struct log2_x;
    template <> struct log2_x<1> {
        enum {value = 0};
    };
    template <> struct log2_x<2> {
        enum {value = 1};
    };
    template <> struct log2_x<4> {
        enum {value = 2};
    };
    template <> struct log2_x<8> {
        enum {value = 3};
    };
}


// Type trait for the type id
template <typename T> struct type_id_of;

// Can't use bool, because it doesn't have a guaranteed sizeof
template <> struct type_id_of<dynd_bool> {enum {value = bool_type_id};};
template <> struct type_id_of<char> {enum {value = ((char)-1) < 0 ? int8_type_id : uint8_type_id};};
template <> struct type_id_of<signed char> {enum {value = int8_type_id};};
template <> struct type_id_of<short> {enum {value = int16_type_id};};
template <> struct type_id_of<int> {enum {value = int32_type_id};};
template <> struct type_id_of<long> {
    enum {value = int8_type_id + detail::log2_x<sizeof(long)>::value};
};
template <> struct type_id_of<long long> {enum {value = int64_type_id};};
template <> struct type_id_of<uint8_t> {enum {value = uint8_type_id};};
template <> struct type_id_of<uint16_t> {enum {value = uint16_type_id};};
template <> struct type_id_of<unsigned int> {enum {value = uint32_type_id};};
template <> struct type_id_of<unsigned long> {
    enum {value = uint8_type_id + detail::log2_x<sizeof(unsigned long)>::value};
};
template <> struct type_id_of<unsigned long long>{enum {value = uint64_type_id};};
template <> struct type_id_of<float> {enum {value = float32_type_id};};
template <> struct type_id_of<double> {enum {value = float64_type_id};};
template <> struct type_id_of<std::complex<float> > {enum {value = complex_float32_type_id};};
template <> struct type_id_of<std::complex<double> > {enum {value = complex_float64_type_id};};
template <> struct type_id_of<void> {enum {value = void_type_id};};

// Type trait for the kind
template <typename T> struct dtype_kind_of;

template <> struct dtype_kind_of<void> {static const dtype_kind_t value = void_kind;};
// Can't use bool, because it doesn't have a guaranteed sizeof
template <> struct dtype_kind_of<dynd_bool> {static const dtype_kind_t value = bool_kind;};
template <> struct dtype_kind_of<char> {
    static const dtype_kind_t value = ((char)-1) < 0 ? int_kind : uint_kind;
};
template <> struct dtype_kind_of<signed char> {static const dtype_kind_t value = int_kind;};
template <> struct dtype_kind_of<short> {static const dtype_kind_t value = int_kind;};
template <> struct dtype_kind_of<int> {static const dtype_kind_t value = int_kind;};
template <> struct dtype_kind_of<long> {static const dtype_kind_t value = int_kind;};
template <> struct dtype_kind_of<long long> {static const dtype_kind_t value = int_kind;};
template <> struct dtype_kind_of<uint8_t> {static const dtype_kind_t value = uint_kind;};
template <> struct dtype_kind_of<uint16_t> {static const dtype_kind_t value = uint_kind;};
template <> struct dtype_kind_of<unsigned int> {static const dtype_kind_t value = uint_kind;};
template <> struct dtype_kind_of<unsigned long> {static const dtype_kind_t value = uint_kind;};
template <> struct dtype_kind_of<unsigned long long>{static const dtype_kind_t value = uint_kind;};
template <> struct dtype_kind_of<float> {static const dtype_kind_t value = real_kind;};
template <> struct dtype_kind_of<double> {static const dtype_kind_t value = real_kind;};
template <typename T> struct dtype_kind_of<std::complex<T> > {static const dtype_kind_t value = complex_kind;};

// Metaprogram for determining if a type is a valid C++ scalar
// of a particular dtype.
template<typename T> struct is_dtype_scalar {enum {value = false};};
template <> struct is_dtype_scalar<dynd_bool> {enum {value = true};};
template <> struct is_dtype_scalar<char> {enum {value = true};};
template <> struct is_dtype_scalar<signed char> {enum {value = true};};
template <> struct is_dtype_scalar<short> {enum {value = true};};
template <> struct is_dtype_scalar<int> {enum {value = true};};
template <> struct is_dtype_scalar<long> {enum {value = true};};
template <> struct is_dtype_scalar<long long> {enum {value = true};};
template <> struct is_dtype_scalar<unsigned char> {enum {value = true};};
template <> struct is_dtype_scalar<unsigned short> {enum {value = true};};
template <> struct is_dtype_scalar<unsigned int> {enum {value = true};};
template <> struct is_dtype_scalar<unsigned long> {enum {value = true};};
template <> struct is_dtype_scalar<unsigned long long> {enum {value = true};};
template <> struct is_dtype_scalar<float> {enum {value = true};};
template <> struct is_dtype_scalar<double> {enum {value = true};};
template <> struct is_dtype_scalar<std::complex<float> > {enum {value = true};};
template <> struct is_dtype_scalar<std::complex<double> > {enum {value = true};};

// Metaprogram for determining scalar alignment
template<typename T> struct scalar_align_of;
template <> struct scalar_align_of<dynd_bool> {enum {value = 1};};
template <> struct scalar_align_of<char> {enum {value = 1};};
template <> struct scalar_align_of<signed char> {enum {value = 1};};
template <> struct scalar_align_of<short> {enum {value = sizeof(short)};};
template <> struct scalar_align_of<int> {enum {value = sizeof(int)};};
template <> struct scalar_align_of<long> {enum {value = sizeof(long)};};
template <> struct scalar_align_of<long long> {enum {value = sizeof(long long)};};
template <> struct scalar_align_of<unsigned char> {enum {value = 1};};
template <> struct scalar_align_of<unsigned short> {enum {value = sizeof(unsigned short)};};
template <> struct scalar_align_of<unsigned int> {enum {value = sizeof(unsigned int)};};
template <> struct scalar_align_of<unsigned long> {enum {value = sizeof(unsigned long)};};
template <> struct scalar_align_of<unsigned long long> {enum {value = sizeof(unsigned long long)};};
template <> struct scalar_align_of<float> {enum {value = sizeof(long)};};
template <> struct scalar_align_of<double> {enum {value = sizeof(double)};};
template <> struct scalar_align_of<std::complex<float> > {enum {value = sizeof(long)};};
template <> struct scalar_align_of<std::complex<double> > {enum {value = sizeof(double)};};

// Metaprogram for determining if a type is the C++ "bool" or not
template<typename T> struct is_type_bool {enum {value = false};};
template<> struct is_type_bool<bool> {enum {value = true};};

/**
 * Increments the offset value so that it is aligned to the requested alignment
 * NOTE: The alignment must be a power of two.
 */
inline size_t inc_to_alignment(size_t offset, size_t alignment) {
    return (offset + alignment - 1) & (std::size_t)(-(std::ptrdiff_t)alignment);
}

/**
 * Increments the pointer value so that it is aligned to the requested alignment
 * NOTE: The alignment must be a power of two.
 */
inline char *inc_to_alignment(char *ptr, size_t alignment) {
    return reinterpret_cast<char *>((reinterpret_cast<std::size_t>(ptr) + alignment - 1) &
                                    (std::size_t)(-(std::ptrdiff_t)alignment));
}

/**
 * Increments the pointer value so that it is aligned to the requested alignment
 * NOTE: The alignment must be a power of two.
 */
inline void *inc_to_alignment(void *ptr, size_t alignment) {
    return reinterpret_cast<char *>((reinterpret_cast<std::size_t>(ptr) + alignment - 1) &
                                    (size_t)(-(std::ptrdiff_t)alignment));
}

/**
 * \brief Tests whether the offset has the requested alignment.
 *
 * NOTE: The alignment must be a power of two.
 *
 * \param offset  The offset whose alignment is tested.
 * \param alignment  The required alignment, must be a power of two.
 *
 * \returns  True if the offset is divisible by the power of two alignment,
 *           False otherwise.
 */
inline bool offset_is_aligned(size_t offset, size_t alignment) {
    return (offset&(alignment - 1)) == 0;
}

/** Prints a single scalar of a builtin dtype to the stream */
void print_builtin_scalar(type_id_t type_id, std::ostream& o, const char *data);

class dtype;
class extended_dtype;
struct iterdata_common;

/** This is the callback function type used by the extended_dtype::foreach function */
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

typedef void (*dtype_transform_fn_t)(const dtype& dt, const void *extra,
                dtype& out_transformed_dtype, bool& out_was_transformed);

struct iterdata_common {
    iterdata_increment_fn_t incr;
    iterdata_reset_fn_t reset;
};

/** Special iterdata which broadcasts to any number of additional dimensions */
struct iterdata_broadcasting_terminator {
    iterdata_common common;
    char *data;
};
char *iterdata_broadcasting_terminator_incr(iterdata_common *iterdata, int level);
char *iterdata_broadcasting_terminator_reset(iterdata_common *iterdata, char *data, int level);

// The extended_dtype class is for dtypes which require more data
// than a type_id, kind, and element_size, and endianness.
class extended_dtype {
    /** Embedded reference counting */
    mutable atomic_refcount m_use_count;
protected:
    /** Standard dtype data */
    uint16_t m_type_id;
    uint8_t m_kind, m_alignment;
    size_t m_data_size;
    

protected:
    // Helper function for uniform dimension dtypes
    void get_nonuniform_ndobject_properties_and_functions(
                    std::vector<std::pair<std::string, gfunc::callable> >& out_properties,
                    std::vector<std::pair<std::string, gfunc::callable> >& out_functions) const;
public:
    /** Starts off the extended dtype instance with a use count of 1. */
    inline extended_dtype(type_id_t type_id, dtype_kind_t kind, size_t data_size, size_t alignment)
        : m_use_count(1), m_type_id(static_cast<uint16_t>(type_id)), m_kind(static_cast<uint8_t>(kind)),
                m_alignment(static_cast<uint8_t>(alignment)), m_data_size(data_size)
    {}

    virtual ~extended_dtype();

    inline type_id_t get_type_id() const {
        return static_cast<type_id_t>(m_type_id);
    }
    inline dtype_kind_t get_kind() const {
        return static_cast<dtype_kind_t>(m_kind);
    }
    inline size_t get_alignment() const {
        return m_alignment;
    }
    inline size_t get_data_size() const {
        return m_data_size;
    }
    virtual size_t get_default_data_size(int ndim, const intptr_t *shape) const;

    /**
     * Print the raw data interpreted as a single value of this dtype.
     *
     * \param o the std::ostream to print to
     * \param data pointer to the data element to print
     */
    virtual void print_data(std::ostream& o, const char *metadata, const char *data) const = 0;

    /**
     * Print a representation of the dtype itself
     *
     * \param o the std::ostream to print to
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
     */
    virtual dtype apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const;

    /**
     * Indexes into an ndobject using the provided linear index, and a dtype and freshly allocated output
     * set to point to the same base data reference.
     *
     * \param nindices     The number of elements in the 'indices' array. This is shrunk by one for each recursive call.
     * \param indices      The indices to apply. This is incremented by one for each recursive call.
     * \param data         The data of the input array.
     * \param metadata     The metadata of the input array.
     * \param result_dtype The result of an apply_linear_index call.
     * \param out_metadata The metadata of the output array. The output data should all be references to the data
     *                     of the input array, so there is no out_data parameter.
     * \param embedded_reference  For references which are NULL, add this reference in the output.
     *                            A NULL means the data was embedded in the original ndobject, so
     *                            when putting it in a new ndobject, need to hold a reference to
     *                            that memory.
     * \param current_i    The current index position. Used for error messages.
     * \param root_dt      The data type in the first call, before any recursion. Used for error messages.
     *
     * @return  An offset to apply to the data pointer.
     */
    virtual intptr_t apply_linear_index(int nindices, const irange *indices, char *data, const char *metadata,
                    const dtype& result_dtype, char *out_metadata,
                    memory_block_data *embedded_reference,
                    int current_i, const dtype& root_dt) const;

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
     * Retrieves the number of initial uniform dimensions.
     */
    virtual int get_undim() const;

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
    virtual dtype get_dtype_at_dimension(char **inout_metadata, int i, int total_ndim = 0) const;

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
    virtual void get_shape(int i, intptr_t *out_shape) const;

    /**
     * Retrieves the shape of the dtype ndobject instance, expanding the vector as needed. For dimensions with
     * variable shape, -1 is returned.
     *
     * The output must be pre-initialized to have get_undim() elements.
     */
    virtual void get_shape(int i, intptr_t *out_shape, const char *metadata) const;

    /**
     * Retrieves the strides of the dtype ndobject instance, expanding the vector as needed. For dimensions
     * where there is not a simple stride (e.g. a tuple/struct dtype), 0 is returned and
     * the caller should handle this.
     *
     * The output must be pre-initialized to have get_undim() elements.
     */
    virtual void get_strides(int i, intptr_t *out_strides, const char *metadata) const;

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
    virtual void get_single_compare_kernel(single_compare_kernel_instance& out_kernel) const;

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

    virtual bool operator==(const extended_dtype& rhs) const = 0;

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
    virtual void get_dynamic_dtype_properties(const std::pair<std::string, gfunc::callable> **out_properties, int *out_count) const;

    /**
     * Additional dynamic functions exposed by the dtype as gfunc::callable.
     */
    virtual void get_dynamic_dtype_functions(const std::pair<std::string, gfunc::callable> **out_functions, int *out_count) const;

    /**
     * Additional dynamic properties exposed by any ndobject of this dtype as gfunc::callable.
     *
     * \note Uniform dtypes copy these properties from the first non-uniform dtype, so such properties must
     *       be able to handle the case where they are the first non-uniform dtype in an array type, not
     *       just strictly of the non-uniform dtype.
     */
    virtual void get_dynamic_ndobject_properties(const std::pair<std::string, gfunc::callable> **out_properties, int *out_count) const;

    /**
     * Additional dynamic functions exposed by any ndobject of this dtype as gfunc::callable.
     *
     * \note Uniform dtypes copy these functions from the first non-uniform dtype, so such properties must
     *       be able to handle the case where they are the first non-uniform dtype in an array type, not
     *       just strictly of the non-uniform dtype.
     */
    virtual void get_dynamic_ndobject_functions(const std::pair<std::string, gfunc::callable> **out_functions, int *out_count) const;

    friend void extended_dtype_incref(const extended_dtype *ed);
    friend void extended_dtype_decref(const extended_dtype *ed);
};

/**
 * Increments the reference count of a memory block object.
 */
inline void extended_dtype_incref(const extended_dtype *ed)
{
    //std::cout << "dtype " << (void *)ed << " inc: " << ed->m_use_count + 1 << "\t"; ed->print_dtype(std::cout); std::cout << std::endl;
    ++ed->m_use_count;
}

/**
 * Decrements the reference count of a memory block object,
 * freeing it if the count reaches zero.
 */
inline void extended_dtype_decref(const extended_dtype *ed)
{
    //std::cout << "dtype " << (void *)ed << " dec: " << ed->m_use_count - 1 << "\t"; ed->print_dtype(std::cout); std::cout << std::endl;
    if (--ed->m_use_count == 0) {
        delete ed;
    }
}



/**
 * Base class for all string extended dtypes. If a dtype
 * has kind string_kind, it must be a subclass of
 * extended_string_dtype.
 */
class extended_string_dtype : public extended_dtype {
public:
    inline extended_string_dtype(type_id_t type_id, dtype_kind_t kind, size_t data_size, size_t alignment)
        : extended_dtype(type_id, kind, data_size, alignment)
    {}

    virtual ~extended_string_dtype();
    /** The encoding used by the string */
    virtual string_encoding_t get_encoding() const = 0;

    /** Retrieves the data range in which a string is stored */
    virtual void get_string_range(const char **out_begin, const char**out_end, const char *metadata, const char *data) const = 0;
    /** Converts a string element into a C++ std::string with a UTF8 encoding */
    std::string get_utf8_string(const char *metadata, const char *data, assign_error_mode errmode) const;
    /** Copies a C++ std::string with a UTF8 encoding to a string element */
    virtual void set_utf8_string(const char *metadata, char *data, assign_error_mode errmode, const std::string& utf8_str) const = 0;

    // String dtypes stop the iterdata chain
    // TODO: Maybe it should be more flexible?
    size_t get_iterdata_size(int ndim) const;

    void get_dynamic_dtype_properties(const std::pair<std::string, gfunc::callable> **out_properties, int *out_count) const;
};

/**
 * Base class for all dtypes of expression_kind.
 */
class extended_expression_dtype : public extended_dtype {
public:
    inline extended_expression_dtype(type_id_t type_id, dtype_kind_t kind, size_t data_size, size_t alignment)
        : extended_dtype(type_id, kind, data_size, alignment)
    {}

    virtual ~extended_expression_dtype();

    /**
     * Should return a reference to the dtype representing the value which
     * is for calculation. This should never be an expression dtype.
     */
    virtual const dtype& get_value_dtype() const = 0;
    /**
     * Should return a reference to a dtype representing the data this dtype
     * uses to produce the value.
     */
    virtual const dtype& get_operand_dtype() const = 0;

    /** Returns a kernel which converts from (operand_dtype().value_dtype()) to (value_dtype()) */
    virtual void get_operand_to_value_kernel(const eval::eval_context *ectx,
                            kernel_instance<unary_operation_pair_t>& out_borrowed_kernel) const = 0;
    /** Returns a kernel which converts from (value_dtype()) to (operand_dtype().value_dtype()) */
    virtual void get_value_to_operand_kernel(const eval::eval_context *ectx,
                            kernel_instance<unary_operation_pair_t>& out_borrowed_kernel) const = 0;

    /**
     * This method is for expression dtypes, and is a way to substitute
     * the storage dtype (deepest operand dtype) of an existing dtype.
     *
     * The value_dtype of the replacement should match the storage dtype
     * of this instance. Implementations of this should raise an exception
     * when this is not true.
     */
    virtual dtype with_replaced_storage_dtype(const dtype& replacement_dtype) const = 0;

    // Always return true for expression dtypes
    bool is_expression() const;

    // The canonical dtype for expression dtypes is always the value dtype
    dtype get_canonical_dtype() const;

    // Expression dtypes use the values from their operand dtype.
    size_t get_metadata_size() const;
    void metadata_default_construct(char *metadata, int ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;

    // Expression dtypes stop the iterdata chain
    // TODO: Maybe it should be more flexible?
    size_t get_iterdata_size(int ndim) const;
};

namespace detail {
    /**
     * Internal implementation detail - makes a builtin dtype from its raw values.
     */
    /* TODO: DYND_CONSTEXPR */ dtype internal_make_raw_dtype(char type_id, char kind, intptr_t element_size, char alignment);

} // namespace detail


/**
 * This class represents a data type.
 *
 * The purpose of this data type is to describe the data layout
 * of elements in ndarrays. The class stores a number of common
 * properties, like a type id, a kind, an alignment, a byte-swapped
 * flag, and an element_size. Some data types have additional data
 * which is stored as a dynamically allocated extended_dtype object.
 *
 * For the simple built-in dtypes, no extended data is needed, in
 * which case this is entirely a value type with no allocated memory.
 *
 */
class dtype {
private:
    const extended_dtype *m_extended;

    inline static bool is_builtin(const extended_dtype *ext) {
        return (reinterpret_cast<uintptr_t>(ext)&(~builtin_type_id_mask)) == 0;
    }
    /**
     * Validates that the given type ID is a proper ID and casts to
     * an extended_dtype pointer if it is. Throws
     * an exception if not.
     *
     * \param type_id  The type id to validate.
     */
    static inline const extended_dtype *validate_builtin_type_id(type_id_t type_id)
    {
        // 0 <= type_id < (builtin_type_id_count + 1)
        // The + 1 is for void_type_id
        if ((unsigned int)type_id < builtin_type_id_count + 1) {
            return reinterpret_cast<const extended_dtype *>(type_id);
        } else {
            throw invalid_type_id((int)type_id);
        }
    }

    static uint8_t builtin_kinds[builtin_type_id_count + 1];
    static uint8_t builtin_data_sizes[builtin_type_id_count + 1];
    static uint8_t builtin_data_alignments[builtin_type_id_count + 1];
public:
    /** Constructor */
    dtype()
        : m_extended(reinterpret_cast<const extended_dtype *>(void_type_id))
    {}
    /** Constructor from an extended_dtype. This claims ownership of the 'extended' reference by default, be careful! */
    explicit dtype(const extended_dtype *extended, bool incref = false)
        : m_extended(extended)
    {
        if (incref && !dtype::is_builtin(extended)) {
            extended_dtype_incref(m_extended);
        }
    }
    /** Copy constructor (should be "= default" in C++11) */
    dtype(const dtype& rhs)
        : m_extended(rhs.m_extended)
    {
        if (!dtype::is_builtin(m_extended)) {
            extended_dtype_incref(m_extended);
        }
    }
    /** Assignment operator (should be "= default" in C++11) */
    dtype& operator=(const dtype& rhs) {
        if (!dtype::is_builtin(m_extended)) {
            extended_dtype_decref(m_extended);
        }
        m_extended = rhs.m_extended;
        if (!dtype::is_builtin(m_extended)) {
            extended_dtype_incref(m_extended);
        }
        return *this;
    }
#ifdef DYND_RVALUE_REFS
    /** Move constructor */
    dtype(dtype&& rhs)
        : m_extended(rhs.m_extended)
    {
        rhs.m_extended = reinterpret_cast<const extended_dtype *>(void_type_id);
    }
    /** Move assignment operator */
    dtype& operator=(dtype&& rhs) {
        if (!dtype::is_builtin(m_extended)) {
            extended_dtype_decref(m_extended);
        }
        m_extended = rhs.m_extended;
        rhs.m_extended = reinterpret_cast<const extended_dtype *>(void_type_id);
        return *this;
    }
#endif // DYND_RVALUE_REFS

    /** Construct from a builtin type ID */
    explicit dtype(type_id_t type_id)
        : m_extended(dtype::validate_builtin_type_id(type_id))
    {}

    /** Construct from a string representation */
    explicit dtype(const std::string& rep);

    ~dtype() {
        if (!is_builtin()) {
            extended_dtype_decref(m_extended);
        }
    }

    void swap(dtype& rhs) {
        std::swap(m_extended, rhs.m_extended);
    }

    inline bool operator==(const dtype& rhs) const {
        if (is_builtin() || rhs.is_builtin()) {
            return m_extended == rhs.m_extended;
        } else {
            return *m_extended == *rhs.m_extended;
        }
    }
    bool operator!=(const dtype& rhs) const {
        return !(operator==(rhs));
    }

    /**
     * Returns true if this dtype is of a builtin dtype, which
     * means the type id is encoded directly in the m_extended
     * pointer.
     */
    inline bool is_builtin() const {
        return dtype::is_builtin(m_extended);
    }

    /**
     * Indexes into the dtype. This function returns the dtype which results
     * from applying the same index to an ndarray of this dtype.
     *
     * \param ndim         The number of elements in the 'indices' array
     * \param indices      The indices to apply.
     */
    dtype at_array(int nindices, const irange *indices) const;

    /**
     * The 'at' function is used for indexing. Overloading operator[] isn't
     * practical for multidimensional objects. Indexing one dimension with
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
    inline dtype at_single(intptr_t i0, const char **inout_metadata = NULL, const char **inout_data = NULL) const {
        if (!is_builtin()) {
            return m_extended->at(i0, inout_metadata, inout_data);
        } else {
            throw too_many_indices(1, 0);
        }
    }

    inline dtype at(intptr_t i0) const {
        return at_single(i0);
    }

    /**
     * The 'at' function is used for indexing. Overloading operator[] isn't
     * practical for multidimensional objects.
     */
    inline dtype at(const irange& i0) const {
        return at_array(1, &i0);
    }

    /** Indexing with two index values */
    inline dtype at(const irange& i0, const irange& i1) const {
        irange i[2] = {i0, i1};
        return at_array(2, i);
    }

    /** Indexing with three index values */
    inline dtype at(const irange& i0, const irange& i1, const irange& i2) const {
        irange i[3] = {i0, i1, i2};
        return at_array(3, i);
    }
    /** Indexing with four index values */
    inline dtype at(const irange& i0, const irange& i1, const irange& i2, const irange& i3) const {
        irange i[4] = {i0, i1, i2, i3};
        return at_array(4, i);
    }

    /**
     * Indexes into the dtype, intended for recursive calls from the extended-dtype version. See
     * the function in extended_dtype with the same name for more details.
     */
    dtype apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const;

    /**
     * Returns the non-expression dtype that this
     * dtype looks like for the purposes of calculation,
     * printing, etc.
     */
    const dtype& value_dtype() const {
        // Only expression_kind dtypes have different value_dtype
        if (is_builtin() || m_extended->get_kind() != expression_kind) {
            return *this;
        } else {
            // All chaining happens in the operand_dtype
            return static_cast<const extended_expression_dtype *>(m_extended)->get_value_dtype();
        }
    }

    /**
     * For expression dtypes, returns the operand dtype,
     * which is the source dtype of this dtype's expression.
     * This is one link down the expression chain.
     */
    const dtype& operand_dtype() const {
        // Only expression_kind dtypes have different operand_dtype
        if (is_builtin() || m_extended->get_kind() != expression_kind) {
            return *this;
        } else {
            return static_cast<const extended_expression_dtype *>(m_extended)->get_operand_dtype();
        }
    }

    /**
     * For expression dtypes, returns the storage dtype,
     * which is the dtype of the underlying input data.
     * This is the bottom of the expression chain.
     */
    const dtype& storage_dtype() const {
        // Only expression_kind dtypes have different storage_dtype
        if (is_builtin() || m_extended->get_kind() != expression_kind) {
            return *this;
        } else {
            // Follow the operand dtype chain to get the storage dtype
            const dtype* dt = &static_cast<const extended_expression_dtype *>(m_extended)->get_operand_dtype();
            while (dt->get_kind() == expression_kind) {
                dt = &static_cast<const extended_expression_dtype *>(dt->m_extended)->get_operand_dtype();
            }
            return *dt;
        }
    }

    /**
     * The type number is an enumeration of data types, starting
     * at 0, with one value for each unique data type. This is
     * inspired by the approach in NumPy, and the intention is
     * to have the default
     */
    type_id_t get_type_id() const {
        if (is_builtin()) {
            return static_cast<type_id_t>(reinterpret_cast<intptr_t>(m_extended));
        } else {
            return m_extended->get_type_id();
        }
    }

    /** The 'kind' of the dtype (int, uint, float, etc) */
    dtype_kind_t get_kind() const {
        if (is_builtin()) {
            return static_cast<dtype_kind_t>(dtype::builtin_kinds[reinterpret_cast<intptr_t>(m_extended)]);
        } else {
            return m_extended->get_kind();
        }
    }

    /** The alignment of the dtype */
    size_t get_alignment() const {
        if (is_builtin()) {
            return static_cast<size_t>(dtype::builtin_data_alignments[reinterpret_cast<intptr_t>(m_extended)]);
        } else {
            return m_extended->get_alignment();
        }
    }

    /** The element size of the dtype */
    size_t get_data_size() const {
        if (is_builtin()) {
            return static_cast<size_t>(dtype::builtin_data_sizes[reinterpret_cast<intptr_t>(m_extended)]);
        } else {
            return m_extended->get_data_size();
        }
    }

    inline dtype_memory_management_t get_memory_management() const {
        if (is_builtin()) {
            return pod_memory_management;
        } else {
            return m_extended->get_memory_management();
        }
    }

    /*
     * Return a comparison kernel that can perform the requested single comparison on
     * data of this dtype
     *
     * \param compare_id the identifier of the comparison
     */
    void get_single_compare_kernel(single_compare_kernel_instance& out_kernel) const;

    inline bool is_scalar() const {
        if (is_builtin()) {
            return true;
        } else {
            return m_extended->is_scalar();
        }
    }

    /**
     * Returns true if the dtype contains any expression
     * dtype within it somewhere.
     */
    inline bool is_expression() const {
        if (is_builtin()) {
            return false;
        } else {
            return m_extended->is_expression();
        }
    }

    /**
     * For array types, recursively applies to each child type, and for
     * scalar types converts to the provided one.
     *
     * \param scalar_dtype  The scalar dtype to convert all scalars to.
     * \param errmode       The error mode for the conversion.
     */
    dtype with_replaced_scalar_types(const dtype& scalar_dtype, assign_error_mode errmode = assign_error_default) const;

    /**
     * Returns a modified dtype with all expression dtypes replaced with
     * their value dtypes, and dtypes replaced with "standard versions"
     * whereever appropriate. For example, an offset-based uniform array
     * would be replaced by a strided uniform array.
     */
    inline dtype get_canonical_dtype() const {
        if (is_builtin()) {
            return *this;
        } else {
            return m_extended->get_canonical_dtype();
        }
    }

    /**
     * Gets the number of uniform dimensions in the dtype.
     */
    inline int get_undim() const {
        if (is_builtin()) {
            return 0;
        } else {
            return m_extended->get_undim();
        }
    }

    /**
     * Gets the dtype with all the unifom dimensions stripped away.
     */
    inline dtype get_udtype() const {
        if (is_builtin()) {
            return *this;
        } else {
            return m_extended->get_dtype_at_dimension(NULL, m_extended->get_undim());
        }
    }

    intptr_t get_dim_size(const char *data, const char *metadata) const {
        if (!is_builtin()) {
            return m_extended->get_dim_size(data, metadata);
        } else {
            std::stringstream ss;
            ss << "Cannot get the leading dimension size of ndobject with scalar dtype " << *this;
            throw std::runtime_error(ss.str());
        }
    }

    inline dtype get_dtype_at_dimension(char **inout_metadata, int i, int total_ndim = 0) const {
        if (!is_builtin()) {
            return m_extended->get_dtype_at_dimension(inout_metadata, i, total_ndim);
        } else if (i == 0) {
            return *this;
        } else {
            throw too_many_indices(total_ndim + i, total_ndim);
        }
    }


    /**
     * Returns a const pointer to the extended_dtype object which
     * contains information about the dtype, or NULL if no extended
     * dtype information exists. The returned pointer is only valid during
     * the lifetime of the dtype.
     */
    inline const extended_dtype* extended() const {
        return m_extended;
    }

    /** The size of the data required for uniform iteration */
    inline size_t get_iterdata_size(int ndim) const {
        if (is_builtin()) {
            return 0;
        } else {
            return m_extended->get_iterdata_size(ndim);
        }
    }
    /**
     * \brief Constructs the iterdata for processing iteration of the specified shape.
     *
     * \param iterdata  The allocated iterdata to construct.
     * \param inout_metadata  The metadata corresponding to the dtype for the iterdata construction.
     *                        This is modified in place to become the metadata for the uniform dtype.
     * \param ndim      Number of iteration dimensions.
     * \param shape     The iteration shape.
     * \param out_uniform_dtype  This is populated with the dtype of each iterated element
     */
    inline void iterdata_construct(iterdata_common *iterdata, const char **inout_metadata,
                    int ndim, const intptr_t* shape, dtype& out_uniform_dtype) const
    {
        if (!is_builtin()) {
            m_extended->iterdata_construct(iterdata, inout_metadata, ndim, shape, out_uniform_dtype);
        }
    }

    /** Destructs any references or other state contained in the iterdata */
    inline void iterdata_destruct(iterdata_common *iterdata, int ndim) const
    {
        if (!is_builtin()) {
            m_extended->iterdata_destruct(iterdata, ndim);
        }
    }

    inline size_t get_broadcasted_iterdata_size(int ndim) const {
        if (is_builtin()) {
            return sizeof(iterdata_broadcasting_terminator);
        } else {
            return m_extended->get_iterdata_size(ndim) + sizeof(iterdata_broadcasting_terminator);
        }
    }

    /**
     * Constructs an iterdata which can be broadcast to the left indefinitely, by capping
     * off the iterdata with a iterdata_broadcasting_terminator.
     * \param iterdata  The allocated iterdata to construct.
     * \param inout_metadata  The metadata corresponding to the dtype for the iterdata construction.
     *                        This is modified in place to become the metadata for the uniform dtype.
     * \param ndim      Number of iteration dimensions.
     * \param shape     The iteration shape.
     * \param out_uniform_dtype  This is populated with the dtype of each iterated element
     */
    inline void broadcasted_iterdata_construct(iterdata_common *iterdata, const char **inout_metadata,
                    int ndim, const intptr_t* shape, dtype& out_uniform_dtype) const
    {
        size_t size;
        if (is_builtin()) {
            size = 0;
        } else {
            size = m_extended->iterdata_construct(iterdata, inout_metadata, ndim, shape, out_uniform_dtype);
        }
        iterdata_broadcasting_terminator *id = reinterpret_cast<iterdata_broadcasting_terminator *>(
                        reinterpret_cast<char *>(iterdata) + size);
        id->common.incr = &iterdata_broadcasting_terminator_incr;
        id->common.reset = &iterdata_broadcasting_terminator_reset;
    }

    /**
     * print data interpreted as a single value of this dtype
     *
     * \param o         the std::ostream to print to
     * \param data      pointer to the data element to print
     * \param metadata  pointer to the ndobject metadata for the data element
     */
    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    friend /* TODO: DYND_CONSTEXPR*/ dtype detail::internal_make_raw_dtype(char type_id, char kind, intptr_t element_size, char alignment);
    friend std::ostream& operator<<(std::ostream& o, const dtype& rhs);
};

// Convenience function which makes a dtype object from a template parameter
template<class T>
dtype make_dtype()
{
    return dtype(static_cast<type_id_t>(type_id_of<T>::value));
}

/**
 * A static array of the builtin dtypes and void.
 * If code is specialized just for a builtin type, like int, it can use
 * static_builtin_dtypes[type_id_of<int>::value] as a fast
 * way to get a const reference to its dtype.
 */
extern const dtype static_builtin_dtypes[builtin_type_id_count + 1];

std::ostream& operator<<(std::ostream& o, const dtype& rhs);
/** Prints raw bytes as hexadecimal */
void hexadecimal_print(std::ostream& o, char value);
void hexadecimal_print(std::ostream& o, unsigned char value);
void hexadecimal_print(std::ostream& o, unsigned short value);
void hexadecimal_print(std::ostream& o, unsigned int value);
void hexadecimal_print(std::ostream& o, unsigned long value);
void hexadecimal_print(std::ostream& o, unsigned long long value);
void hexadecimal_print(std::ostream& o, const char *data, intptr_t element_size);

} // namespace dynd

#endif // _DYND__DTYPE_HPP_
