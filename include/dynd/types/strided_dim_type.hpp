//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRIDED_DIM_TYPE_HPP_
#define _DYND__STRIDED_DIM_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/types/base_uniform_dim_type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/view_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/func/callable.hpp>

namespace dynd {

struct strided_dim_type_metadata {
    intptr_t size;
    intptr_t stride;
};

struct strided_dim_type_iterdata {
    iterdata_common common;
    char *data;
    intptr_t stride;
};

class strided_dim_type : public base_uniform_dim_type {
public:
    strided_dim_type(const ndt::type& element_tp);

    virtual ~strided_dim_type();

    size_t get_default_data_size(intptr_t ndim, const intptr_t *shape) const;

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_type(std::ostream& o) const;

    bool is_expression() const;
    bool is_unique_data_owner(const char *metadata) const;
    void transform_child_types(type_transform_fn_t transform_fn, void *extra,
                    ndt::type& out_transformed_tp, bool& out_was_transformed) const;
    ndt::type get_canonical_type() const;
    bool is_strided() const;
    void process_strided(const char *metadata, const char *data,
                    ndt::type& out_dt, const char *&out_origin,
                    intptr_t& out_stride, intptr_t& out_dim_size) const;

    ndt::type apply_linear_index(intptr_t nindices, const irange *indices,
                size_t current_i, const ndt::type& root_tp, bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *metadata,
                    const ndt::type& result_tp, char *out_metadata,
                    memory_block_data *embedded_reference,
                    size_t current_i, const ndt::type& root_tp,
                    bool leading_dimension, char **inout_data,
                    memory_block_data **inout_dataref) const;
    ndt::type at_single(intptr_t i0, const char **inout_metadata, const char **inout_data) const;

    ndt::type get_type_at_dimension(char **inout_metadata, intptr_t i, intptr_t total_ndim = 0) const;

    intptr_t get_dim_size(const char *metadata, const char *data) const;
        void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *metadata, const char *data) const;
    void get_strides(size_t i, intptr_t *out_strides, const char *metadata) const;

    axis_order_classification_t classify_axis_order(const char *metadata) const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    void metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_reset_buffers(char *metadata) const;
    void metadata_finalize_buffers(char *metadata) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;
    size_t metadata_copy_construct_onedim(char *dst_metadata, const char *src_metadata,
                    memory_block_data *embedded_reference) const;

    size_t get_iterdata_size(intptr_t ndim) const;
    size_t iterdata_construct(iterdata_common *iterdata, const char **inout_metadata, intptr_t ndim, const intptr_t* shape, ndt::type& out_uniform_tp) const;
    size_t iterdata_destruct(iterdata_common *iterdata, intptr_t ndim) const;

    void data_destruct(const char *metadata, char *data) const;
    void data_destruct_strided(const char *metadata, char *data,
                    intptr_t stride, size_t count) const;

    size_t make_assignment_kernel(
                    ckernel_builder *out, size_t offset_out,
                    const ndt::type& dst_tp, const char *dst_metadata,
                    const ndt::type& src_tp, const char *src_metadata,
                    kernel_request_t kernreq, assign_error_mode errmode,
                    const eval::eval_context *ectx) const;

    void foreach_leading(const char *metadata, char *data,
                         foreach_fn_t callback, void *callback_data) const;

    /**
     * Modifies metadata allocated using the metadata_default_construct function, to be used
     * immediately after nd::array construction. Given an input type/metadata, edits the output
     * metadata in place to match.
     *
     * \param dst_metadata  The metadata created by metadata_default_construct, which is modified in place
     * \param src_tp  The type of the input nd::array whose stride ordering is to be matched.
     * \param src_metadata  The metadata of the input nd::array whose stride ordering is to be matched.
     */
    void reorder_default_constructed_strides(
                    char *dst_metadata,
                    const ndt::type& src_tp, const char *src_metadata) const;

    void get_dynamic_type_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
    void get_dynamic_array_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
    void get_dynamic_array_functions(
                    const std::pair<std::string, gfunc::callable> **out_functions,
                    size_t *out_count) const;
};

/**
 * Does a value lookup into an array of type "strided * T", without
 * bounds checking the index ``i`` or validating that ``a`` has the
 * required type. Use only when these checks have been done externally.
 */
template<typename T>
inline const T& unchecked_strided_dim_get(const nd::array& a, intptr_t i)
{
    const strided_dim_type_metadata *md =
        reinterpret_cast<const strided_dim_type_metadata *>(a.get_arrmeta());
    return *reinterpret_cast<const T *>(a.get_readonly_originptr() +
                                        i * md->stride);
}

/**
 * Does a writable value lookup into an array of type "strided * T", without
 * bounds checking the index ``i`` or validating that ``a`` has the
 * required type. Use only when these checks have been done externally.
 */
template<typename T>
inline T& unchecked_strided_dim_get_rw(const nd::array& a, intptr_t i)
{
    const strided_dim_type_metadata *md =
        reinterpret_cast<const strided_dim_type_metadata *>(a.get_arrmeta());
    return *reinterpret_cast<T *>(a.get_readwrite_originptr() + i * md->stride);
}

namespace ndt {
    ndt::type make_strided_dim(const ndt::type& element_tp);

    inline ndt::type make_strided_dim(const ndt::type& uniform_tp, intptr_t ndim) {
        if (ndim > 0) {
            ndt::type result = make_strided_dim(uniform_tp);
            for (intptr_t i = 1; i < ndim; ++i) {
                result = make_strided_dim(result);
            }
            return result;
        } else {
            return uniform_tp;
        }
    }
} // namespace ndt

namespace nd {
    // nd::string, used as a tag for the ensure_immutable_contig function
    class string;

    namespace detail {
        template<typename T>
        struct ensure_immutable_contig {
            inline static enable_if<is_dynd_scalar<T>::value, bool> run(nd::array& a)
            {
                const ndt::type& tp = a.get_type();
                if (a.is_immutable() && tp.get_type_id() == strided_dim_type_id) {
                    // It's immutable and "strided * <something>"
                    const ndt::type &et =
                        tp.tcast<strided_dim_type>()->get_element_type();
                    const strided_dim_type_metadata *md =
                        reinterpret_cast<const strided_dim_type_metadata *>(
                            a.get_arrmeta());
                    if (et.get_type_id() == type_id_of<T>::value &&
                            md->stride == sizeof(T)) {
                        // It also has the right type and is contiguous,
                        // so no modification necessary.
                        return true;
                    }
                }
                // We have to make a copy, check that it's a 1D array, and that
                // it has the same array kind as the requested type.
                if (tp.get_ndim() == 1) {
                    // It's a 1D array
                    const ndt::type& et = tp.get_type_at_dimension(NULL, 1).value_type();
                    if(et.get_kind() == dynd_kind_of<T>::value) {
                        // It also has the same array kind as requested
                        nd::array tmp = nd::empty(
                            a.get_dim_size(),
                            ndt::make_strided_dim(ndt::make_type<T>()));
                        tmp.vals() = a;
                        tmp.flag_as_immutable();
                        a.swap(tmp);
                        return true;
                    }
                }
                // It's not compatible, so return false
                return false;
            }
        };

        template<>
        struct ensure_immutable_contig<nd::string> {
            inline static bool run(nd::array& a)
            {
                const ndt::type& tp = a.get_type();
                if (a.is_immutable() && tp.get_type_id() == strided_dim_type_id) {
                    // It's immutable and "strided * <something>"
                    const ndt::type &et =
                        tp.tcast<strided_dim_type>()->get_element_type();
                    const strided_dim_type_metadata *md =
                        reinterpret_cast<const strided_dim_type_metadata *>(
                            a.get_arrmeta());
                    if (et.get_type_id() == string_type_id &&
                            et.tcast<string_type>()->get_encoding() == string_encoding_utf_8 &&
                            md->stride == sizeof(string_type_data)) {
                        // It also has the right type and is contiguous,
                        // so no modification necessary.
                        return true;
                    }
                }
                // We have to make a copy, check that it's a 1D array, and that
                // it has the same array kind as the requested type.
                if (tp.get_ndim() == 1) {
                    // It's a 1D array
                    const ndt::type& et = tp.get_type_at_dimension(NULL, 1).value_type();
                    if(et.get_kind() == string_kind) {
                        // It also has the same array kind as requested
                        nd::array tmp = nd::empty(a.get_dim_size(), ndt::make_strided_of_string());
                        tmp.vals() = a;
                        tmp.flag_as_immutable();
                        a.swap(tmp);
                        return true;
                    }
                }
                // It's not compatible, so return false
                return false;
            }
        };

        template<>
        struct ensure_immutable_contig<ndt::type> {
            inline static bool run(nd::array& a)
            {
                const ndt::type& tp = a.get_type();
                if (a.is_immutable() && tp.get_type_id() == strided_dim_type_id) {
                    // It's immutable and "strided * <something>"
                    const ndt::type &et =
                        tp.tcast<strided_dim_type>()->get_element_type();
                    const strided_dim_type_metadata *md =
                        reinterpret_cast<const strided_dim_type_metadata *>(
                            a.get_arrmeta());
                    if (et.get_type_id() == type_type_id &&
                            md->stride == sizeof(ndt::type)) {
                        // It also has the right type and is contiguous,
                        // so no modification necessary.
                        return true;
                    }
                }
                // We have to make a copy, check that it's a 1D array, and that
                // it has the same array kind as the requested type.
                if (tp.get_ndim() == 1) {
                    // It's a 1D array
                    const ndt::type& et = tp.get_type_at_dimension(NULL, 1).value_type();
                    if(et.get_type_id() == type_type_id) {
                        // It also has the same array type as requested
                        nd::array tmp = nd::empty(a.get_dim_size(), ndt::make_strided_of_type());
                        tmp.vals() = a;
                        tmp.flag_as_immutable();
                        a.swap(tmp);
                        return true;
                    }
                }
                // It's not compatible, so return false
                return false;
            }
        };
    } // namespace detail

    /**
     * Makes sure the array is an immutable, contiguous, "strided * T"
     * array, modifying it in place if necessary.
     */
    template<typename T>
    bool ensure_immutable_contig(nd::array& a)
    {
        if (!a.is_null()) {
            return detail::ensure_immutable_contig<T>::run(a);
        } else {
            return false;
        }
    }
} // namespace nd

} // namespace dynd

#endif // _DYND__STRIDED_DIM_TYPE_HPP_
