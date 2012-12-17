//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DATE_DTYPE_HPP_
#define _DYND__DATE_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {

class date_dtype : public extended_dtype {
public:
    date_dtype();

    virtual ~date_dtype();

    void set_ymd(const char *metadata, char *data, assign_error_mode errmode,
                    int32_t year, int32_t month, int32_t day) const;

    void get_ymd(const char *metadata, const char *data,
                    int32_t &out_year, int32_t &out_month, int32_t &out_day) const;

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return pod_memory_management;
    }

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(single_compare_kernel_instance& out_kernel) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    kernel_instance<unary_operation_pair_t>& out_kernel) const;

    bool operator==(const extended_dtype& rhs) const;

    size_t get_metadata_size() const {
        return 0;
    }
    void metadata_default_construct(char *DYND_UNUSED(metadata), int DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const {
    }
    void metadata_copy_construct(char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata), memory_block_data *DYND_UNUSED(embedded_reference)) const {
    }
    void metadata_destruct(char *DYND_UNUSED(metadata)) const {
    }
    void metadata_debug_print(const char *DYND_UNUSED(metadata), std::ostream& DYND_UNUSED(o), const std::string& DYND_UNUSED(indent)) const {
    }

    void get_dynamic_dtype_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;
    void get_dynamic_dtype_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const;
    void get_dynamic_ndobject_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;
    void get_dynamic_ndobject_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const;

    /**
     * Returns a kernel to transform instances of this date dtype into values of the
     * property.
     *
     * \param property_name  The name of the requested property.
     * \param out_value_dtype  This is filled with the dtype of the property.
     * \param out_to_value_kernel  This is filled with a kernel extracting the property from the dtype.
     */
    void get_property_getter_kernel(const std::string& property_name,
                    dtype& out_value_dtype, kernel_instance<unary_operation_pair_t>& out_to_value_kernel) const;

};

inline dtype make_date_dtype() {
    return dtype(new date_dtype());
}

} // namespace dynd

#endif // _DYND__DATE_DTYPE_HPP_
