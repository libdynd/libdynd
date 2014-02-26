//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/types/base_uniform_dim_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/buffer_storage.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/gfunc/make_callable.hpp>
#include <dynd/gfunc/call_callable.hpp>

#include <dynd/types/convert_type.hpp>
#include <dynd/types/datashape_parser.hpp>

#include <sstream>
#include <cstring>
#include <vector>

using namespace std;
using namespace dynd;

char *dynd::iterdata_broadcasting_terminator_incr(iterdata_common *iterdata, intptr_t DYND_UNUSED(level))
{
    // This repeats the same data over and over again, broadcasting additional leftmost dimensions
    iterdata_broadcasting_terminator *id = reinterpret_cast<iterdata_broadcasting_terminator *>(iterdata);
    return id->data;
}

char *dynd::iterdata_broadcasting_terminator_reset(iterdata_common *iterdata, char *data, intptr_t DYND_UNUSED(level))
{
    iterdata_broadcasting_terminator *id = reinterpret_cast<iterdata_broadcasting_terminator *>(iterdata);
    id->data = data;
    return data;
}

const ndt::type ndt::static_builtin_types[builtin_type_id_count] = {
    ndt::type(uninitialized_type_id),
    ndt::type(bool_type_id),
    ndt::type(int8_type_id),
    ndt::type(int16_type_id),
    ndt::type(int32_type_id),
    ndt::type(int64_type_id),
    ndt::type(int128_type_id),
    ndt::type(uint8_type_id),
    ndt::type(uint16_type_id),
    ndt::type(uint32_type_id),
    ndt::type(uint64_type_id),
    ndt::type(uint128_type_id),
    ndt::type(float16_type_id),
    ndt::type(float32_type_id),
    ndt::type(float64_type_id),
    ndt::type(float128_type_id),
    ndt::type(complex_float32_type_id),
    ndt::type(complex_float64_type_id),
    ndt::type(void_type_id)
};


ndt::type::type(const std::string& rep)
    : m_extended(NULL)
{
    type_from_datashape(rep).swap(*this);
}

ndt::type::type(const char *rep_begin, const char *rep_end)
    : m_extended(NULL)
{
    type_from_datashape(rep_begin, rep_end).swap(*this);
}


ndt::type ndt::type::at_array(int nindices, const irange *indices) const
{
    if (this->is_builtin()) {
        if (nindices == 0) {
            return *this;
        } else {
            throw too_many_indices(*this, nindices, 0);
        }
    } else {
        return m_extended->apply_linear_index(nindices, indices, 0, *this, true);
    }
}

nd::array ndt::type::p(const char *property_name) const
{
    if (!is_builtin()) {
        const std::pair<std::string, gfunc::callable> *properties;
        size_t count;
        extended()->get_dynamic_type_properties(&properties, &count);
        // TODO: We probably want to make some kind of acceleration structure for the name lookup
        if (count > 0) {
            for (size_t i = 0; i < count; ++i) {
                if (properties[i].first == property_name) {
                    return properties[i].second.call(*this);
                }
            }
        }
    }

    stringstream ss;
    ss << "dynd type does not have property " << property_name;
    throw runtime_error(ss.str());
}

nd::array ndt::type::p(const std::string& property_name) const
{
    if (!is_builtin()) {
        const std::pair<std::string, gfunc::callable> *properties;
        size_t count;
        extended()->get_dynamic_type_properties(&properties, &count);
        // TODO: We probably want to make some kind of acceleration structure for the name lookup
        if (count > 0) {
            for (size_t i = 0; i < count; ++i) {
                if (properties[i].first == property_name) {
                    return properties[i].second.call(*this);
                }
            }
        }
    }

    stringstream ss;
    ss << "dynd type does not have property " << property_name;
    throw runtime_error(ss.str());
}

ndt::type ndt::type::apply_linear_index(intptr_t nindices, const irange *indices,
                size_t current_i, const ndt::type& root_tp, bool leading_dimension) const
{
    if (is_builtin()) {
        if (nindices == 0) {
            return *this;
        } else {
            throw too_many_indices(*this, nindices + current_i, current_i);
        }
    } else {
        return m_extended->apply_linear_index(nindices, indices, current_i, root_tp, leading_dimension);
    }
}

namespace {
    struct replace_scalar_type_extra {
        replace_scalar_type_extra(const ndt::type& dt, assign_error_mode em)
            : scalar_tp(dt), errmode(em)
        {
        }
        const ndt::type& scalar_tp;
        assign_error_mode errmode;
    };
    static void replace_scalar_types(const ndt::type& dt, void *extra,
                ndt::type& out_transformed_tp, bool& out_was_transformed)
    {
        const replace_scalar_type_extra *e = reinterpret_cast<const replace_scalar_type_extra *>(extra);
        if (dt.is_scalar()) {
            out_transformed_tp = ndt::make_convert(e->scalar_tp, dt, e->errmode);
            out_was_transformed = true;
        } else {
            dt.extended()->transform_child_types(&replace_scalar_types, extra, out_transformed_tp, out_was_transformed);
        }
    }
} // anonymous namespace

ndt::type ndt::type::with_replaced_scalar_types(const ndt::type& scalar_tp, assign_error_mode errmode) const
{
    ndt::type result;
    bool was_transformed;
    replace_scalar_type_extra extra(scalar_tp, errmode);
    replace_scalar_types(*this, &extra, result, was_transformed);
    return result;
}

namespace {
    struct replace_dtype_extra {
        replace_dtype_extra(const ndt::type& replacement_tp, intptr_t replace_ndim)
            : m_replacement_tp(replacement_tp), m_replace_ndim(replace_ndim)
        {
        }
        const ndt::type& m_replacement_tp;
        intptr_t m_replace_ndim;
    };
    static void replace_dtype(const ndt::type& tp, void *extra,
                ndt::type& out_transformed_tp, bool& out_was_transformed)
    {
        const replace_dtype_extra *e = reinterpret_cast<const replace_dtype_extra *>(extra);
        if (tp.get_ndim() == e->m_replace_ndim) {
            out_transformed_tp = e->m_replacement_tp;
            out_was_transformed = true;
        } else {
            tp.extended()->transform_child_types(&replace_dtype, extra, out_transformed_tp, out_was_transformed);
        }
    }
} // anonymous namespace

ndt::type ndt::type::with_replaced_dtype(const ndt::type& replacement_tp, intptr_t replace_ndim) const
{
    ndt::type result;
    bool was_transformed;
    replace_dtype_extra extra(replacement_tp, replace_ndim);
    replace_dtype(*this, &extra, result, was_transformed);
    return result;
}

intptr_t ndt::type::get_dim_size(const char *metadata, const char *data) const {
    if (get_kind() == uniform_dim_kind) {
        return static_cast<const base_uniform_dim_type *>(m_extended)->get_dim_size(metadata, data);
    } else if (get_kind() == struct_kind) {
        return static_cast<const base_struct_type *>(m_extended)->get_field_count();
    } else if (get_ndim() > 0) {
        intptr_t dim_size = -1;
        m_extended->get_shape(1, 0, &dim_size, metadata, data);
        if (dim_size >= 0) {
            return dim_size;
        }
    }

    std::stringstream ss;
    ss << "Cannot get the leading dimension size of dynd array with type " << *this;
    throw dynd::type_error(ss.str());
}

bool ndt::type::data_layout_compatible_with(const ndt::type& rhs) const
{
    if (extended() == rhs.extended()) {
        // If they're trivially identical, quickly return true
        return true;
    }
    if (get_data_size() != rhs.get_data_size() ||
                    get_metadata_size() != rhs.get_metadata_size()) {
        // The size of the data and metadata must be the same
        return false;
    }
    if (get_metadata_size() == 0 && is_pod() && rhs.is_pod()) {
        // If both are POD with no metadata, then they're compatible
        return true;
    }
    if (get_kind() == expression_kind || rhs.get_kind() == expression_kind) {
        // If either is an expression type, check compatibility with
        // the storage types
        return storage_type().data_layout_compatible_with(rhs.storage_type());
    }
    // Rules for the rest of the types
    switch (get_type_id()) {
        case string_type_id:
        case bytes_type_id:
        case json_type_id:
            switch (rhs.get_type_id()) {
                case string_type_id:
                case bytes_type_id:
                case json_type_id:
                    // All of string, bytes, json are compatible
                    return true;
                default:
                    return false;
            }
        case fixed_dim_type_id:
            // For fixed dimensions, it's data layout compatible if
            // the shape and strides match, and the element is data
            // layout compatible.
            if (rhs.get_type_id() == fixed_dim_type_id) {
                const fixed_dim_type *fdd = static_cast<const fixed_dim_type *>(extended());
                const fixed_dim_type *rhs_fdd = static_cast<const fixed_dim_type *>(rhs.extended());
                return fdd->get_fixed_dim_size() == rhs_fdd->get_fixed_dim_size() &&
                    fdd->get_fixed_stride() == rhs_fdd->get_fixed_stride() &&
                    fdd->get_element_type().data_layout_compatible_with(
                                    rhs_fdd->get_element_type());
            }
            break;
        case strided_dim_type_id:
        case var_dim_type_id:
            // For strided and var dimensions, it's data layout
            // compatible if the element is
            if (rhs.get_type_id() == get_type_id()) {
                const base_uniform_dim_type *budd = static_cast<const base_uniform_dim_type *>(extended());
                const base_uniform_dim_type *rhs_budd = static_cast<const base_uniform_dim_type *>(rhs.extended());
                return budd->get_element_type().data_layout_compatible_with(
                                    rhs_budd->get_element_type());
            }
            break;
        default:
            break;
    }
    return false;
}

std::ostream& dynd::ndt::operator<<(std::ostream& o, const ndt::type& rhs)
{
    switch (rhs.get_type_id()) {
        case uninitialized_type_id:
            o << "uninitialized";
            break;
        case bool_type_id:
            o << "bool";
            break;
        case int8_type_id:
            o << "int8";
            break;
        case int16_type_id:
            o << "int16";
            break;
        case int32_type_id:
            o << "int32";
            break;
        case int64_type_id:
            o << "int64";
            break;
        case int128_type_id:
            o << "int128";
            break;
        case uint8_type_id:
            o << "uint8";
            break;
        case uint16_type_id:
            o << "uint16";
            break;
        case uint32_type_id:
            o << "uint32";
            break;
        case uint64_type_id:
            o << "uint64";
            break;
        case uint128_type_id:
            o << "uint128";
            break;
        case float16_type_id:
            o << "float16";
            break;
        case float32_type_id:
            o << "float32";
            break;
        case float64_type_id:
            o << "float64";
            break;
        case float128_type_id:
            o << "float128";
            break;
        case complex_float32_type_id:
            o << "complex[float32]";
            break;
        case complex_float64_type_id:
            o << "complex[float64]";
            break;
        case void_type_id:
            o << "void";
            break;
        default:
            rhs.extended()->print_type(o);
            break;
    }

    return o;
}

ndt::type ndt::make_type(intptr_t ndim, const intptr_t *shape, const ndt::type& dtp)
{
    if (ndim > 0) {
        ndt::type result_tp = shape[ndim-1] >= 0
                        ? ndt::make_strided_dim(dtp)
                        : ndt::make_var_dim(dtp);
        for (intptr_t i = ndim-2; i >= 0; --i) {
            if (shape[i] >= 0) {
                result_tp = ndt::make_strided_dim(result_tp);
            } else {
                result_tp = ndt::make_var_dim(result_tp);
            }
        }
        return result_tp;
    } else {
        return dtp;
    }
}

ndt::type ndt::make_type(intptr_t ndim, const intptr_t *shape, const ndt::type& dtp, bool& out_any_var)
{
    if (ndim > 0) {
        ndt::type result_tp = dtp;
        for (intptr_t i = ndim - 1; i >= 0; --i) {
            if (shape[i] >= 0) {
                result_tp = ndt::make_strided_dim(result_tp);
            }
            else {
                result_tp = ndt::make_var_dim(result_tp);
                out_any_var = true;
            }
        }
        return result_tp;
    }
    else {
        return dtp;
    }
}

template<class T, class Tas>
static void print_as(std::ostream& o, const char *data)
{
    T value;
    memcpy(&value, data, sizeof(value));
    o << static_cast<Tas>(value);
}

void dynd::hexadecimal_print(std::ostream& o, char value)
{
    static char hexadecimal[] = "0123456789abcdef";
    unsigned char v = (unsigned char)value;
    o << hexadecimal[v >> 4] << hexadecimal[v & 0x0f];
}

void dynd::hexadecimal_print(std::ostream& o, unsigned char value)
{
    hexadecimal_print(o, static_cast<char>(value));
}

void dynd::hexadecimal_print(std::ostream& o, unsigned short value)
{
    // Standard printing is in big-endian order
    hexadecimal_print(o, static_cast<char>((value >> 8) & 0xff));
    hexadecimal_print(o, static_cast<char>(value & 0xff));
}

void dynd::hexadecimal_print(std::ostream& o, unsigned int value)
{
    // Standard printing is in big-endian order
    hexadecimal_print(o, static_cast<char>(value >> 24));
    hexadecimal_print(o, static_cast<char>((value >> 16) & 0xff));
    hexadecimal_print(o, static_cast<char>((value >> 8) & 0xff));
    hexadecimal_print(o, static_cast<char>(value & 0xff));
}

void dynd::hexadecimal_print(std::ostream& o, unsigned long value)
{
    if (sizeof(unsigned int) == sizeof(unsigned long)) {
        hexadecimal_print(o, static_cast<unsigned int>(value));
    } else {
        hexadecimal_print(o, static_cast<unsigned long long>(value));
    }
}

void dynd::hexadecimal_print(std::ostream& o, unsigned long long value)
{
    // Standard printing is in big-endian order
    hexadecimal_print(o, static_cast<char>(value >> 56));
    hexadecimal_print(o, static_cast<char>((value >> 48) & 0xff));
    hexadecimal_print(o, static_cast<char>((value >> 40) & 0xff));
    hexadecimal_print(o, static_cast<char>((value >> 32) & 0xff));
    hexadecimal_print(o, static_cast<char>((value >> 24) & 0xff));
    hexadecimal_print(o, static_cast<char>((value >> 16) & 0xff));
    hexadecimal_print(o, static_cast<char>((value >> 8) & 0xff));
    hexadecimal_print(o, static_cast<char>(value & 0xff));
}

void dynd::hexadecimal_print(std::ostream& o, const char *data, intptr_t element_size)
{
    for (int i = 0; i < element_size; ++i, ++data) {
        hexadecimal_print(o, *data);
    }
}

void dynd::print_builtin_scalar(type_id_t type_id, std::ostream& o, const char *data)
{
    switch (type_id) {
        case bool_type_id:
            o << (*data ? "true" : "false");
            break;
        case int8_type_id:
            print_as<int8_t, int32_t>(o, data);
            break;
        case int16_type_id:
            print_as<int16_t, int32_t>(o, data);
            break;
        case int32_type_id:
            print_as<int32_t, int32_t>(o, data);
            break;
        case int64_type_id:
            print_as<int64_t, int64_t>(o, data);
            break;
        case int128_type_id:
            print_as<dynd_int128, dynd_int128>(o, data);
            break;
        case uint8_type_id:
            print_as<uint8_t, uint32_t>(o, data);
            break;
        case uint16_type_id:
            print_as<uint16_t, uint32_t>(o, data);
            break;
        case uint32_type_id:
            print_as<uint32_t, uint32_t>(o, data);
            break;
        case uint64_type_id:
            print_as<uint64_t, uint64_t>(o, data);
            break;
        case uint128_type_id:
            print_as<dynd_uint128, dynd_uint128>(o, data);
            break;
        case float16_type_id:
            print_as<dynd_float16, float>(o, data);
            break;
        case float32_type_id:
            print_as<float, float>(o, data);
            break;
        case float64_type_id:
            print_as<double, double>(o, data);
            break;
        case float128_type_id:
            print_as<dynd_float128, dynd_float128>(o, data);
            break;
        case complex_float32_type_id:
            print_as<dynd_complex<float>, dynd_complex<float> >(o, data);
            break;
        case complex_float64_type_id:
            print_as<dynd_complex<double>, dynd_complex<double> >(o, data);
            break;
        case void_type_id:
            o << "(void)";
            break;
        default:
            stringstream ss;
            ss << "printing of dynd builtin type id " << type_id << " isn't supported yet";
            throw dynd::type_error(ss.str());
    }
}

void ndt::type::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    if (is_builtin()) {
        print_builtin_scalar(get_type_id(), o, data);
    } else {
        extended()->print_data(o, metadata, data);
    }
}
