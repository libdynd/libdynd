//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtype.hpp>
#include <dynd/dtypes/base_uniform_dim_dtype.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/buffer_storage.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/gfunc/make_callable.hpp>
#include <dynd/gfunc/call_callable.hpp>

#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/dtypes/datashape_parser.hpp>

#include <sstream>
#include <cstring>
#include <vector>

using namespace std;
using namespace dynd;

char *dynd::iterdata_broadcasting_terminator_incr(iterdata_common *iterdata, size_t DYND_UNUSED(level))
{
    // This repeats the same data over and over again, broadcasting additional leftmost dimensions
    iterdata_broadcasting_terminator *id = reinterpret_cast<iterdata_broadcasting_terminator *>(iterdata);
    return id->data;
}

char *dynd::iterdata_broadcasting_terminator_reset(iterdata_common *iterdata, char *data, size_t DYND_UNUSED(level))
{
    iterdata_broadcasting_terminator *id = reinterpret_cast<iterdata_broadcasting_terminator *>(iterdata);
    id->data = data;
    return data;
}

const dtype dynd::static_builtin_dtypes[builtin_type_id_count] = {
    dtype(uninitialized_type_id),
    dtype(bool_type_id),
    dtype(int8_type_id),
    dtype(int16_type_id),
    dtype(int32_type_id),
    dtype(int64_type_id),
    dtype(int128_type_id),
    dtype(uint8_type_id),
    dtype(uint16_type_id),
    dtype(uint32_type_id),
    dtype(uint64_type_id),
    dtype(uint128_type_id),
    dtype(float16_type_id),
    dtype(float32_type_id),
    dtype(float64_type_id),
    dtype(float128_type_id),
    dtype(complex_float32_type_id),
    dtype(complex_float64_type_id),
    dtype(void_type_id)
};

uint8_t dtype::builtin_kinds[builtin_type_id_count] = {
        void_kind,
        bool_kind,
        int_kind,
        int_kind,
        int_kind,
        int_kind,
        int_kind,
        uint_kind,
        uint_kind,
        uint_kind,
        uint_kind,
        uint_kind,
        real_kind,
        real_kind,
        real_kind,
        real_kind,
        complex_kind,
        complex_kind,
        void_kind
    };
uint8_t dtype::builtin_data_sizes[builtin_type_id_count] = {
        0,
        sizeof(dynd_bool),
        sizeof(int8_t),
        sizeof(int16_t),
        sizeof(int32_t),
        sizeof(int64_t),
        sizeof(dynd_int128),
        sizeof(uint8_t),
        sizeof(uint16_t),
        sizeof(uint32_t),
        sizeof(uint64_t),
        sizeof(dynd_uint128),
        sizeof(dynd_float16),
        sizeof(float),
        sizeof(double),
        sizeof(dynd_float128),
        sizeof(complex<float>),
        sizeof(complex<double>),
        0
    };
uint8_t dtype::builtin_data_alignments[builtin_type_id_count] = {
        1,
        1,
        1,
        scalar_align_of<int16_t>::value,
        scalar_align_of<int32_t>::value,
        scalar_align_of<int64_t>::value,
        scalar_align_of<dynd_int128>::value,
        1,
        scalar_align_of<uint16_t>::value,
        scalar_align_of<uint32_t>::value,
        scalar_align_of<uint64_t>::value,
        scalar_align_of<dynd_uint128>::value,
        scalar_align_of<dynd_float16>::value,
        scalar_align_of<float>::value,
        scalar_align_of<double>::value,
        scalar_align_of<dynd_float128>::value,
        scalar_align_of<complex<float> >::value,
        scalar_align_of<complex<double> >::value,
        1
    };



dtype::dtype(const std::string& rep)
    : m_extended(NULL)
{
    dtype_from_datashape(rep).swap(*this);
}

dtype::dtype(const char *rep_begin, const char *rep_end)
    : m_extended(NULL)
{
    dtype_from_datashape(rep_begin, rep_end).swap(*this);
}


dtype dtype::at_array(int nindices, const irange *indices) const
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

ndobject dtype::p(const char *property_name) const
{
    if (!is_builtin()) {
        const std::pair<std::string, gfunc::callable> *properties;
        size_t count;
        extended()->get_dynamic_dtype_properties(&properties, &count);
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
    ss << "dynd dtype does not have property " << property_name;
    throw runtime_error(ss.str());
}

ndobject dtype::p(const std::string& property_name) const
{
    if (!is_builtin()) {
        const std::pair<std::string, gfunc::callable> *properties;
        size_t count;
        extended()->get_dynamic_dtype_properties(&properties, &count);
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
    ss << "dynd dtype does not have property " << property_name;
    throw runtime_error(ss.str());
}

dtype dtype::apply_linear_index(size_t nindices, const irange *indices,
                size_t current_i, const dtype& root_dt, bool leading_dimension) const
{
    if (is_builtin()) {
        if (nindices == 0) {
            return *this;
        } else {
            throw too_many_indices(*this, nindices + current_i, current_i);
        }
    } else {
        return m_extended->apply_linear_index(nindices, indices, current_i, root_dt, leading_dimension);
    }
}

namespace {
    struct replace_scalar_type_extra {
        replace_scalar_type_extra(const dtype& dt, assign_error_mode em)
            : scalar_dtype(dt), errmode(em)
        {
        }
        const dtype& scalar_dtype;
        assign_error_mode errmode;
    };
    static void replace_scalar_types(const dtype& dt, void *extra,
                dtype& out_transformed_dtype, bool& out_was_transformed)
    {
        const replace_scalar_type_extra *e = reinterpret_cast<const replace_scalar_type_extra *>(extra);
        if (dt.is_scalar()) {
            out_transformed_dtype = make_convert_dtype(e->scalar_dtype, dt, e->errmode);
            out_was_transformed = true;
        } else {
            dt.extended()->transform_child_dtypes(&replace_scalar_types, extra, out_transformed_dtype, out_was_transformed);
        }
    }
} // anonymous namespace

dtype dtype::with_replaced_scalar_types(const dtype& scalar_dtype, assign_error_mode errmode) const
{
    dtype result;
    bool was_transformed;
    replace_scalar_type_extra extra(scalar_dtype, errmode);
    replace_scalar_types(*this, &extra, result, was_transformed);
    return result;
}

namespace {
    struct replace_udtype_extra {
        replace_udtype_extra(const dtype& udtype_, size_t replace_undim_)
            : udtype(udtype_), replace_undim(replace_undim_)
        {
        }
        const dtype& udtype;
        size_t replace_undim;
    };
    static void replace_udtype(const dtype& dt, void *extra,
                dtype& out_transformed_dtype, bool& out_was_transformed)
    {
        const replace_udtype_extra *e = reinterpret_cast<const replace_udtype_extra *>(extra);
        if (dt.get_undim() == e->replace_undim) {
            out_transformed_dtype = e->udtype;
            out_was_transformed = true;
        } else {
            dt.extended()->transform_child_dtypes(&replace_udtype, extra, out_transformed_dtype, out_was_transformed);
        }
    }
} // anonymous namespace

dtype dtype::with_replaced_udtype(const dtype& udtype, size_t replace_undim) const
{
    dtype result;
    bool was_transformed;
    replace_udtype_extra extra(udtype, replace_undim);
    replace_udtype(*this, &extra, result, was_transformed);
    return result;
}

intptr_t dtype::get_dim_size(const char *metadata, const char *data) const {
    if (get_kind() == uniform_dim_kind) {
        return static_cast<const base_uniform_dim_dtype *>(m_extended)->get_dim_size(metadata, data);
    } else if (get_kind() == struct_kind) {
        return static_cast<const base_struct_dtype *>(m_extended)->get_field_count();
    } else if (get_undim() > 0) {
        dimvector shape(get_undim());
        m_extended->get_shape(0, shape.get(), metadata);
        return shape[0];
    } else {
        std::stringstream ss;
        ss << "Cannot get the leading dimension size of ndobject with dtype " << *this;
        throw std::runtime_error(ss.str());
    }
}

bool dtype::data_layout_compatible_with(const dtype& rhs) const
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
        // If either is an expression dtype, check compatibility with
        // the storage dtypes
        return storage_dtype().data_layout_compatible_with(rhs.storage_dtype());
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
                const fixed_dim_dtype *fdd = static_cast<const fixed_dim_dtype *>(extended());
                const fixed_dim_dtype *rhs_fdd = static_cast<const fixed_dim_dtype *>(rhs.extended());
                return fdd->get_fixed_dim_size() == rhs_fdd->get_fixed_dim_size() &&
                    fdd->get_fixed_stride() == rhs_fdd->get_fixed_stride() &&
                    fdd->get_element_dtype().data_layout_compatible_with(
                                    rhs_fdd->get_element_dtype());
            }
            break;
        case strided_dim_type_id:
        case var_dim_type_id:
            // For strided and var dimensions, it's data layout
            // compatible if the element is
            if (rhs.get_type_id() == get_type_id()) {
                const base_uniform_dim_dtype *budd = static_cast<const base_uniform_dim_dtype *>(extended());
                const base_uniform_dim_dtype *rhs_budd = static_cast<const base_uniform_dim_dtype *>(rhs.extended());
                return budd->get_element_dtype().data_layout_compatible_with(
                                    rhs_budd->get_element_dtype());
            }
            break;
        default:
            break;
    }
    return false;
}

std::ostream& dynd::operator<<(std::ostream& o, const dtype& rhs)
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
            o << "complex<float32>";
            break;
        case complex_float64_type_id:
            o << "complex<float64>";
            break;
        case void_type_id:
            o << "void";
            break;
        default:
            rhs.extended()->print_dtype(o);
            break;
    }

    return o;
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
            print_as<complex<float>, complex<float> >(o, data);
            break;
        case complex_float64_type_id:
            print_as<complex<double>, complex<double> >(o, data);
            break;
        case void_type_id:
            o << "(void)";
            break;
        default:
            stringstream ss;
            ss << "printing of dynd builtin type id " << type_id << " isn't supported yet";
            throw std::runtime_error(ss.str());
    }
}

void dtype::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    if (is_builtin()) {
        print_builtin_scalar(get_type_id(), o, data);
    } else {
        extended()->print_data(o, metadata, data);
    }
}
