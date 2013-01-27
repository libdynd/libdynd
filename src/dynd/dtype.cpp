//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtype.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/buffer_storage.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/buffered_unary_kernels.hpp>
#include <dynd/gfunc/make_callable.hpp>

#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/dtypes/datashape_parser.hpp>

#include <sstream>
#include <cstring>
#include <vector>

using namespace std;
using namespace dynd;

char *dynd::iterdata_broadcasting_terminator_incr(iterdata_common *iterdata, int DYND_UNUSED(level))
{
    // This repeats the same data over and over again, broadcasting additional leftmost dimensions
    iterdata_broadcasting_terminator *id = reinterpret_cast<iterdata_broadcasting_terminator *>(iterdata);
    return id->data;
}

char *dynd::iterdata_broadcasting_terminator_reset(iterdata_common *iterdata, char *data, int DYND_UNUSED(level))
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
    dtype(uint8_type_id),
    dtype(uint16_type_id),
    dtype(uint32_type_id),
    dtype(uint64_type_id),
    dtype(float32_type_id),
    dtype(float64_type_id),
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
        uint_kind,
        uint_kind,
        uint_kind,
        uint_kind,
        real_kind,
        real_kind,
        complex_kind,
        complex_kind,
        void_kind
    };
uint8_t dtype::builtin_data_sizes[builtin_type_id_count] = {
        0,
        1,
        1,
        2,
        4,
        8,
        1,
        2,
        4,
        8,
        4,
        8,
        8,
        16,
        0
    };
uint8_t dtype::builtin_data_alignments[builtin_type_id_count] = {
        1,
        1,
        1,
        2,
        4,
        8,
        1,
        2,
        4,
        8,
        4,
        8,
        4,
        8,
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

dtype dtype::apply_linear_index(int nindices, const irange *indices,
                int current_i, const dtype& root_dt, bool leading_dimension) const
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
    static void replace_scalar_types(const dtype& dt, const void *extra,
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


void dtype::get_single_compare_kernel(kernel_instance<compare_operations_t> &out_kernel) const {
    if (is_builtin()) {
        memcpy(out_kernel.kernel.ops,
                        builtin_dtype_comparisons_table[reinterpret_cast<intptr_t>(m_extended)-bool_type_id],
                        sizeof(out_kernel.kernel.ops));
    } else {
        extended()->get_single_compare_kernel(out_kernel);
    }
}

std::ostream& dynd::operator<<(std::ostream& o, const dtype& rhs)
{
    switch (rhs.get_type_id()) {
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
        case float32_type_id:
            o << "float32";
            break;
        case float64_type_id:
            o << "float64";
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
        case float32_type_id:
            print_as<float, float>(o, data);
            break;
        case float64_type_id:
            print_as<double, double>(o, data);
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
