//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/dtype.hpp>
#include <dnd/exceptions.hpp>
#include <dnd/dtype_assign.hpp>
#include <dnd/buffer_storage.hpp>
#include <dnd/kernels/assignment_kernels.hpp>
#include <dnd/kernels/buffered_unary_kernels.hpp>

#include <sstream>
#include <cstring>
#include <vector>

using namespace std;
using namespace dnd;

// Default destructor for the extended dtype does nothing
extended_dtype::~extended_dtype()
{
}

void dnd::extended_dtype::get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                assign_error_mode DND_UNUSED(errmode),
                unary_specialization_kernel_instance& DND_UNUSED(out_kernel)) const
{
    stringstream ss;
    ss << "get_dtype_assignment_kernel has not been implemented for ";
    if (this == dst_dt.extended()) {
        ss << dst_dt;
    } else {
        ss << src_dt;
    }
    throw std::runtime_error(ss.str());
}

const unary_specialization_kernel_instance& extended_dtype::get_operand_to_value_kernel() const
{
    throw std::runtime_error("get_operand_to_value_kernel: this operation is only for expression_kind dtypes");
}

const unary_specialization_kernel_instance& extended_dtype::get_value_to_operand_kernel() const
{
    throw std::runtime_error("get_value_to_operand_kernel: this operation is only for expression_kind dtypes");
}

dtype extended_dtype::with_replaced_storage_dtype(const dtype& DND_UNUSED(replacement_dtype)) const
{
    throw std::runtime_error("with_replaced_storage_dtype: this operation is only for expression_kind dtypes");
}

void extended_dtype::get_single_compare_kernel(single_compare_kernel_instance& DND_UNUSED(out_kernel)) const
{
        throw std::runtime_error("get_single_compare_kernel: this dtypes does not support comparisons");
}

extended_string_dtype::~extended_string_dtype()
{
}

inline /* TODO: DND_CONSTEXPR */ dtype dnd::detail::internal_make_raw_dtype(char type_id, char kind, intptr_t element_size, char alignment)
{
    return dtype(type_id, kind, element_size, alignment);
}

const dtype dnd::static_builtin_dtypes[builtin_type_id_count + 1] = {
    dnd::detail::internal_make_raw_dtype(bool_type_id, bool_kind, 1, 1),
    dnd::detail::internal_make_raw_dtype(int8_type_id, int_kind, 1, 1),
    dnd::detail::internal_make_raw_dtype(int16_type_id, int_kind, 2, 2),
    dnd::detail::internal_make_raw_dtype(int32_type_id, int_kind, 4, 4),
    dnd::detail::internal_make_raw_dtype(int64_type_id, int_kind, 8, 8),
    dnd::detail::internal_make_raw_dtype(uint8_type_id, uint_kind, 1, 1),
    dnd::detail::internal_make_raw_dtype(uint16_type_id, uint_kind, 2, 2),
    dnd::detail::internal_make_raw_dtype(uint32_type_id, uint_kind, 4, 4),
    dnd::detail::internal_make_raw_dtype(uint64_type_id, uint_kind, 8, 8),
    dnd::detail::internal_make_raw_dtype(float32_type_id, real_kind, 4, 4),
    dnd::detail::internal_make_raw_dtype(float64_type_id, real_kind, 8, 8),
    dnd::detail::internal_make_raw_dtype(complex_float32_type_id, complex_kind, 8, 4),
    dnd::detail::internal_make_raw_dtype(complex_float64_type_id, complex_kind, 16, 8),
    dnd::detail::internal_make_raw_dtype(void_type_id, void_kind, 0, 1)
};

/**
 * Validates that the given type ID is a proper ID. Throws
 * an exception if not.
 *
 * @param type_id  The type id to validate.
 */
static inline int validate_type_id(type_id_t type_id)
{
    // 0 <= type_id < builtin_type_id_count
    if ((unsigned int)type_id < builtin_type_id_count + 1) {
        return type_id;
    } else {
        throw invalid_type_id((int)type_id);
    }
}

dtype::dtype()
    : m_type_id(pattern_type_id), m_kind(pattern_kind), m_alignment(1),
      m_element_size(0), m_data()
{
    // Default to a generic type with zero size
}

dtype::dtype(type_id_t type_id)
    : m_type_id(validate_type_id(type_id)),
      m_kind(static_builtin_dtypes[type_id].m_kind),
      m_alignment(static_builtin_dtypes[type_id].m_alignment),
      m_element_size(static_builtin_dtypes[type_id].m_element_size),
      m_data()
{
}

dtype::dtype(int type_id)
    : m_type_id(validate_type_id((type_id_t)type_id)),
      m_kind(static_builtin_dtypes[type_id].m_kind),
      m_alignment(static_builtin_dtypes[type_id].m_alignment),
      m_element_size(static_builtin_dtypes[type_id].m_element_size),
      m_data()
{
}

dtype::dtype(const std::string& rep)
    : m_data()
{
    static const char *type_id_names[builtin_type_id_count] = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "complex<float32>",
        "complex<float64>",
    };

    // TODO: make a decent efficient parser
    for (int id = 0; id < builtin_type_id_count; ++id) {
        if (rep == type_id_names[id]) {
            m_type_id = (type_id_t)id;
            m_kind = static_builtin_dtypes[id].m_kind;
            m_alignment = static_builtin_dtypes[id].m_alignment;
            m_element_size = static_builtin_dtypes[id].m_element_size;
            return;
        }
    }

    if (rep == "void") {
        m_type_id = void_type_id;
        m_kind = void_kind;
        m_alignment = 1;
        m_element_size = 0;
        return;
    }

    throw std::runtime_error(std::string() + "invalid type string \"" + rep + "\"");
}

void dtype::get_single_compare_kernel(single_compare_kernel_instance &out_kernel) const {
    if (extended() != NULL) {
        return extended()->get_single_compare_kernel(out_kernel);
    } else if (type_id() >= 0 && type_id() < builtin_type_id_count) {
        out_kernel.comparisons = builtin_dtype_comparisons_table[type_id()];
    } else {
        stringstream ss;
        ss << "Cannot get single compare kernels for dtype " << *this;
        throw runtime_error(ss.str());
    }
}

std::ostream& dnd::operator<<(std::ostream& o, const dtype& rhs)
{
    switch (rhs.type_id()) {
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
        case fixedbytes_type_id:
            o << "fixedbytes<" << rhs.element_size() << "," << rhs.alignment() << ">";
            break;
        case void_type_id:
            o << "void";
            break;
        case pattern_type_id:
            o << "pattern";
            break;
        default:
            if (rhs.extended()) {
                rhs.extended()->print_dtype(o);
            } else {
                o << "<internal error: builtin dtype without formatting support>";
            }
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

void dnd::hexadecimal_print(std::ostream& o, char value)
{
    static char hexadecimal[] = "0123456789abcdef";
    unsigned char v = (unsigned char)value;
    o << hexadecimal[v >> 4] << hexadecimal[v & 0x0f];
}

void dnd::hexadecimal_print(std::ostream& o, uint16_t value)
{
    // Standard printing is in big-endian order
    hexadecimal_print(o, static_cast<char>((value >> 8) & 0xff));
    hexadecimal_print(o, static_cast<char>(value & 0xff));
}

void dnd::hexadecimal_print(std::ostream& o, uint32_t value)
{
    // Standard printing is in big-endian order
    hexadecimal_print(o, static_cast<char>(value >> 24));
    hexadecimal_print(o, static_cast<char>((value >> 16) & 0xff));
    hexadecimal_print(o, static_cast<char>((value >> 8) & 0xff));
    hexadecimal_print(o, static_cast<char>(value & 0xff));
}

void dnd::hexadecimal_print(std::ostream& o, const char *data, intptr_t element_size)
{
    for (int i = 0; i < element_size; ++i, ++data) {
        hexadecimal_print(o, *data);
    }
}

void dnd::dtype::print_element(std::ostream& o, const char * data) const
{
    if (extended() != NULL) {
        extended()->print_element(o, data);
    } else {
        switch (type_id()) {
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
            case fixedbytes_type_id:
                o << "0x";
                hexadecimal_print(o, data, m_element_size);
                break;
            case void_type_id:
                o << "(void)";
                break;
            default:
                stringstream ss;
                ss << "printing of dtype " << *this << " isn't supported yet";
                throw std::runtime_error(ss.str());
        }
    }
}

dtype dnd::make_fixedbytes_dtype(intptr_t element_size, intptr_t alignment)
{
    if (alignment > element_size) {
        std::stringstream ss;
        ss << "Cannot make a fixedbytes<" << element_size << "," << alignment << "> dtype, its alignment is greater than its size";
        throw std::runtime_error(ss.str());
    }
    if (alignment != 1 && alignment != 2 && alignment != 4 && alignment != 8 && alignment != 16) {
        std::stringstream ss;
        ss << "Cannot make a fixedbytes<" << element_size << "," << alignment << "> dtype, its alignment is not a small power of two";
        throw std::runtime_error(ss.str());
    }
    if ((element_size&(alignment-1)) != 0) {
        std::stringstream ss;
        ss << "Cannot make a fixedbytes<" << element_size << "," << alignment << "> dtype, its alignment does not divide into its element size";
        throw std::runtime_error(ss.str());
    }
    return dtype(fixedbytes_type_id, bytes_kind, element_size, alignment);
}
