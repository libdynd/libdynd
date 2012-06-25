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

/** The maximum number of type ids which can be defined */
#define DND_MAX_NUM_TYPE_IDS 64

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


/**
 * A static look-up table structure which contains data about the type ids.
 * This must match up with the type id enumeration, and has space that
 * is intended to be filled up with more data when custom dtypes are added.
 */
static struct {
    unsigned char kind, alignment, element_size;
} basic_type_id_info[DND_MAX_NUM_TYPE_IDS] = {
    {bool_kind, 1, 1},         // bool
    {int_kind, 1, 1},          // int8
    {int_kind, 2, 2},          // int16
    {int_kind, 4, 4},          // int32
    {int_kind, 8, 8},          // int64
    {uint_kind, 1, 1},         // uint8
    {uint_kind, 2, 2},         // uint16
    {uint_kind, 4, 4},         // uint32
    {uint_kind, 8, 8},         // uint64
    {real_kind, 4, 4},         // float32
    {real_kind, 8, 8},         // float64
    {complex_kind, 4, 8},      // complex<float32>
    {complex_kind, 8, 16},     // complex<float64>
    {string_kind, 1, 0},       // utf8
    {composite_kind, 1, 0},    // struct
    {composite_kind, 1, 0},    // subarray
    {pattern_kind, 1, 0}       // pattern
};

/**
 * A static look-up table which contains the names of all the type ids.
 * This must match up with the type id enumeration, and has space that
 * is intended to be filled up with more data when custom dtypes are added.
 */
static char type_id_names[DND_MAX_NUM_TYPE_IDS][32] = {
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
    "bytes<>",
    "fixedstring<>",
    "struct",
    "array",
    "convert",
    "pattern"
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
    if ((unsigned int)type_id < builtin_type_id_count) {
        return type_id;
    } else {
        throw invalid_type_id((int)type_id);
    }
}

const char *dnd::get_type_id_basename(type_id_t type_id)
{
    return type_id_names[validate_type_id(type_id)];
}

dtype::dtype()
    : m_type_id(pattern_type_id), m_kind(pattern_kind), m_alignment(1),
      m_element_size(0), m_data()
{
    // Default to a generic type with zero size
}

dtype::dtype(type_id_t type_id)
    : m_type_id(validate_type_id(type_id)),
      m_kind(basic_type_id_info[type_id].kind),
      m_alignment(basic_type_id_info[type_id].alignment),
      m_element_size(basic_type_id_info[type_id].element_size),
      m_data()
{
}

dtype::dtype(int type_id)
    : m_type_id(validate_type_id((type_id_t)type_id)),
      m_kind(basic_type_id_info[type_id].kind),
      m_alignment(basic_type_id_info[type_id].alignment),
      m_element_size(basic_type_id_info[type_id].element_size),
      m_data()
{
}

dtype::dtype(const std::string& rep)
    : m_data()
{
    // TODO: make a decent efficient parser
    for (int id = 0; id < builtin_type_id_count; ++id) {
        if (rep == type_id_names[id]) {
            m_type_id = (type_id_t)id;
            m_kind = basic_type_id_info[id].kind;
            m_alignment = basic_type_id_info[id].alignment;
            m_element_size = basic_type_id_info[id].element_size;
            return;
        }
    }

    throw std::runtime_error(std::string() + "invalid type string \"" + rep + "\"");
}

void dtype::get_single_compare_kernel(single_compare_kernel_instance &out_kernel) const {
    if (extended() != NULL) {
        return extended()->get_single_compare_kernel(out_kernel);
    }
    out_kernel.comparisons = builtin_dtype_comparisons_table[type_id()];
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
        case bytes_type_id:
            o << "bytes<" << rhs.element_size() << "," << rhs.alignment() << ">";
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
            case bytes_type_id:
                o << "0x";
                hexadecimal_print(o, data, m_element_size);
                break;
            default:
                stringstream ss;
                ss << "printing of dtype " << *this << " isn't supported yet";
                throw std::runtime_error(ss.str());
        }
    }
}

dtype dnd::make_bytes_dtype(intptr_t element_size, intptr_t alignment)
{
    if (alignment > element_size) {
        std::stringstream ss;
        ss << "Cannot make a bytes<" << element_size << "," << alignment << "> dtype, its alignment is greater than its size";
        throw std::runtime_error(ss.str());
    }
    if (alignment != 1 && alignment != 2 && alignment != 4 && alignment != 8 && alignment != 16) {
        std::stringstream ss;
        ss << "Cannot make a bytes<" << element_size << "," << alignment << "> dtype, its alignment is not a small power of two";
        throw std::runtime_error(ss.str());
    }
    if ((element_size&(alignment-1)) != 0) {
        std::stringstream ss;
        ss << "Cannot make a bytes<" << element_size << "," << alignment << "> dtype, its alignment does not divide into its element size";
        throw std::runtime_error(ss.str());
    }
    return dtype(bytes_type_id, bytes_kind, alignment, element_size);
}
