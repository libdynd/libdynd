//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
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
dnd::extended_dtype::~extended_dtype()
{
}

void dnd::extended_dtype::get_operand_to_value_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride,
                            kernel_instance<unary_operation_t>& out_kernel) const
{
    throw std::runtime_error("get_operand_to_value_operation: this operation is only for expression_kind dtypes");
}

void dnd::extended_dtype::get_value_to_operand_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride,
                            kernel_instance<unary_operation_t>& out_kernel) const
{
    throw std::runtime_error("get_value_to_operand_operation: this operation is only for expression_kind dtypes");
}

dtype dnd::extended_dtype::with_replaced_storage_dtype(const dtype& replacement_dtype) const
{
    throw std::runtime_error("with_replaced_storage_dtype: this operation is only for expression_kind dtypes");
}



/**
 * A static look-up table structure which contains data about the type ids.
 * This must match up with the type id enumeration, and has space that
 * is intended to be filled up with more data when custom dtypes are added.
 */
static struct {
    unsigned char kind, alignment, itemsize;
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
    "utf8",
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
    // 0 <= type_id <= utf8_type_id
    if ((unsigned int)type_id <= utf8_type_id) {
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
      m_itemsize(0), m_data()
{
    // Default to a generic type with zero size
}

dtype::dtype(type_id_t type_id)
    : m_type_id(validate_type_id(type_id)),
      m_kind(basic_type_id_info[type_id].kind),
      m_alignment(basic_type_id_info[type_id].alignment),
      m_itemsize(basic_type_id_info[type_id].itemsize),
      m_data()
{
}

dtype::dtype(int type_id)
    : m_type_id(validate_type_id((type_id_t)type_id)),
      m_kind(basic_type_id_info[type_id].kind),
      m_alignment(basic_type_id_info[type_id].alignment),
      m_itemsize(basic_type_id_info[type_id].itemsize),
      m_data()
{
}

dtype::dtype(type_id_t type_id, uintptr_t size)
    : m_type_id(validate_type_id(type_id)),
      m_kind(basic_type_id_info[type_id].kind),
      m_alignment(basic_type_id_info[type_id].alignment),
      m_itemsize(basic_type_id_info[type_id].itemsize),
      m_data()
{
    if (m_itemsize != 0) {
        if (m_itemsize != size) {
            throw std::runtime_error(std::string() + "invalid itemsize for type id "
                                                    + get_type_id_basename(type_id));
        }
    } else {
        m_itemsize = size;
    }
}

dtype::dtype(int type_id, uintptr_t size)
    : m_type_id(validate_type_id((type_id_t)type_id)),
      m_kind(basic_type_id_info[type_id].kind),
      m_alignment(basic_type_id_info[type_id].alignment),
      m_itemsize(basic_type_id_info[type_id].itemsize),
      m_data()
{
    if (m_itemsize != 0) {
        if (m_itemsize != size) {
            throw std::runtime_error(std::string() + "invalid itemsize for type id "
                                                    + get_type_id_basename((type_id_t)type_id));
        }
    } else {
        m_itemsize = size;
    }
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
            m_itemsize = basic_type_id_info[id].itemsize;
            return;
        }
    }

    throw std::runtime_error(std::string() + "invalid type string \"" + rep + "\"");
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
        case utf8_type_id:
            if (rhs.itemsize() == 0) {
                o << "utf8";
            } else {
                o << "utf8<" << rhs.itemsize() << ">";
            }
            break;
        case pattern_type_id:
            o << "pattern";
            break;
        default:
            if (rhs.extended()) {
                rhs.extended()->print(o);
            } else {
                o << "<dtype without formatting support>";
            }
            break;
    }

    return o;
}

template<class T>
static void strided_print(std::ostream& o, const char *data, intptr_t stride, intptr_t size, const char *separator)
{
    T value;
    memcpy(&value, data, sizeof(value));
    o << value;
    for (intptr_t i = 1; i < size; ++i) {
        data += stride;
        memcpy(&value, data, sizeof(value));
        o << separator << value;
    }
}

void dnd::dtype::print_data(std::ostream& o, const char *data, intptr_t stride, intptr_t size, const char *separator) const
{
    if (size > 0) {
        if (extended() != NULL) {
            extended()->print_data(o, *this, data, stride, size, separator);
        } else {
            // TODO: Handle byte-swapped dtypes
            switch (type_id()) {
                case bool_type_id:
                    o << (*data ? "true" : "false");
                    for (intptr_t i = 1; i < size; ++i) {
                        data += stride;
                        o << separator << (*data ? "true" : "false");
                    }
                    break;
                case int8_type_id:
                    strided_print<int8_t>(o, data, stride, size, separator);
                    break;
                case int16_type_id:
                    strided_print<int16_t>(o, data, stride, size, separator);
                    break;
                case int32_type_id:
                    strided_print<int32_t>(o, data, stride, size, separator);
                    break;
                case int64_type_id:
                    strided_print<int64_t>(o, data, stride, size, separator);
                    break;
                case uint8_type_id:
                    strided_print<uint8_t>(o, data, stride, size, separator);
                    break;
                case uint16_type_id:
                    strided_print<uint16_t>(o, data, stride, size, separator);
                    break;
                case uint32_type_id:
                    strided_print<uint32_t>(o, data, stride, size, separator);
                    break;
                case uint64_type_id:
                    strided_print<uint64_t>(o, data, stride, size, separator);
                    break;
                case float32_type_id:
                    strided_print<float>(o, data, stride, size, separator);
                    break;
                case float64_type_id:
                    strided_print<double>(o, data, stride, size, separator);
                    break;
                case complex_float32_type_id:
                    strided_print<complex<float> >(o, data, stride, size, separator);
                    break;
                case complex_float64_type_id:
                    strided_print<complex<double> >(o, data, stride, size, separator);
                    break;
                default:
                    stringstream ss;
                    ss << "printing of dtype " << *this << " isn't supported yet";
                    throw std::runtime_error(ss.str());
            }
        }
    }
}

void dnd::dtype::get_storage_to_value_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride,
                            kernel_instance<unary_operation_t>& out_kernel) const
{
    if (m_kind != expression_kind) {
        // If it's not an expression_kind dtype, return a simple copy operation
        get_dtype_assignment_kernel(*this, dst_fixedstride, src_fixedstride, out_kernel);
        return;
    } else {
        const dtype& next_dt = m_data->operand_dtype(*this);
        if (next_dt.kind() != expression_kind) {
            // If there is no chained expressions, return the function unchanged
            m_data->get_operand_to_value_operation(dst_fixedstride, src_fixedstride, out_kernel);
        } else {
            std::deque<kernel_instance<unary_operation_t> > kernels;
            std::deque<intptr_t> element_sizes;

            push_front_dtype_storage_to_value_kernels(*this, dst_fixedstride, src_fixedstride,
                                kernels, element_sizes);

            make_buffered_chain_unary_kernel(kernels, element_sizes, out_kernel);
        }
    }
}

void dnd::dtype::get_value_to_storage_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride,
                            kernel_instance<unary_operation_t>& out_kernel) const
{
    if (m_kind != expression_kind) {
        // If it's not an expression_kind dtype, return a simple copy operation
        get_dtype_assignment_kernel(*this, dst_fixedstride, src_fixedstride, out_kernel);
        return;
    } else {
        const dtype& next_dt = m_data->operand_dtype(*this);
        if (next_dt.kind() != expression_kind) {
            // If there is no chained expressions, return the function unchanged
            m_data->get_value_to_operand_operation(dst_fixedstride, src_fixedstride, out_kernel);
        } else {
            std::deque<kernel_instance<unary_operation_t> > kernels;
            std::deque<intptr_t> element_sizes;

            push_back_dtype_value_to_storage_kernels(*this, dst_fixedstride, src_fixedstride,
                                kernels, element_sizes);

            make_buffered_chain_unary_kernel(kernels, element_sizes, out_kernel);
        }
    }
}