//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtype.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/buffer_storage.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/buffered_unary_kernels.hpp>

#include <sstream>
#include <cstring>
#include <vector>

using namespace std;
using namespace dynd;

// Default destructor for the extended dtype does nothing
extended_dtype::~extended_dtype()
{
}

dtype extended_dtype::apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& DYND_UNUSED(root_dt)) const
{
    // Default to scalar behavior
    if (nindices == 0) {
        return dtype(this);
    } else {
        throw too_many_indices(current_i + nindices, current_i);
    }
}

void dynd::extended_dtype::get_shape(int DYND_UNUSED(i), std::vector<intptr_t>& DYND_UNUSED(out_shape)) const
{
    // Default to scalar behavior
}


size_t extended_dtype::get_default_element_size(int DYND_UNUSED(ndim), const intptr_t *DYND_UNUSED(shape)) const
{
    return get_element_size();
}

// TODO: Make this a pure virtual function eventually
size_t extended_dtype::get_metadata_size() const
{
    stringstream ss;
    ss << "TODO: get_metadata_size for " << dtype(this) << " is not implemented";
    throw std::runtime_error(ss.str());
}

// TODO: Make this a pure virtual function eventually
void extended_dtype::metadata_default_construct(char *metadata, int ndim, const intptr_t* shape) const
{
    stringstream ss;
    ss << "TODO: metadata_construct for " << dtype(this) << " is not implemented";
    throw std::runtime_error(ss.str());
}


// TODO: Make this a pure virtual function eventually
void extended_dtype::metadata_destruct(char *DYND_UNUSED(metadata)) const
{
    stringstream ss;
    ss << "TODO: metadata_destruct for " << dtype(this) << " is not implemented";
    throw std::runtime_error(ss.str());
}

// TODO: Make this a pure virtual function eventually
void extended_dtype::metadata_debug_dump(const char *DYND_UNUSED(metadata), std::ostream& DYND_UNUSED(o), const std::string& DYND_UNUSED(indent)) const
{
    stringstream ss;
    ss << "TODO: metadata_debug_dump for " << dtype(this) << " is not implemented";
    throw std::runtime_error(ss.str());
}


void dynd::extended_dtype::get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                assign_error_mode DYND_UNUSED(errmode),
                unary_specialization_kernel_instance& DYND_UNUSED(out_kernel)) const
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

void extended_dtype::get_single_compare_kernel(single_compare_kernel_instance& DYND_UNUSED(out_kernel)) const
{
        throw std::runtime_error("get_single_compare_kernel: this dtypes does not support comparisons");
}

extended_string_dtype::~extended_string_dtype()
{
}

inline /* TODO: DYND_CONSTEXPR */ dtype dynd::detail::internal_make_raw_dtype(char type_id, char kind, intptr_t element_size, char alignment)
{
    return dtype(type_id, kind, element_size, alignment);
}

const dtype dynd::static_builtin_dtypes[builtin_type_id_count + 1] = {
    dynd::detail::internal_make_raw_dtype(bool_type_id, bool_kind, 1, 1),
    dynd::detail::internal_make_raw_dtype(int8_type_id, int_kind, 1, 1),
    dynd::detail::internal_make_raw_dtype(int16_type_id, int_kind, 2, 2),
    dynd::detail::internal_make_raw_dtype(int32_type_id, int_kind, 4, 4),
    dynd::detail::internal_make_raw_dtype(int64_type_id, int_kind, 8, 8),
    dynd::detail::internal_make_raw_dtype(uint8_type_id, uint_kind, 1, 1),
    dynd::detail::internal_make_raw_dtype(uint16_type_id, uint_kind, 2, 2),
    dynd::detail::internal_make_raw_dtype(uint32_type_id, uint_kind, 4, 4),
    dynd::detail::internal_make_raw_dtype(uint64_type_id, uint_kind, 8, 8),
    dynd::detail::internal_make_raw_dtype(float32_type_id, real_kind, 4, 4),
    dynd::detail::internal_make_raw_dtype(float64_type_id, real_kind, 8, 8),
    dynd::detail::internal_make_raw_dtype(complex_float32_type_id, complex_kind, 8, 4),
    dynd::detail::internal_make_raw_dtype(complex_float64_type_id, complex_kind, 16, 8),
    dynd::detail::internal_make_raw_dtype(void_type_id, void_kind, 0, 1)
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
    : m_type_id(void_type_id), m_kind(void_kind), m_alignment(1),
      m_element_size(0), m_extended(NULL)
{
    // Default to a generic type with zero size
}

dtype::dtype(type_id_t type_id)
    : m_type_id(validate_type_id(type_id)),
      m_kind(static_builtin_dtypes[type_id].m_kind),
      m_alignment(static_builtin_dtypes[type_id].m_alignment),
      m_element_size(static_builtin_dtypes[type_id].m_element_size),
      m_extended(NULL)
{
}

dtype::dtype(int type_id)
    : m_type_id(validate_type_id((type_id_t)type_id)),
      m_kind(static_builtin_dtypes[type_id].m_kind),
      m_alignment(static_builtin_dtypes[type_id].m_alignment),
      m_element_size(static_builtin_dtypes[type_id].m_element_size),
      m_extended(NULL)
{
}

dtype::dtype(const std::string& rep)
    : m_extended(NULL)
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

dtype dynd::dtype::index(int nindices, const irange *indices) const
{
    if (m_extended == NULL) {
        if (nindices == 0) {
            return *this;
        } else {
            throw too_many_indices(nindices, 0);
        }
    } else {
        return m_extended->apply_linear_index(nindices, indices, 0, *this);
    }
}

dtype dynd::dtype::apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const
{
    if (m_extended == NULL) {
        if (nindices == 0) {
            return *this;
        } else {
            throw too_many_indices(nindices + current_i, current_i);
        }
    } else {
        return m_extended->apply_linear_index(nindices, indices, current_i, root_dt);
    }
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

std::ostream& dynd::operator<<(std::ostream& o, const dtype& rhs)
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

void dynd::dtype::print_element(std::ostream& o, const char *data, const char *metadata) const
{
    if (extended() != NULL) {
        extended()->print_element(o, data, metadata);
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

dtype dynd::make_fixedbytes_dtype(intptr_t element_size, intptr_t alignment)
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
