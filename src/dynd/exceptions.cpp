//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <sstream>
#include <iostream> // for DEBUG

#include <dynd/exceptions.hpp>
#include <dynd/array.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/types/datashape_formatter.hpp>
#include <dynd/kernels/comparison_kernels.hpp>

#include <utf8.h>

using namespace std;
using namespace dynd;

broadcast_error::broadcast_error(const std::string& m)
    : dynd_exception("broadcast error", m)
{
}

inline string broadcast_error_message(intptr_t dst_ndim, const intptr_t *dst_shape,
                    intptr_t src_ndim, const intptr_t *src_shape)
{
    stringstream ss;

    ss << "cannot broadcast shape ";
    print_shape(ss, src_ndim, src_shape);
    ss << " to shape ";
    print_shape(ss, dst_ndim, dst_shape);

    return ss.str();
}

broadcast_error::broadcast_error(intptr_t dst_ndim, const intptr_t *dst_shape,
                    intptr_t src_ndim, const intptr_t *src_shape)
    : dynd_exception("broadcast error", broadcast_error_message(dst_ndim, dst_shape, src_ndim, src_shape))
{
}

inline string broadcast_error_message(const nd::array& dst, const nd::array& src)
{
    vector<intptr_t> dst_shape = dst.get_shape(), src_shape = src.get_shape();
    stringstream ss;

    ss << "cannot broadcast dynd array with type ";
    ss << src.get_type() << " and shape ";
    print_shape(ss, src_shape);
    ss << " to type " << dst.get_type() << " and shape ";
    print_shape(ss, dst_shape);

    return ss.str();
}

broadcast_error::broadcast_error(const nd::array& dst, const nd::array& src)
    : dynd_exception("broadcast error", broadcast_error_message(dst, src))
{
}

inline string broadcast_error_message(intptr_t ninputs, const nd::array* inputs)
{
    stringstream ss;

    ss << "cannot broadcast input dynd operands with shapes ";
    for (intptr_t i = 0; i < ninputs; ++i) {
        intptr_t undim = inputs[i].get_ndim();
        dimvector shape(undim);
        inputs[i].get_shape(shape.get());
        print_shape(ss, undim, shape.get());
        if (i + 1 != ninputs) {
            ss << " ";
        }
    }

    return ss.str();
}

broadcast_error::broadcast_error(intptr_t ninputs, const nd::array *inputs)
    : dynd_exception("broadcast error", broadcast_error_message(ninputs, inputs))
{
}

inline string broadcast_error_message(const ndt::type& dst_tp, const char *dst_metadata,
                const ndt::type& src_tp, const char *src_metadata)
{
    stringstream ss;
    ss << "cannot broadcast input datashape '";
    format_datashape(ss, src_tp, src_metadata, NULL, false);
    ss << "' into datashape '";
    format_datashape(ss, dst_tp, dst_metadata, NULL, false);
    ss << "'";
    return ss.str();
}

broadcast_error::broadcast_error(const ndt::type& dst_tp, const char *dst_metadata,
                const ndt::type& src_tp, const char *src_metadata)
    : dynd_exception("broadcast error", broadcast_error_message(
                    dst_tp, dst_metadata, src_tp, src_metadata))
{
}

inline string broadcast_error_message(const ndt::type& dst_tp, const char *dst_metadata,
                const char *src_name)
{
    stringstream ss;
    ss << "cannot broadcast input " << src_name << " into datashape '";
    format_datashape(ss, dst_tp, dst_metadata, NULL, false);
    ss << "'";
    return ss.str();
}

broadcast_error::broadcast_error(const ndt::type& dst_tp, const char *dst_metadata,
                const char *src_name)
    : dynd_exception("broadcast error", broadcast_error_message(
                    dst_tp, dst_metadata, src_name))
{
}

inline string broadcast_error_message(intptr_t dst_size, intptr_t src_size,
                const char *dst_name, const char *src_name)
{
    stringstream ss;
    ss << "cannot broadcast input " << src_name << " with size " << src_size;
    ss << " into output " << dst_name << " with size " << dst_size;
    return ss.str();
}

broadcast_error::broadcast_error(intptr_t dst_size, intptr_t src_size,
                const char *dst_name, const char *src_name)
    : dynd_exception("broadcast error", broadcast_error_message(
                    dst_size, src_size, dst_name, src_name))
{
}

inline string too_many_indices_message(const ndt::type& dt, intptr_t nindices, intptr_t ndim)
{
    std::stringstream ss;

    ss << "provided " << nindices << " indices to dynd type " << dt << ", but only ";
    ss << ndim << " dimensions available";

    return ss.str();
}

dynd::too_many_indices::too_many_indices(const ndt::type& dt, intptr_t nindices, intptr_t ndim)
    : dynd_exception("too many indices", too_many_indices_message(dt, nindices, ndim))
{
    //cout << "throwing too_many_indices\n";
}

inline string index_out_of_bounds_message(intptr_t i, size_t axis, intptr_t ndim, const intptr_t *shape)
{
    stringstream ss;

    ss << "index " << i << " is out of bounds for axis " << axis;
    ss << " in shape ";
    print_shape(ss, ndim, shape);

    return ss.str();
}

inline string index_out_of_bounds_message(intptr_t i, intptr_t dimension_size)
{
    stringstream ss;

    ss << "index " << i << " is out of bounds for dimension of size " << dimension_size;

    return ss.str();
}

index_out_of_bounds::index_out_of_bounds(intptr_t i, size_t axis, intptr_t ndim, const intptr_t *shape)
    : dynd_exception("index out of bounds", index_out_of_bounds_message(i, axis, ndim, shape))
{
}

index_out_of_bounds::index_out_of_bounds(intptr_t i, size_t axis, const std::vector<intptr_t>& shape)
    : dynd_exception("index out of bounds", index_out_of_bounds_message(i, axis, (int)shape.size(), shape.empty() ? NULL : &shape[0]))
{
}

index_out_of_bounds::index_out_of_bounds(intptr_t i, intptr_t dimension_size)
    : dynd_exception("index out of bounds", index_out_of_bounds_message(i, dimension_size))
{
}

inline string axis_out_of_bounds_message(size_t i, intptr_t ndim)
{
    stringstream ss;

    ss << "axis " << i << " is not a valid axis for an " << ndim << " dimensional operation";

    return ss.str();
}

dynd::axis_out_of_bounds::axis_out_of_bounds(size_t i, intptr_t ndim)
    : dynd_exception("axis out of bounds", axis_out_of_bounds_message(i, ndim))
{
    //cout << "throwing axis_out_of_bounds\n";
}

inline void print_slice(std::ostream& o, const irange& i)
{
    if (i.step() == 0) {
        o << '[' << i.start() << ']';
    } else {
        o << '[';
        if (i.start() != std::numeric_limits<intptr_t>::min()) {
            o << i.start();
        }
        o << ':';
        if (i.finish() != std::numeric_limits<intptr_t>::max()) {
            o << i.finish();
        }
        if (i.step() != 1) {
            o << ':';
            o << i.step();
        }
        o << ']';
    }
}

inline string irange_out_of_bounds_message(const irange& i, size_t axis, intptr_t ndim, const intptr_t *shape)
{
    stringstream ss;

    ss << "index range ";
    print_slice(ss, i);
    ss << " is out of bounds for axis " << axis;
    ss << " in shape ";
    print_shape(ss, ndim, shape);

    return ss.str();
}

inline string irange_out_of_bounds_message(const irange& i, intptr_t dimension_size)
{
    stringstream ss;

    ss << "index range ";
    print_slice(ss, i);
    ss << " is out of bounds for dimension of size " << dimension_size;

    return ss.str();
}

irange_out_of_bounds::irange_out_of_bounds(const irange& i, size_t axis, intptr_t ndim, const intptr_t *shape)
    : dynd_exception("irange out of bounds", irange_out_of_bounds_message(i, axis, ndim, shape))
{
    //cout << "throwing irange_out_of_bounds\n";
}

irange_out_of_bounds::irange_out_of_bounds(const irange& i, size_t axis, const std::vector<intptr_t>& shape)
    : dynd_exception("irange out of bounds", irange_out_of_bounds_message(i, axis, (int)shape.size(), shape.empty() ? NULL : &shape[0]))
{
}

irange_out_of_bounds::irange_out_of_bounds(const irange& i, intptr_t dimension_size)
    : dynd_exception("irange out of bounds", irange_out_of_bounds_message(i, dimension_size))
{
}

inline string invalid_type_id_message(int type_id)
{
    stringstream ss;

    ss << "the id " << type_id << " is not valid";

    return ss.str();
}

invalid_type_id::invalid_type_id(int type_id)
    : type_error("invalid type id", invalid_type_id_message(type_id))
{
    //cout << "throwing invalid_type_id\n";
}

inline string string_decode_error_message(const char *begin, const char *end, string_encoding_t encoding)
{
    stringstream ss;
    ss << "encoded bytes ";
    hexadecimal_print(ss, begin, end - begin);
    ss << " are invalid in " << encoding << " input.";
    return ss.str();
}

string_decode_error::string_decode_error(const char *begin, const char *end, string_encoding_t encoding)
    : dynd_exception("string decode error", string_decode_error_message(begin, end, encoding)),
                    m_bytes(begin, end), m_encoding(encoding)
{
}

inline string string_encode_error_message(uint32_t cp, string_encoding_t encoding)
{
    stringstream ss;
    if (!utf8::internal::is_code_point_valid(cp)) {
        ss << "Cannot encode invalid code point U+";
        hexadecimal_print(ss, cp);
        ss << " as " << encoding;
        return ss.str();
    } else {
        ss << "Cannot encode input code point U+";
        hexadecimal_print(ss, cp);
        ss << " as " << encoding;
        return ss.str();
    }
}

string_encode_error::string_encode_error(uint32_t cp, string_encoding_t encoding)
    : dynd_exception("string encode error", string_encode_error_message(cp, encoding)),
                    m_cp(cp), m_encoding(encoding)
{
}

inline string not_comparable_error_message(const ndt::type& lhs, const ndt::type& rhs,
                comparison_type_t comptype)
{
    stringstream ss;
    ss << "Cannot compare values of types " << lhs << " and " << rhs;
    ss << " with comparison operator ";
    switch (comptype) {
        case comparison_type_sorting_less:
            ss << "'sorting <'";
            break;
        case comparison_type_less:
            ss << "'<'";
            break;
        case comparison_type_less_equal:
            ss << "'<='";
            break;
        case comparison_type_equal:
            ss << "'=='";
            break;
        case comparison_type_not_equal:
            ss << "'!='";
            break;
        case comparison_type_greater_equal:
            ss << "'>='";
            break;
        case comparison_type_greater:
            ss << "'>'";
            break;
    }
    return ss.str();
}

not_comparable_error::not_comparable_error(const ndt::type& lhs, const ndt::type& rhs,
                comparison_type_t comptype)
    : dynd_exception("not comparable error",
                    not_comparable_error_message(lhs, rhs, comptype))
{
}

#ifdef DYND_CUDA

inline string cuda_runtime_error_message(cudaError_t error)
{
    return cudaGetErrorString(error);
}

cuda_runtime_error::cuda_runtime_error(cudaError_t error)
    : std::runtime_error(cuda_runtime_error_message(error)), m_error(error)
{
}

#endif // DYND_CUDA
