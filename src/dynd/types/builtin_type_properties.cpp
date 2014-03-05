//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/builtin_type_properties.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/property_type.hpp>
#include <dynd/gfunc/make_callable.hpp>

using namespace std;
using namespace dynd;

static nd::array property_complex_real(const nd::array& n) {
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "real"));
}

static nd::array property_complex_imag(const nd::array& n) {                        
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "imag"));
}

static nd::array property_complex_conj(const nd::array& n) {
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "conj"));
}

static pair<string, gfunc::callable> complex_array_properties[] = {
    pair<string, gfunc::callable>("real", gfunc::make_callable(&property_complex_real, "self")),
    pair<string, gfunc::callable>("imag", gfunc::make_callable(&property_complex_imag, "self")),
    pair<string, gfunc::callable>("conj", gfunc::make_callable(&property_complex_conj, "self")),
};

void dynd::get_builtin_type_dynamic_array_properties(
                type_id_t builtin_type_id,
                const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count)
{
    switch (builtin_type_id) {
        case complex_float32_type_id:
        case complex_float64_type_id:
            *out_properties = complex_array_properties;
            *out_count = sizeof(complex_array_properties) /
                            sizeof(complex_array_properties[0]);
            break;
        default:
            *out_properties = NULL;
            *out_count = 0;
            break;
    }
}

size_t dynd::get_builtin_type_elwise_property_index(
                type_id_t builtin_type_id,
                const std::string& property_name)
{
    switch (builtin_type_id) {
        case complex_float32_type_id:
        case complex_float64_type_id:
            if (property_name == "real") {
                return 0;
            } else if (property_name == "imag") {
                return 1;
            } else if (property_name == "conj") {
                return 2;
            }
        default:
            break;
    }

    std::stringstream ss;
    ss << "the dynd type " << ndt::type(builtin_type_id);
    ss << " doesn't have a property \"" << property_name << "\"";
    throw std::runtime_error(ss.str());
}

ndt::type dynd::get_builtin_type_elwise_property_type(
                type_id_t builtin_type_id,
                size_t elwise_property_index,
                bool& out_readable, bool& out_writable)
{
    switch (builtin_type_id) {
        case complex_float32_type_id:
            switch (elwise_property_index) {
                case 0: // real
                case 1: // imag
                    out_readable = true;
                    out_writable = false;
                    return ndt::type(float32_type_id);
                case 2: // conj
                    out_readable = true;
                    out_writable = true;
                    return ndt::type(complex_float32_type_id);
                default:
                    break;
            }
            break;
        case complex_float64_type_id:
            switch (elwise_property_index) {
                case 0: // real
                case 1: // imag
                    out_readable = true;
                    out_writable = false;
                    return ndt::type(float64_type_id);
                case 2: // conj
                    out_readable = true;
                    out_writable = true;
                    return ndt::type(complex_float64_type_id);
                default:
                    break;
            }
            break;
        default:
            break;

    }
    out_readable = false;
    out_writable = false;
    return ndt::type();
}

static void get_property_kernel_complex_float32_real(char *dst, const char *src,
                ckernel_prefix *DYND_UNUSED(extra))
{
    *reinterpret_cast<uint32_t *>(dst) = reinterpret_cast<const uint32_t *>(src)[0];
}

static void get_property_kernel_complex_float32_imag(char *dst, const char *src,
                ckernel_prefix *DYND_UNUSED(extra))
{
    *reinterpret_cast<uint32_t *>(dst) = reinterpret_cast<const uint32_t *>(src)[1];
}

static void get_property_kernel_complex_float64_real(char *dst, const char *src,
                ckernel_prefix *DYND_UNUSED(extra))
{
    *reinterpret_cast<uint64_t *>(dst) = reinterpret_cast<const uint64_t *>(src)[0];
}

static void get_property_kernel_complex_float64_imag(char *dst, const char *src,
                ckernel_prefix *DYND_UNUSED(extra))
{
    *reinterpret_cast<uint64_t *>(dst) = reinterpret_cast<const uint64_t *>(src)[1];
}

static void get_or_set_property_kernel_complex_float32_conj(char *dst, const char *src,
                ckernel_prefix *DYND_UNUSED(extra))
{
    dynd_complex<float> value = *reinterpret_cast<const dynd_complex<float> *>(src);
    *reinterpret_cast<dynd_complex<float> *>(dst) = dynd_complex<float>(value.real(), -value.imag());
}

static void get_or_set_property_kernel_complex_float64_conj(char *dst, const char *src,
                ckernel_prefix *DYND_UNUSED(extra))
{
    dynd_complex<double> value = *reinterpret_cast<const dynd_complex<double> *>(src);
    *reinterpret_cast<dynd_complex<double> *>(dst) = dynd_complex<double>(value.real(), -value.imag());
}

size_t dynd::make_builtin_type_elwise_property_getter_kernel(
                ckernel_builder *out, size_t offset_out,
                type_id_t builtin_type_id,
                const char *DYND_UNUSED(dst_metadata),
                const char *DYND_UNUSED(src_metadata),
                size_t src_elwise_property_index,
                kernel_request_t kernreq,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    offset_out = make_kernreq_to_single_kernel_adapter(
                    out, offset_out, kernreq);
    ckernel_prefix *e = out->get_at<ckernel_prefix>(offset_out);
    switch (builtin_type_id) {
        case complex_float32_type_id:
            switch (src_elwise_property_index) {
                case 0:
                    e->set_function<unary_single_operation_t>(
                                    &get_property_kernel_complex_float32_real);
                    return offset_out + sizeof(ckernel_prefix);
                case 1:
                    e->set_function<unary_single_operation_t>(
                                    &get_property_kernel_complex_float32_imag);
                    return offset_out + sizeof(ckernel_prefix);
                case 2:
                    e->set_function<unary_single_operation_t>(
                                    &get_or_set_property_kernel_complex_float32_conj);
                    return offset_out + sizeof(ckernel_prefix);
                default:
                    break;
            }
            break;
        case complex_float64_type_id:
            switch (src_elwise_property_index) {
                case 0:
                    e->set_function<unary_single_operation_t>(
                                    &get_property_kernel_complex_float64_real);
                    return offset_out + sizeof(ckernel_prefix);
                case 1:
                    e->set_function<unary_single_operation_t>(
                                    &get_property_kernel_complex_float64_imag);
                    return offset_out + sizeof(ckernel_prefix);
                case 2:
                    e->set_function<unary_single_operation_t>(
                                    &get_or_set_property_kernel_complex_float64_conj);
                    return offset_out + sizeof(ckernel_prefix);
                default:
                    break;
            }
            break;
        default:
            break;
    }
    stringstream ss;
    ss << "dynd type " << ndt::type(builtin_type_id) << " given an invalid property index " << src_elwise_property_index;
    throw runtime_error(ss.str());
}

size_t dynd::make_builtin_type_elwise_property_setter_kernel(
                ckernel_builder *out, size_t offset_out,
                type_id_t builtin_type_id,
                const char *DYND_UNUSED(dst_metadata), size_t dst_elwise_property_index,
                const char *DYND_UNUSED(src_metadata),
                kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx))
{
    offset_out = make_kernreq_to_single_kernel_adapter(
                    out, offset_out, kernreq);
    ckernel_prefix *e = out->get_at<ckernel_prefix>(offset_out);
    switch (builtin_type_id) {
        case complex_float32_type_id:
            switch (dst_elwise_property_index) {
                case 2:
                    e->set_function<unary_single_operation_t>(
                                    &get_or_set_property_kernel_complex_float32_conj);
                    return offset_out + sizeof(ckernel_prefix);
                default:
                    break;
            }
            break;
        case complex_float64_type_id:
            switch (dst_elwise_property_index) {
                case 2:
                    e->set_function<unary_single_operation_t>(
                                    &get_or_set_property_kernel_complex_float64_conj);
                    return offset_out + sizeof(ckernel_prefix);
                default:
                    break;
            }
            break;
        default:
            break;
    }
    stringstream ss;
    ss << "dynd type " << ndt::type(builtin_type_id) << " given an invalid property index " << dst_elwise_property_index;
    throw runtime_error(ss.str());
}
