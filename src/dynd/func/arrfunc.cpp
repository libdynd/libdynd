//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/types/expr_type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/property_type.hpp>
#include <dynd/type.hpp>

using namespace std;
using namespace dynd;

namespace {

////////////////////////////////////////////////////////////////
// Functions for the unary assignment as an arrfunc

static intptr_t instantiate_unary_assignment_ckernel(
    const arrfunc_type_data *self, dynd::ckernel_builder *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, const ndt::type *src_tp,
    const char *const *src_arrmeta, uint32_t kernreq,
    const eval::eval_context *ectx)
{
    assign_error_mode errmode = static_cast<assign_error_mode>(
        reinterpret_cast<uintptr_t>(self->data_ptr));
    if (dst_tp.value_type() == self->get_return_type() &&
            src_tp[0].value_type() == self->get_param_type(0)) {
        return make_assignment_kernel(
            ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0],
            (kernel_request_t)kernreq, errmode, ectx);
    } else {
        stringstream ss;
        ss << "Cannot instantiate arrfunc for assigning from ";
        ss << self->get_param_type(0) << " to " << self->get_return_type();
        ss << " using input type " << src_tp[0];
        ss << " and output type " << dst_tp;
        throw type_error(ss.str());
    }
}

static intptr_t instantiate_adapted_expr_assignment_ckernel(
    const arrfunc_type_data *self, dynd::ckernel_builder *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, const ndt::type *src_tp,
    const char *const *src_arrmeta, uint32_t kernreq,
    const eval::eval_context *ectx)
{
    ckb_offset = kernels::wrap_unary_as_expr_ckernel(ckb, ckb_offset,
                                                     (kernel_request_t)kernreq);
    return instantiate_unary_assignment_ckernel(self, ckb, ckb_offset, dst_tp,
                                                dst_arrmeta, src_tp,
                                                src_arrmeta, kernreq, ectx);
}

////////////////////////////////////////////////////////////////
// Functions for property access as an arrfunc

static void delete_property_arrfunc_data(void *self_data_ptr)
{
    const base_type *data = reinterpret_cast<const base_type *>(self_data_ptr);
    base_type_xdecref(data);
}

static intptr_t instantiate_unary_property_ckernel(
    const arrfunc_type_data *self, dynd::ckernel_builder *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, const ndt::type *src_tp,
    const char *const *src_arrmeta, uint32_t kernreq,
    const eval::eval_context *ectx)
{
    ndt::type prop_src_tp(reinterpret_cast<const base_type *>(self->data_ptr), true);

    if (dst_tp.value_type() == prop_src_tp.value_type()) {
        if (src_tp[0] == prop_src_tp.operand_type()) {
            return make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta,
                                          prop_src_tp, src_arrmeta[0],
                                          (kernel_request_t)kernreq,
                                          assign_error_default, ectx);
        } else if (src_tp[0].value_type() == prop_src_tp.operand_type()) {
            return make_assignment_kernel(
                ckb, ckb_offset, dst_tp, dst_arrmeta,
                prop_src_tp.tcast<base_expression_type>()
                    ->with_replaced_storage_type(src_tp[0]),
                src_arrmeta[0], (kernel_request_t)kernreq, assign_error_default,
                ectx);
        }
    }

    stringstream ss;
    ss << "Cannot instantiate arrfunc for assigning from ";
    ss << self->get_param_type(0) << " to " << self->get_return_type();
    ss << " using input type " << src_tp[0];
    ss << " and output type " << dst_tp;
    throw type_error(ss.str());
}

static intptr_t instantiate_expr_property_ckernel(
    const arrfunc_type_data *self, dynd::ckernel_builder *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, const ndt::type *src_tp,
    const char *const *src_arrmeta, uint32_t kernreq,
    const eval::eval_context *ectx)
{
    ckb_offset = kernels::wrap_unary_as_expr_ckernel(ckb, ckb_offset,
                                                     (kernel_request_t)kernreq);
    return instantiate_unary_property_ckernel(self, ckb, ckb_offset, dst_tp,
                                              dst_arrmeta, src_tp, src_arrmeta,
                                              kernreq, ectx);
}

////////////////////////////////////////////////////////////////
// Structure and functions for the expr as an arrfunc

struct expr_arrfunc_data {
    assign_error_mode errmode;
    const dynd::expr_type *expr_type;
    size_t data_types_size;
    ndt::type data_types[1];
};

static void delete_expr_arrfunc_data(void *self_data_ptr)
{
    expr_arrfunc_data *data =
                    reinterpret_cast<expr_arrfunc_data *>(self_data_ptr);
    base_type_xdecref(data->expr_type);
    ndt::type *data_types = &data->data_types[0];
    for (size_t i = 0; i < data->data_types_size; ++i) {
        // Reset all the types to NULL
        data_types[i] = ndt::type();
    }
    // Call the destructor and free the memory
    data->~expr_arrfunc_data();
    free(data);
}

static intptr_t
instantiate_expr_ckernel(const arrfunc_type_data *self, dynd::ckernel_builder *ckb,
                         intptr_t ckb_offset, const ndt::type &dst_tp,
                         const char *dst_arrmeta, const ndt::type *src_tp,
                         const char *const *src_arrmeta, uint32_t kernreq,
                         const eval::eval_context *ectx)
{
    expr_arrfunc_data *data =
        reinterpret_cast<expr_arrfunc_data *>(self->data_ptr);
    const expr_kernel_generator &kgen = data->expr_type->get_kgen();
    return kgen.make_expr_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta,
                                 data->data_types_size - 1, src_tp, src_arrmeta,
                                 (kernel_request_t)kernreq, ectx);
}



} // anonymous namespace

void dynd::make_arrfunc_from_assignment(
                const ndt::type& dst_tp, const ndt::type& src_tp,
                arrfunc_proto_t funcproto,
                assign_error_mode errmode, arrfunc_type_data &out_af)
{
    if (dst_tp.get_kind() == expression_kind ||
            (src_tp.get_kind() == expression_kind &&
            src_tp.get_type_id() != expr_type_id)) {
        stringstream ss;
        ss << "Creating an arrfunc from an assignment requires non-expression"
           << "src and dst types, got " << src_tp << " and " << dst_tp;
        throw type_error(ss.str());
    }
    memset(&out_af, 0, sizeof(arrfunc_type_data));
    if (funcproto == unary_operation_funcproto) {
        // Since a unary operation was requested, it's a straightforward unary assignment ckernel
        out_af.data_ptr = reinterpret_cast<void *>(errmode);
        out_af.free_func = NULL;
        out_af.instantiate_func = &instantiate_unary_assignment_ckernel;
        out_af.ckernel_funcproto = unary_operation_funcproto;
        out_af.func_proto = ndt::make_funcproto(src_tp, dst_tp);
    } else if (funcproto == expr_operation_funcproto) {
        if (src_tp.get_type_id() == expr_type_id) {
            const expr_type *etp = src_tp.tcast<expr_type>();
            const base_struct_type *operands_type = etp->get_operand_type().tcast<base_struct_type>();
            // Expose the expr type's expression
            intptr_t nargs = operands_type->get_field_count();
            size_t sizeof_data_mem = sizeof(expr_arrfunc_data) + sizeof(void *) * nargs;
            void *data_mem = malloc(sizeof_data_mem);
            memset(data_mem, 0, sizeof_data_mem);
            expr_arrfunc_data *data = reinterpret_cast<expr_arrfunc_data *>(data_mem);
            out_af.data_ptr = data;
            out_af.free_func = &delete_expr_arrfunc_data;
            data->data_types_size = nargs + 1;
            ndt::type *data_types_arr = &data->data_types[0];
            data_types_arr[0] = dst_tp;
            for (intptr_t i = 0; i < nargs; ++i) {
                // Dereference the pointer type in each field
                const pointer_type *field_ptr_type =
                    operands_type->get_field_type(i).tcast<pointer_type>();
                data_types_arr[i+1] = field_ptr_type->get_target_type();
            }
            data->expr_type = static_cast<const expr_type *>(ndt::type(etp, true).release());
            data->errmode = errmode;
            out_af.instantiate_func = &instantiate_expr_ckernel;
            out_af.ckernel_funcproto = expr_operation_funcproto;
            out_af.func_proto = ndt::make_funcproto(nargs, data->data_types + 1, data->data_types[0]);
        } else {
            // Adapt the assignment to an expr kernel
            out_af.data_ptr = reinterpret_cast<void *>(errmode);
            out_af.free_func = NULL;
            out_af.instantiate_func = &instantiate_adapted_expr_assignment_ckernel;
            out_af.ckernel_funcproto = expr_operation_funcproto;
            out_af.func_proto = ndt::make_funcproto(src_tp, dst_tp);
        }
    } else {
        stringstream ss;
        ss << "unrecognized ckernel function prototype enum value " << funcproto;
        throw runtime_error(ss.str());
    }
}

void dynd::make_arrfunc_from_property(const ndt::type &tp,
                                      const std::string &propname,
                                      arrfunc_proto_t funcproto,
                                      assign_error_mode errmode,
                                      arrfunc_type_data &out_af)
{
    if (tp.get_kind() == expression_kind) {
        stringstream ss;
        ss << "Creating an arrfunc from a property requires a non-expression"
           << ", got " << tp;
        throw type_error(ss.str());
    }
    ndt::type prop_tp = ndt::make_property(tp, propname);
    out_af.func_proto = ndt::make_funcproto(tp, prop_tp.value_type());
    out_af.free_func = &delete_property_arrfunc_data;
    out_af.data_ptr =
        const_cast<void *>(reinterpret_cast<const void *>(prop_tp.release()));
    if (funcproto == unary_operation_funcproto) {
        out_af.ckernel_funcproto = unary_operation_funcproto;
        out_af.instantiate_func = &instantiate_unary_property_ckernel;
    } else if (funcproto == expr_operation_funcproto) {
        out_af.ckernel_funcproto = expr_operation_funcproto;
        out_af.instantiate_func = &instantiate_expr_property_ckernel;
    } else {
        stringstream ss;
        ss << "unrecognized ckernel function prototype enum value " << funcproto;
        throw runtime_error(ss.str());
    }
}

nd::arrfunc::arrfunc(const nd::array &rhs)
{
    if (!rhs.is_null()) {
        if (rhs.get_type().get_type_id() == arrfunc_type_id) {
            if (rhs.is_immutable()) {
                const arrfunc_type_data *af =
                    reinterpret_cast<const arrfunc_type_data *>(
                        rhs.get_readonly_originptr());
                if (af->instantiate_func != NULL) {
                    // It's valid: immutable, arrfunc type, contains an
                    // instantiate function.
                    m_value = rhs;
                } else {
                    throw invalid_argument("Require a non-empty arrfunc, "
                                           "provided arrfunc has NULL "
                                           "instantiate_func");
                }
            } else {
                stringstream ss;
                ss << "Require an immutable arrfunc, provided arrfunc";
                rhs.get_type().extended()->print_data(
                    ss, rhs.get_arrmeta(), rhs.get_readonly_originptr());
                ss << " is not immutable";
                throw invalid_argument(ss.str());
            }
        } else {
            stringstream ss;
            ss << "Cannot implicitly convert nd::array of type "
               << rhs.get_type().value_type() << " to  arrfunc";
            throw type_error(ss.str());
        }
    }
}
