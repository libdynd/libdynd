//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/types/ckernel_deferred_type.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/gfunc/make_callable.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/gfunc/make_callable.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

ckernel_deferred_type::ckernel_deferred_type()
    : base_type(ckernel_deferred_type_id, custom_kind, sizeof(ckernel_deferred_type_data),
                    sizeof(void *),
                    type_flag_scalar|type_flag_zeroinit|type_flag_destructor,
                    0, 0)
{
}

ckernel_deferred_type::~ckernel_deferred_type()
{
}

void ckernel_deferred_type::print_data(std::ostream& o,
                const char *DYND_UNUSED(metadata), const char *data) const
{
    const ckernel_deferred_type_data *ddd = reinterpret_cast<const ckernel_deferred_type_data *>(data);
    o << "<ckernel_deferred at " << (const void *)data;
    o << ", types [";
    for (size_t i = 0; i != ddd->data_types_size; ++i) {
        o << ddd->data_dynd_types[i];
        if (i != ddd->data_types_size - 1) {
            o << ", ";
        }
    }
    o << "]>";
}

void ckernel_deferred_type::print_type(std::ostream& o) const
{
    o << "ckernel_deferred";
}

bool ckernel_deferred_type::operator==(const base_type& rhs) const
{
    return this == &rhs || rhs.get_type_id() == ckernel_deferred_type_id;
}

void ckernel_deferred_type::metadata_default_construct(char *DYND_UNUSED(metadata),
                intptr_t DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const
{
}

void ckernel_deferred_type::metadata_copy_construct(char *DYND_UNUSED(dst_metadata),
                const char *DYND_UNUSED(src_metadata), memory_block_data *DYND_UNUSED(embedded_reference)) const
{
}

void ckernel_deferred_type::metadata_reset_buffers(char *DYND_UNUSED(metadata)) const
{
}

void ckernel_deferred_type::metadata_finalize_buffers(char *DYND_UNUSED(metadata)) const
{
}

void ckernel_deferred_type::metadata_destruct(char *DYND_UNUSED(metadata)) const
{
}

void ckernel_deferred_type::data_destruct(const char *DYND_UNUSED(metadata), char *data) const
{
    const ckernel_deferred_type_data *d = reinterpret_cast<ckernel_deferred_type_data *>(data);
    if (d->data_ptr != NULL && d->free_func != NULL) {
        d->free_func(d->data_ptr);
    }
}

void ckernel_deferred_type::data_destruct_strided(const char *DYND_UNUSED(metadata), char *data,
                intptr_t stride, size_t count) const
{
    for (size_t i = 0; i != count; ++i, data += stride) {
        const ckernel_deferred_type_data *d = reinterpret_cast<ckernel_deferred_type_data *>(data);
        if (d->data_ptr != NULL && d->free_func != NULL) {
            d->free_func(d->data_ptr);
        }
    }
}

size_t ckernel_deferred_type::make_assignment_kernel(
                ckernel_builder *DYND_UNUSED(out), size_t DYND_UNUSED(offset_out),
                const ndt::type& dst_tp, const char *DYND_UNUSED(dst_metadata),
                const ndt::type& src_tp, const char *DYND_UNUSED(src_metadata),
                kernel_request_t DYND_UNUSED(kernreq), assign_error_mode DYND_UNUSED(errmode),
                const eval::eval_context *DYND_UNUSED(ectx)) const
{
    // Nothing can be assigned to/from ckernel_deferred
    stringstream ss;
    ss << "Cannot assign from " << src_tp << " to " << dst_tp;
    throw runtime_error(ss.str());
}

///////// properties on the nd::array

static nd::array property_ndo_get_types(const nd::array& n) {
    if (n.get_type().get_type_id() != ckernel_deferred_type_id) {
        throw runtime_error("ckernel_deferred property 'types' only works on scalars presently");
    }
    const ckernel_deferred *ckd = reinterpret_cast<const ckernel_deferred *>(n.get_readonly_originptr());
    nd::array result = nd::empty(ckd->data_types_size, ndt::make_strided_dim(ndt::make_type()));
    ndt::type *out_data = reinterpret_cast<ndt::type *>(result.get_readwrite_originptr());
    for (intptr_t i = 0; i < ckd->data_types_size; ++i) {
        out_data[i] = ckd->data_dynd_types[i];
    }
    return result;
}

static pair<string, gfunc::callable> ckernel_deferred_array_properties[] = {
    pair<string, gfunc::callable>("types", gfunc::make_callable(&property_ndo_get_types, "self"))
};

void ckernel_deferred_type::get_dynamic_array_properties(
                const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    *out_properties = ckernel_deferred_array_properties;
    *out_count = sizeof(ckernel_deferred_array_properties) / sizeof(ckernel_deferred_array_properties[0]);
}

///////// functions on the nd::array

// Maximum number of args (including out) for now
// (need to add varargs capability to this calling convention)
static const int max_args = 6;

static array_preamble *function___call__(const array_preamble *params, void *DYND_UNUSED(extra))
{
    // TODO: Remove the const_cast
    nd::array par(const_cast<array_preamble *>(params), true);
    const nd::array *par_arrs = reinterpret_cast<const nd::array *>(par.get_readonly_originptr());
    if (par_arrs[0].get_type().get_type_id() != ckernel_deferred_type_id) {
        throw runtime_error("ckernel_deferred method '__call__' only works on individual ckernel_deferred instances presently");
    }
    // Figure out how many args were provided
    int nargs;
    for (nargs = 2; nargs < max_args; ++nargs) {
        // Stop at the first NULL arg (means it was default)
        if (par_arrs[nargs+1].get_ndo() == NULL) {
            break;
        }
    }

    const ckernel_deferred *ckd = reinterpret_cast<const ckernel_deferred *>(par_arrs[0].get_readonly_originptr());
    // Validate the number of arguments
    if (nargs != ckd->data_types_size) {
        stringstream ss;
        ss << "ckernel expected " << (ckd->data_types_size - 1) << " arguments, got " << nargs;
        throw runtime_error(ss.str());
    }
    // Validate that the types match exactly
    for (int i = 0; i < nargs; ++i) {
        if (par_arrs[i+1].get_type() != ckd->data_dynd_types[i]) {
            stringstream ss;
            ss << "ckernel argument " << i << " expected type (" << ckd->data_dynd_types[i];
            ss << "), got type (" << par_arrs[i].get_type() << ")";
            throw runtime_error(ss.str());
        }
    }
    // Instantiate the ckernel
    ckernel_builder ckb;
    const char *dynd_metadata[max_args];
    for (int i = 0; i < nargs; ++i) {
        dynd_metadata[i] = par_arrs[i+1].get_ndo_meta();
    }
    ckd->instantiate_func(ckd->data_ptr,
                    &ckb, 0,
                    dynd_metadata, kernel_request_single);
    // Call the ckernel
    if (ckd->ckernel_funcproto == unary_operation_funcproto) {
        throw runtime_error("TODO: unary ckernel call is not implemented");
    } else if (ckd->ckernel_funcproto == expr_operation_funcproto) {
        expr_single_operation_t usngo = ckb.get()->get_function<expr_single_operation_t>();
        const char *in_ptrs[max_args];
        for (int i = 0; i < nargs - 1; ++i) {
            in_ptrs[i] = par_arrs[i+2].get_readonly_originptr();
        }
        usngo(par_arrs[1].get_readwrite_originptr(), in_ptrs, ckb.get());
    } else {
        throw runtime_error("unrecognized ckernel function prototype");
    }
    // Return void
    return nd::empty(ndt::make_type<void>()).release();
}

static pair<string, gfunc::callable> ckernel_deferred_array_functions[] = {
    pair<string, gfunc::callable>("__call__", gfunc::callable(
            ndt::type("{self:pointer(void);out:pointer(void);p0:pointer(void);"
                       "p1:pointer(void);p2:pointer(void);"
                       "p3:pointer(void);p4:pointer(void)}"),
            &function___call__,
            NULL,
            3,
            nd::empty("{self:pointer(void);out:pointer(void);p0:pointer(void);"
                       "p1:pointer(void);p2:pointer(void);"
                       "p3:pointer(void);p4:pointer(void)}")
                    ))
};

void ckernel_deferred_type::get_dynamic_array_functions(
                const std::pair<std::string, gfunc::callable> **out_functions,
                size_t *out_count) const
{
    *out_functions = ckernel_deferred_array_functions;
    *out_count = sizeof(ckernel_deferred_array_functions) / sizeof(ckernel_deferred_array_functions[0]);
}

