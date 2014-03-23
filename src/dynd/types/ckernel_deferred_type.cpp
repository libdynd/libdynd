//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/view.hpp>
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

static void print_ckernel_deferred(std::ostream& o, const ckernel_deferred *ckd)
{
    if (ckd->instantiate_func == NULL) {
        o << "<uninitialized ckernel_deferred>";
    } else {
        o << "<ckernel_deferred ";
        if (ckd->ckernel_funcproto == unary_operation_funcproto) {
            o << "unary ";
        } else if (ckd->ckernel_funcproto == expr_operation_funcproto) {
            o << "expr ";
        } else if (ckd->ckernel_funcproto == binary_predicate_funcproto) {
            o << "binary_predicate ";
        } else {
            o << "<unknown function prototype> ";
        }
        o << ", types [";
        for (intptr_t i = 0; i != ckd->data_types_size; ++i) {
            o << ckd->data_dynd_types[i];
            if (i != ckd->data_types_size - 1) {
                o << "; ";
            }
        }
        o << "]>";
    }
}

void ckernel_deferred_type::print_data(std::ostream& o,
                const char *DYND_UNUSED(metadata), const char *data) const
{
    const ckernel_deferred_type_data *ckd = reinterpret_cast<const ckernel_deferred_type_data *>(data);
    print_ckernel_deferred(o, ckd);
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

/////////////////////////////////////////
// date to string assignment

namespace {
    struct ckernel_deferred_to_string_kernel_extra {
        typedef ckernel_deferred_to_string_kernel_extra extra_type;

        ckernel_prefix base;
        const base_string_type *dst_string_dt;
        const char *dst_metadata;
        assign_error_mode errmode;

        static void single(char *dst, const char *src, ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const ckernel_deferred *ckd = reinterpret_cast<const ckernel_deferred *>(src);
            stringstream ss;
            print_ckernel_deferred(ss, ckd);
            e->dst_string_dt->set_utf8_string(e->dst_metadata, dst, e->errmode, ss.str());
        }

        static void destruct(ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            base_type_xdecref(e->dst_string_dt);
        }
    };
} // anonymous namespace

static intptr_t make_ckernel_deferred_to_string_assignment_kernel(
                ckernel_builder *out_ckb, size_t ckb_offset,
                const ndt::type& dst_string_dt, const char *dst_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    ckb_offset = make_kernreq_to_single_kernel_adapter(out_ckb, ckb_offset, kernreq);
    intptr_t ckb_end_offset = ckb_offset + sizeof(ckernel_deferred_to_string_kernel_extra);
    out_ckb->ensure_capacity_leaf(ckb_end_offset);
    ckernel_deferred_to_string_kernel_extra *e =
                    out_ckb->get_at<ckernel_deferred_to_string_kernel_extra>(ckb_offset);
    e->base.set_function<unary_single_operation_t>(&ckernel_deferred_to_string_kernel_extra::single);
    e->base.destructor = &ckernel_deferred_to_string_kernel_extra::destruct;
    // The kernel data owns a reference to this type
    e->dst_string_dt = static_cast<const base_string_type *>(ndt::type(dst_string_dt).release());
    e->dst_metadata = dst_metadata;
    e->errmode = errmode;
    return ckb_end_offset;
}

size_t ckernel_deferred_type::make_assignment_kernel(
                ckernel_builder *out_ckb, size_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_metadata,
                const ndt::type& src_tp, const char *DYND_UNUSED(src_metadata),
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_tp.extended()) {
    } else {
        if (dst_tp.get_kind() == string_kind) {
            // Assignment to strings
            return make_ckernel_deferred_to_string_assignment_kernel(out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            kernreq, errmode, ectx);
        }
    }
    
    // Nothing can be assigned to/from ckernel_deferred
    stringstream ss;
    ss << "Cannot assign from " << src_tp << " to " << dst_tp;
    throw dynd::type_error(ss.str());
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
    nd::array args[max_args];
    for (nargs = 1; nargs < max_args; ++nargs) {
        // Stop at the first NULL arg (means it was default)
        if (par_arrs[nargs].get_ndo() == NULL) {
            break;
        } else {
            args[nargs-1] = par_arrs[nargs];
        }
    }

    const ckernel_deferred *ckd = reinterpret_cast<const ckernel_deferred *>(par_arrs[0].get_readonly_originptr());

    nargs -= 1;

    // Validate the number of arguments
    if (nargs != ckd->data_types_size) {
        stringstream ss;
        ss << "ckernel expected " << ckd->data_types_size << " arguments, got " << nargs;
        throw runtime_error(ss.str());
    }
    // Validate that the types match exactly, attempting to take a view when they don't
    for (int i = 0; i < nargs; ++i) {
        try {
            args[i] = nd::view(args[i], ckd->data_dynd_types[i]);
        } catch(const type_error&) {
            // Make a type error with a more specific message
            stringstream ss;
            ss << "ckernel argument " << i << " expected type (" << ckd->data_dynd_types[i];
            ss << "), got type (" << args[i].get_type() << ")";
            throw type_error(ss.str());
        }
    }
    // Instantiate the ckernel
    ckernel_builder ckb;
    const char *dynd_metadata[max_args];
    for (int i = 0; i < nargs; ++i) {
        dynd_metadata[i] = args[i].get_ndo_meta();
    }
    ckd->instantiate_func(ckd->data_ptr,
                    &ckb, 0,
                    dynd_metadata, kernel_request_single);
    // Call the ckernel
    if (ckd->ckernel_funcproto == unary_operation_funcproto) {
        unary_single_operation_t usngo = ckb.get()->get_function<unary_single_operation_t>();
        usngo(args[0].get_readwrite_originptr(), args[1].get_readonly_originptr(), ckb.get());
    } else if (ckd->ckernel_funcproto == expr_operation_funcproto) {
        expr_single_operation_t usngo = ckb.get()->get_function<expr_single_operation_t>();
        const char *in_ptrs[max_args];
        for (int i = 0; i < nargs - 1; ++i) {
            in_ptrs[i] = args[i+1].get_readonly_originptr();
        }
        usngo(args[0].get_readwrite_originptr(), in_ptrs, ckb.get());
    } else {
        throw runtime_error("unrecognized ckernel function prototype");
    }
    // Return void
    return nd::empty(ndt::make_type<void>()).release();
}

static pair<string, gfunc::callable> ckernel_deferred_array_functions[] = {
    pair<string, gfunc::callable>("__call__", gfunc::callable(
            ndt::type("{self:pointer[void],out:pointer[void],p0:pointer[void],"
                       "p1:pointer[void],p2:pointer[void],"
                       "p3:pointer[void],p4:pointer[void]}"),
            &function___call__,
            NULL,
            3,
            nd::empty("{self:pointer[void],out:pointer[void],p0:pointer[void],"
                       "p1:pointer[void],p2:pointer[void],"
                       "p3:pointer[void],p4:pointer[void]}")
                    ))
};

void ckernel_deferred_type::get_dynamic_array_functions(
                const std::pair<std::string, gfunc::callable> **out_functions,
                size_t *out_count) const
{
    *out_functions = ckernel_deferred_array_functions;
    *out_count = sizeof(ckernel_deferred_array_functions) / sizeof(ckernel_deferred_array_functions[0]);
}

