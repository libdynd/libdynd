//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/view.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/make_callable.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/func/make_callable.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

arrfunc_type::arrfunc_type()
    : base_type(arrfunc_type_id, custom_kind, sizeof(arrfunc_type_data),
                    scalar_align_of<uint64_t>::value,
                    type_flag_scalar|type_flag_zeroinit|type_flag_destructor,
                    0, 0, 0)
{
}

arrfunc_type::~arrfunc_type()
{
}

static void print_arrfunc(std::ostream& o, const arrfunc_type_data *af)
{
    if (af->instantiate == NULL) {
        o << "<uninitialized arrfunc>";
    } else {
        o << "arrfunc: " << af->func_proto;
    }
}

void arrfunc_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta),
                              const char *data) const
{
  const arrfunc_type_data *af =
      reinterpret_cast<const arrfunc_type_data *>(data);
  print_arrfunc(o, af);
}

void arrfunc_type::print_type(std::ostream& o) const
{
    o << "arrfunc";
}

bool arrfunc_type::operator==(const base_type& rhs) const
{
    return this == &rhs || rhs.get_type_id() == arrfunc_type_id;
}

void arrfunc_type::arrmeta_default_construct(char *DYND_UNUSED(arrmeta),
                intptr_t DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const
{
}

void arrfunc_type::arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta),
                const char *DYND_UNUSED(src_arrmeta), memory_block_data *DYND_UNUSED(embedded_reference)) const
{
}

void arrfunc_type::arrmeta_reset_buffers(char *DYND_UNUSED(arrmeta)) const
{
}

void arrfunc_type::arrmeta_finalize_buffers(char *DYND_UNUSED(arrmeta)) const
{
}

void arrfunc_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const
{
}

void arrfunc_type::data_destruct(const char *DYND_UNUSED(arrmeta), char *data) const
{
    const arrfunc_type_data *d = reinterpret_cast<arrfunc_type_data *>(data);
    d->~arrfunc_type_data();
}

void arrfunc_type::data_destruct_strided(const char *DYND_UNUSED(arrmeta), char *data,
                intptr_t stride, size_t count) const
{
    for (size_t i = 0; i != count; ++i, data += stride) {
        const arrfunc_type_data *d = reinterpret_cast<arrfunc_type_data *>(data);
        d->~arrfunc_type_data();
    }
}

/////////////////////////////////////////
// arrfunc to string assignment

namespace {
    struct arrfunc_to_string_ck : public kernels::unary_ck<arrfunc_to_string_ck> {
        ndt::type m_dst_string_dt;
        const char *m_dst_arrmeta;
        eval::eval_context m_ectx;

        inline void single(char *dst, const char *src)
        {
            const arrfunc_type_data *af =
                reinterpret_cast<const arrfunc_type_data *>(src);
            stringstream ss;
            print_arrfunc(ss, af);
            m_dst_string_dt.tcast<base_string_type>()->set_from_utf8_string(
                m_dst_arrmeta, dst, ss.str(), &m_ectx);
        }
    };
} // anonymous namespace

static intptr_t make_arrfunc_to_string_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_string_dt,
    const char *dst_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
    typedef arrfunc_to_string_ck self_type;
    self_type *self = self_type::create_leaf(ckb, kernreq, ckb_offset);
    // The kernel data owns a reference to this type
    self->m_dst_string_dt = dst_string_dt;
    self->m_dst_arrmeta = dst_arrmeta;
    self->m_ectx = *ectx;
    return ckb_offset;
}

size_t arrfunc_type::make_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp,
    const char *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
    const eval::eval_context *ectx) const
{
    if (this == dst_tp.extended()) {
    } else {
        if (dst_tp.get_kind() == string_kind) {
            // Assignment to strings
            return make_arrfunc_to_string_assignment_kernel(
                ckb, ckb_offset, dst_tp, dst_arrmeta, kernreq, ectx);
        }
    }
    
    // Nothing can be assigned to/from arrfunc
    stringstream ss;
    ss << "Cannot assign from " << src_tp << " to " << dst_tp;
    throw dynd::type_error(ss.str());
}

///////// properties on the nd::array

static nd::array property_ndo_get_proto(const nd::array& n) {
    if (n.get_type().get_type_id() != arrfunc_type_id) {
        throw runtime_error("arrfunc property 'types' only works on scalars presently");
    }
    const arrfunc_type_data *af =
        reinterpret_cast<const arrfunc_type_data *>(n.get_readonly_originptr());
    return af->func_proto;
}

void arrfunc_type::get_dynamic_array_properties(
                const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    static pair<string, gfunc::callable> arrfunc_array_properties[] = {
        pair<string, gfunc::callable>(
            "proto", gfunc::make_callable(&property_ndo_get_proto, "self"))};

    *out_properties = arrfunc_array_properties;
    *out_count = sizeof(arrfunc_array_properties) / sizeof(arrfunc_array_properties[0]);
}

///////// functions on the nd::array

// Maximum number of args (including out) for now
// (need to add varargs capability to this calling convention)
static const int max_args = 6;

static array_preamble *function___call__(const array_preamble *params, void *DYND_UNUSED(self))
{
    // TODO: Remove the const_cast
    nd::array par(const_cast<array_preamble *>(params), true);
    const nd::array *par_arrs = reinterpret_cast<const nd::array *>(par.get_readonly_originptr());
    if (par_arrs[0].get_type().get_type_id() != arrfunc_type_id) {
        throw runtime_error("arrfunc method '__call__' only works on individual arrfunc instances presently");
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

    const arrfunc_type_data *af = reinterpret_cast<const arrfunc_type_data *>(
        par_arrs[0].get_readonly_originptr());
    const funcproto_type *proto = af->func_proto.tcast<funcproto_type>();

    nargs -= 1;

    // Validate the number of arguments
    if (nargs != proto->get_param_count() + 1) {
        stringstream ss;
        ss << "arrfunc expected " << (proto->get_param_count() + 1) << " arguments, got " << nargs;
        throw runtime_error(ss.str());
    }
    // Instantiate the ckernel
    ndt::type src_tp[max_args];
    for (int i = 0; i < nargs - 1; ++i) {
        src_tp[i] = args[i + 1].get_type();
    }
    const char *dynd_arrmeta[max_args];
    for (int i = 0; i < nargs - 1; ++i) {
        dynd_arrmeta[i] = args[i + 1].get_arrmeta();
    }
    ckernel_builder ckb;
    af->instantiate(af, &ckb, 0, args[0].get_type(),
                         args[0].get_arrmeta(), src_tp,
                         dynd_arrmeta, kernel_request_single,
                         &eval::default_eval_context);
    // Call the ckernel
    expr_single_t usngo = ckb.get()->get_function<expr_single_t>();
    const char *in_ptrs[max_args];
    for (int i = 0; i < nargs - 1; ++i) {
        in_ptrs[i] = args[i+1].get_readonly_originptr();
    }
    usngo(args[0].get_readwrite_originptr(), in_ptrs, ckb.get());
    // Return void
    return nd::empty(ndt::make_type<void>()).release();
}

void arrfunc_type::get_dynamic_array_functions(
                const std::pair<std::string, gfunc::callable> **out_functions,
                size_t *out_count) const
{
    static pair<string, gfunc::callable> arrfunc_array_functions[] = {
        pair<string, gfunc::callable>(
            "execute",
            gfunc::callable(
                ndt::type("c{self:ndarrayarg,out:ndarrayarg,p0:ndarrayarg,"
                          "p1:ndarrayarg,p2:ndarrayarg,"
                          "p3:ndarrayarg,p4:ndarrayarg}"),
                &function___call__, NULL, 3,
                nd::empty("c{self:ndarrayarg,out:ndarrayarg,p0:ndarrayarg,"
                          "p1:ndarrayarg,p2:ndarrayarg,"
                          "p3:ndarrayarg,p4:ndarrayarg}")))};

    *out_functions = arrfunc_array_functions;
    *out_count = sizeof(arrfunc_array_functions) / sizeof(arrfunc_array_functions[0]);
}

const ndt::type &ndt::make_arrfunc()
{
    static arrfunc_type aft;
    static const ndt::type static_instance(&aft, true);
    return static_instance;
}
