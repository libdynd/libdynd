//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FUNC_REDUCTION_ARRFUNC_HPP_
#define _DYND__FUNC_REDUCTION_ARRFUNC_HPP_

#include <dynd/func/arrfunc.hpp>

namespace dynd { namespace nd {

template <typename R, typename A0>
struct reduction_ck {
    typedef reduction_ck self_type; 
    typedef void (*func_type)(R &, const vals<A0> &);

    ckernel_prefix base;
    func_type func;
    vals<A0> src[1];

    static void single(char *dst, const char *const *src, ckernel_prefix *ckp) {
        self_type *self = reinterpret_cast<self_type *>(ckp);

        vals<A0> &src0_vals = reinterpret_cast<self_type *>(self)->src[0];
        src0_vals.set_readonly_originptr(src[0]);

        self->func(*reinterpret_cast<R *>(dst), src0_vals);
    }

    static intptr_t instantiate(const arrfunc_type_data *af_self, dynd::ckernel_builder *ckb,
                                intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                                const char *DYND_UNUSED(dst_arrmeta), const ndt::type *src_tp,
                                const char *const *src_arrmeta, kernel_request_t DYND_UNUSED(kernreq),
                                const eval::eval_context *DYND_UNUSED(ectx)) {
        self_type *self = ckb->alloc_ck_leaf<self_type>(ckb_offset);
        self->base.template set_function<expr_single_t>(self_type::single);
        self->func = *af_self->get_data_as<func_type>();
        self->src[0].init(src_tp[0].get_ndim(), reinterpret_cast<const size_stride_t *>(src_arrmeta[0]));

        return ckb_offset;
    }
};

template <typename func_type>
struct arrfunc_from_func;

template <typename R, typename A0>
struct arrfunc_from_func<void (*)(R &, const vals<A0> &)> {
    typedef void (*func_type)(R &, const vals<A0> &);

    static void make(func_type func, arrfunc_type_data *out_af) {
        out_af->func_proto = ndt::make_funcproto(ndt::type("strided * strided * float32"), ndt::make_type<R>());
        *out_af->get_data_as<func_type>() = func;
        out_af->instantiate = &reduction_ck<R, A0>::instantiate;
        out_af->free_func = NULL;
    }
};

template <typename func_type>
inline void make_reduction_arrfunc(func_type func, arrfunc_type_data *out_af) {
    arrfunc_from_func<func_type>::make(func, out_af);
}

template <typename func_type>
nd::arrfunc make_reduction_arrfunc(func_type func) {
    nd::array af = nd::empty(ndt::make_arrfunc());
    make_reduction_arrfunc(func,
        reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()));
    af.flag_as_immutable();
    return af;
}

}} // namespace dynd::nd

#endif // _DYND__FUNC_REDUCTION_ARRFUNC_HPP_
