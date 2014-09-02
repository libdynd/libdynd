//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FUNC_REDUCTION_ARRFUNC_HPP_
#define _DYND__FUNC_REDUCTION_ARRFUNC_HPP_

#include <dynd/func/arrfunc.hpp>

template <typename T>
class vals {
    const char *m_origin;

public:
    vals() {
    }

    void set_origin(const char *origin) { m_origin = origin; }

    dynd::size_stride_t ss[2];

    T operator()(intptr_t i, intptr_t j) const {
        return *reinterpret_cast<const T *>(m_origin + i * ss[0].stride + j * ss[1].stride);
    }

    intptr_t get_ndim() const {
        return 2;
    }

    intptr_t get_dim_size(intptr_t i) const {
        return ss[i].dim_size;
    }
};

template <typename T>
class vals_iter {

};

namespace dynd { namespace nd {

template <typename R, typename A0>
struct reduction_ck {
    ckernel_prefix base;
    void (*func)(R &dst, vals<A0> src);
    vals<A0> src;

    static void single(char *dst, const char *const *src, ckernel_prefix *self)
    {
        vals<A0> srcvals =
            reinterpret_cast<reduction_ck *>(self)->src;
        srcvals.set_origin(src[0]);

        reinterpret_cast<reduction_ck *>(self)->func(*reinterpret_cast<R *>(dst), srcvals);
    }
};

template <typename R, typename A0>
struct reduction_arrfunc_data {
    void (*m_func)(R &dst, vals<A0> src);
};


template <typename R, typename A0>
intptr_t instantiate(const arrfunc_type_data *af_self, dynd::ckernel_builder *ckb,
                    intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                    const char *DYND_UNUSED(dst_arrmeta), const ndt::type *DYND_UNUSED(src_tp),
                    const char *const *src_arrmeta, kernel_request_t DYND_UNUSED(kernreq),
                    const eval::eval_context *DYND_UNUSED(ectx))
{
    typedef reduction_ck<R, A0> self_type;
    reduction_arrfunc_data<R, A0> *data = *af_self->get_data_as<reduction_arrfunc_data<R, A0> *>();

    self_type *self =
        ckb->alloc_ck_leaf<self_type>(ckb_offset);
    self->base.template set_function<expr_single_t>(self_type::single);

    self->func = data->m_func;

    const size_stride_t *ss = reinterpret_cast<const size_stride_t *>(src_arrmeta[0]);
    self->src.ss[0] = *ss;
    self->src.ss[1] = *(ss + 1);

    return ckb_offset;
}

template <typename R, typename A0>
inline void make_reduction_arrfunc(arrfunc_type_data *out_af, void (*func)(R &dst, vals<A0> src)) {
    reduction_arrfunc_data<R, A0> *data = new reduction_arrfunc_data<R, A0>;
    data->m_func = func;

    *out_af->get_data_as<reduction_arrfunc_data<R, A0> *>() = data;
    out_af->func_proto = ndt::make_funcproto(ndt::type("strided * strided * float32"), ndt::make_type<R>());
    out_af->instantiate = &instantiate<R, A0>;
}

template <typename R, typename A0>
nd::arrfunc make_reduction_arrfunc(void (*func)(R &dst, vals<A0> src)) {
    nd::array af = nd::empty(ndt::make_arrfunc());
    make_reduction_arrfunc(
        reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()),
        func);
    af.flag_as_immutable();
    return af;
}

}} // namespace dynd::nd

#endif // _DYND__FUNC_REDUCTION_ARRFUNC_HPP_
