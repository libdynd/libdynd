//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/kernels/option_assignment_kernels.hpp>
#include <dynd/types/type_pattern_match.hpp>

using namespace std;
using namespace dynd;

namespace {

/**
 * A ckernel which assigns option[S] to option[T].
 */
struct option_to_option_ck
        : public kernels::assignment_ck<option_to_option_ck> {
    // The default child is the src is_avail ckernel
    // This child is the dst assign_na ckernel
    size_t m_dst_assign_na_offset;
    size_t m_value_assign_offset;

    inline void single(char *dst, const char *src)
    {
        // Check whether the value is available
        // TODO: Would be nice to do this as a predicate
        //       instead of having to go through a dst pointer
        ckernel_prefix *src_is_avail = get_child_ckernel();
        unary_single_operation_t src_is_avail_fn =
            src_is_avail->get_function<unary_single_operation_t>();
        dynd_bool avail = false;
        src_is_avail_fn(reinterpret_cast<char *>(&avail), src, src_is_avail);
        if (avail) {
            // It's available, copy using value assignment
            ckernel_prefix *value_assign =
                get_child_ckernel(m_value_assign_offset);
            unary_single_operation_t value_assign_fn =
                value_assign->get_function<unary_single_operation_t>();
            value_assign_fn(dst, src, value_assign);
        } else {
            // It's not available, assign an NA
            ckernel_prefix *dst_assign_na =
                get_child_ckernel(m_dst_assign_na_offset);
            expr_single_operation_t dst_assign_na_fn =
                dst_assign_na->get_function<expr_single_operation_t>();
            dst_assign_na_fn(dst, NULL, dst_assign_na);
        }
    }

    inline void strided(char *dst, intptr_t dst_stride, const char *src,
                        intptr_t src_stride, size_t count)
    {
        // Three child ckernels
        ckernel_prefix *src_is_avail = get_child_ckernel();
        unary_strided_operation_t src_is_avail_fn =
            src_is_avail->get_function<unary_strided_operation_t>();
        ckernel_prefix *value_assign =
            get_child_ckernel(m_value_assign_offset);
        unary_strided_operation_t value_assign_fn =
            value_assign->get_function<unary_strided_operation_t>();
        ckernel_prefix *dst_assign_na =
            get_child_ckernel(m_dst_assign_na_offset);
        expr_strided_operation_t dst_assign_na_fn =
            dst_assign_na->get_function<expr_strided_operation_t>();
        // Process in chunks using the dynd default buffer size
        dynd_bool avail[DYND_BUFFER_CHUNK_SIZE];
        while (count > 0) {
            size_t chunk_size = min(count, DYND_BUFFER_CHUNK_SIZE);
            count -= chunk_size;
            src_is_avail_fn(reinterpret_cast<char *>(avail), 1, src, src_stride,
                            chunk_size, src_is_avail);
            void *avail_ptr = avail;
            do {
                // Process a run of available values
                void *next_avail_ptr = memchr(avail_ptr, 0, chunk_size);
                if (!next_avail_ptr) {
                    value_assign_fn(dst, dst_stride, src, src_stride,
                                    chunk_size, value_assign);
                    dst += chunk_size * dst_stride;
                    src += chunk_size * src_stride;
                    break;
                } else if (next_avail_ptr > avail_ptr) {
                    size_t segment_size = (char *)next_avail_ptr - (char *)avail_ptr;
                    value_assign_fn(dst, dst_stride, src, src_stride,
                                    segment_size, value_assign);
                    dst += segment_size * dst_stride;
                    src += segment_size * src_stride;
                    chunk_size -= segment_size;
                    avail_ptr = next_avail_ptr;
                }
                // Process a run of not available values
                next_avail_ptr = memchr(avail_ptr, 1, chunk_size);
                if (!next_avail_ptr) {
                    dst_assign_na_fn(dst, dst_stride, NULL, NULL, chunk_size,
                                     dst_assign_na);
                    dst += chunk_size * dst_stride;
                    src += chunk_size * src_stride;
                    break;
                } else if (next_avail_ptr > avail_ptr) {
                    size_t segment_size = (char *)next_avail_ptr - (char *)avail_ptr;
                    dst_assign_na_fn(dst, dst_stride, NULL, NULL,
                                    segment_size, dst_assign_na);
                    dst += segment_size * dst_stride;
                    src += segment_size * src_stride;
                    chunk_size -= segment_size;
                    avail_ptr = next_avail_ptr;
                }
            } while (chunk_size > 0);
        }
    }


    inline void destruct_children()
    {
        // src_is_avail
        get_child_ckernel()->destroy();
        // dst_assign_na
        base.destroy_child_ckernel(m_dst_assign_na_offset);
        // value_assign
        base.destroy_child_ckernel(m_value_assign_offset);
    }
};

/**
 * A ckernel which assigns option[S] to T.
 */
struct option_to_value_ck
        : public kernels::assignment_ck<option_to_value_ck> {
    // The default child is the src_is_avail ckernel
    size_t m_value_assign_offset;

    inline void single(char *dst, const char *src)
    {
        ckernel_prefix *src_is_avail = get_child_ckernel();
        unary_single_operation_t src_is_avail_fn =
            src_is_avail->get_function<unary_single_operation_t>();
        ckernel_prefix *value_assign =
            get_child_ckernel(m_value_assign_offset);
        unary_single_operation_t value_assign_fn =
            value_assign->get_function<unary_single_operation_t>();
        // Make sure it's not an NA
        dynd_bool avail = false;
        src_is_avail_fn(reinterpret_cast<char *>(&avail), src, src_is_avail);
        if (!avail) {
            throw overflow_error(
                "cannot assign an NA value to a non-option type");
        }
        // Copy using value assignment
        value_assign_fn(dst, src, value_assign);
    }

    inline void strided(char *dst, intptr_t dst_stride, const char *src,
                        intptr_t src_stride, size_t count)
    {
        // Two child ckernels
        ckernel_prefix *src_is_avail = get_child_ckernel();
        unary_strided_operation_t src_is_avail_fn =
            src_is_avail->get_function<unary_strided_operation_t>();
        ckernel_prefix *value_assign =
            get_child_ckernel(m_value_assign_offset);
        unary_strided_operation_t value_assign_fn =
            value_assign->get_function<unary_strided_operation_t>();
        // Process in chunks using the dynd default buffer size
        dynd_bool avail[DYND_BUFFER_CHUNK_SIZE];
        while (count > 0) {
            size_t chunk_size = min(count, DYND_BUFFER_CHUNK_SIZE);
            src_is_avail_fn(reinterpret_cast<char *>(avail), 1, src, src_stride,
                            chunk_size, src_is_avail);
            if (memchr(avail, 0, chunk_size) != NULL) {
                throw overflow_error(
                    "cannot assign an NA value to a non-option type");
            }
            value_assign_fn(dst, dst_stride, src, src_stride, chunk_size,
                            value_assign);
            dst += chunk_size * dst_stride;
            src += chunk_size * src_stride;
            count -= chunk_size;
        }
    }


    inline void destruct_children()
    {
        // src_is_avail
        get_child_ckernel()->destroy();
        // value_assign
        base.destroy_child_ckernel(m_value_assign_offset);
    }
};

} // anonymous namespace

static intptr_t instantiate_option_to_option_assignment_kernel(
    const arrfunc_type_data *DYND_UNUSED(self), dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta, uint32_t kernreq,
    const eval::eval_context *ectx)
{
    typedef option_to_option_ck self_type;
    if (dst_tp.get_type_id() != option_type_id ||
            src_tp[0].get_type_id() != option_type_id) {
        stringstream ss;
        ss << "option to option kernel needs option types, got " << dst_tp
           << " and " << src_tp[0];
        throw invalid_argument(ss.str());
    }
    const ndt::type &dst_val_tp = dst_tp.tcast<option_type>()->get_value_type();
    const ndt::type &src_val_tp = src_tp[0].tcast<option_type>()->get_value_type();
    self_type *self = self_type::create(ckb, ckb_offset, (kernel_request_t)kernreq);
    size_t ckb_end = ckb_offset + sizeof(self_type);
    // instantiate src_is_avail
    const arrfunc_type_data *af = src_tp[0].tcast<option_type>()->get_is_avail_arrfunc();
    ckb_end = af->instantiate(af, ckb, ckb_end, ndt::make_type<dynd_bool>(),
                              NULL, src_tp, src_arrmeta, kernreq, ectx);
    // instantiate dst_assign_na
    ckb->ensure_capacity_leaf(ckb_end + sizeof(ckernel_prefix));
    self = ckb->get_at<self_type>(ckb_offset);
    self->m_dst_assign_na_offset = ckb_end - ckb_offset;
    af = dst_tp.tcast<option_type>()->get_assign_na_arrfunc();
    ckb_end = af->instantiate(af, ckb, ckb_end, dst_tp, dst_arrmeta, NULL, NULL,
                              kernreq, ectx);
    // instantiate value_assign
    ckb->ensure_capacity(ckb_end);
    self = ckb->get_at<self_type>(ckb_offset);
    self->m_value_assign_offset = ckb_end- ckb_offset;
    ckb_end = make_assignment_kernel(ckb, ckb_end, dst_val_tp, dst_arrmeta,
                                     src_val_tp, src_arrmeta[0],
                                     (kernel_request_t)kernreq, ectx);
    return ckb_end;
}

static intptr_t instantiate_option_to_value_assignment_kernel(
    const arrfunc_type_data *DYND_UNUSED(self), dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta, uint32_t kernreq,
    const eval::eval_context *ectx)
{
    typedef option_to_value_ck self_type;
    if (dst_tp.get_type_id() == option_type_id ||
            src_tp[0].get_type_id() != option_type_id) {
        stringstream ss;
        ss << "option to value kernel needs value/option types, got " << dst_tp
           << " and " << src_tp[0];
        throw invalid_argument(ss.str());
    }
    const ndt::type &src_val_tp = src_tp[0].tcast<option_type>()->get_value_type();
    self_type *self = self_type::create(ckb, ckb_offset, (kernel_request_t)kernreq);
    size_t ckb_end = ckb_offset + sizeof(self_type);
    // instantiate src_is_avail
    const arrfunc_type_data *af = src_tp[0].tcast<option_type>()->get_is_avail_arrfunc();
    ckb_end = af->instantiate(af, ckb, ckb_end, ndt::make_type<dynd_bool>(),
                              NULL, src_tp, src_arrmeta, kernreq, ectx);
    // instantiate value_assign
    ckb->ensure_capacity_leaf(ckb_end + sizeof(ckernel_prefix));
    self = ckb->get_at<self_type>(ckb_offset);
    self->m_value_assign_offset = ckb_end - ckb_offset;
    ckb_end = make_assignment_kernel(ckb, ckb_end, dst_tp, dst_arrmeta, src_val_tp,
                                     src_arrmeta[0], (kernel_request_t)kernreq, ectx);
    return ckb_end + sizeof(ckernel_prefix);
}

static intptr_t instantiate_option_as_value_assignment_kernel(
    const arrfunc_type_data *DYND_UNUSED(self), dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta, uint32_t kernreq,
    const eval::eval_context *ectx)
{
    // In all cases not handled, we use the
    // regular S to T assignment kernel.
    //
    // Note that this does NOT catch the case where a value
    // which was ok with type S, but equals the NA
    // token in type T, is assigned. Checking this
    // properly across all the cases would add
    // fairly significant cost, and it seems maybe ok
    // to skip it.
    ndt::type val_dst_tp =
        dst_tp.get_type_id() == option_type_id
            ? dst_tp.tcast<option_type>()->get_value_type()
            : dst_tp;
    ndt::type val_src_tp =
        src_tp[0].get_type_id() == option_type_id
            ? src_tp[0].tcast<option_type>()->get_value_type()
            : src_tp[0];
    return ::make_assignment_kernel(ckb, ckb_offset, val_dst_tp, dst_arrmeta,
                                    val_src_tp, src_arrmeta[0],
                                    (kernel_request_t)kernreq, ectx);
}

namespace {

struct option_arrfunc_list {
    arrfunc_type_data af[3];

    option_arrfunc_list() {
        int i = 0;
        af[i].func_proto = ndt::type("(?T) -> ?S");
        af[i].ckernel_funcproto = unary_operation_funcproto;
        af[i].data_ptr = NULL;
        af[i].instantiate = &instantiate_option_to_option_assignment_kernel;
        ++i;
        af[i].func_proto = ndt::type("(?T) -> S");
        af[i].ckernel_funcproto = unary_operation_funcproto;
        af[i].data_ptr = NULL;
        af[i].instantiate = &instantiate_option_to_value_assignment_kernel;
        ++i;
        af[i].func_proto = ndt::type("(T) -> S");
        af[i].ckernel_funcproto = unary_operation_funcproto;
        af[i].data_ptr = NULL;
        af[i].instantiate = &instantiate_option_as_value_assignment_kernel;
    }

    inline intptr_t size() const {
        return sizeof(af) / sizeof(af[0]);
    }

    const arrfunc_type_data *get() const {
        return af;
    }
};
} // anonymous namespace

size_t kernels::make_option_assignment_kernel(
    ckernel_builder *ckb, size_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
    static option_arrfunc_list afl;
    intptr_t size = afl.size();
    const arrfunc_type_data *af = afl.get();
    map<nd::string, ndt::type> typevars;
    for (intptr_t i = 0; i < size; ++i, ++af) {
        typevars.clear();
        if (ndt::type_pattern_match(src_tp, af->get_param_type(0), typevars) &&
                ndt::type_pattern_match(dst_tp, af->get_return_type(), typevars)) {
            return af->instantiate(af, ckb, ckb_offset, dst_tp, dst_arrmeta,
                                   &src_tp, &src_arrmeta, kernreq, ectx);
        }
    }

    stringstream ss;
    ss << "Could not instantiate option assignment kernel from " << src_tp
       << " to " << dst_tp;
    throw invalid_argument(ss.str());
}
