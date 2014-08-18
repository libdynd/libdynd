//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/kernels/option_assignment_kernels.hpp>
#include <dynd/types/type_pattern_match.hpp>
#include <dynd/parser_util.hpp>

using namespace std;
using namespace dynd;

namespace {

/**
 * A ckernel which assigns option[S] to option[T].
 */
struct option_to_option_ck
        : public kernels::unary_ck<option_to_option_ck> {
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
        expr_single_t src_is_avail_fn =
            src_is_avail->get_function<expr_single_t>();
        dynd_bool avail = false;
        src_is_avail_fn(reinterpret_cast<char *>(&avail), &src, src_is_avail);
        if (avail) {
            // It's available, copy using value assignment
            ckernel_prefix *value_assign =
                get_child_ckernel(m_value_assign_offset);
            expr_single_t value_assign_fn =
                value_assign->get_function<expr_single_t>();
            value_assign_fn(dst, &src, value_assign);
        } else {
            // It's not available, assign an NA
            ckernel_prefix *dst_assign_na =
                get_child_ckernel(m_dst_assign_na_offset);
            expr_single_t dst_assign_na_fn =
                dst_assign_na->get_function<expr_single_t>();
            dst_assign_na_fn(dst, NULL, dst_assign_na);
        }
    }

    inline void strided(char *dst, intptr_t dst_stride, const char *src,
                        intptr_t src_stride, size_t count)
    {
        // Three child ckernels
        ckernel_prefix *src_is_avail = get_child_ckernel();
        expr_strided_t src_is_avail_fn =
            src_is_avail->get_function<expr_strided_t>();
        ckernel_prefix *value_assign =
            get_child_ckernel(m_value_assign_offset);
        expr_strided_t value_assign_fn =
            value_assign->get_function<expr_strided_t>();
        ckernel_prefix *dst_assign_na =
            get_child_ckernel(m_dst_assign_na_offset);
        expr_strided_t dst_assign_na_fn =
            dst_assign_na->get_function<expr_strided_t>();
        // Process in chunks using the dynd default buffer size
        dynd_bool avail[DYND_BUFFER_CHUNK_SIZE];
        while (count > 0) {
            size_t chunk_size = min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
            count -= chunk_size;
            src_is_avail_fn(reinterpret_cast<char *>(avail), 1, &src,
                            &src_stride, chunk_size, src_is_avail);
            void *avail_ptr = avail;
            do {
                // Process a run of available values
                void *next_avail_ptr = memchr(avail_ptr, 0, chunk_size);
                if (!next_avail_ptr) {
                    value_assign_fn(dst, dst_stride, &src, &src_stride,
                                    chunk_size, value_assign);
                    dst += chunk_size * dst_stride;
                    src += chunk_size * src_stride;
                    break;
                } else if (next_avail_ptr > avail_ptr) {
                    size_t segment_size = (char *)next_avail_ptr - (char *)avail_ptr;
                    value_assign_fn(dst, dst_stride, &src, &src_stride,
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
        : public kernels::unary_ck<option_to_value_ck> {
    // The default child is the src_is_avail ckernel
    size_t m_value_assign_offset;

    inline void single(char *dst, const char *src)
    {
        ckernel_prefix *src_is_avail = get_child_ckernel();
        expr_single_t src_is_avail_fn =
            src_is_avail->get_function<expr_single_t>();
        ckernel_prefix *value_assign =
            get_child_ckernel(m_value_assign_offset);
        expr_single_t value_assign_fn =
            value_assign->get_function<expr_single_t>();
        // Make sure it's not an NA
        dynd_bool avail = false;
        src_is_avail_fn(reinterpret_cast<char *>(&avail), &src, src_is_avail);
        if (!avail) {
            throw overflow_error(
                "cannot assign an NA value to a non-option type");
        }
        // Copy using value assignment
        value_assign_fn(dst, &src, value_assign);
    }

    inline void strided(char *dst, intptr_t dst_stride, const char *src,
                        intptr_t src_stride, size_t count)
    {
        // Two child ckernels
        ckernel_prefix *src_is_avail = get_child_ckernel();
        expr_strided_t src_is_avail_fn =
            src_is_avail->get_function<expr_strided_t>();
        ckernel_prefix *value_assign =
            get_child_ckernel(m_value_assign_offset);
        expr_strided_t value_assign_fn =
            value_assign->get_function<expr_strided_t>();
        // Process in chunks using the dynd default buffer size
        dynd_bool avail[DYND_BUFFER_CHUNK_SIZE];
        while (count > 0) {
            size_t chunk_size = min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
            src_is_avail_fn(reinterpret_cast<char *>(avail), 1, &src, &src_stride,
                            chunk_size, src_is_avail);
            if (memchr(avail, 0, chunk_size) != NULL) {
                throw overflow_error(
                    "cannot assign an NA value to a non-option type");
            }
            value_assign_fn(dst, dst_stride, &src, &src_stride, chunk_size,
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
    const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
  intptr_t root_ckb_offset = ckb_offset;
  typedef option_to_option_ck self_type;
  if (dst_tp.get_type_id() != option_type_id ||
      src_tp[0].get_type_id() != option_type_id) {
    stringstream ss;
    ss << "option to option kernel needs option types, got " << dst_tp
       << " and " << src_tp[0];
    throw invalid_argument(ss.str());
  }
  const ndt::type &dst_val_tp = dst_tp.tcast<option_type>()->get_value_type();
  const ndt::type &src_val_tp =
      src_tp[0].tcast<option_type>()->get_value_type();
  self_type *self = self_type::create(ckb, kernreq, ckb_offset);
  // instantiate src_is_avail
  const arrfunc_type_data *af =
      src_tp[0].tcast<option_type>()->get_is_avail_arrfunc();
  ckb_offset = af->instantiate(af, ckb, ckb_offset, ndt::make_type<dynd_bool>(),
                               NULL, src_tp, src_arrmeta, kernreq, ectx);
  // instantiate dst_assign_na
  ckb->ensure_capacity_leaf(ckb_offset);
  self = ckb->get_at<self_type>(root_ckb_offset);
  self->m_dst_assign_na_offset = ckb_offset - root_ckb_offset;
  af = dst_tp.tcast<option_type>()->get_assign_na_arrfunc();
  ckb_offset = af->instantiate(af, ckb, ckb_offset, dst_tp, dst_arrmeta, NULL,
                               NULL, kernreq, ectx);
  // instantiate value_assign
  ckb->ensure_capacity(ckb_offset);
  self = ckb->get_at<self_type>(root_ckb_offset);
  self->m_value_assign_offset = ckb_offset - root_ckb_offset;
  ckb_offset =
      make_assignment_kernel(ckb, ckb_offset, dst_val_tp, dst_arrmeta,
                             src_val_tp, src_arrmeta[0], kernreq, ectx);
  return ckb_offset;
}

static intptr_t instantiate_option_to_value_assignment_kernel(
    const arrfunc_type_data *DYND_UNUSED(self), dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
  intptr_t root_ckb_offset = ckb_offset;
  typedef option_to_value_ck self_type;
  if (dst_tp.get_type_id() == option_type_id ||
      src_tp[0].get_type_id() != option_type_id) {
    stringstream ss;
    ss << "option to value kernel needs value/option types, got " << dst_tp
       << " and " << src_tp[0];
    throw invalid_argument(ss.str());
  }
  const ndt::type &src_val_tp =
      src_tp[0].tcast<option_type>()->get_value_type();
  self_type *self = self_type::create(ckb, kernreq, ckb_offset);
  // instantiate src_is_avail
  const arrfunc_type_data *af =
      src_tp[0].tcast<option_type>()->get_is_avail_arrfunc();
  ckb_offset = af->instantiate(af, ckb, ckb_offset, ndt::make_type<dynd_bool>(),
                               NULL, src_tp, src_arrmeta, kernreq, ectx);
  // instantiate value_assign
  ckb->ensure_capacity_leaf(ckb_offset);
  self = ckb->get_at<self_type>(root_ckb_offset);
  self->m_value_assign_offset = ckb_offset - root_ckb_offset;
  return make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta,
                                src_val_tp, src_arrmeta[0], kernreq, ectx);
}

namespace {
struct string_to_option_bool_ck
    : public kernels::unary_ck<string_to_option_bool_ck> {
  assign_error_mode m_errmode;

  inline void single(char *dst, const char *src)
  {
    const string_type_data *std =
        reinterpret_cast<const string_type_data *>(src);
    parse::string_to_bool(dst, std->begin, std->end, true, m_errmode);
  }
};

struct string_to_option_number_ck : public kernels::unary_ck<string_to_option_number_ck> {
    type_id_t m_tid;
    assign_error_mode m_errmode;

    inline void single(char *dst, const char *src)
    {
        const string_type_data *std =
            reinterpret_cast<const string_type_data *>(src);
        parse::string_to_number(dst, m_tid, std->begin, std->end, true,
                                m_errmode);
    }
};

struct string_to_option_tp_ck : public kernels::unary_ck<string_to_option_tp_ck> {
    intptr_t m_dst_assign_na_offset;

    inline void single(char *dst, const char *src)
    {
        const string_type_data *std =
            reinterpret_cast<const string_type_data *>(src);
        if (parse::matches_option_type_na_token(std->begin, std->end)) {
          // It's not available, assign an NA
          ckernel_prefix *dst_assign_na =
              get_child_ckernel(m_dst_assign_na_offset);
          expr_single_t dst_assign_na_fn =
              dst_assign_na->get_function<expr_single_t>();
          dst_assign_na_fn(dst, NULL, dst_assign_na);
        } else {
          // It's available, copy using value assignment
          ckernel_prefix *value_assign = get_child_ckernel();
          expr_single_t value_assign_fn =
              value_assign->get_function<expr_single_t>();
          value_assign_fn(dst, &src, value_assign);
        }
    }

    inline void destruct_children()
    {
        // value_assign
        get_child_ckernel()->destroy();
        // dst_assign_na
        base.destroy_child_ckernel(m_dst_assign_na_offset);
    }
};
}

static intptr_t instantiate_string_to_option_assignment_kernel(
    const arrfunc_type_data *DYND_UNUSED(self), dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
  // Deal with some string to option[T] conversions where string values
  // might mean NA
  if (dst_tp.get_type_id() != option_type_id ||
      !(src_tp[0].get_kind() == string_kind ||
        (src_tp[0].get_type_id() == option_type_id &&
         src_tp[0].tcast<option_type>()->get_value_type().get_kind() ==
             string_kind))) {
    stringstream ss;
    ss << "string to option kernel needs string/option types, got ("
       << src_tp[0] << ") -> " << dst_tp;
    throw invalid_argument(ss.str());
  }

  type_id_t tid = dst_tp.tcast<option_type>()->get_value_type().get_type_id();
  switch (tid) {
  case bool_type_id: {
    string_to_option_bool_ck *self =
        string_to_option_bool_ck::create_leaf(ckb, kernreq, ckb_offset);
    self->m_errmode = ectx->errmode;
    return ckb_offset;
  }
  case int8_type_id:
  case int16_type_id:
  case int32_type_id:
  case int64_type_id:
  case int128_type_id:
  case float16_type_id:
  case float32_type_id:
  case float64_type_id: {
    string_to_option_number_ck *self =
        string_to_option_number_ck::create_leaf(ckb, kernreq, ckb_offset);
    self->m_tid = tid;
    self->m_errmode = ectx->errmode;
    return ckb_offset;
  }
  case string_type_id: {
    // Just a string to string assignment
    return ::make_assignment_kernel(
        ckb, ckb_offset, dst_tp.tcast<option_type>()->get_value_type(),
        dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq, ectx);
  }
  default:
    break;
  }

  // Fall back to an adaptor that checks for a few standard
  // missing value tokens, then uses the standard value assignment
  intptr_t root_ckb_offset = ckb_offset;
  string_to_option_tp_ck *self =
      string_to_option_tp_ck::create(ckb, kernreq, ckb_offset);
  // First child ckernel is the value assignment
  ckb_offset = ::make_assignment_kernel(
      ckb, ckb_offset, dst_tp.tcast<option_type>()->get_value_type(),
      dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq, ectx);
  // Re-acquire self because the address may have changed
  self = ckb->get_at<string_to_option_tp_ck>(root_ckb_offset);
  // Second child ckernel is the NA assignment
  self->m_dst_assign_na_offset = ckb_offset - root_ckb_offset;
  const arrfunc_type_data *af =
      dst_tp.tcast<option_type>()->get_assign_na_arrfunc();
  ckb_offset = af->instantiate(af, ckb, ckb_offset, dst_tp, dst_arrmeta, NULL,
                               NULL, kernreq, ectx);
  return ckb_offset;

}

static intptr_t instantiate_option_as_value_assignment_kernel(
    const arrfunc_type_data *DYND_UNUSED(self), dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
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
  ndt::type val_dst_tp = dst_tp.get_type_id() == option_type_id
                             ? dst_tp.tcast<option_type>()->get_value_type()
                             : dst_tp;
  ndt::type val_src_tp = src_tp[0].get_type_id() == option_type_id
                             ? src_tp[0].tcast<option_type>()->get_value_type()
                             : src_tp[0];
  return ::make_assignment_kernel(ckb, ckb_offset, val_dst_tp, dst_arrmeta,
                                  val_src_tp, src_arrmeta[0], kernreq, ectx);
}

namespace {

struct option_arrfunc_list {
    arrfunc_type_data af[5];

    option_arrfunc_list() {
        int i = 0;
        af[i].func_proto = ndt::type("(?string) -> ?S");
        af[i].instantiate = &instantiate_string_to_option_assignment_kernel;
        ++i;
        af[i].func_proto = ndt::type("(?T) -> ?S");
        af[i].instantiate = &instantiate_option_to_option_assignment_kernel;
        ++i;
        af[i].func_proto = ndt::type("(?T) -> S");
        af[i].instantiate = &instantiate_option_to_value_assignment_kernel;
        ++i;
        af[i].func_proto = ndt::type("(string) -> ?S");
        af[i].instantiate = &instantiate_string_to_option_assignment_kernel;
        ++i;
        af[i].func_proto = ndt::type("(T) -> S");
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
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
    static option_arrfunc_list afl;
    intptr_t size = afl.size();
    const arrfunc_type_data *af = afl.get();
    map<nd::string, ndt::type> typevars;
    for (intptr_t i = 0; i < size; ++i, ++af) {
        typevars.clear();
        if (ndt::pattern_match(src_tp, af->get_param_type(0), typevars) &&
                ndt::pattern_match(dst_tp, af->get_return_type(), typevars)) {
            return af->instantiate(af, ckb, ckb_offset, dst_tp, dst_arrmeta,
                                   &src_tp, &src_arrmeta, kernreq, ectx);
        }
    }

    stringstream ss;
    ss << "Could not instantiate option assignment kernel from " << src_tp
       << " to " << dst_tp;
    throw invalid_argument(ss.str());
}
