//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/date_adapter_kernels.hpp>
#include <dynd/parser_util.hpp>
#include <dynd/types/date_parser.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/funcproto_type.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/string.hpp>

using namespace std;
using namespace dynd;

/**
 * Matches netcdf date metadata like "days since 2001-1-1".
 */
static bool parse_days_since(const char *begin, const char *end,
                             int32_t &out_epoch_date)
{
    if (!parse::parse_token(begin, end, "days")) {
        return false;
    }
    if (!parse::skip_required_whitespace(begin, end)) {
        return false;
    }
    // The tokens supported by netcdf, as from the udunits libarary
    if (!parse::parse_token(begin, end, "since") &&
            !parse::parse_token(begin, end, "after") &&
            !parse::parse_token(begin, end, "from") &&
            !parse::parse_token(begin, end, "ref") &&
            !parse::parse_token(begin, end, '@')) {
        return false;
    }
    if (!parse::skip_required_whitespace(begin, end)) {
        return false;
    }
    date_ymd epoch;
    if (!parse::parse_date(begin, end, epoch, date_parse_no_ambig, 0)) {
      int year;
      if (parse::parse_4digit_int_no_ws(begin, end, year)) {
        epoch.year = static_cast<int16_t>(year);
        epoch.month = 1;
        epoch.day = 1;
      } else {
        return false;
      }
    }
    parse::skip_whitespace(begin, end);
    out_epoch_date = epoch.to_days();
    return begin == end;
}

namespace {
template<class Tsrc, class Tdst>
struct int_offset_ck : public kernels::unary_ck<int_offset_ck<Tsrc, Tdst> > {
    Tdst m_offset;

    Tdst operator()(Tsrc value) {
        return value != std::numeric_limits<Tsrc>::min()
                   ? static_cast<Tdst>(value) + m_offset
                   : std::numeric_limits<Tdst>::min();
    }
};

template <class Tsrc, class Tdst>
static intptr_t instantiate_int_offset_arrfunc(
    const arrfunc_type_data *self_af, dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *DYND_UNUSED(dst_arrmeta), const ndt::type *src_tp,
    const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
    const nd::array &DYND_UNUSED(args), const nd::array &DYND_UNUSED(kwds), const eval::eval_context *DYND_UNUSED(ectx))
{
    typedef int_offset_ck<Tsrc, Tdst> self_type;
    if (dst_tp !=
                self_af->func_proto.tcast<funcproto_type>()->get_return_type() ||
            src_tp[0] !=
                self_af->func_proto.tcast<funcproto_type>()->get_param_type(0)) {
        stringstream ss;
        ss << "Cannot instantiate arrfunc with signature ";
        ss << self_af->func_proto << " with types (";
        ss << src_tp[0] << ") -> " << dst_tp;
        throw type_error(ss.str());
    }
    self_type *self = self_type::create_leaf(ckb, kernreq, ckb_offset);
    self->m_offset = *self_af->get_data_as<Tdst>();
    return ckb_offset;
}

template <class Tsrc, class Tdst>
nd::arrfunc make_int_offset_arrfunc(Tdst offset, const ndt::type &func_proto)
{
    nd::array out_af = nd::empty(ndt::make_arrfunc());
    arrfunc_type_data *af =
        reinterpret_cast<arrfunc_type_data *>(out_af.get_readwrite_originptr());
    af->func_proto = func_proto;
    *af->get_data_as<Tdst>() = offset;
    af->instantiate = &instantiate_int_offset_arrfunc<Tsrc, Tdst>;
    out_af.flag_as_immutable();
    return out_af;
}
} // anonymous namespace

bool dynd::make_date_adapter_arrfunc(const ndt::type &operand_tp,
                                     const nd::string &op,
                                     nd::arrfunc &out_forward,
                                     nd::arrfunc &out_reverse)
{
    int32_t epoch_date;
    if (parse_days_since(op.begin(), op.end(), epoch_date)) {
        switch (operand_tp.get_type_id()) {
            case int32_type_id:
                out_forward = make_int_offset_arrfunc<int32_t, int32_t>(
                    epoch_date, ndt::make_funcproto(ndt::make_type<int32_t>(),
                                                    ndt::make_date()));
                out_reverse = make_int_offset_arrfunc<int32_t, int32_t>(
                    -epoch_date,
                    ndt::make_funcproto(ndt::make_date(),
                                        ndt::make_type<int32_t>()));
                return true;
            case int64_type_id:
                out_forward = make_int_offset_arrfunc<int64_t, int32_t>(
                    epoch_date, ndt::make_funcproto(ndt::make_type<int64_t>(),
                                                    ndt::make_date()));
                out_reverse = make_int_offset_arrfunc<int32_t, int64_t>(
                    -epoch_date,
                    ndt::make_funcproto(ndt::make_date(),
                                        ndt::make_type<int64_t>()));
                return true;
            default:
                return false;
        }
    } else {
        return false;
    }
}
