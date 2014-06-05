//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/ndarrayarg_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

void ndarrayarg_type::print_data(std::ostream &o,
                                 const char *DYND_UNUSED(arrmeta),
                                 const char *data) const
{
    o << *reinterpret_cast<const nd::array *>(data);
}

void ndarrayarg_type::print_type(std::ostream& o) const
{
    o << "ndarrayarg";
}

 bool ndarrayarg_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
{
    if (dst_tp.extended() == this) {
        if (src_tp.extended() == this) {
            return true;
        } else {
            return src_tp.get_type_id() == ndarrayarg_type_id;
        }
    } else {
        return false;
    }
}

bool ndarrayarg_type::operator==(const base_type& rhs) const
{
    return rhs.get_type_id() == ndarrayarg_type_id;
}

namespace {
    struct ndarrayarg_assign_ck : public kernels::assignment_ck<ndarrayarg_assign_ck> {
        inline void single(char *dst, const char *src)
        {
            if (*reinterpret_cast<void *const *>(src) == NULL) {
                *reinterpret_cast<void **>(dst) = NULL;
            } else {
                throw invalid_argument(
                    "Cannot make a copy of a non-NULL dynd ndarrayarg value");
            }
        }
    };
} // anonymous namespace

size_t ndarrayarg_type::make_assignment_kernel(
    ckernel_builder *ckb, size_t ckb_offset, const ndt::type &dst_tp,
    const char *DYND_UNUSED(dst_arrmeta), const ndt::type &src_tp,
    const char *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
    const eval::eval_context *DYND_UNUSED(ectx)) const
{
    if (this == dst_tp.extended() && src_tp.get_type_id() == ndarrayarg_type_id) {
        ndarrayarg_assign_ck::create_leaf(ckb, ckb_offset, kernreq);
        return ckb_offset + sizeof(ndarrayarg_assign_ck);
    }

    stringstream ss;
    ss << "Cannot assign from " << src_tp << " to " << dst_tp;
    throw dynd::type_error(ss.str());
}

size_t ndarrayarg_type::make_comparison_kernel(
    ckernel_builder *DYND_UNUSED(ckb), size_t DYND_UNUSED(ckb_offset), const ndt::type &src0_tp,
    const char *DYND_UNUSED(src0_arrmeta), const ndt::type &src1_tp,
    const char *DYND_UNUSED(src1_arrmeta), comparison_type_t comptype,
    const eval::eval_context *DYND_UNUSED(ectx)) const
{
    throw not_comparable_error(src0_tp, src1_tp, comptype);
}

ndt::type ndt::make_ndarrayarg()
{
    // Static instance of ndarrayarg_type, which has a reference count > 0 for the
    // lifetime of the program. This static construction is inside a
    // function to ensure correct creation order during startup.
    static ndarrayarg_type nat;
    const ndt::type static_instance(&nat, true);
    return static_instance;
}
