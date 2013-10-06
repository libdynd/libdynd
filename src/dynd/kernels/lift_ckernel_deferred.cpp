//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/lift_ckernel_deferred.hpp>

using namespace std;
using namespace dynd;

void dynd::lift_ckernel_deferred(ckernel_deferred *out_ckd,
                const nd::array& ckd_arr,
                const std::vector<ndt::type>& lifted_types)
{
    // Validate the input ckernel_deferred
    if (ckd_arr.get_type().get_type_id() != ckernel_deferred_type_id) {
        stringstream ss;
        ss << "lift_ckernel_deferred() 'ckd' must have type "
           << "ckernel_deferred, not " << ckd_arr.get_type();
        throw runtime_error(ss.str());
    }
    const ckernel_deferred *ckd = reinterpret_cast<const ckernel_deferred *>(ckd_arr.get_readonly_originptr());
    if (ckd->instantiate_func == NULL) {
        throw runtime_error("lift_ckernel_deferred() 'ckd' must contain a"
                        " non-null ckernel_deferred object");
    }
    // Validate that all the types are subarrays as needed for lifting
    intptr_t ntypes = ckd->data_types_size;
    if (ntypes != lifted_types.size()) {
        stringstream ss;
        ss << "lift_ckernel_deferred() 'lifted_types' list must have "
           << "the same number of types as the input ckernel_deferred "
           << "(" << lifted_types.size() << " vs " << ntypes << ")";
        throw runtime_error(ss.str());
    }
    const ndt::type *ckd_types = ckd->data_dynd_types;
    for (intptr_t i = 0; i < ntypes; ++i) {
        if (!lifted_types[i].is_type_subarray(ckd_types[i])) {
            stringstream ss;
            ss << "lift_ckernel_deferred() 'lifted_types[" << i << "]' value must "
               << "have the corresponding input ckernel_deferred type as a subarray "
               << "(" << ckd_types[i] << " is not a subarray of " << lifted_types[i] << ")";
            throw runtime_error(ss.str());
        }
    }

    throw runtime_error("lift_ckernel_deferred() is not finished");
}
