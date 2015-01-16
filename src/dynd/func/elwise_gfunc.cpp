//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#if 0 // TODO reenable

#include <algorithm>

#include <dynd/gfunc/elwise_gfunc.hpp>

using namespace std;
using namespace dynd;

const dynd::gfunc::elwise_kernel *
dynd::gfunc::elwise::find_matching_kernel(const std::vector<ndt::type>& paramtypes) const
{
    for(size_t i = 0, i_end = m_kernels.size(); i != i_end; ++i) {
        const std::vector<ndt::type>& kparamtypes = m_kernels[i].m_paramtypes;
        if (kparamtypes == paramtypes) {
            return &m_kernels[i];
        }
    }

    return NULL;
}

void dynd::gfunc::elwise::add_kernel(elwise_kernel& egk)
{
    const elwise_kernel *check = find_matching_kernel(egk.m_paramtypes);
    if (check == NULL) {
        m_kernels.push_back(elwise_kernel());
        m_kernels.back().swap(egk);
    } else {
        stringstream ss;
        ss << "Cannot add kernel to gfunc " << m_name << " because a kernel with the same arguments, (";
        for (size_t j = 0, j_end = egk.m_paramtypes.size(); j != j_end; ++j) {
            ss << egk.m_paramtypes[j];
            if (j != j_end - 1) {
                ss << ", ";
            }
        }
        ss << "), already exists in the gfunc";
        throw runtime_error(ss.str());
    }
}

void dynd::gfunc::elwise::debug_print(std::ostream& o, const std::string& indent) const
{
    o << indent << "------ elwise_gfunc\n";
    o << indent << "name: " << m_name << "\n";
    o << indent << "kernel count: " << m_kernels.size() << "\n";
    for (deque<elwise_kernel>::size_type i = 0; i < m_kernels.size(); ++i) {
        const elwise_kernel &k = m_kernels[i];
        o << indent << "kernel " << i << "\n";
        o << indent << "   " << k.m_returntype << " (";
        for (size_t j = 0, j_end = k.m_paramtypes.size(); j != j_end; ++j) {
            o << k.m_paramtypes[j];
            if (j != j_end - 1) {
                o << ", ";
            }
        }
        o << ")\n";
        if (k.m_paramtypes.size() == 1) {
            o << indent << "unary aux data: " << (const void *)(const dynd::AuxDataBase *)k.m_unary_kernel.auxdata << "\n";
        } else if (k.m_paramtypes.size() == 2) {
            o << indent << "binary aux data: " << (const void *)(const dynd::AuxDataBase *)k.m_binary_kernel.auxdata << "\n";
        }
    }
    o << indent << "------" << endl;
}

#endif // TODO reenable
