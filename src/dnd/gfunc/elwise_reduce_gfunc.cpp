//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dnd/nodes/elwise_reduce_kernel_node.hpp>
#include <dnd/gfunc/elwise_reduce_gfunc.hpp>

using namespace std;
using namespace dnd;

const dnd::gfunc::elwise_reduce_gfunc_kernel *
dnd::gfunc::elwise_reduce_gfunc::find_matching_kernel(const std::vector<dtype>& paramtypes) const
{
    for(size_t i = 0, i_end = m_kernels.size(); i != i_end; ++i) {
        const std::vector<dtype>& kparamtypes = m_kernels[i].m_paramtypes;
        if (kparamtypes == paramtypes) {
            return &m_kernels[i];
        }
    }

    return NULL;
}

void dnd::gfunc::elwise_reduce_gfunc::add_kernel(elwise_reduce_gfunc_kernel& ergk)
{
    const elwise_reduce_gfunc_kernel *check = find_matching_kernel(ergk.m_paramtypes);
    if (check == NULL) {
        m_kernels.push_back(elwise_reduce_gfunc_kernel());
        m_kernels.back().swap(ergk);
    } else {
        stringstream ss;
        ss << "Cannot add kernel to gfunc " << m_name << " because a kernel with the same arguments, (";
        for (size_t j = 0, j_end = ergk.m_paramtypes.size(); j != j_end; ++j) {
            ss << ergk.m_paramtypes[j];
            if (j != j_end - 1) {
                ss << ", ";
            }
        }
        ss << "), already exists in the gfunc";
        throw runtime_error(ss.str());
    }
}

void dnd::gfunc::elwise_reduce_gfunc::debug_dump(std::ostream& o, const std::string& indent) const
{
    o << indent << "------ elwise_reduce_gfunc\n";
    o << indent << "name: " << m_name << "\n";
    o << indent << "kernel count: " << m_kernels.size() << "\n";
    for (deque<elwise_reduce_gfunc_kernel>::size_type i = 0; i < m_kernels.size(); ++i) {
        const elwise_reduce_gfunc_kernel &k = m_kernels[i];
        o << indent << "kernel " << i << "\n";
        o << indent << " signature: " << k.m_returntype << " (";
        for (size_t j = 0, j_end = k.m_paramtypes.size(); j != j_end; ++j) {
            o << k.m_paramtypes[j];
            if (j != j_end - 1) {
                o << ", ";
            }
        }
        o << ")\n";
        if (k.m_left_associative_reduction_kernel.kernel != NULL) {
            o << indent << " left associative kernel aux data: " << (const void *)(const dnd::AuxDataBase *)k.m_left_associative_reduction_kernel.auxdata << "\n";
        }
        if (k.m_right_associative_reduction_kernel.kernel != NULL) {
            o << indent << " right associative kernel aux data: " << (const void *)(const dnd::AuxDataBase *)k.m_right_associative_reduction_kernel.auxdata << "\n";
        }
        if (k.m_identity.get_node().get()) {
            o << indent << " reduction identity:\n";
            k.m_identity.debug_dump(o, indent + "  ");
        } else {
            o << indent << " reduction identity: NULL\n";
        }
    }
    o << indent << "------" << endl;
}
