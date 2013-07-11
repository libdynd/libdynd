//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/shortvector.hpp>
#include <dynd/type.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/string_algorithm_kernels.hpp>
#include <dynd/types/string_type.hpp>

using namespace std;
using namespace dynd;

void kernels::string_concatenation_kernel::init(
                size_t nop,
                const char *dst_metadata,
                const char **DYND_UNUSED(src_metadata))
{
    const string_type_metadata *sdm = reinterpret_cast<const string_type_metadata *>(dst_metadata);
    m_nop = nop;
    // This is a borrowed reference
    m_dst_blockref = sdm->blockref;
}

inline void concat_one_string(
                size_t nop, string_type_data *d,
                const string_type_data * const *s,
                memory_block_pod_allocator_api *allocator,
                memory_block_data *dst_blockref)
{
    // Get the size of the concatenated string
    size_t size = 0;
    for (size_t i = 0; i != nop; ++i) {
        size += (s[i]->end - s[i]->begin);
    }
    // Allocate the output
    size_t alignment = 1; // NOTE: This kernel is hardcoded for UTF-8, alignment 1
    allocator->allocate(dst_blockref, size, alignment,
                    &d->begin, &d->end);
    // Copy the string data
    char *dst = d->begin;
    for (size_t i = 0; i != nop; ++i) {
        size_t op_size = (s[i]->end - s[i]->begin);
        DYND_MEMCPY(dst, s[i]->begin, op_size);
        dst += op_size;
    }
}

void kernels::string_concatenation_kernel::single(
                char *dst, const char * const *src,
                kernel_data_prefix *extra)
{
    const extra_type *e = reinterpret_cast<const extra_type *>(extra);
    string_type_data *d = reinterpret_cast<string_type_data *>(dst);
    const string_type_data * const *s = reinterpret_cast<const string_type_data * const *>(src);
    memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(e->m_dst_blockref);

    concat_one_string(e->m_nop, d, s, allocator, e->m_dst_blockref);
}

void kernels::string_concatenation_kernel::strided(
                char *dst, intptr_t dst_stride,
                const char * const *src, const intptr_t *src_stride,
                size_t count, kernel_data_prefix *extra)
{
    const extra_type *e = reinterpret_cast<const extra_type *>(extra);
    size_t nop = e->m_nop;
    memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(e->m_dst_blockref);

    // Loop to concatenate all the strings3
    shortvector<const char *> src_vec(nop, src); 
    for (size_t i = 0; i != count; ++i) {
        string_type_data *d = reinterpret_cast<string_type_data *>(dst);
        const string_type_data * const *s = reinterpret_cast<const string_type_data * const *>(src_vec.get());
        concat_one_string(nop, d, s, allocator, e->m_dst_blockref);
        dst += dst_stride;
        for (size_t op = 0; op < nop; ++op) {
            src_vec[op] += src_stride[op];
        }
    }
}
