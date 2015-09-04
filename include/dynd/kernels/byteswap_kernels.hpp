//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

namespace dynd {

/**
 * Function for byteswapping a single value.
 */
inline uint16_t byteswap_value(uint16_t value) {
    return ((value&0xffu) << 8) | (value >> 8);
}

/**
 * Function for byteswapping a single value.
 */
inline uint32_t byteswap_value(uint32_t value) {
    return ((value&0xffu) << 24) |
            ((value&0xff00u) << 8) |
            ((value&0xff0000u) >> 8) |
            (value >> 24);
}

/**
 * Function for byteswapping a single value.
 */
inline uint64_t byteswap_value(uint64_t value) {
    return ((value&0xffULL) << 56) |
            ((value&0xff00ULL) << 40) |
            ((value&0xff0000ULL) << 24) |
            ((value&0xff000000ULL) << 8) |
            ((value&0xff00000000ULL) >> 8) |
            ((value&0xff0000000000ULL) >> 24) |
            ((value&0xff000000000000ULL) >> 40) |
            (value >> 56);
}

/**
 * Creates an assignment kernel which does a byteswap
 * of the specified data size.
 */
DYND_API size_t make_byteswap_assignment_function(
                void *ckb, intptr_t ckb_offset,
                intptr_t data_size, intptr_t data_alignment,
                kernel_request_t kernreq);

/**
 * Creates an assignment kernel which does a byteswap
 * of the specified data size.
 */
DYND_API size_t make_pairwise_byteswap_assignment_function(
                void *ckb, intptr_t ckb_offset,
                intptr_t data_size, intptr_t data_alignment,
                kernel_request_t kernreq);

} // namespace dynd
