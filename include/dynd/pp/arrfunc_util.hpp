//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND__PP_ARRFUNC_UTIL_HPP
#define DYND__PP_ARRFUNC_UTIL_HPP

#include <dynd/pp/list.hpp>
#include <dynd/pp/meta.hpp>

/**
 * Generates comma-separated names typically used for function arguments.
 *
 * NAME0, NAME1, ...
 */
#define DYND_PP_ARGRANGE_1(NAME, N)                                            \
  DYND_PP_JOIN_1((, ), DYND_PP_META_NAME_RANGE(NAME, N))

/**
 * Generates comma-separated typename names typically used for template
 * arguments.
 *
 * typename NAME0, typename NAME1, ...
 */
#define DYND_PP_TYPENAME_ARGRANGE_1(NAME, N)                                   \
  DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (, ),                              \
                     DYND_PP_META_NAME_RANGE(NAME, N))

/**
 * Applies a template metafunction to a range of named types.
 *
 * (TMPFUNC<NAME0>::value, TMPFUNC<NAME1>::value, ...)
 */
#define DYND_PP_APPLY_TMPFUNC_RANGE_1(TMPFUNC, NAME, N)                        \
  DYND_PP_ELWISE_1(DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE,                  \
                   DYND_PP_REPEAT(TMPFUNC, N),                                 \
                   DYND_PP_META_NAME_RANGE(NAME, N), DYND_PP_REPEAT(value, N))

/**
 * Greates a range of named types which are cleaned versions of an input
 * range of named types.
 *
 * typedef remove_const<remove_reference<TYPE0>::type>::type CLEANED_TYPE0;
 * typedef remove_const<remove_reference<TYPE1>::type>::type CLEANED_TYPE1;
 * ...
 */
#define DYND_PP_CLEAN_TYPE_RANGE_1(CLEANED_TYPE, TYPE, N)                      \
  DYND_PP_JOIN_ELWISE_1(                                                       \
      DYND_PP_META_TYPEDEF_TYPENAME, (;),                                      \
      DYND_PP_ELWISE_1(                                                        \
          DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE,                           \
          DYND_PP_REPEAT(remove_const, N),                                     \
          DYND_PP_ELWISE_1(DYND_PP_META_TYPENAME_TEMPLATE_INSTANTIATION_SCOPE, \
                           DYND_PP_REPEAT(remove_reference, N),       \
                           DYND_PP_META_NAME_RANGE(TYPE, N),                   \
                           DYND_PP_REPEAT(type, N)),                           \
          DYND_PP_REPEAT(type, N)),                                            \
      DYND_PP_META_NAME_RANGE(CLEANED_TYPE, N));

/**
 * Static_assert that a template metafunction returns true for each type in a
 * range of named types.
 *
 * static_assert(TMPFUNC<NAME0>::value, MSG);
 * static_assert(TMPFUNC<NAME1>::value, MSG);
 * ...
 */
#define DYND_PP_STATIC_ASSERT_RANGE_1(MSG, TMPFUNC, NAME, N)                   \
  DYND_PP_JOIN_ELWISE_1(DYND_PP_META_STATIC_ASSERT, (;),                       \
                        DYND_PP_APPLY_TMPFUNC_RANGE_1(TMPFUNC, NAME, N),       \
                        DYND_PP_REPEAT(MSG, N));

/**
 * For generating ckernel function calls, casts each ``ARG_NAME[#]`` input to
 * the type ``TYPE#`` and dereferences it, output with a comma separator.
 *
 * *reinterpret_cast<const TYPE0 *>(ARG_NAME[0]), ...",
 * *reinterpret_cast<const TYPE1 *>(ARG_NAME[1]), ...",
 * ...
 */
#define DYND_PP_DEREF_CAST_ARRAY_RANGE_1(TYPE, ARG_NAME, N)                    \
  DYND_PP_JOIN_MAP_1(                                                          \
      DYND_PP_META_DEREFERENCE, (, ),                                          \
      DYND_PP_ELWISE_1(DYND_PP_META_REINTERPRET_CAST,                          \
                       DYND_PP_MAP_1(DYND_PP_META_MAKE_CONST_PTR,              \
                                     DYND_PP_META_NAME_RANGE(TYPE, N)),        \
                       DYND_PP_META_AT_RANGE(ARG_NAME, N)))

/**
 * For generating ckernel function calls, casts each ``ARG_NAME#`` input to
 * the type ``TYPE#`` and dereferences it, output with a comma separator.
 *
 * *reinterpret_cast<const TYPE0 *>(ARG_NAME0), ...",
 * *reinterpret_cast<const TYPE1 *>(ARG_NAME1), ...",
 * ...
 */
#define DYND_PP_DEREF_CAST_ARGRANGE_1(TYPE, ARG_NAME, N)                      \
  DYND_PP_JOIN_MAP_1(                                                          \
      DYND_PP_META_DEREFERENCE, (, ),                                          \
      DYND_PP_ELWISE_1(DYND_PP_META_REINTERPRET_CAST,                          \
                       DYND_PP_MAP_1(DYND_PP_META_MAKE_CONST_PTR,              \
                                     DYND_PP_META_NAME_RANGE(TYPE, N)),        \
                       DYND_PP_META_NAME_RANGE(ARG_NAME, N)))

/**
 * Declares a range of ``TYPE`` variables ``ARG_NAME#`` and
 * initializes them to ``ARG_NAME[#]``.
 *
 * const char *ARG_NAME0 = ARG_NAME[0];
 * const char *ARG_NAME1 = ARG_NAME[1];
 * ...
 */
#define DYND_PP_VARIABLE_RANGE_FROM_ARRAY_1(TYPE, ARG_NAME, N)                 \
  DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;),                           \
                        DYND_PP_REPEAT_1(TYPE, N),                             \
                        DYND_PP_META_NAME_RANGE(ARG_NAME, N),                  \
                        DYND_PP_META_AT_RANGE(ARG_NAME, N));

/**
 * Initializes ``src#`` and ``src_stride#`` variables from the ``src`` and
 * ``src_stride`` input arrays.
 *
 * const char *src0 = src[0];
 * const char *src1 = src[1];
 * ...
 * intptr_t src_stride0 = src_stride[0];
 * intptr_t src_stride1 = src_stride[1];
 * ...
 *
 */
#define DYND_PP_INIT_SRC_VARIABLES(N)                                          \
  DYND_PP_VARIABLE_RANGE_FROM_ARRAY_1(const char *, src, N);                   \
  DYND_PP_VARIABLE_RANGE_FROM_ARRAY_1(intptr_t, src_stride, N);

/**
 * Increment ``dst``, ``src0``, etc. with ``dst_stride``, ``src_stride0``, etc.
 *
 * dst += dst_stride;
 * src0 += src_stride0;
 * src1 += src_stride1;
 */
#define DYND_PP_STRIDED_INCREMENT(N)                                           \
  dst += dst_stride;                                                           \
  DYND_PP_JOIN_ELWISE_1(DYND_PP_META_ADD_ASGN, (;),                            \
                        DYND_PP_META_NAME_RANGE(src, N),                       \
                        DYND_PP_META_NAME_RANGE(src_stride, N));

/**
 * Create ``dst_tp`` and ``src_tp[N]``, etc. from the types DST_TYPE, SRC_TYPE0,
 * etc.
 *
 */
#define DYND_PP_NDT_TYPES_FROM_TYPES(DST_TYPE, SRC_TYPE, N)                    \
  ndt::type dst_tp = ndt::cfixed_dim_from_array<DST_TYPE>::make();             \
  ndt::type src_tp[N] = {DYND_PP_JOIN_ELWISE_1(                                \
      DYND_PP_META_SCOPE_CALL, (, ),                                           \
      DYND_PP_ELWISE_1(DYND_PP_META_TEMPLATE_INSTANTIATION,                    \
                       DYND_PP_REPEAT_1(ndt::cfixed_dim_from_array, N),        \
                       DYND_PP_META_NAME_RANGE(SRC_TYPE, N)),                  \
      DYND_PP_REPEAT_1(make, N))};

#endif // DYND__PP_ARRFUNC_UTIL_HPP
