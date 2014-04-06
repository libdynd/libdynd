//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__PP_META_HPP_
#define _DYND__PP_META_HPP_

// #define DYND_PP_META_ARRAY_DIMS(A) DYND_PP_MAP(DYND_PP_META_AT, 

#define DYND_PP_META_EQ(LHS, RHS) LHS = RHS
#define DYND_PP_META_NE(LHS, RHS) LHS != RHS

#define DYND_PP_META_ADD_EQ(A, B) A += B 
#define DYND_PP_META_SUB_EQ(A, B) A -= B 

#define DYND_PP_META_DECL(TYPE, VAR) TYPE VAR
#define DYND_PP_META_DECL_EQ(TYPE, VAR, VALUE) DYND_PP_META_EQ(DYND_PP_META_DECL(TYPE, VAR), VALUE)


#define DYND_PP_META_AT(VAR, INDEX) VAR[INDEX]
#define DYND_PP_META_DEREFERENCE(VAR) *VAR
#define DYND_PP_META_ADDRESS(VAR) &VAR

#define DYND_PP_META_CALL(FUNC) FUNC()

#define DYND_PP_META_DOT(LHS, RHS) LHS.RHS
#define DYND_PP_META_DOT_CALL(OBJ, FUNC) DYND_PP_META_DOT(OBJ, DYND_PP_META_CALL(FUNC))

#define DYND_PP_META_SCOPE(LHS, RHS) LHS::RHS
#define DYND_PP_META_SCOPE_CALL(OBJ, FUNC) DYND_PP_META_SCOPE(OBJ, DYND_PP_CALL(FUNC))

#define DYND_PP_META_TYPENAME(TYPE) typename TYPE

#define DYND_PP_META_TEMPLATE(TEMP, ARGS) TEMP<ARGS>

#define DYND_PP_META_STATIC_CAST(TYPE, VAR) static_cast<TYPE>(VAR)
#define DYND_PP_META_REINTERPRET_CAST(TYPE, VAR) reinterpret_cast<TYPE>(VAR)
#define DYND_PP_META_CONST_CAST(TYPE, VAR) const_cast<TYPE>(VAR)
#define DYND_PP_META_DYNAMIC_CAST(TYPE, VAR) dynamic_cast<TYPE>(VAR)

#define DYND_PP_META_NAME_RANGE(NAME, STOP) DYND_PP_OUTER(DYND_PP_PASTE, (NAME), DYND_PP_RANGE(STOP))
#define DYND_PP_META_AT_RANGE(VAR, STOP) DYND_PP_OUTER(DYND_PP_META_AT, (VAR), DYND_PP_RANGE(STOP))


#define DYND_PP_META_AS_CONST_PTR(TOKEN) const TOKEN *

#endif
