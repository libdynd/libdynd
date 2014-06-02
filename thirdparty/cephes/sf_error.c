#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

#include "sf_error.h"

const char *sf_error_messages[] = {
    "no error",
    "singularity",
    "underflow",
    "overflow",
    "too slow convergence",
    "loss of precision",
    "no result obtained",
    "domain error",
    "invalid input argument",
    "other error",
    NULL
};

static int print_error_messages = 0;

extern int wrap_PyUFunc_getfperr();

int sf_error_set_print(int flag)
{
    int old_flag = print_error_messages;
    print_error_messages = flag;
    return old_flag;
}

int sf_error_get_print()
{
    return print_error_messages;
}

void sf_error(char *func_name, sf_error_t code, char *fmt, ...)
{
    printf("%s error\n", sf_error_messages[(int)code]);
}

#define UFUNC_FPE_DIVIDEBYZERO  1
#define UFUNC_FPE_OVERFLOW      2
#define UFUNC_FPE_UNDERFLOW     4
#define UFUNC_FPE_INVALID       8
