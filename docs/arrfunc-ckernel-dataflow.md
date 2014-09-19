# Data Flow of ArrFuncs/CKernels

DyND arrfuncs represent array computations in DyND,
divied up between dynamically made execution choices and
the ability to prepare an execution plan to run repeatedly
over many elements with the same memory layout.

The basic steps to use an arrfunc include:

1. Call the arrfunc's type resolution to
   determine the type of the output if it isn't already
   known.
2. Create the output array (e.g. with nd.empty) if it
   doesn't already exist.
3. Instantiate the arrfunc into a ckernel.
4. Execute the ckernel on raw memory addresses.


