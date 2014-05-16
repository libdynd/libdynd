rem  Batch file to compile Cephes Math Library in Microsoft C
rem
rem  You might have to comment out the definitions of INFINITIES,
rem  NANS, and MINUSZERO in mconf.h.
rem  Use the following for no 8087 support:
rem  cl /c polevl.c/FPa;
rem  Use the following instead if you have an 8087 chip
rem  or software emulator and change the /FPa everywhere
rem  to /FPi or /FPi87
rem  masm polevl.asm/r;
cl /c polevl.c
rem
cl /c acosh.c
cl /c airy.c
cl /c asin.c
cl /c asinh.c
cl /c atan.c
cl /c atanh.c
cl /c bdtr.c
cl /c beta.c
cl /c btdtr.c
cl /c cbrt.c
cl /c chbevl.c
cl /c chdtr.c
cl /c clog.c
cl /c cmplx.c
cl /c const.c
cl /c cosh.c
cl /c dawsn.c
cl /c drand.c
cl /c ellie.c
cl /c ellik.c
cl /c ellpe.c
cl /c ellpj.c
cl /c ellpk.c
cl /c exp.c
cl /c exp2.c
cl /c exp10.c
cl /c expn.c
cl /c expx2.c
cl /c fac.c
cl /c fdtr.c
cl /c fresnl.c
cl /c gamma.c
cl /c gdtr.c
cl /c hyperg.c
cl /c hyp2f1.c
cl /c incbet.c
cl /c incbi.c
cl /c igam.c
cl /c igami.c
cl /c iv.c
cl /c i0.c
cl /c i1.c
cl /c jn.c
cl /c jv.c
cl /c j0.c
cl /c j1.c
cl /c k0.c
cl /c k1.c
cl /c kn.c
cl /c log.c
cl /c log10.c
cl /c mtherr.c
cl /c nbdtr.c
cl /c ndtr.c
cl /c ndtri.c
cl /c pdtr.c
cl /c pow.c
cl /c powi.c
cl /c psi.c
cl /c rgamma.c
cl /c round.c
cl /c shichi.c
cl /c sici.c
cl /c sin.c
cl /c sindg.c
cl /c sinh.c
cl /c spence.c
cl /c sqrt.c
cl /c stdtr.c
cl /c struve.c
cl /c tan.c
cl /c tandg.c
cl /c tanh.c
cl /c yn.c
cl /c zeta.c
cl /c zetac.c
lib @ftiasm.rsp
