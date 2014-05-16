#
# Borland C++ IDE generated makefile
#
.AUTODEPEND


#
# Borland C++ tools
#
IMPLIB  = Implib
BCCDOS  = Bcc +BccDos.cfg 
TLINK   = TLink
TLIB    = TLib
TASM    = Tasm
#
# IDE macros
#


#
# Options
#
IDE_LFLAGSDOS =  -LC:\BC45\LIB
IDE_BFLAGS = 
LLATDOS_doubledlib =  -c -Tde
RLATDOS_doubledlib = 
BLATDOS_doubledlib = 
CNIEAT_doubledlib = -IC:\BC45\INCLUDE;C:\MATH\DOUBLE -D
LNIEAT_doubledlib = -x
LEAT_doubledlib = $(LLATDOS_doubledlib)
REAT_doubledlib = $(RLATDOS_doubledlib)
BEAT_doubledlib = $(BLATDOS_doubledlib)

#
# Dependency List
#
Dep_double = \
   double.lib

double : BccDos.cfg $(Dep_double)
  echo MakeNode 

Dep_doubledlib = \
   setprbor.obj\
   unity.obj\
   yn.obj\
   zeta.obj\
   zetac.obj\
   stdtr.obj\
   struve.obj\
   tan.obj\
   tandg.obj\
   tanh.obj\
   shichi.obj\
   sici.obj\
   simpsn.obj\
   simq.obj\
   sin.obj\
   sincos.obj\
   sindg.obj\
   sinh.obj\
   spence.obj\
   polrt.obj\
   polylog.obj\
   polyn.obj\
   pow.obj\
   powi.obj\
   psi.obj\
   revers.obj\
   rgamma.obj\
   round.obj\
   pdtr.obj\
   planck.obj\
   polevl.obj\
   polmisc.obj\
   nbdtr.obj\
   ndtr.obj\
   ndtri.obj\
   mtherr.obj\
   kolmogor.obj\
   levnsn.obj\
   log.obj\
   log10.obj\
   log2.obj\
   lrand.obj\
   lsqrt.obj\
   incbet.obj\
   incbi.obj\
   isnan.obj\
   iv.obj\
   j0.obj\
   j1.obj\
   jn.obj\
   jv.obj\
   k0.obj\
   k1.obj\
   kn.obj\
   fresnl.obj\
   gamma.obj\
   gdtr.obj\
   gels.obj\
   hyp2f1.obj\
   hyperg.obj\
   i0.obj\
   i1.obj\
   igam.obj\
   igami.obj\
   exp.obj\
   exp10.obj\
   exp2.obj\
   expn.obj\
   fabs.obj\
   fac.obj\
   fdtr.obj\
   fftr.obj\
   floor.obj\
   euclid.obj\
   ei.obj\
   eigens.obj\
   ellie.obj\
   ellik.obj\
   ellpe.obj\
   ellpj.obj\
   ellpk.obj\
   drand.obj\
   const.obj\
   cosh.obj\
   cpmul.obj\
   dawsn.obj\
   clog.obj\
   cmplx.obj\
   chbevl.obj\
   chdtr.obj\
   acosh.obj\
   airy.obj\
   arcdot.obj\
   asin.obj\
   asinh.obj\
   atan.obj\
   atanh.obj\
   bdtr.obj\
   beta.obj\
   btdtr.obj\
   cbrt.obj

double.lib : $(Dep_doubledlib)
  $(TLIB) $< $(IDE_BFLAGS) $(BEAT_doubledlib) @&&|
 -+setprbor.obj &
-+unity.obj &
-+yn.obj &
-+zeta.obj &
-+zetac.obj &
-+stdtr.obj &
-+struve.obj &
-+tan.obj &
-+tandg.obj &
-+tanh.obj &
-+shichi.obj &
-+sici.obj &
-+simpsn.obj &
-+simq.obj &
-+sin.obj &
-+sincos.obj &
-+sindg.obj &
-+sinh.obj &
-+spence.obj &
-+polrt.obj &
-+polylog.obj &
-+polyn.obj &
-+pow.obj &
-+powi.obj &
-+psi.obj &
-+revers.obj &
-+rgamma.obj &
-+round.obj &
-+pdtr.obj &
-+planck.obj &
-+polevl.obj &
-+polmisc.obj &
-+nbdtr.obj &
-+ndtr.obj &
-+ndtri.obj &
-+mtherr.obj &
-+kolmogor.obj &
-+levnsn.obj &
-+log.obj &
-+log10.obj &
-+log2.obj &
-+lrand.obj &
-+lsqrt.obj &
-+incbet.obj &
-+incbi.obj &
-+isnan.obj &
-+iv.obj &
-+j0.obj &
-+j1.obj &
-+jn.obj &
-+jv.obj &
-+k0.obj &
-+k1.obj &
-+kn.obj &
-+fresnl.obj &
-+gamma.obj &
-+gdtr.obj &
-+gels.obj &
-+hyp2f1.obj &
-+hyperg.obj &
-+i0.obj &
-+i1.obj &
-+igam.obj &
-+igami.obj &
-+exp.obj &
-+exp10.obj &
-+exp2.obj &
-+expn.obj &
-+fabs.obj &
-+fac.obj &
-+fdtr.obj &
-+fftr.obj &
-+floor.obj &
-+euclid.obj &
-+ei.obj &
-+eigens.obj &
-+ellie.obj &
-+ellik.obj &
-+ellpe.obj &
-+ellpj.obj &
-+ellpk.obj &
-+drand.obj &
-+const.obj &
-+cosh.obj &
-+cpmul.obj &
-+dawsn.obj &
-+clog.obj &
-+cmplx.obj &
-+chbevl.obj &
-+chdtr.obj &
-+acosh.obj &
-+airy.obj &
-+arcdot.obj &
-+asin.obj &
-+asinh.obj &
-+atan.obj &
-+atanh.obj &
-+bdtr.obj &
-+beta.obj &
-+btdtr.obj &
-+cbrt.obj
|

setprbor.obj :  double\setprbor.asm
  $(TASM) @&&|
 /ml C:\MATH\double\setprbor.asm ,setprbor.obj
|

unity.obj :  double\unity.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\unity.c
|

yn.obj :  double\yn.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\yn.c
|

zeta.obj :  double\zeta.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\zeta.c
|

zetac.obj :  double\zetac.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\zetac.c
|

stdtr.obj :  double\stdtr.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\stdtr.c
|

struve.obj :  double\struve.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\struve.c
|

tan.obj :  double\tan.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\tan.c
|

tandg.obj :  double\tandg.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\tandg.c
|

tanh.obj :  double\tanh.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\tanh.c
|

shichi.obj :  double\shichi.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\shichi.c
|

sici.obj :  double\sici.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\sici.c
|

simpsn.obj :  double\simpsn.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\simpsn.c
|

simq.obj :  double\simq.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\simq.c
|

sin.obj :  double\sin.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\sin.c
|

sincos.obj :  double\sincos.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\sincos.c
|

sindg.obj :  double\sindg.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\sindg.c
|

sinh.obj :  double\sinh.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\sinh.c
|

spence.obj :  double\spence.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\spence.c
|

polrt.obj :  double\polrt.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\polrt.c
|

polylog.obj :  double\polylog.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\polylog.c
|

polyn.obj :  double\polyn.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\polyn.c
|

pow.obj :  double\pow.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\pow.c
|

powi.obj :  double\powi.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\powi.c
|

psi.obj :  double\psi.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\psi.c
|

revers.obj :  double\revers.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\revers.c
|

rgamma.obj :  double\rgamma.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\rgamma.c
|

round.obj :  double\round.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\round.c
|

pdtr.obj :  double\pdtr.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\pdtr.c
|

planck.obj :  double\planck.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\planck.c
|

polevl.obj :  double\polevl.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\polevl.c
|

polmisc.obj :  double\polmisc.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\polmisc.c
|

nbdtr.obj :  double\nbdtr.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\nbdtr.c
|

ndtr.obj :  double\ndtr.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\ndtr.c
|

ndtri.obj :  double\ndtri.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\ndtri.c
|

mtherr.obj :  double\mtherr.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\mtherr.c
|

kolmogor.obj :  double\kolmogor.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\kolmogor.c
|

levnsn.obj :  double\levnsn.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\levnsn.c
|

## The following section is generated by duplicate target log.obj.

log.obj :  double\log.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\log.c
|

## The above section is generated by duplicate target log.obj.

log10.obj :  double\log10.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\log10.c
|

log2.obj :  double\log2.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\log2.c
|

lrand.obj :  double\lrand.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\lrand.c
|

lsqrt.obj :  double\lsqrt.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\lsqrt.c
|

incbet.obj :  double\incbet.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\incbet.c
|

incbi.obj :  double\incbi.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\incbi.c
|

isnan.obj :  double\isnan.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\isnan.c
|

iv.obj :  double\iv.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\iv.c
|

j0.obj :  double\j0.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\j0.c
|

j1.obj :  double\j1.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\j1.c
|

jn.obj :  double\jn.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\jn.c
|

jv.obj :  double\jv.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\jv.c
|

k0.obj :  double\k0.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\k0.c
|

k1.obj :  double\k1.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\k1.c
|

kn.obj :  double\kn.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\kn.c
|

fresnl.obj :  double\fresnl.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\fresnl.c
|

## The following section is generated by duplicate target gamma.obj.

gamma.obj :  double\gamma.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\gamma.c
|

## The above section is generated by duplicate target gamma.obj.

gdtr.obj :  double\gdtr.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\gdtr.c
|

gels.obj :  double\gels.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\gels.c
|

hyp2f1.obj :  double\hyp2f1.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\hyp2f1.c
|

hyperg.obj :  double\hyperg.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\hyperg.c
|

i0.obj :  double\i0.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\i0.c
|

i1.obj :  double\i1.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\i1.c
|

igam.obj :  double\igam.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\igam.c
|

igami.obj :  double\igami.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\igami.c
|

exp.obj :  double\exp.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\exp.c
|

exp10.obj :  double\exp10.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\exp10.c
|

exp2.obj :  double\exp2.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\exp2.c
|

expn.obj :  double\expn.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\expn.c
|

fabs.obj :  double\fabs.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\fabs.c
|

fac.obj :  double\fac.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\fac.c
|

fdtr.obj :  double\fdtr.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\fdtr.c
|

fftr.obj :  double\fftr.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\fftr.c
|

floor.obj :  double\floor.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\floor.c
|

euclid.obj :  double\euclid.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\euclid.c
|

ei.obj :  double\ei.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\ei.c
|

eigens.obj :  double\eigens.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\eigens.c
|

ellie.obj :  double\ellie.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\ellie.c
|

ellik.obj :  double\ellik.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\ellik.c
|

ellpe.obj :  double\ellpe.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\ellpe.c
|

ellpj.obj :  double\ellpj.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\ellpj.c
|

ellpk.obj :  double\ellpk.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\ellpk.c
|

drand.obj :  double\drand.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\drand.c
|

const.obj :  double\const.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\const.c
|

cosh.obj :  double\cosh.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\cosh.c
|

cpmul.obj :  double\cpmul.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\cpmul.c
|

dawsn.obj :  double\dawsn.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\dawsn.c
|

clog.obj :  double\clog.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\clog.c
|

cmplx.obj :  double\cmplx.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\cmplx.c
|

chbevl.obj :  double\chbevl.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\chbevl.c
|

chdtr.obj :  double\chdtr.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\chdtr.c
|

acosh.obj :  double\acosh.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\acosh.c
|

airy.obj :  double\airy.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\airy.c
|

arcdot.obj :  double\arcdot.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\arcdot.c
|

asin.obj :  double\asin.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\asin.c
|

asinh.obj :  double\asinh.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\asinh.c
|

atan.obj :  double\atan.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\atan.c
|

atanh.obj :  double\atanh.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\atanh.c
|

## The following section is generated by duplicate target bdtr.obj.

bdtr.obj :  double\bdtr.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\bdtr.c
|

## The above section is generated by duplicate target bdtr.obj.

beta.obj :  double\beta.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\beta.c
|

btdtr.obj :  double\btdtr.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\btdtr.c
|

cbrt.obj :  double\cbrt.c
  $(BCCDOS) -P- -c @&&|
 $(CEAT_doubledlib) $(CNIEAT_doubledlib) -o$@ double\cbrt.c
|

# Compiler configuration file
BccDos.cfg : 
   Copy &&|
-W-
-R
-v
-vi
-H
-H=double.csm
-ml
-f
| $@


