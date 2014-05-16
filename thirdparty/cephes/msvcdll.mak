# Microsoft Developer Studio Generated NMAKE File, Based on libmd.dsp
!IF "$(CFG)" == ""
CFG=libmd - Win32 Debug
!MESSAGE No configuration specified. Defaulting to libmd - Win32 Debug.
!ENDIF 

!IF "$(CFG)" != "libmd - Win32 Release" && "$(CFG)" != "libmd - Win32 Debug"
!MESSAGE Invalid configuration "$(CFG)" specified.
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "libmd.mak" CFG="libmd - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "libmd - Win32 Release" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "libmd - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 

CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "libmd - Win32 Release"

OUTDIR=.\Release
INTDIR=.\Release
# Begin Custom Macros
OutDir=.\Release
# End Custom Macros

ALL : "$(OUTDIR)\libmd.dll"


CLEAN :
	-@erase "$(INTDIR)\acosh.obj"
	-@erase "$(INTDIR)\airy.obj"
	-@erase "$(INTDIR)\asin.obj"
	-@erase "$(INTDIR)\asinh.obj"
	-@erase "$(INTDIR)\atan.obj"
	-@erase "$(INTDIR)\atanh.obj"
	-@erase "$(INTDIR)\bdtr.obj"
	-@erase "$(INTDIR)\beta.obj"
	-@erase "$(INTDIR)\btdtr.obj"
	-@erase "$(INTDIR)\cbrt.obj"
	-@erase "$(INTDIR)\chbevl.obj"
	-@erase "$(INTDIR)\chdtr.obj"
	-@erase "$(INTDIR)\clog.obj"
	-@erase "$(INTDIR)\cmplx.obj"
	-@erase "$(INTDIR)\const.obj"
	-@erase "$(INTDIR)\cosh.obj"
	-@erase "$(INTDIR)\cpmul.obj"
	-@erase "$(INTDIR)\dawsn.obj"
	-@erase "$(INTDIR)\drand.obj"
	-@erase "$(INTDIR)\ei.obj"
	-@erase "$(INTDIR)\eigens.obj"
	-@erase "$(INTDIR)\ellie.obj"
	-@erase "$(INTDIR)\ellik.obj"
	-@erase "$(INTDIR)\ellpe.obj"
	-@erase "$(INTDIR)\ellpj.obj"
	-@erase "$(INTDIR)\ellpk.obj"
	-@erase "$(INTDIR)\euclid.obj"
	-@erase "$(INTDIR)\exp.obj"
	-@erase "$(INTDIR)\exp10.obj"
	-@erase "$(INTDIR)\exp2.obj"
	-@erase "$(INTDIR)\expn.obj"
	-@erase "$(INTDIR)\expx2.obj"
	-@erase "$(INTDIR)\fabs.obj"
	-@erase "$(INTDIR)\fac.obj"
	-@erase "$(INTDIR)\fdtr.obj"
	-@erase "$(INTDIR)\fftr.obj"
	-@erase "$(INTDIR)\fresnl.obj"
	-@erase "$(INTDIR)\gamma.obj"
	-@erase "$(INTDIR)\gdtr.obj"
	-@erase "$(INTDIR)\gels.obj"
	-@erase "$(INTDIR)\hyp2f1.obj"
	-@erase "$(INTDIR)\hyperg.obj"
	-@erase "$(INTDIR)\i0.obj"
	-@erase "$(INTDIR)\i1.obj"
	-@erase "$(INTDIR)\igam.obj"
	-@erase "$(INTDIR)\igami.obj"
	-@erase "$(INTDIR)\incbet.obj"
	-@erase "$(INTDIR)\incbi.obj"
	-@erase "$(INTDIR)\isnan.obj"
	-@erase "$(INTDIR)\iv.obj"
	-@erase "$(INTDIR)\j0.obj"
	-@erase "$(INTDIR)\j1.obj"
	-@erase "$(INTDIR)\jn.obj"
	-@erase "$(INTDIR)\jv.obj"
	-@erase "$(INTDIR)\k0.obj"
	-@erase "$(INTDIR)\k1.obj"
	-@erase "$(INTDIR)\kn.obj"
	-@erase "$(INTDIR)\kolmogorov.obj"
	-@erase "$(INTDIR)\log.obj"
	-@erase "$(INTDIR)\log10.obj"
	-@erase "$(INTDIR)\log2.obj"
	-@erase "$(INTDIR)\lrand.obj"
	-@erase "$(INTDIR)\lsqrt.obj"
	-@erase "$(INTDIR)\mtherr.obj"
	-@erase "$(INTDIR)\nbdtr.obj"
	-@erase "$(INTDIR)\ndtr.obj"
	-@erase "$(INTDIR)\ndtri.obj"
	-@erase "$(INTDIR)\pdtr.obj"
	-@erase "$(INTDIR)\planck.obj"
	-@erase "$(INTDIR)\polevl.obj"
	-@erase "$(INTDIR)\polmisc.obj"
	-@erase "$(INTDIR)\polrt.obj"
	-@erase "$(INTDIR)\polylog.obj"
	-@erase "$(INTDIR)\polyn.obj"
	-@erase "$(INTDIR)\pow.obj"
	-@erase "$(INTDIR)\powi.obj"
	-@erase "$(INTDIR)\psi.obj"
	-@erase "$(INTDIR)\revers.obj"
	-@erase "$(INTDIR)\rgamma.obj"
	-@erase "$(INTDIR)\round.obj"
	-@erase "$(INTDIR)\setpmsvc.obj"
	-@erase "$(INTDIR)\shichi.obj"
	-@erase "$(INTDIR)\sici.obj"
	-@erase "$(INTDIR)\simpsn.obj"
	-@erase "$(INTDIR)\simq.obj"
	-@erase "$(INTDIR)\sin.obj"
	-@erase "$(INTDIR)\sinh.obj"
	-@erase "$(INTDIR)\spence.obj"
	-@erase "$(INTDIR)\sqrt.obj"
	-@erase "$(INTDIR)\stdtr.obj"
	-@erase "$(INTDIR)\struve.obj"
	-@erase "$(INTDIR)\tan.obj"
	-@erase "$(INTDIR)\tanh.obj"
	-@erase "$(INTDIR)\unity.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\yn.obj"
	-@erase "$(INTDIR)\zeta.obj"
	-@erase "$(INTDIR)\zetac.obj"
	-@erase "$(OUTDIR)\libmd.dll"
	-@erase "$(OUTDIR)\libmd.exp"
	-@erase "$(OUTDIR)\libmd.lib"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP_PROJ=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 
MTL_PROJ=/nologo /D "NDEBUG" /mktyplib203 /win32 
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\libmd.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /incremental:no /pdb:"$(OUTDIR)\libmd.pdb" /machine:I386 /out:"$(OUTDIR)\libmd.dll" /implib:"$(OUTDIR)\libmd.lib" 
LINK32_OBJS= \
	"$(INTDIR)\acosh.obj" \
	"$(INTDIR)\airy.obj" \
	"$(INTDIR)\asin.obj" \
	"$(INTDIR)\asinh.obj" \
	"$(INTDIR)\atan.obj" \
	"$(INTDIR)\atanh.obj" \
	"$(INTDIR)\bdtr.obj" \
	"$(INTDIR)\beta.obj" \
	"$(INTDIR)\btdtr.obj" \
	"$(INTDIR)\cbrt.obj" \
	"$(INTDIR)\chbevl.obj" \
	"$(INTDIR)\chdtr.obj" \
	"$(INTDIR)\clog.obj" \
	"$(INTDIR)\cmplx.obj" \
	"$(INTDIR)\const.obj" \
	"$(INTDIR)\cosh.obj" \
	"$(INTDIR)\cpmul.obj" \
	"$(INTDIR)\dawsn.obj" \
	"$(INTDIR)\drand.obj" \
	"$(INTDIR)\ei.obj" \
	"$(INTDIR)\eigens.obj" \
	"$(INTDIR)\ellie.obj" \
	"$(INTDIR)\ellik.obj" \
	"$(INTDIR)\ellpe.obj" \
	"$(INTDIR)\ellpj.obj" \
	"$(INTDIR)\ellpk.obj" \
	"$(INTDIR)\euclid.obj" \
	"$(INTDIR)\exp.obj" \
	"$(INTDIR)\exp10.obj" \
	"$(INTDIR)\exp2.obj" \
	"$(INTDIR)\expn.obj" \
	"$(INTDIR)\expx2.obj" \
	"$(INTDIR)\fabs.obj" \
	"$(INTDIR)\fac.obj" \
	"$(INTDIR)\fdtr.obj" \
	"$(INTDIR)\fftr.obj" \
	"$(INTDIR)\fresnl.obj" \
	"$(INTDIR)\gamma.obj" \
	"$(INTDIR)\gdtr.obj" \
	"$(INTDIR)\gels.obj" \
	"$(INTDIR)\hyp2f1.obj" \
	"$(INTDIR)\hyperg.obj" \
	"$(INTDIR)\i0.obj" \
	"$(INTDIR)\igam.obj" \
	"$(INTDIR)\igami.obj" \
	"$(INTDIR)\i1.obj" \
	"$(INTDIR)\incbet.obj" \
	"$(INTDIR)\incbi.obj" \
	"$(INTDIR)\isnan.obj" \
	"$(INTDIR)\iv.obj" \
	"$(INTDIR)\j0.obj" \
	"$(INTDIR)\j1.obj" \
	"$(INTDIR)\jn.obj" \
	"$(INTDIR)\jv.obj" \
	"$(INTDIR)\k0.obj" \
	"$(INTDIR)\k1.obj" \
	"$(INTDIR)\kn.obj" \
	"$(INTDIR)\kolmogorov.obj" \
	"$(INTDIR)\log.obj" \
	"$(INTDIR)\log10.obj" \
	"$(INTDIR)\log2.obj" \
	"$(INTDIR)\lrand.obj" \
	"$(INTDIR)\lsqrt.obj" \
	"$(INTDIR)\mtherr.obj" \
	"$(INTDIR)\nbdtr.obj" \
	"$(INTDIR)\ndtr.obj" \
	"$(INTDIR)\ndtri.obj" \
	"$(INTDIR)\pdtr.obj" \
	"$(INTDIR)\planck.obj" \
	"$(INTDIR)\polevl.obj" \
	"$(INTDIR)\polmisc.obj" \
	"$(INTDIR)\polrt.obj" \
	"$(INTDIR)\polylog.obj" \
	"$(INTDIR)\polyn.obj" \
	"$(INTDIR)\pow.obj" \
	"$(INTDIR)\powi.obj" \
	"$(INTDIR)\psi.obj" \
	"$(INTDIR)\revers.obj" \
	"$(INTDIR)\rgamma.obj" \
	"$(INTDIR)\round.obj" \
	"$(INTDIR)\shichi.obj" \
	"$(INTDIR)\sici.obj" \
	"$(INTDIR)\simpsn.obj" \
	"$(INTDIR)\simq.obj" \
	"$(INTDIR)\sin.obj" \
	"$(INTDIR)\sinh.obj" \
	"$(INTDIR)\spence.obj" \
	"$(INTDIR)\sqrt.obj" \
	"$(INTDIR)\stdtr.obj" \
	"$(INTDIR)\struve.obj" \
	"$(INTDIR)\tan.obj" \
	"$(INTDIR)\tanh.obj" \
	"$(INTDIR)\unity.obj" \
	"$(INTDIR)\yn.obj" \
	"$(INTDIR)\zeta.obj" \
	"$(INTDIR)\zetac.obj" \
	"$(INTDIR)\setpmsvc.obj"

"$(OUTDIR)\libmd.dll" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

OUTDIR=.\Debug
INTDIR=.\Debug
# Begin Custom Macros
OutDir=.\Debug
# End Custom Macros

ALL : "$(OUTDIR)\libmd.dll"


CLEAN :
	-@erase "$(INTDIR)\acosh.obj"
	-@erase "$(INTDIR)\airy.obj"
	-@erase "$(INTDIR)\asin.obj"
	-@erase "$(INTDIR)\asinh.obj"
	-@erase "$(INTDIR)\atan.obj"
	-@erase "$(INTDIR)\atanh.obj"
	-@erase "$(INTDIR)\bdtr.obj"
	-@erase "$(INTDIR)\beta.obj"
	-@erase "$(INTDIR)\btdtr.obj"
	-@erase "$(INTDIR)\cbrt.obj"
	-@erase "$(INTDIR)\chbevl.obj"
	-@erase "$(INTDIR)\chdtr.obj"
	-@erase "$(INTDIR)\clog.obj"
	-@erase "$(INTDIR)\cmplx.obj"
	-@erase "$(INTDIR)\const.obj"
	-@erase "$(INTDIR)\cosh.obj"
	-@erase "$(INTDIR)\cpmul.obj"
	-@erase "$(INTDIR)\dawsn.obj"
	-@erase "$(INTDIR)\drand.obj"
	-@erase "$(INTDIR)\ei.obj"
	-@erase "$(INTDIR)\eigens.obj"
	-@erase "$(INTDIR)\ellie.obj"
	-@erase "$(INTDIR)\ellik.obj"
	-@erase "$(INTDIR)\ellpe.obj"
	-@erase "$(INTDIR)\ellpj.obj"
	-@erase "$(INTDIR)\ellpk.obj"
	-@erase "$(INTDIR)\euclid.obj"
	-@erase "$(INTDIR)\exp.obj"
	-@erase "$(INTDIR)\exp10.obj"
	-@erase "$(INTDIR)\exp2.obj"
	-@erase "$(INTDIR)\expn.obj"
	-@erase "$(INTDIR)\expx2.obj"
	-@erase "$(INTDIR)\fabs.obj"
	-@erase "$(INTDIR)\fac.obj"
	-@erase "$(INTDIR)\fdtr.obj"
	-@erase "$(INTDIR)\fftr.obj"
	-@erase "$(INTDIR)\fresnl.obj"
	-@erase "$(INTDIR)\gamma.obj"
	-@erase "$(INTDIR)\gdtr.obj"
	-@erase "$(INTDIR)\gels.obj"
	-@erase "$(INTDIR)\hyp2f1.obj"
	-@erase "$(INTDIR)\hyperg.obj"
	-@erase "$(INTDIR)\i0.obj"
	-@erase "$(INTDIR)\i1.obj"
	-@erase "$(INTDIR)\igam.obj"
	-@erase "$(INTDIR)\igami.obj"
	-@erase "$(INTDIR)\incbet.obj"
	-@erase "$(INTDIR)\incbi.obj"
	-@erase "$(INTDIR)\isnan.obj"
	-@erase "$(INTDIR)\iv.obj"
	-@erase "$(INTDIR)\j0.obj"
	-@erase "$(INTDIR)\j1.obj"
	-@erase "$(INTDIR)\jn.obj"
	-@erase "$(INTDIR)\jv.obj"
	-@erase "$(INTDIR)\k0.obj"
	-@erase "$(INTDIR)\k1.obj"
	-@erase "$(INTDIR)\kn.obj"
	-@erase "$(INTDIR)\kolmogorov.obj"
	-@erase "$(INTDIR)\log.obj"
	-@erase "$(INTDIR)\log10.obj"
	-@erase "$(INTDIR)\log2.obj"
	-@erase "$(INTDIR)\lrand.obj"
	-@erase "$(INTDIR)\lsqrt.obj"
	-@erase "$(INTDIR)\mtherr.obj"
	-@erase "$(INTDIR)\nbdtr.obj"
	-@erase "$(INTDIR)\ndtr.obj"
	-@erase "$(INTDIR)\ndtri.obj"
	-@erase "$(INTDIR)\pdtr.obj"
	-@erase "$(INTDIR)\planck.obj"
	-@erase "$(INTDIR)\polevl.obj"
	-@erase "$(INTDIR)\polmisc.obj"
	-@erase "$(INTDIR)\polrt.obj"
	-@erase "$(INTDIR)\polylog.obj"
	-@erase "$(INTDIR)\polyn.obj"
	-@erase "$(INTDIR)\pow.obj"
	-@erase "$(INTDIR)\powi.obj"
	-@erase "$(INTDIR)\psi.obj"
	-@erase "$(INTDIR)\revers.obj"
	-@erase "$(INTDIR)\rgamma.obj"
	-@erase "$(INTDIR)\round.obj"
	-@erase "$(INTDIR)\setpmsvc.obj"
	-@erase "$(INTDIR)\shichi.obj"
	-@erase "$(INTDIR)\sici.obj"
	-@erase "$(INTDIR)\simpsn.obj"
	-@erase "$(INTDIR)\simq.obj"
	-@erase "$(INTDIR)\sin.obj"
	-@erase "$(INTDIR)\sinh.obj"
	-@erase "$(INTDIR)\spence.obj"
	-@erase "$(INTDIR)\sqrt.obj"
	-@erase "$(INTDIR)\stdtr.obj"
	-@erase "$(INTDIR)\struve.obj"
	-@erase "$(INTDIR)\tan.obj"
	-@erase "$(INTDIR)\tanh.obj"
	-@erase "$(INTDIR)\unity.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(INTDIR)\yn.obj"
	-@erase "$(INTDIR)\zeta.obj"
	-@erase "$(INTDIR)\zetac.obj"
	-@erase "$(OUTDIR)\libmd.dll"
	-@erase "$(OUTDIR)\libmd.exp"
	-@erase "$(OUTDIR)\libmd.ilk"
	-@erase "$(OUTDIR)\libmd.lib"
	-@erase "$(OUTDIR)\libmd.pdb"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP_PROJ=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 
MTL_PROJ=/nologo /D "_DEBUG" /mktyplib203 /win32 
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\libmd.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /incremental:yes /pdb:"$(OUTDIR)\libmd.pdb" /debug /machine:I386 /out:"$(OUTDIR)\libmd.dll" /implib:"$(OUTDIR)\libmd.lib" /pdbtype:sept 
LINK32_OBJS= \
	"$(INTDIR)\acosh.obj" \
	"$(INTDIR)\airy.obj" \
	"$(INTDIR)\asin.obj" \
	"$(INTDIR)\asinh.obj" \
	"$(INTDIR)\atan.obj" \
	"$(INTDIR)\atanh.obj" \
	"$(INTDIR)\bdtr.obj" \
	"$(INTDIR)\beta.obj" \
	"$(INTDIR)\btdtr.obj" \
	"$(INTDIR)\cbrt.obj" \
	"$(INTDIR)\chbevl.obj" \
	"$(INTDIR)\chdtr.obj" \
	"$(INTDIR)\clog.obj" \
	"$(INTDIR)\cmplx.obj" \
	"$(INTDIR)\const.obj" \
	"$(INTDIR)\cosh.obj" \
	"$(INTDIR)\cpmul.obj" \
	"$(INTDIR)\dawsn.obj" \
	"$(INTDIR)\drand.obj" \
	"$(INTDIR)\ei.obj" \
	"$(INTDIR)\eigens.obj" \
	"$(INTDIR)\ellie.obj" \
	"$(INTDIR)\ellik.obj" \
	"$(INTDIR)\ellpe.obj" \
	"$(INTDIR)\ellpj.obj" \
	"$(INTDIR)\ellpk.obj" \
	"$(INTDIR)\euclid.obj" \
	"$(INTDIR)\exp.obj" \
	"$(INTDIR)\exp10.obj" \
	"$(INTDIR)\exp2.obj" \
	"$(INTDIR)\expn.obj" \
	"$(INTDIR)\expx2.obj" \
	"$(INTDIR)\fabs.obj" \
	"$(INTDIR)\fac.obj" \
	"$(INTDIR)\fdtr.obj" \
	"$(INTDIR)\fftr.obj" \
	"$(INTDIR)\fresnl.obj" \
	"$(INTDIR)\gamma.obj" \
	"$(INTDIR)\gdtr.obj" \
	"$(INTDIR)\gels.obj" \
	"$(INTDIR)\hyp2f1.obj" \
	"$(INTDIR)\hyperg.obj" \
	"$(INTDIR)\i0.obj" \
	"$(INTDIR)\igam.obj" \
	"$(INTDIR)\igami.obj" \
	"$(INTDIR)\i1.obj" \
	"$(INTDIR)\incbet.obj" \
	"$(INTDIR)\incbi.obj" \
	"$(INTDIR)\isnan.obj" \
	"$(INTDIR)\iv.obj" \
	"$(INTDIR)\j0.obj" \
	"$(INTDIR)\j1.obj" \
	"$(INTDIR)\jn.obj" \
	"$(INTDIR)\jv.obj" \
	"$(INTDIR)\k0.obj" \
	"$(INTDIR)\k1.obj" \
	"$(INTDIR)\kn.obj" \
	"$(INTDIR)\kolmogorov.obj" \
	"$(INTDIR)\log.obj" \
	"$(INTDIR)\log10.obj" \
	"$(INTDIR)\log2.obj" \
	"$(INTDIR)\lrand.obj" \
	"$(INTDIR)\lsqrt.obj" \
	"$(INTDIR)\mtherr.obj" \
	"$(INTDIR)\nbdtr.obj" \
	"$(INTDIR)\ndtr.obj" \
	"$(INTDIR)\ndtri.obj" \
	"$(INTDIR)\pdtr.obj" \
	"$(INTDIR)\planck.obj" \
	"$(INTDIR)\polevl.obj" \
	"$(INTDIR)\polmisc.obj" \
	"$(INTDIR)\polrt.obj" \
	"$(INTDIR)\polylog.obj" \
	"$(INTDIR)\polyn.obj" \
	"$(INTDIR)\pow.obj" \
	"$(INTDIR)\powi.obj" \
	"$(INTDIR)\psi.obj" \
	"$(INTDIR)\revers.obj" \
	"$(INTDIR)\rgamma.obj" \
	"$(INTDIR)\round.obj" \
	"$(INTDIR)\shichi.obj" \
	"$(INTDIR)\sici.obj" \
	"$(INTDIR)\simpsn.obj" \
	"$(INTDIR)\simq.obj" \
	"$(INTDIR)\sin.obj" \
	"$(INTDIR)\sinh.obj" \
	"$(INTDIR)\spence.obj" \
	"$(INTDIR)\sqrt.obj" \
	"$(INTDIR)\stdtr.obj" \
	"$(INTDIR)\struve.obj" \
	"$(INTDIR)\tan.obj" \
	"$(INTDIR)\tanh.obj" \
	"$(INTDIR)\unity.obj" \
	"$(INTDIR)\yn.obj" \
	"$(INTDIR)\zeta.obj" \
	"$(INTDIR)\zetac.obj" \
	"$(INTDIR)\setpmsvc.obj"

"$(OUTDIR)\libmd.dll" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ENDIF 

.c{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.c{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<


!IF "$(NO_EXTERNAL_DEPS)" != "1"
!IF EXISTS("libmd.dep")
!INCLUDE "libmd.dep"
!ELSE 
!MESSAGE Warning: cannot find "libmd.dep"
!ENDIF 
!ENDIF 


!IF "$(CFG)" == "libmd - Win32 Release" || "$(CFG)" == "libmd - Win32 Debug"
SOURCE=.\acosh.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\acosh.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\acosh.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\airy.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\airy.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\airy.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\asin.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\asin.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\asin.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\asinh.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\asinh.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\asinh.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\atan.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\atan.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\atan.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\atanh.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\atanh.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\atanh.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\bdtr.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\bdtr.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\bdtr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\beta.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\beta.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\beta.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\btdtr.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\btdtr.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\btdtr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\cbrt.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\cbrt.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\cbrt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\chbevl.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\chbevl.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\chbevl.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\chdtr.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\chdtr.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\chdtr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\clog.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\clog.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\clog.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\cmplx.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\cmplx.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\cmplx.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\const.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\const.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\const.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\cosh.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\cosh.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\cosh.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\cpmul.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\cpmul.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\cpmul.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\dawsn.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\dawsn.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\dawsn.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\drand.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\drand.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\drand.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\ei.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ei.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\ei.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\eigens.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\eigens.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\eigens.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\ellie.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ellie.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\ellie.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\ellik.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ellik.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\ellik.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\ellpe.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ellpe.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\ellpe.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\ellpj.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ellpj.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\ellpj.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\ellpk.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ellpk.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\ellpk.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\euclid.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\euclid.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\euclid.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\exp.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\exp.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\exp.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\exp10.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\exp10.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\exp10.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\exp2.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\exp2.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\exp2.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\expn.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\expn.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\expn.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\expx2.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\expx2.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\expx2.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\fabs.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\fabs.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\fabs.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\fac.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\fac.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\fac.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\fdtr.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\fdtr.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\fdtr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\fftr.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\fftr.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\fftr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\fresnl.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\fresnl.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\fresnl.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\gamma.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\gamma.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\gamma.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\gdtr.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\gdtr.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\gdtr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\gels.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\gels.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\gels.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\hyp2f1.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\hyp2f1.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\hyp2f1.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\hyperg.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\hyperg.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\hyperg.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\i0.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\i0.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\i0.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\i1.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\i1.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\i1.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\igam.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\igam.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\igam.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\igami.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\igami.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\igami.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\incbet.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\incbet.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\incbet.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\incbi.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\incbi.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\incbi.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\isnan.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\isnan.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\isnan.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\iv.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\iv.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\iv.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\j0.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\j0.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\j0.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\j1.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\j1.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\j1.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\jn.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\jn.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\jn.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\jv.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\jv.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\jv.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\k0.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\k0.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\k0.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\k1.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\k1.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\k1.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\kn.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\kn.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\kn.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\kolmogorov.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\kolmogorov.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\kolmogorov.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\log.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\log.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\log.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\log10.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\log10.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\log10.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\log2.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\log2.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\log2.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\lrand.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\lrand.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\lrand.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\lsqrt.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\lsqrt.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\lsqrt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\mtherr.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\mtherr.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\mtherr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\nbdtr.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\nbdtr.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\nbdtr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\ndtr.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ndtr.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\ndtr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\ndtri.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ndtri.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\ndtri.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\pdtr.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\pdtr.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\pdtr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\planck.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\planck.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\planck.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\polevl.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\polevl.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\polevl.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\polmisc.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\polmisc.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\polmisc.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\polrt.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\polrt.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\polrt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\polylog.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\polylog.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\polylog.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\polyn.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\polyn.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\polyn.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\pow.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\pow.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\pow.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\powi.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\powi.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\powi.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\psi.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\psi.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\psi.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\revers.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\revers.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\revers.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\rgamma.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\rgamma.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\rgamma.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\round.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\round.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\round.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\setpmsvc.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\setpmsvc.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\setpmsvc.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\shichi.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\shichi.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\shichi.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\sici.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\sici.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\sici.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\simpsn.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\simpsn.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\simpsn.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\simq.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\simq.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\simq.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\sin.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\sin.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\sin.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\sinh.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\sinh.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\sinh.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\spence.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\spence.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\spence.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\sqrt.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\sqrt.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\sqrt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\stdtr.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\stdtr.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\stdtr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\struve.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\struve.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\struve.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\tan.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\tan.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\tan.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\tanh.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\tanh.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\tanh.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\unity.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\unity.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\unity.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\yn.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\yn.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\yn.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\zeta.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\zeta.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\zeta.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\zetac.c

!IF  "$(CFG)" == "libmd - Win32 Release"

CPP_SWITCHES=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fp"$(INTDIR)\libmd.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\zetac.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\libmd.pch"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "libmd - Win32 Debug"

CPP_SWITCHES=/nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LIBMD_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ  /c 

"$(INTDIR)\zetac.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 


!ENDIF 

