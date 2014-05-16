mtst.obj: mtst.c mconf.h
	cl /c mtst.c

mtst.exe: mtst.obj fti.lib
	link mtst,,,fti.lib;

