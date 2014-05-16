CFLAGS = -g
eltst: eltst.o ellie.o ellie2.o
	gcc $(CFLAGS) -o eltst eltst.o ellie.o ellie2.o -lmd
