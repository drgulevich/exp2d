.PHONY : clean

CFLAGS  = -O3
#CFLAGS  = -g -Wall

# -Wl,xx,yy: pass option as an options xx,yy to the linker
# On MAC: replace -soname by -install_name 
libexp2d.so : exp2d.o
	gcc $(CFLAGS) -shared -Wl,-soname,libexp2d.so -o libexp2d.so exp2d.o -lfftw3 -lm

# -fPIC: position-independent code
exp2d.o : exp2d.c
	gcc $(CFLAGS) -c -fPIC exp2d.c -o exp2d.o $(INCLUDES)

clean :
	-rm -vf libexp2d.so exp2d.o exp2d.pyc

